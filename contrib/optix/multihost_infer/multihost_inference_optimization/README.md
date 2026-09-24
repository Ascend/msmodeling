# vLLM 多机混布 插件

vLLM 多机混合部署插件，基于 **多节点 DP（Data Parallel）+ headless** 方案：node 主节点对外提供 API 服务并承载一个模型副本，work 节点以 `--headless` 仅作工作节点（不对外提供 API），通过 RPC 接收任务。配合 PSO 优化器自动寻优 vLLM 服务参数。

> **部署前提**：优化器进程与 node 主节点**运行在同一台机器**上。因此 node 节点的启动脚本在本机以子进程方式拉起（无需 SSH，配置中也无需填写 SSH 信息），work 节点则通过 SSH 远程拉起。健康检查走 `127.0.0.1:<port>`。

## 基础知识

### 多节点 DP + headless 方案流程图

#### 整体架构

```text
                          客户端（OpenAI 兼容请求）
                                    │
                                    │ HTTP  /v1/completions、/health
                                    ▼
┌───────────────────────────────────────────────────────────────┐
│  node 主节点（与优化器同机）                                  │
│                                                               │
│   ┌─────────────────────────┐                                 │
│   │  PSO 优化器             │  本地子进程                     │
│   │  msserviceprofiler      │──────────────┐                  │
│   │  optimizer              │  bash -l     │                  │
│   └───────────┬─────────────┘  start_node.sh                  │
│               │ 轮询 127.0.0.1:<port>/health                  │
│               │ + 一次真实推理                ▼               │
│               │            ┌──────────────────────────────┐   │
│               └───────────▶│  API Server                  │   │
│                            │  --host 0.0.0.0 --port PORT  │   │
│                            │  --api-server-count N        │   │
│                            └───────────────┬──────────────┘   │
│                                            ▼                  │
│                            ┌──────────────────────────────┐   │
│                            │  DP Coordinator              │   │
│                            │  --data-parallel-rpc-port    │◀──┼──┐
│                            └───────────────┬──────────────┘   │  │
│                                            ▼                  │  │
│                            ┌──────────────────────────────┐   │  │
│                            │  EngineCore                  │   │  │
│                            │  DP rank 0 .. local-1        │   │  │
│                            │  TP=n（机内 HCCS）           │   │  │
│                            └───────────────┬──────────────┘   │  │
│                                            ▼                  │  │
│                                   NPU 0 .. chips_per_node-1   │  │
└───────────────────────────────────────────┬───────────────────┘  │
                                            │                      │
        SSH + scp / nohup start_work_i.sh   │  HCCL 跨节点集合通信  │ ZMQ RPC
        （由优化器远程拉起）                 │  HCCL_SOCKET_IFNAME  │ 任务下发
                                            │  = nic_name          │ 结果回传
                                            ▼                      │
┌───────────────────────────────────────────────────────────────┐  │
│  work 工作节点（--headless，无 API 端口）                     │  │
│                                                               │  │
│                            ┌──────────────────────────────┐   │  │
│                            │  EngineCore                  │───┼──┘
│                            │  DP rank start-rank ..       │   │
│                            │  TP=n（机内 HCCS）           │   │
│                            └───────────────┬──────────────┘   │
│                                            ▼                  │
│                                   NPU 0 .. chips_per_node-1   │
└───────────────────────────────────────────────────────────────┘
```

节点角色一眼看清：node 承载 API Server 与 DP Coordinator，是客户端唯一入口；work 只有 EngineCore，对客户端完全透明。

#### 请求处理流程

```text
 客户端        API Server        DP Coordinator      EngineCore        EngineCore
                (node)              (node)            (node)        (work, headless)
   │               │                   │                 │                 │
   │ POST /v1/...  │                   │                 │                 │
   ├──────────────▶│                   │                 │                 │
   │               │  提交请求          │                 │                 │
   │               ├──────────────────▶│                 │                 │
   │               │                   │ 按各 DP 副本负载 │                 │
   │               │                   │ 选择目标 rank    │                 │
   │               │                   │                 │                 │
   │               │        ┌──────────┴─────── 调度到本机副本 ────────┐    │
   │               │        │          │ 本地 ZMQ 下发   │             │    │
   │               │        │          ├────────────────▶│             │    │
   │               │        │          │                 │ prefill     │    │
   │               │        │          │                 │ + decode    │    │
   │               │        │          │  输出 token 流   │ (TP 内HCCS) │    │
   │               │        │          │◀╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤             │    │
   │               │        └──────────┬───────────────────────────────┘    │
   │               │                   │                 │                 │
   │               │        ┌──────────┴─────── 调度到远端副本 ────────────┐│
   │               │        │          │ 跨节点 ZMQ RPC 下发              ││
   │               │        │          ├────────────────────────────────▶ ││
   │               │        │          │                 │      prefill  ││
   │               │        │          │                 │      + decode ││
   │               │        │          │  输出 token 流   │   (TP 内HCCS) ││
   │               │        │          │◀╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤ ││
   │               │        └──────────┬──────────────────────────────────┘│
   │               │  聚合结果          │                 │                 │
   │               │◀╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤                 │                 │
   │ SSE / JSON    │                   │                 │                 │
   │◀╌╌╌╌╌╌╌╌╌╌╌╌╌┤                   │                 │                 │
   │               │                   │                 │◀═══════════════▶│
   │               │                   │                 │  每步 decode 各 DP
   │               │                   │                 │  副本需 HCCL 同步
   │               │                   │                 │ （dummy batch 对齐）
```

关键点：work 节点从不接收 HTTP，只通过 RPC 拿任务；一个请求虽只落在单个 DP 副本上，但 decode 阶段所有副本要参与 HCCL 同步，因此任一节点未就绪整集群都无法出正确结果——这也是 `health()` 在 200 之后还要补一次真实推理的原因。

### 流程关键点说明

| 阶段       | 主节点 (Node 0)                    | 工作节点 (Work 1)                 |
| ---------- | ---------------------------------- | --------------------------------- |
| 启动命令   | 正常启动，暴露API端口              | 添加 --headless 参数              |
| 角色       | 对外服务 + 承载1个模型副本         | 仅承载1个模型副本                 |
| RPC端口    | 监听 rpc端口，等待连接             | 主动连接主节点的 rpc端口          |
| 请求处理   | 接收所有HTTP请求，可调度到任意节点 | 不接收HTTP请求，只通过RPC接收任务 |
| 对外可见性 | 客户端可见的唯一入口               | 对客户端完全透明                  |

### TP / DP 关系

| 概念                  | 作用           | 通信                  |
| --------------------- | -------------- | --------------------- |
| TP（Tensor Parallel） | 切模型到多卡   | HCCS 机内互联         |
| DP（Data Parallel）   | 加副本分担请求 | ZMQ ↔ DP Coordinator |

## 环境准备

```bash

# 安装multihost_infer插件
cd contrib/optix/multihost_infer
pip install -e .
```

---

## 多节点部署通信验证

在启动多节点 vLLM 服务之前，必须确保节点间的 NPU 通信链路正常。以下验证步骤参考 [vLLM-ascend 官方文档](https://docs.vllm.ai/projects/ascend/zh-cn/latest/installation.html#verify-multi-node-communication)。

### 物理层要求

- 所有物理机必须位于同一局域网内，且网络互通
- 所有 NPU 均通过光模块连接，且连接状态必须正常

### 单节点验证

在**每个节点**上依次执行以下命令，结果必须全部为 success 且状态为 UP。

A2 系列（8 卡）：

```bash
# 检查远端交换机端口
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done

# 获取以太网端口链路状态（UP or DOWN）
for i in {0..7}; do hccn_tool -i $i -link -g; done

# 检查网络健康状态
for i in {0..7}; do hccn_tool -i $i -net_health -g; done

# 查看网络检测 IP 配置
for i in {0..7}; do hccn_tool -i $i -netdetect -g; done

# 查看网关配置
for i in {0..7}; do hccn_tool -i $i -gateway -g; done

# 查看 NPU 网络配置
cat /etc/hccn.conf
```

A3 系列（16 卡）：

```bash
# 检查远端交换机端口
for i in {0..15}; do hccn_tool -i $i -lldp -g | grep Ifname; done

# 获取以太网端口链路状态（UP or DOWN）
for i in {0..15}; do hccn_tool -i $i -link -g; done

# 检查网络健康状态
for i in {0..15}; do hccn_tool -i $i -net_health -g; done

# 查看网络检测 IP 配置
for i in {0..15}; do hccn_tool -i $i -netdetect -g; done

# 查看网关配置
for i in {0..15}; do hccn_tool -i $i -gateway -g; done

# 查看 NPU 网络配置
cat /etc/hccn.conf
```

### 互联验证

**1. 获取 NPU IP 地址**

```bash
# A2 系列
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done

# A3 系列
for i in {0..15}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

**2. 跨节点 PING 测试**

在一个节点上对另一个节点的 NPU IP 执行 ping（将 `x.x.x.x` 替换为目标节点实际 IP）：

```bash
hccn_tool -i 0 -ping -g address x.x.x.x
```

> 所有验证通过后，方可进行多节点 vLLM 部署。如果任一步骤失败，请检查光模块连接、网络配置和防火墙规则。

---

## 配置文件

本插件涉及两个配置文件：

| 配置文件 | 位置                               | 职责                                        |
| -------- | ---------------------------------- | ------------------------------------------- |
| 集群配置 | 插件目录下`config.toml`            | 节点 SSH 连接信息、网卡名等集群拓扑         |
| 寻优配置 | `optix/config.toml` 的 `[vllm]` 段 | vllm 服务参数、寻优变量定义（target_field） |

### 集群配置（插件 config.toml）

配置 Node 主节点和 Worker 工作节点的连接信息：

```toml
# 集群配置

# 是否使用非交互 sudo -n；启用前须由用户配置远端免密 sudo 权限
docker_use_sudo = false

[vllm_mix]
chips_per_node = 16              # 每节点芯片数：A3=16, A2=8，用于校验单节点 DP 不超配
# data_parallel_rpc_port = 13389 # 可选，DP 握手 RPC 端口；不配置时每次生成脚本自动选空闲端口

# 静态环境变量：渲染进每个生成脚本的头部（export KEY=VALUE）
[vllm_mix.env]
OMP_PROC_BIND = false            # 布尔值渲染为小写 true/false
OMP_NUM_THREADS = 1
HCCL_BUFFSIZE = 1024
TASK_QUEUE_ENABLE = 1
HCCL_OP_EXPANSION_MODE = "AIV"

[[vllm_mix.node]]
host = "<node_ip>"               # 主节点 IP，例如 192.168.0.10
# nic_name = "<nic_name>"  # 可选，不配置时寻优前自动探测（detect_nic.py）

[[vllm_mix.workers]]
host = "<worker_ip>"             # 工作节点 IP，例如 192.168.0.11
ssh_port = 22
ssh_user = "<ssh_user>"
# 用户须提前配置 SSH 免密登录；此文件不配置密码、私钥或私钥口令
# nic_name = "<nic_name>"  # 可选，不配置时寻优前自动探测（detect_nic.py）
```

### 寻优配置（优化器 config.toml）

在 `/config.toml` 的 `[vllm]` 段配置服务命令和寻优变量：

```toml
[vllm]
[vllm.command]
host = "0.0.0.0"
port = "9975"
model = "/data/models/Qwen3-8B"
served_model_name = "qwen"
others = ""

# 寻优变量（config_position="env" 的变量会注入到生成的启动脚本中）
[[vllm.target_field]]
name = "MAX_NUM_BATCHED_TOKENS"
config_position = "env"
min = 8192
max = 8192          # min=max 时为常量，不参与 PSO 寻优
dtype = "int"
value = 8192

[[vllm.target_field]]
name = "MAX_NUM_SEQS"
config_position = "env"
min = 32
max = 512           # min≠max，参与 PSO 寻优
dtype = "int"
value = 64

[[vllm.target_field]]
name = "TP"
config_position = "env"
dtype = "enum"
dtype_param = [1, 2, 4, 8, 16]
value = 4

[[vllm.target_field]]
name = "DP"
config_position = "env"
dtype = "enum"
dtype_param = [1, 2, 4, 8, 16]
value = 4

[[vllm.target_field]]
name = "ASCEND_RT_VISIBLE_DEVICES"
config_position = "env"
dtype = "enum"
dtype_param = [5, 7]
value = 7

[[vllm.target_field]]
name = "CONCURRENCY"
config_position = "env"
min = 1
max = 1000
dtype = "int"
value = 100

[[vllm.target_field]]
name = "REQUESTRATE"
config_position = "env"
min = 1
max = 100
dtype = "float"
value = 1
```

寻优变量分为两类，注入目标不同：

| 变量                        | 注入目标                               | 说明                                    |
| --------------------------- | -------------------------------------- | --------------------------------------- |
| `MAX_NUM_BATCHED_TOKENS`    | vllm CLI（`--max-num-batched-tokens`） | 自动映射，无需写在 others 里            |
| `MAX_NUM_SEQS`              | vllm CLI（`--max-num-seqs`）           | 自动映射，无需写在 others 里            |
| `TP`                        | 脚本环境变量 export                    | 可在 others 中通过`$TP` 引用为 CLI 参数 |
| `DP`                        | 脚本环境变量 export                    | 同上                                    |
| `ASCEND_RT_VISIBLE_DEVICES` | 脚本环境变量 export                    | NPU 驱动通过环境变量识别                |
| `CONCURRENCY`               | benchmark 命令行                       | 仅传给 benchmark 插件，不注入 vllm 脚本 |
| `REQUESTRATE`               | benchmark 命令行                       | 仅传给 benchmark 插件，不注入 vllm 脚本 |

> 详细注入机制见下方"寻优变量注入"章节。

### SSH 认证方式

本插件**仅支持免密登录**。用户须自行保证：在优化器实际运行的机器或容器内，以运行优化器的同一操作系统用户，能够按配置的 `host`、`ssh_port`、`ssh_user` 无交互登录每个 worker 的 SSH 目标。

插件复用运行环境中的 SSH agent 或本地 SSH 密钥，不接收、保存或传递登录密码、私钥内容或私钥口令。密钥由用户在插件之外管理；使用带口令的私钥时，须提前加载到优化器可访问的 SSH agent 中。SSH 连接、文件上传和网卡探测均使用同一免密认证方式，认证失败时不会回退到密码登录。

推荐使用业界标准的 **Ed25519** 算法配置 SSH 免密。在优化器实际运行的同一用户或容器中执行，使用 `-t ed25519` 指定密钥类型（替换邮箱或机器标识及远端连接信息）：

```bash
ssh-keygen -t ed25519 -C "你的邮箱或机器标识"
ssh-copy-id -i ~/.ssh/id_ed25519.pub -p <ssh_port> <ssh_user>@<worker_ip>
```

生成时使用默认保存路径 `~/.ssh/id_ed25519`，已有密钥请勿直接覆盖。首次连接应核对远端主机指纹；复制公钥通常需要输入一次远端账户密码。私钥只保存在本地，不写入配置文件或发送给他人。

**加密私钥**：生成密钥时在 `Enter passphrase` 提示中设置非空密码短语即可加密私钥；直接回车则生成无口令密钥。如使用带口令的私钥，请先通过 `ssh-add ~/.ssh/id_ed25519` 加载到 SSH agent；无口令密钥无需此步。

SSH 客户端默认不会打印远端命令的返回码。远端可能先输出登录提示语；下面的 `ssh exit code` 为 `0`，就表示免密登录和远端命令执行成功。必须在 SSH 命令后立即读取返回码：

```bash
ssh -o BatchMode=yes -p <ssh_port> <ssh_user>@<worker_ip> true
rc=$?
printf 'ssh exit code: %s\n' "$rc"
```

配置了 `docker_container_id` 时，免密登录的目标是容器所在宿主机；直接 SSH 到容器时，则需保证容器的 SSH 端口可免密访问。

启用 `docker_use_sudo = true` 时，用户还须为插件执行的远端命令配置免密 sudo 权限。插件使用 `sudo -n`，没有相应权限会直接失败，不会请求或传递 sudo 密码。

**旧配置迁移**：删除节点配置中的 `password`（包括明文、Base64 或空字符串）以及 `ssh_password`、`private_key`、`key_filename`、`pkey`、`passphrase` 等认证字段。配置加载时会拒绝这些字段并提示迁移；SSH 配置仅保留主机、端口和用户名，免密认证由用户在运行环境中配置。

为保证多机寻优不受控制节点环境影响，插件使用 `config.toml` 中显式配置的 worker IP、端口和用户，默认不加载
控制节点的 `~/.ssh/config` 或 `/etc/ssh/ssh_config`。SSH agent、标准默认密钥及节点配置中已支持的密码认证仍可使用；
`Host` 别名、`ProxyJump` 和自定义 `IdentityFile` 不会从 SSH config 中继承。

### 节点配置字段说明

**node 主节点**（`[[vllm_mix.node]]`）与优化器同机，本地拉起，无需 SSH 信息：

| 字段       | 必填 | 说明                                                                               |
| ---------- | ---- | ---------------------------------------------------------------------------------- |
| `host`     | 是   | 主节点 IP，作为所有节点的`--data-parallel-address`                                 |
| `nic_name` | 否   | 通信网卡名（HCCL / GLOO / TP socket 绑定）；不配置时寻优前自动探测，见"网卡名探测" |

**集群级字段**（`[vllm_mix]`，作用于所有节点）：

| 字段                     | 必填 | 说明                                                                          |
| ------------------------ | ---- | ----------------------------------------------------------------------------- |
| `chips_per_node`         | 否   | 每节点芯片数（A3=16, A2=8），默认 8，用于校验单节点 DP 不超配                 |
| `data_parallel_rpc_port` | 否   | DP 握手 RPC 端口（`--data-parallel-rpc-port`）；不配置或配 0 时自动选空闲端口 |
| `env`                    | 否   | 静态环境变量表（`[vllm_mix.env]`），以 `export KEY=VALUE` 注入每个生成脚本     |

**work 工作节点**（`[[vllm_mix.workers]]`）通过 SSH 远程拉起：

| 字段                  | 必填 | 说明                                               |
| --------------------- | ---- | -------------------------------------------------- |
| `host`                | 是   | 服务器 IP，作为该节点的`LOCAL_IP`                  |
| `nic_name`            | 否   | 通信网卡名；不配置时寻优前自动探测，见"网卡名探测" |
| `ssh_port`            | 否   | SSH 端口（默认 22）                                |
| `ssh_user`            | 否   | SSH 用户名（默认 root）                            |
| `docker_container_id` | 否   | 配置后通过`docker exec / docker cp` 操作容器       |

---

## 使用

### 使用前提：确保默认参数可运行

寻优流程的第一步是**基线运行**——用配置文件中各 `target_field` 的初始 `value` 启动多机 vLLM 服务并执行一次压测。如果默认参数跑不起来，寻优在基线阶段就会失败退出。

因此，用户在启动寻优前必须确认：

1. 用配置中的默认参数（model、TP、DP、MAX_NUM_SEQS 等）在多机环境上能**正常启动 vllm serve**
2. `min`/`max` 范围只包含物理上可行的值（例如 8 卡机器不要把 TP 设为 16）
3. 优化器运行环境到各 worker 已配置并验证 SSH 免密登录，NPU 通信正常（参考上方"多节点部署通信验证"章节）

建议在寻优前先手动执行一次验证：

```bash
# 用默认参数生成脚本并启动，确认服务能正常响应
cd <插件目录>
python build_shell_scripts.py --model-name <模型路径> --port <端口> \
    --vllm-params "--served-model-name <名称> --max-num-batched-tokens <值> --max-num-seqs <值>" \
    --output-dir ./scripts --template-file template.sh --config-file config.toml
# 在 node 节点执行
bash scripts/start_node.sh
# 在 worker 节点执行
bash scripts/start_work_0.sh
# 验证服务可用
curl http://127.0.0.1:<端口>/health
curl -X POST http://127.0.0.1:<端口>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"<served_model_name>","prompt":"Hello","max_tokens":10}'
```

确认手动启动无问题后，再使用寻优命令：

```bash
msserviceprofiler optimizer -e multihost_infer -b evalscopeperf --backup
```

### 启动流程

执行 `run()` 时，先完成准备（生成脚本 → 上传 → 清理残留），再**非阻塞**地并发拉起所有节点，随后立即返回，由优化器轮询 `health()` 判定整集群就绪。

- **Stage 0 探测网卡名**：为每个未在 `config.toml` 中配置 `nic_name` 的节点探测其 IP 对应的网卡名（详见下方"网卡名探测"）。仅在第一个寻优周期执行一次，结果缓存复用
- **Stage 1 生成脚本**：依据 `config.toml` 与 `template.sh`，为主节点生成 `start_node.sh`、各 worker 生成 `start_work_{i}.sh`。寻优变量与上一步探测到的网卡名在此阶段注入（详见下方"寻优变量注入"）
- **Stage 2 上传脚本**：将 `script_template/` 全部文件 + 对应 `start_work_{i}.sh` scp 到各 worker 的 `/tmp/vllm/`
- **Stage 3 残留清理**：在各 worker 上执行清理脚本，并 kill 本机 node 残留 vllm 进程
- **Stage 4 启动 node**：本机以子进程 `bash -l start_node.sh` 拉起主节点（DP 协调者），非阻塞
- **Stage 5 启动 worker**：各 worker 通过 SSH `nohup` 后台拉起 `start_work_{i}.sh`，非阻塞
- **快速失败检查**：每个 worker 启动后等待约 3s 回探进程是否存活，起步即崩则抓日志尾部统一报错
- **健康等待**：优化器轮询 `127.0.0.1:<port>/health` 并发一次真实推理，确认整集群就绪

> **为什么先 node 再 worker**：所有节点的 `--data-parallel-address` 都指向 node，node 是 DP 握手的协调者。先拉起协调者，再让各 worker 连入 RPC 端口，rendezvous 更稳。两步均为非阻塞启动，进程在各节点上**并发运行**。
>
> **RPC 端口**：`--data-parallel-rpc-port` 不再硬编码。`build_shell_scripts.py` 在生成脚本时统一定一次值，渲染进所有节点脚本（node 监听、worker 连接，必须一致）。取值优先级为 `--dp-rpc-port` > `config.toml` 的 `[vllm_mix].data_parallel_rpc_port` > 本机在 `[20000, 32000)` 内自动挑选的空闲端口。默认不配置即每个寻优周期换一个空闲端口，避免上一周期残留进程或 `TIME_WAIT` 让固定端口起不来。
>
> **健康判定**：在多节点 DP 架构下，一个请求需所有节点协同才能完成。node 的 `/health` 在 worker 未就绪时也可能返回 200，因此 `health()` 在 200 之后会再发一次真实推理请求，成功才判定为 `running`。

### 停止流程

1. 停止本机 node 进程（按追踪到的 PID 定向 kill 及其子进程）
2. 在各 worker 上执行 `stop_vllm_process.sh` 兜底清理残留（按进程名 `pkill -9 -i -f`）

### 网卡名探测

`nic_name`（HCCL / GLOO / TP socket 绑定的通信网卡）在 **Stage 0 生成脚本之前** 确定，然后直接渲染进各节点的启动脚本。

启动脚本里**不再用 `ip -o addr show` 现场检测网卡名**——节点上可能没有安装 iproute2（`ip` 命令缺失），检测会静默失败。但每个节点一定有 Python，因此探测统一走 `detect_nic.py`。

取值优先级：

| 优先级 | 来源                                | 说明                         |
| ------ | ----------------------------------- | ---------------------------- |
| 1      | `config.toml` 中该节点的 `nic_name` | 用户手工指定，不做任何探测   |
| 2      | `detect_nic.py` 探测结果            | 未配置时按节点 IP 反查网卡名 |

探测方式（`detect_nic.py` 按顺序尝试，任一成功即返回）：

1. **pyroute2**：走 netlink，能看到一张网卡上配置的全部地址（含 secondary / alias 地址）。节点未安装 pyroute2 时跳过
2. **纯标准库**：用 `socket.if_nameindex()`（回退解析 `/proc/net/dev`）枚举网卡，再用 `ioctl(SIOCGIFADDR)` 取每张网卡的 IPv4 地址逐一比对。**只需 Python 本身**，适用于未安装任何第三方库的节点

探测在哪里执行：

- **node 主节点**：与优化器同机，直接本地调用
- **work 工作节点**：`detect_nic.py` 上传到节点的 `/tmp/ms_optix_detect_nic.py`，用**该节点自己的 Python** 执行并回读结果。依次尝试 `python3` / `python`；配了 `docker_container_id` 时优先在容器内执行（vllm 进程实际所在的网络命名空间），失败再回退宿主机

> **探测时机**：只在第一个寻优周期执行一次，结果按 host 缓存。网卡拓扑在寻优过程中不会变，后续周期直接复用，不重复付 SSH 开销。
>
> **探测失败即报错退出**：任一节点拿不到网卡名时，寻优在生成脚本阶段就明确失败，并提示在 `config.toml` 中手工配置该节点的 `nic_name`。这比让服务起来后 HCCL 绑到错误网卡、卡在通信超时更容易定位。

单独验证探测结果（不启动寻优）：

```bash
cd <插件目录>
# 探测集群内所有节点（读 config.toml），输出 "host<TAB>nic_name"
python nic_resolver.py
# 或在某个节点上直接探测单个 IP
python detect_nic.py <node_ip>
```

### 寻优变量注入

优化器的寻优变量（`config_position="env"` 的 `target_field`）在 Stage 0 生成脚本时自动注入到所有节点的启动脚本中。注入通过两条通道实现：

**1. 环境变量 export**

所有 `config_position="env"` 的寻优变量会以 `export KEY=VALUE` 的形式写入生成脚本的头部。例如：

```bash
# 寻优变量（由优化器注入）
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=64
export TP=4
export DP=4
export ASCEND_RT_VISIBLE_DEVICES=5
```

适用于不作为 vllm CLI 参数、而是被运行时环境读取的变量（如 `ASCEND_RT_VISIBLE_DEVICES` 由 NPU 驱动识别）。

> **静态环境变量**：除寻优变量外，插件 `config.toml` 的 `[vllm_mix.env]` 段也可以配置固定环境变量（如 `OMP_PROC_BIND`、`HCCL_BUFFSIZE`），以同样方式注入脚本头部。二者同名时寻优变量优先。

**2. CLI 参数（通过 `others` 字段）**

在优化器配置的 `vllm.command.others` 中可使用 `$VAR_NAME` 占位符引用寻优变量，生成脚本时会被替换为实际值，拼入 `vllm serve` 命令行。例如：

```toml
[vllm.command]
others = "--max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS --max-num-seqs $MAX_NUM_SEQS --tensor-parallel-size $TP"
```

生成后脚本中 vllm 命令尾部：

```bash
vllm serve /data/models/Qwen3-8B \
    ...
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --tensor-parallel-size 4
```

**注意事项**：

- 环境变量通过 `export` 注入脚本子进程，脚本退出后自动消亡，不会污染系统环境
- 非 vllm 参数的寻优变量（如 `CONCURRENCY`、`REQUESTRATE`）是 benchmark 参数，不会注入到 vllm 启动脚本中——它们仍由优化器直接传给 benchmark 插件
- `--data-parallel-size` 通过 `others` 传入作为寻优变量（见下方"data-parallel-size 与本地 DP 派生"），未传入时自动按节点数补入

### data-parallel-size 与本地 DP 派生

`--data-parallel-size`（全局 DP 副本总数）**由 `others` 传入作为寻优变量**，不再在 `template.sh` 中硬编码。`build_shell_scripts.py` 在生成脚本时据此派生每节点的本地 DP 参数：

| 参数                         | 派生规则                               | 说明                         |
| ---------------------------- | -------------------------------------- | ---------------------------- |
| `--data-parallel-size`       | 由`others` 传入；未传入时默认 = 节点数 | 全局 DP 副本总数             |
| `--data-parallel-size-local` | `data-parallel-size / 节点数`          | 每节点承载的 DP 副本数       |
| `--data-parallel-start-rank` | `节点序号 * data-parallel-size-local`  | 仅 worker 节点，DP 起始 rank |

在 `vllm.command.others` 中通过 `$DP` 引用寻优变量即可：

```toml
[vllm.command]
others = "--tensor-parallel-size $TP --data-parallel-size $DP"
```

**校验规则**（任一不满足则生成脚本报错退出）：

- `data-parallel-size` 必须能被节点数整除（保证各节点均分 DP 副本）
- `data-parallel-size-local * tensor-parallel-size <= chips_per_node`（防止单节点芯片超配；`chips_per_node` 在插件 `config.toml` 的 `[vllm_mix]` 段配置，A3=16、A2=8）

> **示例**：2 节点、`chips_per_node=16`、`TP=8`、`DP=4` 时，派生出 `data-parallel-size-local=2`（每节点 2 个副本，占用 2×8=16 卡，正好占满），worker 的 `data-parallel-start-rank=2`。若不传 `DP`，则默认 `DP=2`、`data-parallel-size-local=1`（每节点 1 个副本，退化为最初行为）。
