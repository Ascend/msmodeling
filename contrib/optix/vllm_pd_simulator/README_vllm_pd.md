# vLLM PD 分离部署插件

> 最后更新：2026-08-13（vllm_pd_simulator 0.1.0）

通过 SSH 远程管理 vLLM [PD](#术语表)（Prefill-Decode）分离部署集群，将多节点的启停、健康检查封装为黑盒，对外提供与单节点一致的 `run()` / `stop()` / `health()` 接口，配合 PSO/TPE 优化器自动寻优 vLLM 服务参数。

## 术语表

| 术语 | 含义 |
|------|------|
| **PD** | Prefill-Decode 分离：将预填充（Prefill）与解码（Decode）阶段部署到不同节点 |
| **TP** | Tensor Parallel，张量并行：将模型按张量维度切分到多卡执行 |
| **DP** | Data Parallel，数据并行：将请求/批次切分到多个实例执行 |
| **KV Cache** | 键值缓存：缓存历史 token 的 K/V 供解码阶段复用 |
| **Mooncake / Mooncake KV Connector** | 跨节点 KV 传输连接器（`MooncakeLayerwiseConnector` 按层传输、`MooncakeConnectorV1` 整体传输） |
| **HCCL** | Huawei Collective Communication Library，昇腾集合通信库 |
| **Ascend Direct Transport** | 昇腾设备直连传输（对应 `ascend_base_port` 配置） |
| **le_enum** | less-than-or-equal enum：约束 decode 枚举值不超过 prefill 对应值 |
| **factories** | 由 target 字段动态计算的派生类型（如 product / TP = DP） |

## 架构设计

```text
┌─────────────────────────────────────────────────────────────┐
│  优化器主进程（主节点）                                        │
│                                                              │
│  config.toml ──► 生成各节点启动脚本（本地）                    │
│                   │                                          │
│                   ├─► fabric SSH 登录各节点                    │
│                   │    ├─ 清场：kill 残留进程                  │
│                   │    ├─ 拷贝：scp 上传脚本到远端              │
│                   │    ├─ 启动：nohup 后台执行                 │
│                   │    └─ 健康检查：kill -0 + HTTP curl 轮询   │
│                   │                                          │
│                   ▼                                          │
│             全部节点 HEALTHY ──► 交还控制权给优化器             │
│                                  运行 benchmark               │
└─────────────────────────────────────────────────────────────┘
```

**核心流程（4 个 Stage）**：

| Stage | 动作 | 说明 |
|-------|------|------|
| 0 | 生成脚本 | 基于 config.toml 为每个节点生成启动脚本，保存在本地临时目录 |
| 1 | 上传脚本 | fabric SSH 登录各节点，scp 拷贝脚本到远端 `/tmp/vllm_pd_{cluster_id}/` |
| 2 | 清场 | 检查并杀掉残留的 vLLM / Proxy 进程（按 NPU ID 和端口精确匹配） |
| 3 | 启动 + 健康检查 | nohup 后台启动各节点，轮询进程存活（`kill -0`）+ HTTP 健康检查（`/health`），全部就绪后返回 |

**脚本命名规则**：

```text
<角色>-I<实例号>-R<DP排名>-T<TP大小>-D<DP大小>_<SSH_IP>_<SSH端口>_<容器ID>.sh
```

示例：

- `P-I0-R0-T4-D1_192.0.2.10_22222_none.sh` — Prefill 节点
- `D-I0-R0-T4-D1_192.0.2.11_22222_none.sh` — Decode 节点
- `proxy_192.0.2.10_22222_none.sh` — Proxy 节点

容器 ID 为 `none` 表示直连场景（无 Docker）。

## 前提条件

在开始部署前，请逐项确认以下环境已就绪：

- 各节点 SSH 服务已启动，且可从主节点连通
- 各节点已安装 NPU 驱动，CANN 环境已加载（对应环境变量 `PD_CANN_HOME`）
- 各节点已安装 vLLM 与 vllm-ascend，模型文件已就位
- 主节点已安装 Python 3.10+ 与 fabric（>= 2.0）
- 节点间 HCCL / RDMA 链路正常，KV 传输端口（`kv_port`）互通

### 硬件规格

| 项目 | 要求 |
|------|------|
| 加速卡 | 昇腾 NPU（型号与数量以实际部署为准，配置示例按每节点 4 卡） |
| CPU / 内存 | 满足 vLLM 推理与多节点寻优任务运行需求 |
| 磁盘 | 预留模型权重 + 运行日志 + 评测结果空间 |
| 网络 | 节点间 RDMA / HCCL 链路，带宽与拓扑以实际部署为准 |

### 软件环境

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（如 openEuler 22.03 / Ubuntu 20.04+） |
| Python | >= 3.10 |
| CANN | 以实际安装为准（配置默认 `cann-9.0.0`） |
| vLLM / vllm-ascend | 以实际安装为准 |
| fabric | >= 2.0 |

## 安装

```bash
# 安装优化器主体
cd MindStudio-Service-Profiler_xql
pip install -e ./ms_serviceparam_optimizer
pip install -e ./evalscopeperf --no-deps 2>/dev/null || true

# 安装 vllm_pd 插件
cd plugins/optimizer/vllm_pd_simulator
pip install -e .
```

预期回显：各 `pip install` 成功时输出 `Successfully installed ...`（如 `Successfully installed vllm_pd_simulator-0.1.0`）。

### 依赖版本

| 组件 | 版本要求 |
|------|----------|
| Python | >= 3.10 |
| fabric | >= 2.0 |
| vllm_pd_simulator | 0.1.0（本插件） |
| msmodeling | 0.2.0 |
| vLLM / vllm-ascend | 以实际安装为准 |
| CANN | 以实际安装为准（默认配置 `cann-9.0.0`） |

> 兼容性约束：插件依赖 fabric >= 2.0 与 Python >= 3.10；vLLM / vllm-ascend / CANN 版本需与实际部署环境配套，请以实际安装为准。

## Proxy 脚本说明

PD 分离部署依赖两个 Proxy 负载均衡脚本（来自 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 上游）：

| 脚本 | 用途 |
|------|------|
| `load_balance_proxy_layerwise_server_example.py` | 按层传输 KV Cache（默认） |
| `load_balance_proxy_server_example.py` | 整体传输 KV Cache |

脚本获取方式（三种，按需选用）：

- **方式一（自动，需外网）**：寻优运行时（Stage 0 生成启动脚本阶段）自动检查，本地缺失则从 vllm-ascend GitHub 仓下载；下载失败但有本地缓存时降级使用缓存。需服务器可连外网。
- **方式二（手动下载）**：运行 `scripts/pd/download_proxy_scripts.sh`，默认仅下载缺失文件；加 `--download` 强制重新下载并覆盖已有文件；加 `--check` 仅检查存在状态不下载。
- **方式三（离线上传）**：在可连外网的机器上从 [vllm-ascend/examples/disaggregated_prefill_v1/](https://github.com/vllm-project/vllm-ascend/tree/main/examples/disaggregated_prefill_v1) 下载上述脚本，上传到插件安装目录下的 `scripts/pd/` 子目录。

> **TLS 证书**：默认启用证书验证。如内网环境证书有问题，可通过 `export WGET_OPTS=--no-check-certificate` 或 `export CURL_OPTS=-k` 临时绕过 TLS 验证。

## 部署场景

插件支持两种 SSH 登录场景，通过 `docker_container_id` 字段切换：

| 场景 | SSH 目标 | `docker_container_id` | 说明 |
|------|----------|----------------------|------|
| 场景1 | Docker 容器内 | 留空（默认） | 容器有独立 SSH 服务，直接 SSH 登录到容器内部 |
| 场景2 | 宿主机 | 填容器 ID 或名称 | SSH 登录到宿主机后，通过 `docker exec` 操作容器 |

两种场景的配置完全一致，唯一区别是 `docker_container_id` 字段是否填写。

---

## 场景1：SSH 直连 Docker 容器

适用条件：Docker 容器内运行了 SSH 服务，有独立 IP 或端口，能直接 `ssh` 登录。

### 配置示例

```toml
# ======================== 主机信息 ========================
[[cluster.hosts]]
id = "rank0"
ssh_ip = "192.0.2.10"             # 容器的 SSH IP
ssh_port = 22222                    # 容器的 SSH 端口
ssh_user = "root"

service_ip = "127.0.0.1"
network_interface = "eth0"

[[cluster.hosts]]
id = "rank1"
ssh_ip = "192.0.2.11"
ssh_port = 22222
ssh_user = "root"

service_ip = "127.0.0.1"
network_interface = "eth0"

# ======================== Prefill 组 ========================
[[vllm_pd.prefill_groups]]
dp_rpc_port = 12345

[[vllm_pd.prefill_groups.nodes]]
hosts = "rank0"
gpu_ids = [0,1,2,3]
service_port = 18000
kv_port = 19000

# ======================== Decode 组 ========================
[[vllm_pd.decode_groups]]
dp_rpc_port = 12356

[[vllm_pd.decode_groups.nodes]]
hosts = "rank1"
gpu_ids = [0,1,2,3]
service_port = 18100
kv_port = 19100

# ======================== Proxy ========================
[vllm_pd.proxy]
hosts = "rank0"
service_port = 8000
```

> `docker_container_id` 留空，插件直接通过 `fabric.Connection` SSH 登录到容器内执行所有操作。

### 启动寻优

启动命令见 [使用](#使用)。

### 常见问题

**容器 sshd / 端口映射（场景专属）**

容器内需运行 sshd 且 SSH 端口已映射到宿主机；连接失败时按通用步骤排查，见 [常见问题](#常见问题)「SSH 连接失败」。

**脚本上传失败**

确认容器内 `/tmp` 目录可写，且 SSH 用户有权限创建目录。

**启动后节点一直卡在等待健康**

通用排查见 [常见问题](#常见问题)「启动后节点一直卡在等待健康」。

---

## 场景2：宿主机 SSH + Docker exec

适用条件：Docker 容器没有独立 SSH，只能通过宿主机跳转。插件 SSH 登录到宿主机后，所有命令通过 `docker exec` 在容器内执行，文件通过 `docker cp` 传入容器。

### 配置示例

```toml
# ======================== 主机信息 ========================
[[cluster.hosts]]
id = "rank0"
ssh_ip = "192.0.2.10"             # 宿主机的 SSH IP
ssh_port = 22                       # 宿主机的 SSH 端口
ssh_user = "root"

service_ip = "127.0.0.1"
network_interface = "eth0"
docker_container_id = "pd-container-0"   # 容器 ID 或名称
docker_use_sudo = false                  # docker 命令是否需要 sudo

[[cluster.hosts]]
id = "rank1"
ssh_ip = "192.0.2.11"
ssh_port = 22
ssh_user = "root"

service_ip = "127.0.0.1"
network_interface = "eth0"
docker_container_id = "pd-container-1"
docker_use_sudo = false

# ======================== Prefill 组 ========================
[[vllm_pd.prefill_groups]]
dp_rpc_port = 12345

[[vllm_pd.prefill_groups.nodes]]
hosts = "rank0"
gpu_ids = [0,1,2,3]
service_port = 18000
kv_port = 19000

# ======================== Decode 组 ========================
[[vllm_pd.decode_groups]]
dp_rpc_port = 12356

[[vllm_pd.decode_groups.nodes]]
hosts = "rank1"
gpu_ids = [0,1,2,3]
service_port = 18100
kv_port = 19100

# ======================== Proxy ========================
[vllm_pd.proxy]
hosts = "rank0"
service_port = 8000
```

> 填写 `docker_container_id` 后，插件会自动在所有 SSH 命令外层包一层 `docker exec`，文件上传走 `scp` 到宿主机 → `docker cp` 进容器两步完成。

### 启动寻优

启动命令见 [使用](#使用)。

### 常见问题

**docker exec 权限不足**

如果 docker 命令需要 sudo，设置 `docker_use_sudo = true`，并在远端配置 NOPASSWD 免密 sudo（互信后免密）。

**docker cp 失败**

确认容器正在运行：`docker ps | grep pd-container-0`。容器停止状态下 `docker cp` 会失败。

**同宿主机多容器共享端口**

多个容器在同一宿主机上时，各自映射不同的 SSH 端口即可。容器的 `service_port` / `kv_port` 在容器内部生效，宿主机层面通过端口映射区分。

**脚本已上传但容器内找不到**

插件先 `scp` 到宿主机 `/tmp/vllm_pd_{cluster_id}/`，再 `docker cp` 到容器内同路径。如果 `docker cp` 失败，检查宿主机 `/tmp` 空间和容器状态。

---

## 配置参考

### 全局字段 `[vllm_pd]`

| 字段 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `ssh_command_timeout` | 否 | `30` | 远端 SSH 短命令超时（秒） |
| `model_path` | 否 | - | 模型路径 |
| `served_model_name` | 否 | - | `--served-model-name` |
| `vllm_others` | 否 | - | vLLM 额外启动参数 |

### 主机段 `[[cluster.hosts]]`

| 字段 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | 是 | - | 主机唯一标识，供节点 `hosts` 字段引用 |
| `ssh_ip` | 是 | `localhost` | SSH 连接 IP |
| `ssh_port` | 是 | `22` | SSH 端口 |
| `ssh_user` | 是 | `root` | SSH 用户名 |
| `docker_container_id` | 否 | - | Docker 容器 ID 或名称，填写后通过 `docker exec` 操作（场景2） |
| `docker_use_sudo` | 否 | `false` | docker 命令是否需要 sudo |
| `service_ip` | 否 | `ssh_ip` | vLLM 服务绑定 IP |
| `network_interface` | 否 | `lo` | 网卡名，用于 HCCL 通信 |

插件使用上述字段中显式配置的 SSH IP、端口和用户，默认不加载控制节点的 `~/.ssh/config` 或
`/etc/ssh/ssh_config`，避免 `Match final all` 等控制节点配置影响多机 PD 拉起。SSH agent、标准默认密钥及节点配置中已支持的
密码认证仍可使用；`Host` 别名、`ProxyJump` 和自定义 `IdentityFile` 不会从 SSH config 中继承。

### 节点段 `[[vllm_pd.prefill_groups.nodes]]` / `[[vllm_pd.decode_groups.nodes]]`

| 字段 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `hosts` | 是 | - | 引用 `[[cluster.hosts]]` 的 `id` |
| `gpu_ids` | 是 | - | NPU 设备 ID 列表 |
| `service_port` | 是 | `18080` | vLLM 服务端口 |
| `kv_port` | 是 | `30100` | KV 传输端口 |
| `rpc_port` | 否 | `29500` | DP 通信端口 |
| `hccl_if_ip` | 否 | `127.0.0.1` | HCCL 通信 IP |
| `timeout_seconds` | 否 | `7200` | 进程启动超时（秒） |
| `check_interval_seconds` | 否 | `10` | 状态检查间隔（秒） |
| `ascend_base_port` | 否 | `20000` | Ascend Direct Transport 基础端口，同机多实例需指定不同值 |

### Proxy 段 `[vllm_pd.proxy]`

| 字段 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `hosts` | 是 | - | 引用 `[[cluster.hosts]]` 的 `id` |
| `service_port` | 是 | `8000` | Proxy 服务端口 |

### 环境变量

插件自动转发所有 `PD_` 前缀的环境变量到远端节点：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PD_CANN_HOME` | `/usr/local/Ascend/cann-9.0.0` | 远端 CANN 安装路径 |
| `PD_CONDA_NAME` | - | 远端 Conda 环境名 |
| `PD_CONDA_HOME` | - | 远端 Conda 安装路径 |
| `PD_KV_CONNECTOR` | `MooncakeLayerwiseConnector` | KV Connector 类型 |
| `PD_*`（其他） | - | 所有 `PD_` 前缀变量自动转发 |

`PD_KV_CONNECTOR` 同时影响 Proxy 使用的负载均衡脚本：

| 值 | Proxy 脚本 | 说明 |
|----|-----------|------|
| `MooncakeLayerwiseConnector` | `load_balance_proxy_layerwise_server_example.py` | 按层传输 KV Cache（默认） |
| `MooncakeConnectorV1` | `load_balance_proxy_server_example.py` | 整体传输 KV Cache |

此外，`[vllm_pd.env]` 配置公共环境变量，`[vllm_pd.env.prefill]` / `[vllm_pd.env.decode]` 配置角色专属环境变量（覆盖同名公共变量）。

### 寻优参数 `[[vllm_pd.target_field]]`

寻优参数有两个关键规则：

**1. `_prefill` / `_decode` 后缀**

参数名通过后缀区分 P/D 节点：

- P 和 D 要求一样的参数，**不加后缀**，对两侧同时生效
- P 和 D 不一样的参数，**必须加 `_prefill` 或 `_decode` 后缀**分别指定

```toml
# P/D 一样，不加后缀
[[vllm_pd.target_field]]
name = "MAX_MODEL_LEN"
config_position = "env"
min = 8192
max = 131072
dtype = "int"
value = 8192

# P/D 不一样，加后缀
[[vllm_pd.target_field]]
name = "MAX_NUM_BATCHED_TOKENS_prefill"
config_position = "run"
min = 1024
max = 16384
dtype = "int"
value = 1024

[[vllm_pd.target_field]]
name = "MAX_NUM_BATCHED_TOKENS_decode"
config_position = "run"
min = 128
max = 512
dtype = "int"
value = 128
```

`config_position` 决定参数注入方式：`"env"` 导出为环境变量，`"run"` 追加为 vLLM 命令行参数。

**2. `le_enum` 类型约束 TP/DP 联动**

P/D 分离场景下通常要求 decode 侧 TP ≤ prefill 侧 TP。`le_enum`（less-than-or-equal enum，见 [术语表](#术语表)）类型自动约束 decode 的枚举值不超过 prefill 对应参数的值：

```toml
# Prefill TP — enum 类型，自由选择
[[vllm_pd.target_field]]
name = "TENSOR_PARALLEL_SIZE_prefill"
config_position = "env"
min = 1
max = 4
dtype = "enum"
dtype_param = [1, 2, 4]
value = 4

# Prefill DP — factories 类型（见 [术语表](#术语表)），由 TP 动态计算（product / TP = DP）
[[vllm_pd.target_field]]
name = "DATA_PARALLEL_SIZE_prefill"
config_position = "env"
min = 1
max = 4
dtype = "factories"
dtype_param = {target_name = "TENSOR_PARALLEL_SIZE_prefill", product = 4, dtype = "int"}
value = 1

# Decode TP — le_enum 类型，自动取 ≤ prefill TP 的最大值
[[vllm_pd.target_field]]
name = "TENSOR_PARALLEL_SIZE_decode"
config_position = "env"
min = 1
max = 4
dtype = "le_enum"
dtype_param = {target_name = "TENSOR_PARALLEL_SIZE_prefill", values = [1, 2, 4]}
value = 1

# Decode DP — factories 类型，由 decode TP 动态计算
[[vllm_pd.target_field]]
name = "DATA_PARALLEL_SIZE_decode"
config_position = "env"
min = 1
max = 4
dtype = "factories"
dtype_param = {target_name = "TENSOR_PARALLEL_SIZE_decode", product = 4, dtype = "int"}
value = 4
```

效果：prefill_tp=4 时 decode_tp=4，prefill_tp=2 时 decode_tp=2，prefill_tp=1 时 decode_tp=1。

> **声明顺序要求**：`le_enum` 字段的 `target_name` 指向的字段必须在 `le_enum` 字段**之前**声明。若 target 未处理，le_enum 会静默回退到 `min(values)` 并打印 warning 日志。配置时把 `TENSOR_PARALLEL_SIZE_prefill`（target）放在 `TENSOR_PARALLEL_SIZE_decode`（le_enum）之前。

## 使用

```bash
ms_serviceparam_optimizer optimizer -e vllm_pd -b evalscopeperf
```

运行过程中自动完成：生成脚本 → SSH 上传 → 清理残留进程 → 启动 P/D/Proxy → 健康检查等待就绪 → 运行 benchmark → 回收结果 → 停止集群。

预期输出：控制台按顺序输出各阶段日志（生成脚本 → SSH 上传 → 清场 → 启动 → 健康检查 → benchmark）。

## 常见问题

**SSH 连接失败**

在本地先验证：`ssh -p <port> <user>@<host> "echo ok"`。常见原因：密钥未配置、密码错误、防火墙未放行。

**启动后节点一直卡在等待健康**

登录对应节点查看日志 `/tmp/optix_*.log`。常见原因：模型显存不足、端口冲突、CANN 环境未加载、KV 传输端口不可达。

**Mooncake KV 传输失败**

1. P/D 节点间网络是否互通
2. `kv_port` 是否被占用（`ss -tlnp | grep <kv_port>`）
3. RDMA 链路是否正常（`rdma link show`）
4. 防火墙是否拦截了 Mooncake 动态端口

**停止后进程残留**

登录远端手动清理：`ps aux | grep -iE 'vllm|load_balance_proxy'`，然后 `kill -9 <pid>`。或运行 `bash /tmp/vllm_pd_{cluster_id}/stop_pd_process.sh --gpus <gpu_ids> --port <proxy_port>`。

**config.toml 加载失败**

插件优先读取优化器主配置 `model_eval_state.toml` 的 `[plugins.optimizer.vllm_pd]` 节，其次读插件本地 `config.toml`。确认至少有一个配置正确。

**ais_bench model 选择错误导致 tpot 指标缺失**

vLLM PD 分离部署默认提供 Chat 接口（`/v1/chat/completions`），压测工具的 model 参数必须匹配：

| model 名称 | stream | API 端点 | 适用场景 |
|---|---|---|---|
| `vllm_api_stream_chat` | `True` | `/v1/chat/completions` | **PD 分离 + 性能寻优推荐**，可正确解析 tpot/ttft |
| `vllm_api_general_chat` | `False` | `/v1/chat/completions` | 只看吞吐量，不测延迟 |
| `vllm_api_general_stream` | `True` | `/v1/completions` | **不适用于 PD 分离部署** |

## 相关资源

- 插件仓库：本目录 `contrib/optix/vllm_pd_simulator`
- vLLM 上游：[vllm-project/vllm](https://github.com/vllm-project/vllm)
- vLLM Ascend 上游：[vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)
- MindStudio-Service-Profiler：服务化参数寻优工具主体（`ms_serviceparam_optimizer`）
- 问题反馈：通过 GitCode / 对应仓库 Issue 提交
