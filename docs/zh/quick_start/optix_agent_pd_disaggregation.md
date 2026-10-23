# 使用案例：从环境准备到 Agent 驱动的 PD 分离寻优

本文面向首次使用 OptiX 的用户，按顺序完成模型准备、镜像与容器部署、工具安装、Agent 寻优及结果查看。

**使用范围**：Linux、昇腾 NPU、vLLM Ascend、OptiX Agent 模式。工具从 [msmodeling 官方仓库](https://gitcode.com/Ascend/msmodeling/) 获取，安装步骤见第 4 节。

**开始前，请确认具备以下条件：**

- **Agent 工具**：可使用 msagent 或 OpenCode，具备读取文件、执行命令及持续监控任务的能力，安装与启动见第 4.5 节。
- **模型服务与额度**：已配置可用的模型服务账号或 API 认证，并有足够的调用额度；分析日志、生成候选及多轮寻优会持续消耗额度，具体费用和限制以所用服务为准。
- **网络访问**：运行 Agent 的环境能够访问所选模型服务和寻优设备。内网环境可使用可达的内网模型服务，无需强制连接公网；模型、镜像、工具依赖和知识库需能下载，或提前离线准备。服务选择及配置、日志的传输需符合组织要求。

若暂不具备可用的 Agent 工具、模型服务或调用额度，或网络限制无法满足上述条件，请改用 [PSO 算法的 PD 分离寻优操作指南](optix_pso_pd_disaggregation.md)。该流程不依赖 Agent 工具和大模型 API 额度，仍需准备相应的部署与压测环境。

操作主线：**安装准备 → 启动 Agent → 确认任务 → 查看结果**。

本指南用于内网可信测试环境。账号、密钥和网络访问权限由管理员配置；Agent 使用组织允许的模型服务，配置和日志是否出域取决于所选服务。

第 1～4 节完成一次性环境准备并启动 Agent；第 5 节发送任务；第 6 节确认后等待寻优；第 7 节查看结果。寻优过程由 Agent 完成，无需手动启动服务、编写配置或执行压测。

## 1. 准备寻优所需信息

本文以两台机器为例：A 运行 Agent，两台机器共同参与 PD 分离寻优。以下均为示例，**可按业务自定义**；修改后，同步替换后续命令和任务说明中的对应值。

| 需要提供的信息 | 本文示例 | 如何自定义 |
|---|---|---|
| 可用机器与登录方式 | A：`192.0.2.10`，B：`192.0.2.11`；SSH 用户 root、端口 22 | 换成实际 IP、用户、端口 |
| 每台机器可用的卡 | A、B 各可用 `0,1,2,3`，共 8 卡 | 列出允许本任务使用的具体卡号，不能只写总卡数 |
| 模型 | `Qwen/Qwen2.5-7B-Instruct` | 可换成部署后端支持的模型 |
| 业务负载 | 输入 1024、输出 256 token；200 个请求；并发 16 | 按实际请求规模填写 |
| 性能目标 | 满足延迟要求后提高吞吐 | 可自定义 TTFT、TPOT 等目标；不了解时先让 Agent 解释、确认 |
| 时间或次数预算 | 总共最多 24 次、4 小时，包含复测 | 可按设备空闲时间调整，由 Agent 分配各阶段额度 |

**不需要提前确定 PD 配比或调优参数。** Agent 会根据模型、可用卡和实测提出候选；已有参数经验或固定配置，也可以作为附加要求提供。

网络等环境信息由 Agent 检查；多网卡或跨机通信信息无法确定时，会集中请你或管理员确认实际网卡和通信 IP，再开始压测。

## 2. 准备模型

从 [Hugging Face](https://huggingface.co/models) 或 [ModelScope（魔搭）](https://modelscope.cn/models) 下载模型，也可复用已有模型文件。本文以 `Qwen/Qwen2.5-7B-Instruct` 为例，将完整模型目录放在两台宿主机的 `/data/models/Qwen2.5-7B-Instruct`，保持模型版本一致。

模型和保存路径均可自定义；修改后同步替换后续容器挂载路径和 Agent 任务说明中的模型路径。

## 3. 下载镜像并启动容器：在两台宿主机分别执行

### 3.1 检查宿主机

```bash
npu-smi info
docker version
```

确认设备和 Docker 正常。设备基础环境由管理员预先准备；后续 Agent 可读取设备及网络信息。

### 3.2 下载配套镜像

从 [vLLM Ascend 安装说明](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html) 选择与硬件、驱动及 PD Connector 配套的镜像。确定与部署环境配套的完整镜像地址或 digest，然后在两台宿主机执行：

```bash
# 替换为与你的部署环境配套的实际镜像标签，不使用 latest。
export IMAGE='quay.io/ascend/vllm-ascend:<已验证的完整标签>'
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'
```

两台使用相同镜像版本，并确认该镜像支持 PD 分离部署。

### 3.3 创建容器

工具通过 SSH 登录宿主机并操作容器。下面示例映射每台机器的 4 张卡，卡号不同时修改对应 `--device`。

先在两台宿主机创建 `/data/msmodeling`：这是后续下载 msmodeling 工具的保存位置，挂载后容器内路径为 `/workspace/msmodeling`。B 节点此目录可暂时为空。目录可自定义，但要同步修改挂载命令和后续路径。

容器名称 `optix-pd` 也可自定义；修改后同步替换所有 `docker exec` 命令及提供给 Agent 的容器名称。本文挂载的宿主机 `/data/models` 在容器内对应 `/models`。

```bash
mkdir -p /data/msmodeling

docker run -d --name optix-pd \
  --network host --ipc host --shm-size 16g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /data/models:/models:ro \
  -v /data/msmodeling:/workspace/msmodeling \
  --entrypoint /bin/bash "$IMAGE" -lc 'sleep infinity'
```

RDMA 部署还需按所选镜像和网络方案映射 `/dev/infiniband` 等设备、设置相应权限及网络参数；由设备管理员确认。上面的基础容器命令不能替代 RDMA/HCCL 配置。host 网络下，同一宿主机的所有服务共享端口空间，必须避开已有业务。

检查两台容器：

```bash
docker exec optix-pd npu-smi info
docker exec optix-pd test -f /models/Qwen2.5-7B-Instruct/config.json
docker exec optix-pd bash -lc 'command -v vllm; vllm --help >/dev/null'
docker exec optix-pd python3 -c 'import vllm, vllm_ascend; print("vLLM imports OK")'
```

通过标准：容器持续运行、4 张卡可见、模型可见、vLLM 命令和 Ascend 插件可用。镜像挂载要求参考 [官方快速开始](https://docs.vllm.ai/projects/ascend/en/main/getting_started/quick_start.html)，以对应版本说明为准。

## 4. 准备控制端：只在 A 节点容器执行

以下命令均在 A 节点通过 `docker exec -it optix-pd bash` 进入后执行。

### 4.1 准备基础命令及 SSH 免密

容器需要 `git`、`ssh`、`scp`、`curl`、`bash`、Python 3.10+ 和 venv 支持。缺少时按镜像操作系统安装，例如 Ubuntu 使用 `apt-get install git openssh-client curl python3-venv`。

```bash
# 已有可用标准密钥或 SSH agent 时，不必重新生成。
ssh-keygen -t ed25519
ssh-copy-id -p 22 root@192.0.2.10
ssh-copy-id -p 22 root@192.0.2.11

ssh -o BatchMode=yes -p 22 root@192.0.2.10 \
  'docker exec optix-pd bash -lc "echo A_OK; command -v vllm"'
ssh -o BatchMode=yes -p 22 root@192.0.2.11 \
  'docker exec optix-pd bash -lc "echo B_OK; command -v vllm"'
```

首次连接按管理员核实的主机指纹确认。使用本次任务专用密钥或加载了该密钥的 SSH agent，不复用个人或团队共享密钥。建议使用专用运维账号；示例 root 仅用于测试说明，修改账号时同步替换命令和任务说明。Docker 权限同样具有较高的宿主机操作能力，须由管理员限定可访问机器及来源，任务结束后撤销授权并清理专用密钥。

工具不读取 SSH config 的别名、`IdentityFile` 或 `ProxyJump`，因此请填写实际地址，使用默认密钥路径或 SSH agent。需要跳板时，请管理员提供可直连的转发地址。

通过标准：上面两次检查均无需输入密码，并找到容器内的 vLLM。

### 4.2 下载 msmodeling 工具

msmodeling 就是包含 OptiX 和寻优 Skill 的工具仓库。下面将主仓 master 下载到上一节创建的挂载目录：

```bash
git clone --branch master https://gitcode.com/Ascend/msmodeling.git /tmp/msmodeling-source
cp -a /tmp/msmodeling-source/. /workspace/msmodeling/
cd /workspace/msmodeling
```

首次执行时目标目录应为空。已有工具时不要重复复制；在该目录执行 `git pull --ff-only origin master` 获取最新代码，若有本地修改先保留并处理。更新后重新执行第 4.3 节的安装命令。

### 4.3 安装 OptiX 和 PD 插件

已有 `uv` 时跳过前两行；以下将 uv 本身放进独立小环境：

```bash
python3 -m venv /opt/uv-tools
/opt/uv-tools/bin/pip install uv
export PATH="/opt/uv-tools/bin:$PATH"

cd /workspace/msmodeling
uv sync && uv pip install --python .venv/bin/python -e contrib/optix/vllm_pd_simulator

uv run --no-sync msmodeling optix --help
.venv/bin/python -c 'from importlib.metadata import version; print(version("vllm_pd_simulator"))'
```

OptiX、AISBench 按本文分别安装到独立环境，避免影响镜像中已有的 vLLM。安装完成后交给 Agent 运行。更新代码或重新执行 `uv sync` 后，需再次安装 PD 插件：该同步操作会移除依赖清单之外的包。

### 4.4 下载并安装 AISBench（推荐）

AISBench 是负责发送请求、统计吞吐和延迟的压测工具，安装在 **A 节点容器的独立环境**，不装进 OptiX 环境或镜像的 vLLM 环境。安装方式参考 [AISBench 官方仓库](https://github.com/AISBench/benchmark)。

```bash
uv venv --python 3.10 /opt/aisbench-venv
uv pip install --python /opt/aisbench-venv/bin/python 'setuptools<81' \
  ais_bench_benchmark==3.1.20260630
/opt/aisbench-venv/bin/ais_bench --help

# 让 OptiX 找到独立环境内的 AISBench；当前终端及启动 Agent 的终端需保持生效。
export OPTIX_DEPLOY_PATH=/opt/aisbench-venv
```

采用固定版本便于复现。安装路径可自定义，修改时同步调整命令及环境变量。压测配置由 Agent 准备，无需手写。

### 4.5 安装并启动 Agent（任选一种）

本文统一将 Agent **安装并运行在 A 节点容器内**，与 OptiX 使用同一份目录和终端环境。B 节点不用安装 Agent。这样可直接读取配置、执行工具并管理两台机器；不要在宿主机启动后照搬容器内路径。

可使用组织提供的 msagent，或按 [OpenCode 官方说明](https://dev.opencode.ai/v2/docs) 安装并配置可访问的模型服务。Agent 需要具备读取文件、执行命令和持续监控任务的能力；此处不是任意 Agent 的兼容性保证。已有可用工具时跳过安装。

在 A 节点容器内打开终端，进入工具环境后启动 Agent：

```bash
cd /workspace/msmodeling
source .venv/bin/activate
export OPTIX_DEPLOY_PATH=/opt/aisbench-venv
opencode
```

使用 msagent 时，将最后一行替换为组织提供的启动命令。然后发送下一节任务说明；若工具未自动加载 Skill，按任务中的路径读取 `SKILL.md`。知识库缺失时由 Agent 按 Skill 初始化，需能访问对应知识仓库；离线环境请管理员预置，不需自行初始化子模块。

## 5. 告诉 Agent 要做什么

启动 Agent 后，复制下面的任务说明，按实际情况修改机器、模型、负载、目标和预算。所有示例值均可自定义；已有固定配比或参数要求时，在末尾补充即可。

```text
请使用本仓 .agents/skills/optix-assistant，进行 PD 分离的 Agent 模式寻优，使用 AISBench 压测。

机器：A 为 192.0.2.10，B 为 192.0.2.11；SSH 用户 root、端口 22。
两台宿主机的容器名均为 optix-pd，每台可用卡为 0、1、2、3。
模型：两台容器内均为 /models/Qwen2.5-7B-Instruct，服务模型名 pd-demo。
负载：输入 1024 token、输出 256 token，200 个请求，固定并发 16。
目标：在 TTFT≤2000ms、TPOT≤100ms 的前提下提高吞吐。
预算：最多 24 次实测、总计 4 小时，包含基线和最终复测。

请自行检查环境、准备配置并安排寻优，先展示配置和计划供我确认。
执行中汇报进度，结束后提供最佳配比、参数、与基线的对比和部署复现文件。
```

服务模型名可自定义，由 Agent 保证服务端与压测一致。不了解延迟目标时，可改为“请先测基线，解释结果后与我确认目标”。

其余准备和执行由 Agent 完成，信息不足时会集中询问。按提示选择交互式模式，可逐轮确认方案；希望在预算内自动推进时，可在任务说明中补充“采用全自动模式”。

## 6. 确认任务，等待 Agent 寻优

Agent 检查环境后，会展示配置和执行计划。核对以下内容并在对话中确认：

- 使用的机器、可用卡和模型是否正确。
- 请求长度、请求数、并发是否符合本次业务。
- 性能目标、时间和次数预算是否符合预期。

确认后，**服务启动、基线测量、配比与参数调整、压测、复测和结果导出均由 Agent 完成**。无需另开终端启动服务、修改配置或运行压测命令。交互式模式下，Agent 会在需要确认下一组方案时提示你。

本案例使用 AISBench 测量完整 PD 部署的表现，根据端到端吞吐、延迟和成功率选择方案，不需要额外准备 P、D 单侧测量环境。

执行期间可直接与 Agent 对话：

| 需要做什么 | 可以这样说 |
|---|---|
| 查看进度 | “汇报当前进度、最佳结果和剩余预算。” |
| 暂停 | “清理当前任务后暂停，保留结果。” |
| 恢复 | “继续之前的寻优，请先检查环境和剩余预算。” |
| 遇到异常 | “请分析这次失败，告诉我原因及是否需要管理员协助。” |

Agent 会保留任务记录。若更换会话，将之前的任务目录交给新会话继续处理；无需自行编辑记录或清理服务进程。

## 7. 查看结果；需要时让 Agent 复现

完成后，Agent 会给出总结及报告路径。打开 `final_report.md`，重点查看：

| 内容 | 看什么 |
|---|---|
| 最佳方案 | P/D 配比、设备分配和服务参数 |
| 性能表现 | 吞吐、首 token 延迟（TTFT）、后续 token 延迟（TPOT）、请求成功率 |
| 与基线对比 | 同一业务负载下是否提升，是否满足目标 |
| 复测结论 | 最佳方案是否经过独立复测，结果是否稳定 |
| 完成范围 | 实际耗时、测量次数，以及未完成项及原因 |

完成的判断是：有成功的完整 PD 实测、基线与胜出方案的独立复测结论，以及报告和可复现的部署文件；性能是否达标单独说明，不预设提升比例。

报告应逐项列出“推荐 P/D 配比、基线与推荐方案的吞吐/TTFT/TPOT、复测结论、未完成项”。这里不提供示意性能数值，实际数据以本次测量为准。

部署文件保存在 `best_deployment/`，具体路径由 Agent 提供。没有完成复测时，报告应明确说明，不能把单次最好成绩当成稳定收益。

需要复现时，对 Agent 说：“请根据导出结果在原环境复现最佳方案，开始前确认资源可用。”Agent 会准备具体方案，确认后执行部署和验证，无需手工复制脚本或逐台启动服务。

遇到无法自动解决的问题，让 Agent 整理错误摘要、环境信息和相关日志，再联系管理员或支持人员。不要在对话中发送 SSH 私钥或访问令牌。
