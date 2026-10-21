# Agent 多机混部

本场景使用 Linux 上已部署的 vLLM 与原 `multihost_infer` 插件。模型权重在每个节点、容器内路径一致；master 与 OptiX 同机，worker 使用免密 SSH。沿用插件的 Ascend/conda 模板环境，先确认所有节点部署版本支持拟搜索的参数，尤其 PCP。

## 配置来源与参数边界

- OptiX 配置仍用 `[vllm]`，`framework="vllm"`、`optimizer_strategy="agent"`；安装并注册原插件后 CLI 使用 `-e multihost_infer`。
- 集群配置使用 `[vllm_mix]`、唯一 `[[vllm_mix.node]]`、有序 `[[vllm_mix.workers]]`、`chips_per_node` 和 `[vllm_mix.env]`。它决定地址、节点数和每节点设备数。
- `collect_context --cluster-config` 从文件推导总设备数。显式 `--world-size/--num-nodes/--num-per-node` 只能与文件一致；context 保存路径及 SHA256，配置变化须重新采集。
- `NODE_COUNT/CHIPS_PER_NODE/WORLD_SIZE/NPU_COUNT` 属于事实，不登记进 search_space，也不发送给服务。服务常量如 max_model_len 仍登记并渲染。
- Round 0 后 TOML 冻结；类型、配置位置、派生关系和冻结常量保持不变，可调字段的范围与默认值以当轮 context.search_space 为准。候选未设置的参数取当轮空间默认值，预检与运行共用装配器。
- `run` 字段由 OptiX 渲染到 CLI，`env` 字段保留环境语义；历史 MAX_NUM_SEQS/MAX_NUM_BATCHED_TOKENS 环境字段转换到容量 flag。候选值覆盖固定 others，JSON 子键合并；benchmark 并发和请求率不进入 serve 命令。

`others` 中的 `$NAME` 仅从 `[vllm_mix.env]` 和本次执行字段组成的映射解析；不能直接引用未登记的启动 shell 环境变量，未解析的占位符会报错。

TOML 的冻结和集群拓扑快照是两个独立门禁，均须通过。单机文档中的 tp×dp 示例仅适用于 pp=pcp=1；本场景须核算完整并行乘积及节点放置。

## 部署约束

原插件平均分配 DP 副本，要求 `DP % 节点数 == 0`。每节点资源检查为 `DP / 节点数 × TP × PP × PCP <= chips_per_node`；当前配置检查以 `TP × DP × PP × PCP == 总设备数` 验证完整利用。

方法卡的全局设备数公式不能替代本部署的节点放置检查；候选必须同时满足上述满配与节点放置约束，不满足时以本部署约束为准。知识路由中的 `mode=multi-dp` 若描述多机 DP 混部，在本功能中对应 `deployment.type=multihost`；单机 DP 和 PD 分离不属于本文场景。

TP、PP、PCP 进程组限制在节点内；当前实现不支持让这些组跨节点。PCP 搜索必须按所部署 vLLM/vLLM-Ascend 版本的方法卡与 CLI 支持核对，资源公式合法不保证所有模型组合可运行。

模板负责 host、port、headless、DP 本地大小/起始 rank/地址/RPC 端口，这些不能作为候选或 others 覆盖。原模板 master 的 api_server_count 固定为 1。配置中的服务端口通过 `[vllm.command].port` 设置。

## 执行步骤

沿用三个阶段工作流、knowledge 路由及 benchmark_spec 门禁。下面只列多机增加的参数，业务负载旗标仍按 environment-preparation.md 提供。

```bash
export CLUSTER_CONFIG_PATH=/absolute/path/cluster.toml
python <SKILL_ROOT>/scripts/config_preflight.py --config /absolute/path/optix.toml --engine vllm --benchmark <benchmark> --cluster-config "$CLUSTER_CONFIG_PATH" --check-only
python <SKILL_ROOT>/scripts/collect_context.py --run-dir <run-dir> --config-toml /absolute/path/optix.toml --engine vllm --cluster-config "$CLUSTER_CONFIG_PATH" --model-name <name> --model-path <weights> <已确认负载旗标>
python <SKILL_ROOT>/scripts/check_search_space.py --run-dir <run-dir>
python <SKILL_ROOT>/scripts/config_preflight.py --config /absolute/path/optix.toml --engine vllm --benchmark <benchmark> --cluster-config "$CLUSTER_CONFIG_PATH" --context <run-dir>/context.json --check-only
# agent 建立当轮 search_space 和一个候选后：
python <SKILL_ROOT>/scripts/validate_candidates.py --run-dir <run-dir> --round N
msmodeling optix -e multihost_infer -b <benchmark> -c /absolute/path/optix.toml
```

OptiX 会核对插件实际加载的节点、设备数和静态环境与该轮集群配置相同。每轮 context 自动归档；进程退出前不能修改配置或准备下一轮。

## 结果与导出

多机 trial 的 `params` 记录有效字段值；`metadata.requested_params` 记录校验解析值，`metadata.launch_plan` 保存实际生成的 master/worker 脚本。归档 context 对应执行时搜索空间。

```bash
python <SKILL_ROOT>/scripts/export_best_config.py --run-dir <run-dir>
```

候选按最新协议填写方向 trial_type 和顶层 routing_review；diagnosis 只作诊断证据，不参与正常判优。导出回查同轮候选，只选 optimization 中无错误、fitness 有限且非 early_exit 的 trial，优先 SLO 达标。多机直接输出胜出轮次的脚本到 `best_deployment/`，`serve_command_multihost.json` 记录文件清单，handoff 记录参数与部署清单。缺少完整脚本时拒绝导出，不根据当前配置猜造。

脚本复制到各自节点运行；worker 脚本不能在 master 上当作单机命令执行。导出脚本与 common.py 可以脱离仓库运行，不导入 optix；单机仍输出 serve_command.json/sh。真实集群验证需覆盖服务就绪、压测、worker 失败清理和导出重启。
