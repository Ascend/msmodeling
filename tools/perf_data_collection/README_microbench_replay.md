# Microbench Replay 与回填工具

本工具链只消费已有性能数据库 CSV，按其中的 Shape、dtype/format 和 runtime metadata 在 NPU 上 replay，然后通过 msprof 聚合实测结果并回填。它不导入 Shape 生成器，也不会替用户生成或扩充 Shape。

## 独立运行

单独 replay 数据库中存在的算子：

```bash
uv run python tools/perf_data_collection/op_replay/run_all_op.py \
  --database-path /path/to/profiling_database/data/... \
  --ops Add SparseFlashAttention
```

采集、聚合并回填：

```bash
uv run python tools/perf_data_collection/start_microbench.py \
  --database-path /path/to/profiling_database/data/... \
  --ops Add SparseFlashAttention
```

使用两张本地 NPU 并行采集：

```bash
uv run python tools/perf_data_collection/start_microbench.py \
  --database-path /path/to/profiling_database/data/... \
  --num-devices 2 \
  --ops Add SparseFlashAttention
```

`--num-devices` 默认值为 `1`。传入大于 1 的值时，主进程会检查可用 NPU，自动为每张卡复制独立数据库、启动 worker、等待完成并合并结果。支持行分片的 Adapter 会把稳定 case 子集分给每个 worker；少数手工编排的 Adapter 会按算子分配给一个 worker，避免重复采集。合并只回写本次 `--ops` 选中的 CSV；`missing-only` 数据已经完整时不会启动 worker。使用者不需要指定 shard 数量或 shard 编号，也不会因为只启动一个 shard 而漏跑部分 case。

DispatchFFNCombine 需要一个完整的 EP 通信进程组，不参与普通的 `--num-devices` case 并行。请使用 `--dispatch-ffn-combine-*` 参数单独运行该算子。

实际运行需要可用的 Ascend NPU、与数据库目录一致的软件栈，以及对应自定义算子环境。无 NPU 的本地测试只能证明 CSV 解析、replay contract、聚合和回填逻辑，不能视为 A3 实测。

## 架构边界

- `operator_metadata.py`：只保存 canonical kernel、profiler aliases、runtime signature mode 和 profiler task type，不注册 callable。
- `replay_framework.py`：提供通用 replay adapter、runtime case 和稳定分片契约。
- `parallel_runner.py`：根据 `--num-devices` 自动完成设备选择、worker 启动、隔离数据库和结果合并。
- 普通 `*_run.py`：保持薄入口；SparseFlashAttention、LightningIndexer、MLA preprocess 保留显式 adapter。
- `start_microbench.py`：负责 msprof 启动、结果聚合和 CSV 回填。

Shape 与 Microbench 仅通过 CSV + runtime metadata schema 衔接，可分别从 `master` 独立运行和验证。

## A3 实测验证（2026-08-21）

### 单卡和 2 卡并行

在 A3 容器（2× Ascend910_9382，CANN 8.5.0）上验证：

- 34 个非 DFC 算子（每算子 5 行 generated shape）单卡 replay 全部通过
- `--num-devices 2` 两卡并行 replay 34 个算子全部通过，176 行 merged

### DFC EP16 单节点

在 16 卡 A3 容器上验证 DFC EP16：

```bash
python tools/perf_data_collection/start_microbench.py \
  --database-path /path/to/db \
  --ops DispatchFFNCombine \
  --dispatch-ffn-combine-ep-size 16 \
  --dispatch-ffn-combine-nproc-per-node 16 \
  --repeat-count 1
```

5 行 replay + msprof profiling + CSV 回填全部成功。

### DFC EP32 双节点（待验证）

需要两台同 Super Pod 的 16 卡 A3 容器：

```bash
# Node 0
python tools/perf_data_collection/op_replay/DispatchFFNCombine_run.py \
  --database-path /path/to/db --ep-size 32 --nproc-per-node 16 \
  --nnodes 2 --node-rank 0 --master-addr <node0-ip> --master-port <port>

# Node 1
python tools/perf_data_collection/op_replay/DispatchFFNCombine_run.py \
  --database-path /path/to/db --ep-size 32 --nproc-per-node 16 \
  --nnodes 2 --node-rank 1 --master-addr <node0-ip> --master-port <port>
```

> 注意：运行前需 `export ASCEND_CUSTOM_OPP_PATH=...` 和 `export LD_LIBRARY_PATH=...` 指向 vLLM-Ascend 自定义算子路径，否则 `ensure_custom_opp_env` 会报错。
