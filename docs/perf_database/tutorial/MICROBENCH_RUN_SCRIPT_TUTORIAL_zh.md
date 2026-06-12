# Microbench `xxx_run.py` 生成教程

本文说明如何为某个 profiling kernel 生成 `tools/perf_data_collection/op_replay/<KernelType>_run.py`，并让它能够在 NPU 上真实回放该算子，供 `run_all_op.py` 和 `profile_and_update_db.py` 使用。

## 1. 目标

输入：

- 一个算子 CSV，例如：
  `tensor_cast/performance_model/profiling_database/data/<device>/vllm_ascend/<version>/<KernelType>.csv`
- 同版本的 `op_mapping.yaml`
- 对应上游仓库里的接口文档、测试或实现

输出：

- `tools/perf_data_collection/op_replay/<KernelType>_run.py`

这个脚本的职责是：

1. 读取 CSV 的每一行
2. 根据 `Input Shapes / Input Data Types / Input Formats` 重建输入
3. 调用真实接口执行算子
4. `torch.npu.synchronize()`
5. 输出一行简洁的 `[OK]` 日志

## 2. 先看 `op_mapping.yaml`

不要先猜接口名。先打开对应版本的数据目录下的 `op_mapping.yaml`，查 `torch_npu_reference`：

```yaml
torch_npu_reference:
  <KernelType>:
    microbench_api: "..."
```

例如：

- `AscendQuantV2 -> torch_npu.npu_quantize`
- `DynamicQuant -> torch_npu.npu_dynamic_quant`
- `TensorMove -> torch.Tensor.copy_`
- `split_qkv_rmsnorm_rope_kernel -> torch.ops.vllm.qkv_rmsnorm_rope`

这里的 `microbench_api` 就是要在 `xxx_run.py` 里真正调用的接口。

## 3. 到哪里找文档和用例

优先使用本地仓库。没有就 clone 到 `msmodeling` 同级目录。

建议的目录布局：

```text
<workspace>/
├─ msmodeling/
├─ vllm/
├─ vllm-ascend/
├─ op-plugin/
├─ pytorch/
├─ cann-ops-nn/
├─ cann-ops-transformer/
├─ cann-ops-math/
└─ ascend-transformer-boost/
```

推荐搜索顺序：

1. `op-plugin/docs/context/` 下的算子文档
2. `op-plugin/test/` 下的测试用例
3. `vllm-ascend` 中的 Python/Triton/custom op 实现
4. `op-plugin` / `pytorch-npu` 中的 op-api 或注册代码
5. CANN / ATB 仓库里的 layout、cache、融合语义

常用搜索命令：

```bash
git grep -n "torch_npu.npu_dynamic_quant"
git grep -n "npu_kv_rmsnorm_rope_cache"
git grep -n "qkv_rmsnorm_rope"
git grep -n "reshape_and_cache"
```

## 4. 先读 CSV，再决定脚本策略

先看 CSV 的这些列：

- `Input Shapes`
- `Input Data Types`
- `Input Formats`
- `Output Shapes`
- `Output Data Types`

要先回答几个问题：

1. 这个 kernel 一行里到底有几个输入槽位？
2. 有没有空槽位？
3. 输出是一个还是多个？
4. CSV 里有没有记录全部参数，还是缺少一部分非 Tensor 参数？
5. 当前版本的数据是不是只有一种形态？

原则：

- 优先先支持“当前 CSV 真实存在的行”
- 不要为了泛化把脚本写得过度复杂

## 5. 生成脚本时的固定套路

新的 `xxx_run.py` 基本都遵循下面结构：

```python
from common import (
    build_input_tensor,
    build_standard_argparser,
    ensure_npu_available,
    get_runtime_modules,
    get_target_data_dir,
    iter_csv_rows,
    parse_list_field,
    parse_shape,
)
```

然后通常包含：

- `build_argparser()`
- `build_row_case(row)`
- `run_row(csv_path, row_index, row)`
- `main()`

行为模式：

1. 用 `parse_list_field` / `parse_shape` 解析 metadata
2. 用 `build_input_tensor` 重建 Tensor 输入
3. 对缺失的非 Tensor 参数做最小推导
4. 调真实 API
5. `runtime_torch.npu.synchronize()`
6. 打印 `[OK]`

## 6. 常见参数怎么补

很多算子 CSV 不会把所有参数都记下来，这时要从文档和测试里补。

常见例子：

- `AscendQuantV2`
  - `axis=-1`
  - `div_mode=False`
  - 输出 dtype 根据 `Output Data Types` 推导

- `DynamicQuant`
  - 当前 CSV 只有 `x`，没有 `smooth_scales` / `group_index`
  - 直接调用 `torch_npu.npu_dynamic_quant(x)`

- `TensorMove`
  - CSV 只有源张量
  - 目标张量要按相同 shape/dtype/format 重建，再执行 `dst.copy_(src)`

- `ReshapeAndCacheNdKernel`
  - `slot_mapping` 必须合法，不能越界
  - 要根据 cache capacity 构造一个有效索引

- `KvRmsNormRopeCache`
  - 可能有 12 个输入槽位，但后几个是空的
  - 必须保留槽位数量，不能把空槽位丢掉
  - `cache_mode`、`epsilon`、`is_output_kv` 往往需要从测试和输出 shape 推导

- `split_qkv_rmsnorm_rope_kernel`
  - 需要先注册 `torch.ops.vllm.*` 自定义 op
  - 可能还要处理 `vllm_ascend` 的 import fallback

## 7. 什么时候需要特别处理格式

优先看 `Input Formats`：

- `ND`
  - 通常直接用 `build_input_tensor`

- `FRACTAL_NZ`
  - `common.py` 里已经有 `normalize_shape` 和 `npu_format_cast`
  - 一般不要自己手写 NZ 展开逻辑，除非当前算子有特殊缓存布局

- 记录值像 `NCL`
  - 在 `build_input_tensor` 里不会做特殊 cast
  - 这种情况通常只需要按记录 shape 重建即可

## 8. 如何处理自定义 op

如果 `microbench_api` 是：

- `torch.ops.vllm.*`
- `torch.ops._C_ascend.*`
- `atb.*`

那就不能只看 `torch_npu` 文档了，必须同时检查：

1. 自定义 op 是否已注册
2. 是否需要手动 import 注册模块
3. 是否依赖额外环境变量或 sibling repo 路径

例如 `vllm_ascend` 场景推荐做法：

- 先直接 `import vllm_ascend...`
- 如果失败，再尝试：
  - 环境变量 `VLLM_ASCEND_PATH`
  - 同级目录 fallback

这样脚本既能在本地开发环境跑，也能在已安装环境跑。

## 9. 最小验证步骤

生成脚本后，至少做下面两步：

```bash
py -3 -m py_compile tools/perf_data_collection/op_replay/<KernelType>_run.py
py -3 tools/perf_data_collection/op_replay/<KernelType>_run.py --help
```

如果环境有 NPU，再做一次真实运行：

```bash
python tools/perf_data_collection/op_replay/<KernelType>_run.py \
  --device ATLAS_800_A3_752T_128G_DIE \
  --vllm-ascend-version 0.13.0
```

## 10. 提交时要注意什么

通常只提交：

- `tools/perf_data_collection/op_replay/<KernelType>_run.py`

不要误提交：

- 本地 clone 下来的上游仓库
- `profiling_database/data/...` 里你临时生成或解压的内容
- profiling 输出目录

推荐先看：

```bash
git status --short
```

再单独 add：

```bash
git add -- tools/perf_data_collection/op_replay/<KernelType>_run.py
```

## 11. 一个实际工作流示例

以 `DynamicQuant` 为例：

1. 打开 `op_mapping.yaml`
2. 找到 `torch_npu_reference.DynamicQuant.microbench_api = torch_npu.npu_dynamic_quant`
3. 读 `op-plugin/docs/context/torch_npu-npu_dynamic_quant.md`
4. 读 `DynamicQuant.csv`
5. 发现当前 CSV 只有单输入 `x`
6. 生成脚本，直接调用：

```python
output, scale = runtime_torch_npu.npu_dynamic_quant(x_tensor)
```

1. 做 `py_compile` 和 `--help` 验证
2. 单文件提交

## 12. 建议的判断原则

- 先信 `op_mapping.yaml` 给出的 `microbench_api`
- 再信测试用例
- 再信实现文件
- 最后才做必要推断

如果要推断参数，遵守两条：

1. 只为“当前 CSV 真实存在的 case”推断
2. 选择最小、最稳定、最不容易跑偏的规则

这样生成出来的 `xxx_run.py` 更容易真实回放 profiling 数据，也更容易后续维护。
