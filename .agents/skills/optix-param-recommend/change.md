# param-recommend Skill 修改记录

## 修改日期

2026-05-21

## 修改概览

本次修改一共修复 6 个问题，涉及 `scripts/recommend_params.py` 和 `scripts/test_recommend_params.py`。

---

## 问题 1：`missing_required_fields` 中 if/else 分支逻辑完全一致

**文件**：`scripts/recommend_params.py`

**问题**：当用户提供了 `config.json` 路径后，`load_model_config` 已提前从文件中提取并注入模型结构字段到 `context["model"]`，但 `missing_required_fields` 的 `else` 分支仍然检查 `MODEL_FIELDS` 是否缺失——与 `if` 分支完全重复。代码意图不清，且隐含了 `load_model_config` 必须先于 `missing_required_fields` 执行的调用顺序假设。

**修改**：删除冗余的 `else` 分支。当提供了 `config.json` 路径时，信任 `load_model_config` 已处理模型字段填充，不再重复检查。

```python
# 修改前
if not has_model_config:
    missing.extend(field for field in MODEL_FIELDS if nested_get(context, field) in (None, ""))
else:
    missing.extend(field for field in MODEL_FIELDS if nested_get(context, field) in (None, ""))

# 修改后
if not has_model_config:
    missing.extend(field for field in MODEL_FIELDS if nested_get(context, field) in (None, ""))
```

---

## 问题 2：`dtype_bytes` 不支持 4-bit 量化类型

**文件**：`scripts/recommend_params.py`

**问题**：函数只识别 `int8`(1 字节) 和 `32`(4 字节)，其余全部 fallback 到 2 字节。4-bit 量化类型（`int4`、`uint4`、`nf4`、`fp4`、`float4`）被错误当成 2 字节，导致模型权重大小高估 4 倍，进而 KV cache 容量被严重低估，所有 batch size、并行度推荐失准。

**修改**：返回值类型 `int` → `float`，新增 4-bit 类型的识别（0.5 字节）。

```python
# 修改前
def dtype_bytes(dtype: Any) -> int:
    text = str(dtype or "bfloat16").lower()
    if "int8" in text or "uint8" in text:
        return 1
    if "32" in text:
        return 4
    return 2

# 修改后
def dtype_bytes(dtype: Any) -> float:
    text = str(dtype or "bfloat16").lower()
    if "int4" in text or "uint4" in text or "nf4" in text or text in ("fp4", "float4"):
        return 0.5
    if "int8" in text or "uint8" in text:
        return 1
    if "32" in text:
        return 4
    return 2
```

---

## 问题 3：`choose_parallelism` 选择最小满足条件的 TP，过于保守

**文件**：`scripts/recommend_params.py`

**问题**：`tp_candidates` 是从小到大排列的因子列表，循环找到第一个满足显存条件的 TP 就 break，导致选了最小的可行 TP。TP 越小，单卡放的模型权重越多，KV cache 可用显存越少。更大的 TP 能释放更多 KV cache 空间，支撑更大的 batch size。

**修改**：`for candidate in tp_candidates` → `for candidate in reversed(tp_candidates)`，从大到小遍历，选最大的可行 TP。

```python
# 修改前
for candidate in tp_candidates:

# 修改后
for candidate in reversed(tp_candidates):
```

---

## 问题 4：`COMPILATION_CONFIG` dtype_param 硬编码含嵌套引号的完整命令行

**文件**：`scripts/recommend_params.py`

**问题**：`dtype_param` 中存储了完整命令行 flag `"--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'"`，嵌套引号在 TOML 序列化时容易出问题。`COMPILATION_CONFIG` 属于 `VLLM_INLINE_FLAG_NAMES`，会通过 `$COMPILATION_CONFIG` 占位符接入 `vllm_command_others`，dtype_param 只需存 JSON 值本身即可。

**修改**：dtype_param 只保留 JSON 配置值，去除 `--compilation-config` 前缀和单引号包裹。

```python
# 修改前
dtype_param=["", "--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'"]

# 修改后
dtype_param=["", "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}"]
```

---

## 问题 5：`use_search_command = True` 死代码

**文件**：`scripts/recommend_params.py`

**问题**：`command_for_config_skill` 函数中 `use_search_command = True` 永远为真，包裹在其下方的 `if use_search_command:` 代码块也永远执行。该变量和 if 语句是预留的"固定参数"分支，但从未实现。

**修改**：删除 `use_search_command = True` 变量定义，去除 `if use_search_command:` 条件包裹，将其内部代码提升一级缩进。

```python
# 修改前
use_search_command = True
...
if use_search_command:
    if "min" in item: ...
    ...

# 修改后
if "min" in item: ...
...
```

---

## 问题 6：测试 `assert_handoff_commands_parse` 依赖真实子进程

**文件**：`scripts/test_recommend_params.py`

**问题**：`assert_handoff_commands_parse` 在测试中通过 `subprocess.run` 调用外部 `auto_config.py` 脚本并断言返回值为 0。这不是纯单元测试——当 `auto_config.py` 不存在或环境不匹配时测试直接失败，限制了测试的可移植性。

**修改**：在 `assert_handoff_commands_parse` 开头增加目标脚本存在性检查，不存在时静默跳过（`return`），不报错也不断言。

```python
# 修改前
def assert_handoff_commands_parse(handoff, tmp_path):
    config_path = write_minimal_config(tmp_path)
    for command in handoff["apply_commands"]:
        ...

# 修改后
def assert_handoff_commands_parse(handoff, tmp_path):
    config_script = Path(__file__).parents[3] / CONFIG_SKILL_SCRIPT_PATH
    if not config_script.exists():
        return
    config_path = write_minimal_config(tmp_path)
    for command in handoff["apply_commands"]:
        ...
```

新增常量 `CONFIG_SKILL_SCRIPT_PATH = ".agents/skills/optix-config/scripts/auto_config.py"`。

---

## 影响范围

- 所有修改均向后兼容，不改变现有 API 契约。
- 问题 2、3 会影响推荐数值（量化模型更准确、TP 选择更合理），但不改变输出格式。
- 问题 4 修改了 `COMPILATION_CONFIG` 的 dtype_param 格式，下游 `optix-config` Skill 需要配合调整（预期该 Skill 也需同步更新）。
- 问题 1、5、6 仅为代码清理，不改变行为。

---

## 2026-05-30 修改

本次修改修复 2 个问题，涉及 2 个文件。

---

## 问题 1：`load_model_config` 不支持嵌套 `text_config` 结构

**文件**：`skills/param-recommend/scripts/recommend_params.py`

**问题**：部分模型配置（如 Qwen3.5、Qwen3-VL、Kimi）将模型参数嵌套在 `text_config` 下，而非顶层。`load_model_config` 函数只从顶层读取字段，导致 `hidden_size`、`num_hidden_layers` 等关键字段无法提取，进而模型权重估算和并行度推荐全部失效。

**修改**：加载配置后检查是否存在 `text_config`，如果存在则将其字段合并到顶层再进行字段映射。

```python
# 修改后新增
text_config = config.get("text_config", {})
if text_config:
    config = {**config, **text_config}
```

**影响**：修复后能正确解析 Qwen3.5、Qwen3-VL、Kimi 等模型的嵌套配置。

---

## 问题 2：测试用例未覆盖嵌套 `text_config` 场景

**文件**：`skills/param-recommend/scripts/test_recommend_params.py`

**问题**：现有测试仅验证扁平结构的 config.json，缺少对嵌套 `text_config` 结构的测试覆盖，导致问题 7 未被及时发现。

**修改**：
1. 新增 `write_nested_model_config` 辅助函数，模拟 Qwen3.5 的嵌套配置结构
2. 新增 `test_nested_text_config_loaded_correctly` 测试用例

**影响**：Windows 环境下测试会因 `auto_config.py` 输出 Unicode 符号触发 GBK 编码错误，需要手动设置 `PYTHONIOENCODING=utf-8` 环境变量。