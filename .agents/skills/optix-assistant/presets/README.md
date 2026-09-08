# vLLM Ascend 推荐配置目录 · 收集教程

`ascend_vllm_presets.json` 是 **vLLM Ascend 官方推荐配置**的静态目录：从
`docs.vllm.com.cn` 离线收割官方验证过的 `vllm serve` 配置，供 `optix-assistant`
运行时本地查询（**agent 从不直接抓网页**）。

完整设计见 `docs/design/vllm-ascend-preset-catalog-design.md`。

---

## 核心原则

1. **离线收割 → 静态提交 → 运行时零抓取**。收割成本只发生一次，agent 运行时只读 JSON。
2. **draft 必须过 review gate** 才能提交为 `ascend_vllm_presets.json`。官方 HTML 结构会随版本变化，解析产物是"粗解析"，需要人/子 agent 抽查校准。
3. **网络用 curl 兜底**：`docs.vllm.com.cn` 会 reset urllib 连接，harvest 脚本内置了 urllib → curl 的回退。

---

## 一条命令全量收割

```bash
python .agents/skills/optix-assistant/presets/scripts/harvest_vllm_ascend_presets.py \
  --out /tmp/ascend.draft.json --doc-version "v0.22.1rc1"
```

默认枚举索引页全部模型页（当前 ~39 个），逐页抓取解析。常用参数：

| 参数 | 用途 |
| --- | --- |
| `--out <path>` | 输出 draft JSON（必填） |
| `--index-url <url>` | 模型索引页（默认官方 models/ 目录） |
| `--model-page <url>` | 只抓指定模型页（可重复） |
| `--html-file <path>` | 解析本地 HTML（离线/调试；或先 curl 再喂给它） |
| `--limit N` | 限制模型数量试水 |
| `--doc-version` | 记录到 `meta.doc_version` 溯源 |

---

## 完整流程（6 步）

### ① 枚举模型清单

```bash
# 全量收割会先枚举索引；也可单独看清单：
python .agents/skills/optix-assistant/presets/scripts/harvest_vllm_ascend_presets.py \
  --out /tmp/ascend.draft.json --limit 0
```

### ② 抓取 + 解析 → draft

即上面那条全量命令。每页打印 `fetch: <url>`，抓取失败会打印 WARNING 并跳过（网络问题，可稍后单独重试）。

### ③ curate（自动清洗确定性杂质）

```bash
python .agents/skills/optix-assistant/presets/scripts/curate_draft.py \
  --draft /tmp/ascend.draft.json \
  --out /tmp/ascend.curated.json
```

`curate_draft.py` 自动完成 review gate 里**机器能确定**的部分，并生成
`<out>.review-notes.md`（待确认清单）：

| 自动清洗 | 自动推断 |
| --- | --- |
| 过滤 eval/LM-harness 场景 | `model.name` 精简 + `aliases` 生成（含 `Qwen3-0.6B/1.7B/4B` 这类前缀重建） |
| 剔除 shell 模板变量（`tp=$7`、`dp=$3`） | `model.quantization` 权重格式推断（w8a8/w4a8/bf16…） |
| 剔除占位符（`your_*`、`/path/to/`） | `params.quantization` 后端 flag 移到 `additional_config` |
| 去 `¶`、清洗 env 残留引号（`"AIV"`） | `is_moe` 保留收割推断 |
| **`additional_config` 按名保留原样**（含完整 `kv-transfer-config` PD 细节） | |

### ④ 校验

```bash
python .agents/skills/optix-assistant/presets/scripts/validate_presets_schema.py \
  --presets /tmp/ascend.curated.json
```

- **error** → 阻断提交（如缺 `scenario.id/source_url`、`params` 非标量、`presets` 为空）
- **warning** → 不阻断（如溯源字段缺失、某场景 params 为空）

### ⑤ Review Gate（只 review 判断项 —— 人/子 agent）

curate 已清掉确定性杂质，这里只需对照 `<out>.review-notes.md` 处理**判断项**：

1. **`hardware` 待补**：从权重表的"推荐硬件"列补 `cards` / `mem_per_card_gb`；
   拿不准就留空 —— 匹配逻辑对未声明硬件的模型是放行的，对已声明但不符的是保守跳过。
2. **`is_moe` / `feature_support` / `model.quantization` 核对**（curate 的推断值）。
3. **空 params 场景**：官方页面未给 serve 命令的场景，需手工补或标记。
4. **语义化 `scenario.id`**：`config-N` → `single-node-40k` / `pd-prefill` 等。
5. **补长上下文场景**：官方"纯 flag 调整块"（无完整 `vllm serve` 行）不会被
   自动捕获，需手工补（参考同页 serve 配置的 `--max-model-len`）。
6. **数值抽查**：随机对 2-3 个场景，把 `MAX_NUM_SEQS` / `MAX_MODEL_LEN` /
   `GPU_MEMORY_UTILIZATION` 与官方页面比对。

### ⑥ 提交

```bash
cp /tmp/ascend.curated.json \
   .agents/skills/optix-assistant/presets/ascend_vllm_presets.json
python .agents/skills/optix-assistant/presets/scripts/validate_presets_schema.py \
  --presets .agents/skills/optix-assistant/presets/ascend_vllm_presets.json   # 必须 OK
```

提交后 pre-commit 的 `check-json` 会再兜底校验 JSON 合法。

**目录状态跟踪**：提交时同步更新 `presets/review-notes.md`（本目录的待确认清单，
按当前 catalog 实际状态逐模型列出 hardware / quantization / is_moe / 空 params /
语义 id / workload 等待办，便于跟踪"哪些模型还没人工校准"）。

---

## 手动补录单个模型（无官方页面时）

按 schema 手工写一个 preset 条目。`source_url` 留空或填相关文档/仓库链接，
在 `meta.note` 注明 `source=manual`。schema 字段含义见设计文档 §2.2。

---

## 常见问题

| 问题 | 处理 |
| --- | --- |
| 抓取失败 / urllib 被 reset | 脚本已内置 curl 兜底；仍失败先确认 `curl` 可用，或用 `--model-page` 单独重试 |
| 解析错位（官方改版 HTML） | golden 测试 `tests/regression/agent_optimizer/test_harvest_parser.py` 会失败 → 更新 fixture 或适配解析层 |
| schema 校验 error | 定位到具体字段修复（错误信息带路径，如 `presets[0].scenarios[2].params`） |

---

## 运行时如何被使用

- `collect_context.py::_inject_vendor_preset`：按 **模型名/aliases + 硬件 + workload 长度**匹配目录，
  命中则把官方推荐参数**填空**注入搜索空间（`source="vendor_preset"`），并写入 `knowledge.vendor_preset`。
- **优先级**：用户显式配置（config.toml / search-space / recommend-result）最高，vendor 只填空不覆盖；
  实测结果（results.round-*.json）最终权威。详见设计文档 §2.2。
- **未命中 / 文件缺失 / 硬件不符** → 静默回退，不影响寻优。
