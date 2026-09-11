# 查询驱动的 Shape 网格生成设计

## 背景

纯理论笛卡尔积无法可靠回答“某个模型在 throughput optimizer 中到底会查询哪些 CANN Kernel Shape”。
模型结构只能给出隐藏维度、专家数、头数等静态约束；batch、prefill/decode、chunked prefill、TP/DP/EP、
MoE-DP、DCP、MTP 和编译融合会共同改变真实查询。因此，Shape 生成以 TensorCast/ServingCast 的实际查询路径
为首选输入；用户显式指定但查询未命中的算子，再由通用理论规则补足，不维护特定模型或 serving 场景的 Shape 表。

## 用户接口

| 参数 | 必选 | 含义 |
|---|---:|---|
| `--database-path` | 是 | 目标性能数据库目录，同时提供设备和软件栈映射 |
| `--rows` | 否 | 每个 CSV 本次最多新增的有效唯一行数，默认 1000 |
| `--target-models` | 二选一 | 一个或多个 HuggingFace 模型 ID，触发内部采样扫描 |
| `--optimizer-args-file` | 二选一 | 描述实际 throughput_optimizer 场景的 YAML/JSON workload spec |
| `--ops` | 否 | 最终需要生成的 replay-supported Kernel；未指定时使用模型查询结果 |
| `--seed` | 否 | Coverage 候选的稳定排序种子，默认 0 |
| `--report-path` | 否 | 机器可读 JSON 生成报告的输出路径；缺省写入自动查询缓存目录 |

`--target-models` 与 `--optimizer-args-file` 至少提供一个，可以同时提供（场景合并）。设备、vLLM/PyTorch/CANN
版本从数据库的 `op_mapping.yaml` 读取，不再作为重复的公开参数。

```powershell
python tools/perf_data_collection/generate_shape_grid.py `
  --database-path <性能数据库目录> `
  --target-models <HuggingFace模型ID> `
  --rows 1000 `
  --seed 0
```

重复执行时，已有行、重复候选和非法候选都不占用 `--rows`；工具会继续追加后续候选。因此 `--rows 1000`
表示“每个目标 CSV 本轮最多新增 1000 行”，不是“最终总行数为 1000”。

## 实际寻优参数驱动的生成模式

`--optimizer-args-file` 接受一个 YAML 或 JSON 文件，描述一个或多个用户实际执行的
`throughput_optimizer` 场景。字段名与 optimizer CLI 的 snake_case 语义一一对应；省略的键与 CLI 省略
flag 的行为一致（含 TP-only 向后兼容默认）。工作负载并行候选先复用 optimizer 自己的
`resolve_parallel_search_candidates` / `resolve_search_sizes` 解析，再展开为单并行配置的独立子进程，
与内部采样模式共享 checkpoint 缓存、并发调度和查询收敛逻辑。

```yaml
workloads:
  - name: prefill-20k
    model: zai-org/GLM-5.1
    device: ATLAS_800_A3_752T_128G_DIE
    num_devices: 32
    input_length: 20000
    output_length: 1024
    disagg: true
    ttft_limit: 10000
    max_batched_tokens: 20000
    batch_range: [1, 32]
    tp_sizes: [1, 2, 4, 8, 16]
    ep_sizes: [4, 8, 16, 32]
    moe_dp_sizes: [1]
    num_mtp_tokens: [0]
    quantize_linear_action: W8A8_DYNAMIC
    quantize_attention_action: DISABLED
    reserved_memory_gb: 10
    compile: true
    compilation_config: [enable_sequence_parallel, enable_dispatch_ffn_combine]
    enable_shared_expert_tp: true
```

失败封闭（fail closed）约定：

- 未知字段直接报错，并列出全部合法字段；
- 历史命令行里存在、但当前工具语义无法兑现的字段（`dp_sizes`、speculative 三件套、multimodal、
  PD-ratio、`performance_model`/`profiling_database_path`、仅影响输出文件或警告的执行参数等）给出定向错误；
- `input_length` 仅接受整数，长度分布文件不支持；
- spec 的 `device` 必须是已注册 DeviceProfile 且与目标数据库设备一致；
- 展开后不存在任何合法并行组合时报错；被过滤的非法组合数量写入报告。

PP 搜索：`pp_sizes`/`pp_layer_partitions` 直接映射 optimizer 的 `--pp-sizes`/`--pp-layer-partitions`；
并行组合合法性按 optimizer 的 stage-local 规则（`dp = num_devices/(tp×pp)`、
`moe_tp = (num_devices/pp)/(ep×moe_dp)`）预过滤；PP>1 要求 `num_mtp_tokens` 包含 0（即不允许
推测性 MTP）；pp 超过模型 `num_hidden_layers` 时与 optimizer CLI 一致按非法候选过滤并计入汇总。

预算语义：optimizer-args workload 产生的 exact demand 行不受 `--rows` 约束，永远全部写入；组合模式下
target-model workload 的 exact demand 仍与 Coverage 插值、constraint fallback 候选共享 `--rows` 预算。
纯 `--target-models` 模式保持原语义。

每条 exact demand 都有确定状态：`existing`（已有有效行）、`generated`（生成新行）、`duplicate`（与另一条
demand 重复）、`projection_rejected`（无法投影到现有 schema）、`validation_rejected`（违反 replay 约束）、
`preflight_rejected`（真实 build_case 预检拒绝）、`budget_truncated`（仅 target-models 来源）或
`unsupported`（Kernel 无 CSV/replay 入口）。状态台账连同
workload 摘要、spec 稳定 digest、每个算子的 preflight 结果和 CSV 行数前后对比写入 JSON 生成报告；
`demand_id` 与 runtime-rich 算子行内的 `Runtime case_id` 同源，可从报告追溯 demand → CSV row。

## 生成行 replay preflight

所有新增行（含内部采样模式的理论兜底行）在写入 CSV 前执行纯 Python replay preflight：加载对应
`op_replay/*_run.py`，stub 掉 NPU 张量构建后真实执行该 Kernel 的 `build_case` 契约，校验输入输出
Shape、dtype、format 和辅助张量约束。preflight 失败的行不写入数据库，失败原因进入生成报告，
避免浪费 A3 microbench 时间或被 microbench 静默删除。

## 数据流

```text
HuggingFace ModelArchitecture ── 或 ── optimizer-args workload spec
        ↓                                  ↓
内部 workload policy              实际寻优参数场景展开
        └────────────┬─────────────────────┘
              多次 throughput_optimizer
                     ↓
      ProfilingDataSource 捕获 HIT/MISS 的实际 Kernel 查询
                     ↓
          版本化 CANNBackendProjector
                     ↓
   查询命中：精确锚点 + CoveragePlanner 插值/边界候选
   查询未命中的显式 --ops：通用 Theory Generator
                     ↓
       replay build_case 纯 Python preflight
                     ↓
        仅写入通过 preflight 的 Kernel CSV
                     ↓
         start_microbench / op_replay 实测回填
```

## 算子选择优先级

`--ops` 的优先级高于 `--target-models`：

- 传入 `--ops` 时，最终只规划这些算子。查询命中的算子使用查询网格，未命中的算子使用通用理论网格；
- 未传入 `--ops` 时，最终算子集合来自目标模型实际查询到、同时具备 CSV 和 replay 入口的 Kernel；
- 用户显式指定的算子不做模型适用性拦截；只有通用配置明确标为 `skip` 时才跳过理论兜底；
- 无 replay 入口或无目标 CSV 时，在写数据库前报错；配置明确为 `skip` 或没有通用 generator 时报告跳过。

查询和理论候选使用同一套 CSV 签名去重和 `--rows` 预算。数据库只保存可 replay Shape，不写入候选来源；运行时 trace
中的模型和 workload 信息仅用于内部诊断。

## 查询捕获

`ProfilingDataSource` 在完成 TensorCast 到数据库查询形状的投影后记录需求。捕获覆盖：

- 普通 compute、elementwise、attention、MoE 和 MTP projection；
- composite 的 compute、attention/cache 子 Kernel；
- 独立通信和 composite 通信查询；
- 数据库 HIT 与 MISS。

每个 optimizer 子进程写独立 JSONL 分片，主进程按稳定查询签名全局去重。签名不包含模型名和 workload 名，
避免同一个 Kernel 查询因为来源不同重复占用网格；来源仍保留在 trace 中用于诊断。trace schema 和 projector
软件栈身份都显式版本化。

映射存在 primary/alternate Kernel 时，trace 会保留这一组版本化 backend candidate；最终只对目标数据库中存在且有
op replay 的 Kernel 写行。cache pool 容量等已由查询规则声明为性能无关的轴使用 replay-minimal 或现有 schema 容量，
实际序列长度、有效 block 数和 runtime metadata 仍来自真实执行，不把 allocator 预留容量误当作性能 Shape 轴。

MISS 也必须记录。profiling performance model 可以继续走 analytic fallback，使一个缺失 CSV 点不会中断后续模型执行，
从而能够收集完整调用链上的其他需求。

### 与实测查询语义的一致性

本实现复用 PR705 验证过的通用查询语义，但代码独立落在本分支，不依赖 PR705，也不引入其中的模型数据或模型特判：

- composite 的 exact 与 interpolation 共用同一份 TP、phase、SP runtime mapping；SP 只投影 Prefill，Decode 保留原 token；
- TP=1 是合法恒等分片，非整除 token 按 `ceil(tokens / TP)` 生成最忙 rank 的物理 Shape；
- cache update 的物理池容量不是性能轴，Scatter 查询只按 cache tail、update tail、dtype/format 分 regime；
- `compute_scale` 保留 FP16/BF16 的物理差异、标量 scale 输出、NCL/ND format 以及 per-token、per-tensor、
  per-channel、per-block regime；这些信息在 CSV 命中前就写入 query trace；
- 单请求 Sparse Attention Prefill 使用一维 token 轴，避免把两个相同自由度表示成共线二维插值；共享 token
  轴的多输入算子通过 mapping 声明联动输入，不把 residual 等输入误当成固定 Shape。

上述内容改变的是数据库实际查询和 Shape 投影的通用契约，不新增公开参数。查询 Shape 与理论兜底 Shape 最终仍按同一
CSV 签名去重，数据库不记录来源。

## 内部 workload policy

公开接口不暴露 workload 调参旋钮。工具根据 HF 架构和数据库设备拓扑自动构造有限但较宽的扫描：

- 单卡基线扫描短、中、长以及连续轴的代表点；
- HF 最大上下文、chunk `boundary ± 1`、TP 非整除和最大长度边界；
- optimizer 的 batch 1～512 搜索；
- TP、EP、MoE-DP 和 decode-only DCP 分轴扫描；
- 少量短/中/长边界上的 TP×EP corner/midpoint 交叉扫描；
- 设备拓扑上限不是二次幂时，额外覆盖真实拓扑边界；
- 模型声明支持时扫描 MTP token 数；
- 长序列额外运行 decode 与 chunked-prefill；
- 数据库声明 MoE fusion 时额外运行 compile、sequence parallel 和 DFC 组合。
- 默认 W8A8 动态量化之外，自动加入 BF16 基线和代表性的 INT8 KV-cache 查询。

长度、TP、EP、MoE-DP、DCP 和 MTP 不会在每个 workload 中做全笛卡尔积。每个合法单轴值都保留；TP×EP 只保留
`min×max`、`max×min`、`max×max` 和 `mid×mid`，compiled MoE 只组合 EP/MTP 的首尾边界。Shape 的连续长度轴由
CoveragePlanner 扩展，查询阶段负责捕获 schema、regime、离散并行值和不连续边界。该 policy 有内部版本，可以在不增加
用户参数的情况下演进。只有模型主 baseline 运行 batch 1～512 搜索；其余并行轴、交叉、编译和量化 workload 拆成
batch 下边界和采样上边界两个查询，避免 optimizer 对每个离散场景重复整轮 batch 二分搜索。上边界由 HF 架构自适应满足
`batch × input_length ≤ max_context_length`，最低保留 batch=1：短序列仍覆盖 batch=512，越接近模型最大 context，
采样 batch 越小。中间 batch 由已有实测点和 Coverage/interpolation 网格承接，不把连续 batch 轴与离散并行轴做全笛卡尔积。

TP、EP、MoE-DP、DCP 和 MTP 候选完成合法性过滤后，会展开成单一并行配置的独立 workload。每个内部 optimizer 因此
只执行一个配置，配置之间由外层自适应并发调度，不再被 optimizer 的进程池串行长尾绑定；查询集合与展开前一致，失败重试
和 checkpoint 也可以精确到单个并行配置。

compile/sequence-parallel/DFC 场景不会执行 `max context × max EP × max MTP` 的三轴极端笛卡尔角点。工具在一个中间长度
锚点查询 EP/MTP 的完整边界交互，在短、长长度锚点只运行基础编译配置；CoveragePlanner 再将长度轴与离散交互轴组合并
加密。这样仍能捕获编译态 schema、EP/MTP 分支和长上下文边界，同时避免单个极端组合占用数分钟甚至更久。

### 调度、进度和断点恢复

每个 optimizer workload 仍在独立子进程中执行，避免 TensorCast 编译和并行配置跨场景泄漏。调度器最多同时运行八个
workload，并按 CPU 核数和可用内存自动下调；每个 optimizer 保持单 job，避免内外两层并发造成 CPU/内存过量。进度输出
包含候选组合数、单 workload 耗时、完成原因、已完成数量和 ETA。Shape 工具只消费查询 trace，不消费 optimizer 的最终
最优解；当 trace 已覆盖该
workload 的 TP/EP 候选，并且 trace 与整个 optimizer 进程树连续 30 秒都没有活动时，认为查询已经收敛，回收子进程并
进入下一个 workload。这样可以跳过 optimizer 在结果汇总或内部缓存等待阶段的无效耗时，同时不会在模型加载、查询仍在
增长或 CPU 仍在计算时提前结束。

成功 workload 的 query trace 会自动写入系统临时目录下的 checkpoint cache。缓存键包含 workload policy、目标模型、
数据库内容以及查询/投影关键源码摘要；数据库、模型或查询语义变化时自动失效。中断后再次执行同一命令会复用已完成
workload，不需要新增公开参数，失败或不完整的 workload 会重新运行。缓存不写入性能数据库，也不改变最终 CSV 来源语义。

PP 改变层在 stage 之间的归属，但不改变单层 Kernel 的 Shape 语义；PP>1 要求 `num_mtp_tokens` 包含 0
（即不允许推测性 MTP）。因此 optimizer 的 PP 搜索
产生的查询主要由 TP/EP/阶段本地并行维度决定，PP 本身不引入新的 Kernel Shape 轴；涉及跨 stage 通信或特殊
pipeline kernel 时，应先在仿真查询层显式建模，再由相同捕获机制自然进入网格。

## CoveragePlanner

优先级如下：

1. 实际查询的 exact demand；
2. 同一 Kernel、dtype/format 和 runtime regime 内的单轴插值及上下边界；
3. 两个真实动态轴的组合；
4. 三个及以上动态轴的确定性组合。

动态轴只从同一 schema 下多个 exact demand 的实际变化推断。输入和输出中按固定比例共同变化的维度会组成一个联动组，
例如 MatMul 的 M 轴、输出 M 轴，以及 FRACTAL_NZ 权重块与逻辑 N/K 的比例关系。未在查询中变化的隐藏维度、头维度或
runtime regime 不会被任意扰动。

Attention、Sparse Attention 和 LightningIndexer 含有 list-valued runtime metadata。查询 CoveragePlanner 不对这些
metadata 做脱离执行上下文的笛卡尔扩展；显式 `--ops` 触发理论兜底时，必须由该算子的专用 generator 同步生成 Shape 和
runtime metadata，不能用普通张量轴拼接代替。

## 物理 Shape 投影

projector 根据目标 CSV 已有 schema 完成 TensorCast Shape 到 replay Shape 的转换，包括：

- MatMul 权重 `(K,N)` 到数据库 `(N,K)`；
- ND 与 FRACTAL_NZ block Shape；
- grouped weight 的 expert 前缀；
- RoPE 输入换序和布局转换；
- SwiGlu 双输入合并；
- ReshapeAndCache 的合并 cache 拆分；
- MoE token permute/unpermute 的 token flatten、index dtype 和辅助输出；
- grouped-list 激活拼接及专家权重前缀；
- compile/SP 下 RmsNorm、AddRmsNorm、DynamicQuant 和 DFC 的物理槽位；
- 3D token/batch 到 2D kernel Shape 的 flatten；
- FIA、SparseFlashAttention、LightningIndexer 的 runtime metadata 和可 replay 物理槽位。

投影规则由数据库软件栈身份版本化。无法匹配现有 CSV schema 的 demand 会被拒绝并计数，不会写入猜测行。

## Replay 支持边界

支持列表运行时从 `tools/perf_data_collection/op_replay/*_run.py` 自动发现。未传 `--ops` 时，只处理模型查询、replay 脚本和
目标数据库同名 CSV 的交集；显式 `--ops` 是最终目标集合，查询未命中时自动走理论兜底。无 replay 脚本或无 CSV 时立即
报错，配置为 `skip` 的理论算子只报告跳过。通信 Kernel 当前没有统一 op replay 入口，虽然会捕获查询，但不会由本工具
写入 compute CSV。

## 验收边界

Shape 生成只负责得到待实测网格，写入行的耗时字段为 0。完整验收还需要：

1. 在目标 A3/对应硬件上执行 microbench 回填；
2. 检查生成行 replay 成功率和无效行；
3. 用独立 holdout workload 检查数据库 coverage；
4. 检查插值误差；
5. 对 optimizer best row 执行 text_generate B2B。

仅靠 Shape 数量“足够密”不能证明 CANN latency 可精确插值；Kernel 选择、tiling 和融合可能存在非连续边界。查询驱动能让
网格围绕真实需求生长，但实测回填和误差验收仍是不可替代的闭环。
