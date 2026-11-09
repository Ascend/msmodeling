# 项目接口

以当前 checkout 的 CLI 注册参数为准。首次运行 workflow.py preflight；接口改变时改薄适配器，
不把 joint 的 --parallel-layout-file 等参数传入此分支。

```bash
<python> -m cli.inference.template_graph capture --help
<python> -m cli.inference.template_graph build --help
<python> -m cli.inference.template_graph capture --model-id <model> --num-queries 4 --query-length 1024 --tp-size 2 --ep-size 4 --moe-dp-size 1 --dcp-size 1 --compile --num-mtp-tokens 0 --quantize-linear-action DISABLED --quantize-attention-action DISABLED --cache-dir <sample-specific-cache>
<python> -m cli.inference.template_graph build --compile-mode on --cache-dir <training-only-dir> --template <run-dir>/template.json.xz
```

构建输出默认使用紧凑 JSON+XZ，不需要压缩参数。`build` 省略 `--template` 时自动写入
`template_engine/templates/`；Skill 显式输出到运行目录，独立验证后将合格部分安装至该目录。

decode 加 --decode，query-length=MTP+1，context-length=S-query-length；
不传 --num-hidden-layers-override。每样本独立缓存，项目读写同时按模型身份限定。读取 stdout 真正的 path/key，再核验 sidecar model_id 和原生层数。

统一生成 CLI 支持 --output 保存 ops；workflow.py 同时保留进程内的薄验证适配：
采集/构建仍调用项目 CLI，验证使用既有 generate、graph_codec、compare_performance_ops
API，不加推导规则。候选 generate(...,return_ops=True) 执行期间禁止
读取捕获/训练目录，候选保存后才取 oracle。不能通过 zero_meta/native fallback 取得答案。
用 decode_document(...,'model_meta/1') 核验 sidecar；用 encode_capture/decode_capture
保存和比较候选。比较是 CPU/Meta 仿真图，不是 NPU 数值或硬件耗时准确性证明。

Skill planner 支持 `--input-length <I> --output-length <O>`，调用官方 optimizer 的代表形状
解析方法增加长请求目标。它们是 workflow.py 的参数，不是 capture/generate 的参数。
B/S 与请求长度的对应及 16K/1K 示例见 [request-shapes.md](request-shapes.md)。

`run`默认通过 `template_engine.workflow_registration.register_validated` 和项目 registry API
安装已验证桶子集。`report.json.registration` 保存路径、SHA、验证数量、排除桶数量和旧模板备份；
`optimizer_template_ready=true` 表示注册后发现及已验证目标展开检查通过，不表示整个计划 passed。
没有合格桶时保留已有注册；写入或回读失败恢复已有注册并报告错误。`--no-register` 只生成/验证。

用户要求实际寻优性能时：用同一 registry 的模板，同模型/原生层数/设备/
后端/请求语义做真实 optimizer on/off，对真实 worker 记录 replay、native forward、
缓存计数、结果和内存。注册、运行成功或缓存命中不等于模板命中。
官方寻优入口为 `python -m cli.inference.throughput_optimizer`，默认启用模板；
增加 `--no-template-graph` 可跳过模板和工作负载复用。两轮均保留 `--compile`。
用 `--log-level info` 查看 `[template HIT]`／`[template FALLBACK]`，区分 request 与 runner 级记录，
不能用日志行数代替不同配置数量。完整 Bash 示例见项目 README。默认自动注册通过的子集，不跑大规模寻优。

覆盖拒绝可在寻优中回退原生路径；输入非法、模板损坏和内部异常应报告并定位，不将其记成普通未覆盖。

表达缺口、无收益或预算耗尽时返回 partial，保留失败证据，不改引擎、不重贴历史来源。
