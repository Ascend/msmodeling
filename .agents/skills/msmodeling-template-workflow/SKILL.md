---
name: msmodeling-template-workflow
description: 给定模型名称，调用 msmodeling 官方 CLI 采集、构建和独立验证模板图，自动将验证通过的固定结构桶注册到寻优使用的模型模板目录。用于制作、扩展和接入模板，不用于改造推导引擎。
metadata:
  version: 5.4.0
  source: repository-local; official cli registry and UserInputConfig interfaces
---

# 从模型名称生成模板图

客户使用入口见 [template_engine/README.md](../../../template_engine/README.md)，包含采集、构建和查询 CLI 的完整示例及解释。

唯一必填业务输入为模型 ID。读取会话中的范围与预算，其余使用
[scope.json](references/scope.json)；不让用户重填已知参数。
本次交付仅制作 compile-on 模板，范围中的 compile 必须为 true。
执行现有 fix 项目能力，不恢复 joint 研究任务、不修改引擎或模型源码。
目标是交付模板、实际命中表和独立比较证据，而不是只写执行计划。

## 入口

在用户指定的 fix checkout 工作，读其 AGENTS/spec，核对 HEAD 和工作树。
复用已装依赖的 Python；用明确 cwd/PYTHONPATH 保证加载当前 checkout，
不因新目录没有 .venv 就重装环境。先读 [project-api.md](references/project-api.md)。

```bash
<python> .agents/skills/msmodeling-template-workflow/scripts/workflow.py preflight
<python> .agents/skills/msmodeling-template-workflow/scripts/workflow.py plan --model-id <model> --run-dir <new-run>
<python> .agents/skills/msmodeling-template-workflow/scripts/workflow.py run --run-dir <new-run>
```

`plan`只读取/下载模型 config，使用官方 AutoModelConfigLoader，不下载权重；遇到需执行远端模型 Python 的配置先报告；
本地配置加 `--model-config <config.json>`。`run`实际启动官方采集。
用户仅要求编写 skill 时，不自行选择模型启动采集。

用户给出输入/输出长度时，`plan` 增加 `--input-length <I> --output-length <O>`，
使用官方寻优的代表请求形状增加长请求独立目标，不将输出长度当成 decode 的 query-length。
用户明确要求覆盖整个 B 区间时增加 `--decode-batch-coverage all`；默认 typical 只代表少量本地批量。

## 执行流程

1. **识别模型**：确认官方 config、原生层数、MoE/稠密类型、专家/头维度、
   上下文上限、MTP 和量化入口，保存配置与 SHA。不能裁层、换模型或改
   MODEL_CONSTANTS。未知支持标为候选，以真实官方结果核验。
2. **规划典型域**：脚本使用当前 ParallelConfig 和量化枚举，输出典型离散单元、
   训练点和独立目标。按 [coverage.md](references/coverage.md) 检查每个轴，
   补上用户强调的配置，不默认完整笛卡尔积。显式 DP/MOE-TP、模块布局或量化
   子选项无法由 CLI 表达时单列缺口。规划通过不等于模型支持。
   B 是全局 `num_queries`，本地批量由官方 DP 切分；Decode 每个拟覆盖的本地批量分别准备
   至少两个不同 S 的训练点及未见 S 验证点。Prefill 不按本地批量强行拆结构桶，
   但同一结构体内需要区分批量与序列的作用。读取官方输入生成和模型分支条件，
   有窗口回绕、压缩空流或 TP 补齐时，在 scope 的 `S_branch_train` 中增加能区分这些条件的点；
   不用全部同余的 S 去辨识余数公式。脚本的典型点不是源码分析或完整分支覆盖的替代品。
3. **校准成本**：可先用 `run --max-cells 1` 执行首个单元，再按实测估时。
   已有预算直接沿用；否则使用 scope.json 的有界默认预算。运行参数可覆盖
   `--max-captures`、`--max-seconds`。先检查外部已有进程；本脚本串行采集，
   scope 的 `max_capture_processes` 是外部采集进程上限，默认 2，不代表脚本会创建两路。
   用户授权四路后台采集时可按该上限配置并发调度，保留独立缓存/日志并按实际内存限制派发。
   失败也计预算。预算耗尽要报告未覆盖，不能称完成。
4. **采集和构建**：`run`调用统一 cli.inference.template_graph capture/build。
   每样本独立缓存；训练目录只含身份/SHA 核验后的 train 图和 sidecar。
   保存 argv、cwd、退出码、日志、耗时、文件路径/SHA、源码纪元和模板 SHA。
   不完整的训练单元不混入本轮构建。构建失败按 cell 隔离诊断，不改引擎。
   只使用官方固定结构构建流程，不调用旧折叠／动态图接口；默认输出紧凑 JSON+XZ，不能为缩小文件裁减桶或算子。
   每个分支一个代表点只解决结构选择，不等于 shape 已可外推；零 failed_formulas 也可能包含
   被误拟合的常量或偶然相等的特征，必须继续独立比较。合法请求被拒绝时区分结构未匹配、
   等价全局 B 处理、样本跨度门槛和公式缺失，不把所有拒绝都归为缺样本，也不直接关闭准入检查。
5. **先生成后取 oracle**：薄 CLI 适配器调用官方
   generate(return_ops=True)，期间禁止读取采集/训练目录。
   候选先落盘，再采独立目标图，用项目 compare_performance_ops 比较。
   算子、容器、标量、shape/dtype/requires_grad 和张量对象引用均需通过；
   stride 差异保留诊断但不阻塞性能仿真验收，strict_passed 另报完整字段比较结果。
   不能用 op_count 或 coverage verdict 代替比较。生成拒绝不触发原生回退找答案。
   独立点排除相同 `(本地批量,S)` 的训练等价配置；若长请求本身就是训练端点，单列锚点回放，
   不计入独立外推通过率。命中率用固定的全部目标作分母，区分生成、正确命中、拒绝、失配和未跑，
   并分别报告 Prefill/Decode。用户给出的通过率门槛仅用于本次验收，不成为其他模型的保证。
6. **定向补齐**：逐项检查典型 cell 是否有通过的未见 B/S。
   有明确辨识目的时在预算内开新轮次补训练；最终留出不得偷偷转为训练。
   无收益、能力缺失或预算不足时停止扩样，保留失败，交付 partial 和具体缺口。
   精确锚点服务、拒绝、失配、未跑、独立通过必须分列。
7. **自动注册寻优**：`run`默认调用项目注册能力，仅安装有独立目标通过、且训练锚点回放通过的固定结构桶；
   同桶有失配或待比较目标时不注册该桶。允许典型域仍为 partial，但必须明确已注册子集与未覆盖项。
   合格模板安装到 `template_engine/templates/`，由 registry 确定模型/compile 文件名。
   写入对应文件前保存旧文件备份，写入后按模型、原生层数和 compile
   回读发现路径、SHA，并重新验证已通过目标可展开；失败则恢复旧模板。
   报告必须包含 registration 和 optimizer_template_ready，不能只凭模板文件存在称已接入。
   默认本地自动注册属于本 Skill 的用途；用户要求只生成验证时使用 `run --no-register`。
   吞吐寻优进程在运行环境中须加载同一 checkout/registry，且开启 template_graph_optimization。
8. **性能验收**：用户要求寻优加速数据时，按 project-api.md 做真实 runner on/off 对照，
   记录重放、forward、缓存计数及结果/内存。注册发现与图比较不替代完整吞吐性能验收；不默认启动大规模寻优。

## 恢复和交付

运行目录保存 plan.json、model-config.json、epoch.json、receipts/、logs/、
training/、template.json.xz、候选图、comparison 报告及 report.json。
`run --resume`只复用相同计划/源码/依赖且 SHA 仍匹配的成功记录；保留失败，
不通过 --force 覆盖旧捕获。源码变化后开新 run，不重贴历史证据来源。

报告列出：典型单元总数、完整训练单元、可生成与独立通过目标、失配/拒绝/未跑、
精确锚点数、耗时、模板 SHA、注册路径/SHA/桶数量/备份及最短使用命令。任一要求单元缺少独立目标通过，
整个典型域只能是 partial。有限点命中不是任意 B/S 的证明；范围内缺口不能藏进域外回退。
默认完成本地验证与 registry 安装，不自动 commit/push、远端发布或打包权重/凭据。


`--model-config` 用于冻结期望模型配置，不覆盖官方模型加载器。每次新采集都比较真实解析配置与冻结配置，隐藏维度、专家数、上下文等字段不一致时拒绝接受；需要修改模型配置时请提供对应的本地模型目录并重新规划。

训练目录按本次选择的样本集合隔离。缩小 `--max-cells` 后恢复不会混入旧单元文件。时间预算同时覆盖模板加载、图生成、解码和独立比较；这些只读计算超时后不发布候选或启动注册。
