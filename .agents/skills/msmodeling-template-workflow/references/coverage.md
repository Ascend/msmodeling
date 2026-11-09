# 典型范围与验收

需求域见 scope.json，不是全部组合的采集承诺。典型 profile 用代表布局覆盖
小/中/大 world、DP/TP/EP，DCP 单独覆盖 decode；代表单元覆盖两种 phase、
量化静态/动态动作、注意力量化、DFC、MTP0～3和部分交互。未执行交叉组合不计命中。
profile 以当前明确覆盖的 cell 列表为交付范围，用户覆盖的轴优先于默认值。

Planner 用真实 ParallelConfig 解析 world/DP/MOE-TP；不假设 EP 必须整除专家数，
不把 B<DP 判为官方非法，不写死 64/EP 饱和点。MoE 默认 EP>=TP 的 profile 中
MOE-TP 可能恒为 1，应显示，不能算 MOE-TP>1 已覆盖。
稠密模型用稠密典型 profile，不套 MoE 约束；用户约束与模型没有交集时明确报告。

B 为全局请求并发数，本地批量由 ParallelConfig 的 DP 推导。Decode 的训练、独立目标
必须属于同一本地批量；不能用 K=1/2/3 的样本验证一个从未采过的 K=4 结构。
`decode_batch_coverage=typical` 只准备典型 K，plan.json 的 `unplanned_decode_local_batches`
列出其余值；明确覆盖整个 B 区间时使用 `--decode-batch-coverage all`，按实际数量估算预算。

B/S 上限继承需求；模型上下文限制、decode query<=S 等限制要记录。
用户强调长上下文时将该长度加入独立目标和必要训练，不能以短 S 验收替代。
输入/输出长度到 Prefill/Decode S 的映射见 [request-shapes.md](request-shapes.md)。
`--input-length` 与 `--output-length` 成对指定，长请求独立目标不得静默裁剪到模型上限。
训练点还应覆盖源码确认的窗口回绕、空压缩流及 TP 补齐条件；`S_branch_train` 是按模型
分析后补入的候选点，不是通用模型的固定必采列表。余数相关分支需要不同余数的样本。
四点非共线只支持候选公式辨识，不保证每个结构体都获得足够样本，也不证明未知台阶。
规划后必须读实际构建报告，检查固定结构分桶、结构适用条件与 failed_formulas；结构不一致的样本分别建桶，不通过折叠或桶合并补出未知结构。

| 状态 | 必要证据 |
|---|---|
| compared_pass | 排除训练等价几何，模板直接生成，算子序、shape、标量、dtype 与引用关系比较通过；stride 另列 strict_passed |
| generated_only | 已保存候选，没有成功独立比较 |
| refused | 模板未能生成，记录原拒绝原因 |
| mismatch | 已生成但比较失败，保留差异 |
| unrun/budget | 需求范围内尚未跑或预算耗尽 |
| rejected | 官方配置/实际捕获拒绝，区分模型限制与未定位错误 |

exact_group/interpolated 不是实测验收。精确锚点可用，但不替代 B/S 外推；相同 `(K,S)`
的全局 B 改写不算独立目标。固定完整目标集后分别报告生成率、正确命中率、拒绝、失配和未跑，
Prefill/Decode 分开统计。失配后的修复回归与最初独立结果分开保存，不能修改训练集掩盖失败。
非训练配置抽查命中率不等于实际寻优流量命中率，后者须真实执行官方 optimizer 并按配置去重统计。
新语料绑定官方模型配置参数。引擎还有经验准入规则；
不同模型必须独立采集/构建/比较，不能从支持 model-id 推出所有模型已适配。
