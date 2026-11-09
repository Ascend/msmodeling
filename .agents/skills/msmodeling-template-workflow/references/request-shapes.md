# 全局并发与请求长度

B 对应官方 CLI 的 `--num-queries`，表示全局并发请求数，不是输入 token 数或单卡批量。
本地批量 `K=ceil(B/DP)` 用官方 ParallelConfig 的 DP 计算，不能误写为 `ceil(B/TP)`。
输入生成在 Prefill/Decode 均进行批量切分；这不等于 Prefill 必须按 K 划分结构桶。

不启用 prefix cache、chunked prefill 或其他 speculative 算法时，官方寻优代表形状为：

| 阶段 | query_length | context_length | 模板 S |
|---|---:|---:|---:|
| Prefill | 输入长度 I | 0 | I |
| Decode | Q=MTP+1 | I+floor(O/2) | I+floor(O/2)+Q |

O 是输出长度。它不是 Decode 的 query_length；普通 Decode 的 Q 为 1。
脚本调用 `BaseThroughputOptimizer._resolve_forward_shape` 得到代表形状，保持与官方一致。
这里检查的是寻优使用的单次代表 forward，不表示逐 token 执行了完整输出过程。

输入 16384、输出 1024、MTP0 时，Prefill S=16384；Decode query=1、context=16896、S=16897。
例如固定 32 卡、EP32、TP16、DCP1、MOE-DP1 时 DP=2；B=255 对应本地批量 128。

```bash
python .agents/skills/msmodeling-template-workflow/scripts/workflow.py plan \
  --model-id deepseek-ai/DeepSeek-V4-Pro \
  --input-length 16384 --output-length 1024 \
  --decode-batch-coverage all \
  --run-dir ./template_runs/pro-long-request
```

该命令沿用默认典型离散配置范围，准备完整请求 B 区间对应的 Decode 本地批量训练点，
并增加 16K/1K 长请求验证点；不启动采集。仅需特定 32 卡/量化布局时使用相应自定义 `--scope`。
计划和采集数量由范围推导，不保证在默认采集预算内完成；执行前检查 plan.json 的计数及缺口。
超出模型上下文上限时报告，不裁剪后冒充原长度。

长请求若已是训练锚点，只能列作端点回放；独立外推目标要排除相同 `(本地批量,S)`。
当前 planner 遇到该冲突会报告错误，需重选训练点，不能更换全局 B 后仍声称独立验证。
