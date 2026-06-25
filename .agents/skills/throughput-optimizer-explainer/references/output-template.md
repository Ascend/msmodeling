# Output Template

Use this compact structure unless the user asks for a different format.

```text
结论：
- 合理性等级：
- 主要判断：
- 证据等级：
- 最大不确定性：

对比条件：
| 项 | 配置 A | 配置 B |

最优结果：
| 硬件 | throughput | TTFT | TPOT | concurrency | batch | parallel |

阶段分析：
1. Prefill
   - breakdown：
   - 解释：
2. Decode
   - breakdown：
   - 解释：

并行策略：
- ...

还需要验证：
- ...
```

If only one result is provided, replace the comparison tables with a single-result summary.
