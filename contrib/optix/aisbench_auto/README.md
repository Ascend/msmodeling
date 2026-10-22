# AISBench Auto OptiX 插件

参考 EvalScope 插件，仅提供 OptiX 寻优接口：启动外部压测命令、替换搜索参数、清理/备份性能结果，并复用内置 AISBench 的 CSV/JSON 解析。

## 预热与多轮结果选择

直接配置原工具的 `aisbench_test.py` 即可，无需额外生成或安装辅助脚本。插件支持 `--dataset_type prefix_cache --prefix_test` 的前缀预热加正式压测，以及普通数据集的 `--repeat N` 重复压测。

插件只读取本轮子进程的 stdout/stderr 日志，按 `Current exp folder: <path>` 的出现顺序选择最后一次 AISBench 实验。`--prefix_test` 场景取预热后的正式结果；`--repeat N` 场景取第 N 轮结果，不计算多轮平均。原始各轮结果全部保留，选中的 CSV/JSON 复制到 `output_path/.optix_selected/`，并生成 `selection.json` 记录选择依据；指标读取及最优并发计算使用同一组结果，备份包含原始轮次和选择记录。

插件校验预期实验次数、子进程退出状态、最终实验路径和完整结果对。最后一轮失败、缺失 CSV/JSON、目录越界、同一结果目录被多轮复用或无法确定最终轮时均报错，不回退使用预热或更早轮次。实验次数按原工具语义判断：普通数据集取 `--repeat`，前缀数据集为一次正式轮，加上可选的一次预热，原工具在此前缀分支不使用 `--repeat`。

兼容仅输出一组 CSV/JSON 的其他命令：未启用多轮或独立预热、且没有实验日志标记时，仍按唯一结果读取。多组结果缺少本轮实验日志时明确报错，不按目录名或修改时间猜测。`.optix_selected` 为插件保留目录，外部工具不要向其中写入文件。

## 配置

```toml
[aisbench_auto]
work_path = "/opt/aisbench_auto_tools_prefix"
output_path = "outputs/default"

[aisbench_auto.command]
executable = "python3"
script = "aisbench_test.py"
others = "--dataset_type prefix_cache --prefix_test --repeat 1 --repeat_rate 50% --prefix_num 1 --seed 2 --input_len 2048 --output_len 512 --data_num 128"
```

`executable` 指定部署环境中的可执行文件，`script` 是必填的 AISBench 工具脚本路径，插件会在启动前校验它存在；`others` 只填写脚本参数。寻优参数通过 `$变量名` 引用，并在 `[[aisbench_auto.target_field]]` 定义，完整示例见 [config.example.toml](config.example.toml)。支持带引号的参数及请求率为 0。

插件自动追加 `--concurrency $CONCURRENCY --request_rate $REQUESTRATE`，取值来自当前寻优候选或 `target_field` 中的初始值，无需在 `others` 中填写这两个参数。`others` 仅用于其余参数。

所有 `$NAME` 占位符都必须由 `target_field` 初始值或本轮候选提供取值，本轮候选优先；变量名使用大写字母、数字和下划线，以字母开头。替换后若仍有占位符，插件会在清理结果目录及启动脚本前抛出 `ValueError`，按名称排序、去重列出缺失变量，例如 `Undefined benchmark parameters: SEED`。自动追加的 `CONCURRENCY` 和 `REQUESTRATE` 也须提供取值。

`output_path` 是工具的 AISBench 原生输出目录，相对路径以 `work_path` 为基准，须与工具自身的 `OUTPUT_DIR` 一致。必须使用专用、可删除的结果目录，每轮运行前会清理其中的内容。原始目录可以包含预热及多轮 CSV/JSON；插件按照上文规则选择唯一正式结果。

插件每轮重新解析实际路径（包含 `..` 和符号链接），禁止结果目录等于或包含工作目录，也禁止其与 OptiX 寻优结果目录（主配置的 `output`）相同、位于该目录内或包含该目录。工作目录下的专用子目录（例如 `outputs/default`）仍可使用，但不能与 OptiX 结果目录重叠；当前进程工作目录和用户家目录也禁止作为输出目录。路径已存在且不是目录时会报错。清理后仍有内容或无法检查时会中止本轮，不启动外部脚本。保护仅覆盖已知目录，其他重要目录仍须由用户避免配置为可清理的输出目录。

模型、服务地址、数据生成、前缀预热、鉴权和工具依赖均在原工具侧配置。插件不复制工具源码，也不读取或修改工具的 `config.py`。

## 安装和运行

```bash
python -m pip install --no-deps -e contrib/optix/aisbench_auto
msmodeling optix -e vllm -b aisbench_auto -c /path/to/config.toml
```

在实际部署机器准备原工具及其依赖，将示例路径替换为部署路径。性能统计、TTFT/TPOT 单位和最优并发计算复用内置 AISBench。

## 详细设计

模块划分、接口定义、运行流程及结果协议见[详细设计](DESIGN.md)。
