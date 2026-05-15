# Serving Simulation

## Supported python versions

3.10+

> [!Warning]
> If you are using Windows, note that PyTorch 2.10 may not run properly on your system. For a solution, please refer to [this issue](https://github.com/pytorch/pytorch/issues/166628). If you have not yet installed PyTorch, for optimal compatibility, we strongly recommend using version 2.8 or earlier to ensure the program functions correctly.

## Run simulation

Its general usage is shown below:

```text
usage: python -m serving_cast.main [-h] --instance_config_path INSTANCE_CONFIG_PATH [INSTANCE_CONFIG_PATH ...] --common_config_path COMMON_CONFIG_PATH

Run a service inference simulation driven by JSON configuration files.

required arguments:
  --instance_config_path INSTANCE_CONFIG_PATH [INSTANCE_CONFIG_PATH ...]
                        Path to a YAML file that declares one or more instance groups.
                        Each group defines a homogeneous pool of nodes (role, count, TP/DP parallelism)
                        and can be mixed-and-matched in a single benchmark run.
  --common_config_path COMMON_CONFIG_PATH
                        Path to a YAML file with global settings: model architecture,
                        request-generation workload, and serving limits.

optional arguments:
  -h, --help            show this help message and exit
  --enable_profiling    Enable profiling during simulation (default: False)​
  --profiling_output_path PROFILING_OUTPUT_PATH​
                        Path to directory where profiling results will be saved (default: ./profiling_results)
```

example:

- basic usage

```bash
python -m serving_cast.main --instance_config_path=./serving_cast/example/instances.yaml --common_config_path=./serving_cast/example/common.yaml
```

- enable profiling

```bash
python -m serving_cast.main --instance_config_path=./serving_cast/example/instances.yaml --common_config_path=./serving_cast/example/common.yaml --enable_profiling​
```

- enable profiling with custom output path

```bash
python -m serving_cast.main --instance_config_path=./serving_cast/example/instances.yaml --common_config_path=./serving_cast/example/common.yaml --enable_profiling --profiling_output_path=/path/to/custom/profiling_dir
```

### Result

After the simulation finishes, a performance summary is printed to the console like following:

```text
         E2E_TIME(s)  TTFT(s)  TPOT(s)  INPUT_TOKENS  OUTPUT_TOKENS  OUTPUT_TOKEN_THROUGHPUT(tok/s)
AVERAGE     1052.591    0.378    0.301        1500.0         3500.0                           3.327
MIN         1050.000    0.300    0.300        1500.0         3500.0                           2.978
MAX         1175.500    0.600    0.336        1500.0         3500.0                           3.334
MEDIAN      1050.100    0.400    0.300        1500.0         3500.0                           3.334
P75         1050.125    0.400    0.300        1500.0         3500.0                           3.334
P90         1050.200    0.500    0.300        1500.0         3500.0                           3.334
P99         1175.500    0.600    0.336        1500.0         3500.0                           3.334
======== Overall Summary ========
benchmark_duration(s)          1225.500
total_requests                 100.000
request_throughput(req/s)      0.082
total_input_tokens             150000.000
input_token_throughput(tok/s)  122.399
total_output_tokens            350000.000
output_token_throughput(tok/s) 285.598
```

Metric descriptions:

- E2E_TIME: End-to-end latency per request (issue → last token)
- TTFT: Time-to-first-token
- TPOT: Time-per-output-token after the first token
- OUTPUT_TOKEN_THROUGHPUT: Per-request output-token rate
- request_throughput: System-wide request rate
- input_token_throughput / output_token_throughput: Aggregate token throughput

### Profiling

Profiling is supported in the simulation. You can get more specific information about the performance of the system by viewing the profiling result.

Use the following command to enable profiling:

- enable profiling

```bash
python -m serving_cast.main --instance_config_path=./serving_cast/example/instances.yaml --common_config_path=./serving_cast/example/common.yaml --enable_profiling​
```

- enable profiling with custom output path

```bash
python -m serving_cast.main --instance_config_path=./serving_cast/example/instances.yaml --common_config_path=./serving_cast/example/common.yaml --enable_profiling --profiling_output_path=/path/to/custom/profiling_dir
```

The original collected profiling result is stored in the directory ```profiling_output_path/{$time_stamp}```.
The parsed profiling result is stored in the directory ```profiling_output_path/{$time_stamp}_parsed_result```.

A ```chrome_tracing.json``` and a ```profiler.db``` will be generated in parsed_result directory, you can view it by ```chrome://tracing``` or MindStudio Insight
