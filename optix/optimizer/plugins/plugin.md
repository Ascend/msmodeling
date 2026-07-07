# Custom Plugin Development Guide

## Introduction

The optimization tool supports custom plugins, through which users can customize search parameter configurations, custom service frameworks, and custom performance testing tools.

## Logging and error behavior

OptiX configures loguru in `optix/logging.py` before `optimizer.main()` runs. Plugin code should follow these rules:

1. **Structured context**: Prefer `logger.bind(stage=...)` or `with logger.contextualize(...)` instead of embedding run context in message strings. Do not log inside helpers that only raise (Scheduler `_handle_error` pattern).
2. **Levels**: Use `info` / `success` for milestones, `warning` for recoverable plugin issues, `debug` for subprocess I/O detail, `trace` for hot-path detail. Do **not** call `logger.error` for fatal business failures inside plugins or Scheduler — raise `OptimizerError` / `SubprocessError` / `BenchmarkResultError` and let `_main()` log once at the boundary.
3. **Fail-fast benchmark output**: Custom `BenchmarkInterface` implementations must raise `BenchmarkResultError` (from `optix.optimizer.errors`) when result CSV files are missing or ambiguous — do not log and continue with `files[0]`. This terminates the CLI run immediately (not per-particle PSO failure).
4. **Executable PATH check**: Declare `required_executable: ClassVar[str | None] = "your_cli"` on `BenchmarkInterface` or `SimulatorInterface` subclasses. After `register_benchmarks` / `register_simulator`, OptiX validates `PATH` for `-b` / `-e` before constructing plugins. Omit or set `None` when the plugin uses a fixed install path (e.g. MindIE daemon) or has no CLI dependency.
5. **Health checks**: Return fatal/retryable results through the health-check hooks; let Scheduler raise `FatalError` / `RetryableError` with `logger.debug` for diagnostics only.
6. **Log level env**: Users set `OPTIX_LOG_LEVEL` (`INFO` / `DEBUG` / `TRACE`); `MODELEVALSTATE_LEVEL` remains a legacy alias. At `DEBUG`/`TRACE`, console lines include `file:line` for troubleshooting.

If optimization ends with no feasible candidate, `_main()` raises `NoFeasibleSolutionError` and exits with code `1`. Other `OptimizerError` subclasses log a single error line inside the run `contextualize` block; unexpected exceptions log one traceback at that same boundary before `SystemExit(1)`.

## Custom Plugin Development Steps

The steps are as follows:

1. Create your own Python project as a plugin.
2. Custom content:

**Custom Configuration**

- Inherit the Settings class

  Settings is implemented through pydantic-settings, and attributes can be added/removed in the class. For example

  ```python
  from ..config.config import Settings


  class CusSettings(Settings):
      name: str = "vllm-inference-optimization"
  ```

- Register settings initialization function

  Add a registration function in your Python project to register the Settings initialization

  ```python
  def register():
      from vllm_inference_optimization.settings import CusSettings
      from ..config.config import register_settings
      register_settings(lambda : CusSettings())
  ```

- Using settings
  Import get_settings to obtain

  ```python
  from ..config.config import get_settings
  settings = get_settings()
  ```

**Custom Service Framework**

- Inherit msserviceprofiler...optimizer.interfaces.simulator.SimulatorInterface, implement the base_url and data_field properties, implement the update_command method, etc.
  For example:

  ```python
   class ms_service_profiler...optimizer.interfaces.simulator.SimulatorInterface()
      Bases: ABC

      Operate the service framework. Used for service-related operations.

      abstract property data_field: Tuple[OptimizerConfigField] | None
          Get data field property
          Returns: Optional[Tuple[OptimizerConfigField]]

      abstract property setter data_field: Tuple[OptimizerConfigField] | None
          Set data field property
          Returns: None

      abstract update_command() → None
          Update the service startup command based on data_field before service startup. Update the self.command property.
          Returns: None

      update_config(params: Tuple[OptimizerConfigField] | None = None) → bool
          Update the service configuration file or other configurations based on parameters. Modify the configuration file before service startup according to the passed parameter values to make the new configuration take effect.
          Args:

              params: List of tuning parameters, a tuple, defined by the value and config position of each element.

          Returns: bool, returns update success or failure.

      abstract stop()
          Other preparations at runtime.
          Returns: None
  ```

- And register the service framework in the init file

  ```python
   from ..optimizer.register import register_simulator
   register_simulator("vllm_infer", VllmSimulator)
  ```

**Custom Performance Testing Benchmark**

- Inherit msserviceprofiler...optimizer.benchmark.BenchmarkInterface, implement the data_field property, get_performance_index method, etc.
  For example:

  ```python
  class VllmBenchMark(BenchmarkInterface):
      def __init__(self, config: Optional[VllmBenchmarkConfig] = None, *args, **kwargs):

          if config:
              self.config = config
          else:
              settings = get_settings()
              if settings.name != "vllm-inference-optimization":
                  raise ValueError("Settings is invalidator.")
              self.config = settings.vllm_benchmark
          super().__init__(*args, **kwargs)
          self.command = VllmBenchmarkCommand(self.config.command).command
  ```

- Register benchmark

  ```python
  from ..optimizer.register import register_benchmarks
  register_benchmarks("vllm_infer_benchmark", VllmBenchMark)
  ```

## III. Setting Plugin Entry Points

Add the registration function of the custom content to the entry group `optix.plugins`.

For example, register by calling the register function of the vllm_inference_optimization module as follows

pyproject.toml

```toml
[project.entry-points.'optix.plugins']
vllm_inference_optimization = "vllm_inference_optimization:register"
```

## IV. Using Plugins

Specify the plugin-implemented module through the optimization tool's invocation parameters.
For example, a newly registered service framework vllm_infer and performance testing client vllm_infer_benchmark
First check whether the supported services and benchmark tools include the newly registered ones

```bash
msserviceprofiler optimizer -h
```

```text
options:
-h, --help show this help message and exit
-lb, --load_breakpoint
Continue from where the last optimization was aborted.
--backup Whether to back up data.
-e {vllm, vllm_infer}, --engine {vllm, vllm_infer}
Specifies the engine to be used.
-b {vllm_benchmark, vllm_infer_benchmark}, --benchmark {vllm_benchmark, vllm_infer_benchmark}
Specified benchmark to be used.
```

Specify the plugin implementation for optimization

msserviceprofiler optimizer -e vllm_infer -b vllm_infer_benchmark

Common Data Structure Definitions

- ..config.config.OptimizerConfigField

  ```python
  class ..config.config.OptimizerConfigField(*, name: str = 'max_batch_size', config_position: str = 'BackendConfig.ScheduleConfig.maxBatchSize', min: float = 0.0, max: float = 100.0, dtype: str = 'float', value: int | float | bool | None = None, dtype_param: Any = None)
      Bases: BaseModel
      Structure definition of optimization parameters.

      config_position: str  # Position definition, currently supports two types: one is BackendConfig.ScheduleConfig, indicating modification of parameters in mindieconfig.json; the other is env, indicating setting this parameter as an environment variable, with the variable name being the name attribute
      dtype: str # Parameter type definition
      dtype_param: Any  # Additional parameters provided when converting data to the specified data type.
      max: float # Indicates the maximum value of this parameter, used as the upper limit of the parameter variation range
      min: float  # Indicates the minimum value of this parameter, used as the lower limit of the parameter variation range
      name: str  # Field name, when setting as an environment variable, convert it to all uppercase as the variable name. config_position is used to distinguish how to update the field value.
      value: int | float | bool | None  # Parameter value, e.g. modify this value into the configuration file, or set it as the value of an environment variable. dtype_param: Conversion parameters needed for type conversion.
  ```

- ..config.config.PerformanceIndex

  ```python
  class ..config.config.PerformanceIndex(*, generate_speed: float | None = None, time_to_first_token: float | None = None, time_per_output_token: float | None = None, success_rate: float | None = None, throughput: float | None = None)

      Bases: BaseModel
      Performance metrics obtained by benchmark.

      generate_speed: float| None  # Output throughput (token/s), recommended to pass
      success_rate: float | None # Percentage of successful request returns, recommended to pass
      throughput: float | None # qps, recommended to pass
      time_per_output_token: float | None  # tpot, recommended to pass
      time_to_first_token: float | None # ttft, recommended to pass
  ```

### Configuration Customization

..config.config.register_settings

```python
..config.config.register_settings(func: Callable | None = None) → None
Register custom settings, can provide a function to generate.
 Args:
 func: Function to generate settings.

 Returns: None
```

### Benchmark Interface

```python
class ..optimizer.benchmark.BenchmarkInterface()
    Bases: ABC
    property num_prompts: Tuple[OptimizerConfigField] | None
        Get the request count for data retrieval property
        Returns: Optional[Tuple[OptimizerConfigField]]

    property setter num_prompts: Tuple[OptimizerConfigField] | None
        Set the request count for data retrieval property
        Returns: None

    property data_field: Tuple[OptimizerConfigField] | None
        Get data field property
        Returns: Optional[Tuple[OptimizerConfigField]]

    abstract property setter data_field: Tuple[OptimizerConfigField] | None
        Set data field property
        Returns: None

    abstract get_performance_index() → PerformanceIndex
        Get performance metrics
        Returns: Metrics data class

    abstract stop()
        Other preparations at runtime.
        Returns: None

    abstract update_command() → None
        Update the service startup command based on data_field before service startup. Update the self.command property.
        Returns: None
```

Register benchmark in the init file

```python
class ..optimizer.interfaces.simulator.SimulatorInterface()
    Bases: ABC

    Operate the service framework. Used for service-related operations.

    abstract property data_field: Tuple[OptimizerConfigField] | None
        Get data field property
        Returns: Optional[Tuple[OptimizerConfigField]]

    abstract property setter data_field: Tuple[OptimizerConfigField] | None
        Set data field property
        Returns: None

    abstract update_command() → None
        Update the service startup command based on data_field before service startup. Update the self.command property.
        Returns: None

    update_config(params: Tuple[OptimizerConfigField] | None = None) → bool
        Update the service configuration file or other configurations based on parameters. Modify the configuration file before service startup according to the passed parameter values to make the new configuration take effect.
        Args:

            params: List of tuning parameters, a tuple, defined by the value and config position of each element.

        Returns: bool, returns update success or failure.

    abstract stop()
        Other preparations at runtime.
        Returns: None
```

Set plugin entry points

Add the registration function of the custom content to the entry group `optix.plugins`.

For example, register by calling the register function of the vllm_inference_optimization module as follows

pyproject.toml

```toml
[project.entry-points.'optix.plugins']
vllm_inference_optimization = "vllm_inference_optimization:register"
```

Install Plugin

Set the entry point group to `optix.plugins`. For example:

```toml
[project.entry-points.'optix.plugins']
vllm_inference_optimization = "vllm_inference_optimization:register"
```

Before using plugin mode, install the plugin in the plugin directory (ensure the current path contains pyproject.toml):

```bash
pip install -e .
```

Using Plugins

Specify the plugin-implemented module through the optimization tool's invocation parameters.

For example, a newly registered service framework vllm_infer and performance testing client vllm_infer_benchmark

First check whether the supported services and benchmark tools include the newly registered ones

```bash
msserviceprofiler optimizer -h
```

```text
options:
-h, --help show this help message and exit
-lb, --load_breakpoint
Continue from where the last optimization was aborted.
--backup Whether to back up data.
-e {vllm, vllm_infer}, --engine {vllm, vllm_infer}
Specifies the engine to be used.
-b {vllm_benchmark, vllm_infer_benchmark}, --benchmark {vllm_benchmark, vllm_infer_benchmark}
Specified benchmark to be used.
```

Specify the plugin implementation for optimization

```bash
msserviceprofiler optimizer -e vllm_infer -b vllm_infer_benchmark
```
