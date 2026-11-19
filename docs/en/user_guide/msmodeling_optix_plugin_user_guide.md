# OptiX Plugin Developer Guide

## Overview

OptiX supports custom plugins. You can develop your own plugins to customize search parameters, service frameworks, and performance benchmark tools.

## Applicable Users and Prerequisites

This document applies to developers who need to extend the OptiX optimization capabilities. Before you start, confirm the following:

- You have installed msModeling and OptiX, and `msmodeling optix -h` runs properly.
- You are familiar with Python package structures, `pyproject.toml`, and the entry points mechanism.
- You have identified what needs to be extended: search parameter configuration, service frameworks, or performance benchmark tools.

> [!NOTE]
> If the plugin starts the deployment stack by creating a `subprocess` on its own, use `optix.deploy_env.build_deploy_env` to build the subprocess environment instead of passing `os.environ` directly. If the first element of the command is an executable name, call `materialize_command(self.command, self.env, ctx)` to resolve it to an absolute path before starting the subprocess. For the API description, see [`optix/deploy_env.py`](../../../optix/deploy_env.py).
> [!NOTE]
> In the following examples, the `optix.*` module paths depend on the installed OptiX version. If the current repository branch does not contain the OptiX source code, switch to a release branch that contains the OptiX code or install the corresponding release package.

The process of developing a custom plugin is as follows:

1. Create a Python project as a plugin.
2. Develop the custom plugin.

## Plugin Integration Process Overview

| Step | Objective | Key Operation |
| --- | --- | --- |
| 1 | Create the plugin project. | Create an independent Python project and prepare `pyproject.toml`. |
| 2 | Implement extension classes. | Inherit from `Settings`, `SimulatorInterface`, or `BenchmarkInterface` as needed. |
| 3 | Register the extension. | Implement `register()` in the plugin package and call the OptiX registration function. |
| 4 | Declare the entry point. | Add `[project.entry-points.'optix.plugins']` to `pyproject.toml`. |
| 5 | Install and verify. | Run `pip install -e .` and confirm that the extension is loaded by running `msmodeling optix -h`. |

## Procedure for Developing the Custom Plugin

### Customizing Search Parameters

1. Inherit from the `Settings` class.

    `settings` is implemented through `pydantic-settings`. You can customize the class by adding or removing attributes. Example:

    ```python
    from optix.config.config import Settings
    class CusSettings(Settings):
        name: str = "vllm-inference-optimization"
    ```

2. Register the `settings` with its initialization function.

    Add a registration function to your Python project to register your `Settings` class with its initialization method. Example:

    ```python
    def register():
        from vllm_inference_optimization.settings import CusSettings
        from optix.config.config import register_settings
        register_settings(lambda : CusSettings())
    ```

3. Use the `settings`.

    Import `get_settings` to retrieve your custom `settings`. Example:

    ```python
    from optix.config.config import get_settings
    settings = get_settings()
    ```

### Customizing a Service Framework

1. Inherit from `optix.optimizer.interfaces.simulator.SimulatorInterface`, and implement the `base_url` property and the `update_command` method. Example:

    ```python
    from optix.optimizer.interfaces.simulator import SimulatorInterface


    class VllmSimulator(SimulatorInterface):
        """Adapter for a vLLM service framework.

        ``data_field``, ``update_config`` and ``stop`` are provided by the base
        classes and work as they are. Implement the required abstract members
        below.
        """

        @property
        def base_url(self) -> str:
            # Return the health-check URL of the vLLM service.
            ...

        def update_command(self) -> None:
            # Build the service startup command from data_field before startup
            # and update self.command.
            ...
    ```

2. Register the service framework in the `__init__.py` file. Example:

    ```python
    from optix.optimizer.register import register_simulator
    register_simulator("vllm_infer", VllmSimulator)
    ```

    If the plugin depends on an executable in `PATH`, such as `vllm`, declare `required_executable = "vllm"` on the implementation class. The framework automatically validates it before constructing the plugin for `-e`. You do not need to write `shutil.which` in the plugin. Engines that use a fixed installation path, such as MindIE, can omit this attribute.

### Customizing a Performance Benchmark Tool

1. Inherit from `optix.optimizer.interfaces.benchmark.BenchmarkInterface` and implement the `update_command` method and the `get_performance_index` method.
Example:

    ```python
    from optix.optimizer.interfaces.benchmark import BenchmarkInterface
    from optix.config.config import PerformanceIndex


    class VllmBenchMark(BenchmarkInterface):
        """Adapter for a vLLM benchmark tool.

        ``data_field``, ``num_prompts`` and ``stop`` are provided by the base
        classes and work as they are. Implement the required abstract members
        below.
        """

        def update_command(self) -> None:
            # Build the benchmark startup command from data_field before startup
            # and update self.command.
            ...

        def get_performance_index(self) -> PerformanceIndex:
            # Collect and return the performance metrics.
            ...
    ```

2. Register the benchmark tool in the `__init__.py` file. Example:

    ```python
    from optix.optimizer.register import register_benchmarks
    register_benchmarks("vllm_infer_benchmark", VllmBenchMark)
    ```

    If the benchmark tool depends on a CLI in `PATH`, declare `required_executable = "your_tool"` on the implementation class. The framework automatically validates it before constructing the plugin for `-b`.

3. Set the plugin entry point.

    Add your custom registration function to the entry point group 'optix.plugins'.
    For example, to register the `register` function from the `vllm_inference_optimization` module:

    ```toml
    [project.entry-points.'optix.plugins']
    vllm_inference_optimization = "vllm_inference_optimization:register"
    ```

4. Install the plugin.
    Set the entry point to `optix.plugins`. Example:

    ```toml
    [project.entry-points.'optix.plugins']
    vllm_inference_optimization="vllm_inference_optimization:register"
    ```

    Before using the plugin mode, install the plugin in the plugin directory (ensure that the current path contains `pyproject.toml`). For example:

    ```bash
    pip install -e .
    ```

5. Use the plugin.

    You can specify which plugin modules to use via the OptiX CLI arguments.
    For example, after registering the service framework `vllm_infer` and the benchmark tool `vllm_infer_benchmark`, check whether they appear in the supported services and benchmark tools.

    ```bash
    msmodeling optix -h
    ```

    ```text
    options:
    -h, --help            show this help message and exit
    --load-breakpoint     Continue from where the last optimization was aborted.
    --backup              Back up optimizer data.
    -e {vllm,vllm_infer}, --engine {vllm,vllm_infer}
                          Engine used for model evaluation.
    -b {vllm_benchmark,vllm_infer_benchmark,ais_bench}, --benchmark-policy {vllm_benchmark,vllm_infer_benchmark,ais_bench}
                          Benchmark used for custom performance indicators.
    ```

    Run optimization using the specified plugin module:

    ```bash
    msmodeling optix -e vllm_infer -b vllm_infer_benchmark
    ```

## P/D Service Parameter Search Plugin Extension Point

The built-in `--mode pd_disagg` still uses the existing `register_simulator()` and `register_benchmarks()` registration entry points. A P/D phase plugin can optionally implement the following:

```python
def configure_pd_phase(self, *, phase_name, phase_config):
    """Refresh the command or traffic semantics based on the prefill/decode phase."""
```

A Prefill benchmark plugin must set the output length to 1 token in the Prefill phase so that the request ends after the first token, minimizing the effect of the Decode
process on the measurement. The parameter name is determined by the specific benchmark. For example, pass the equivalent `output_len=1` through its own command configuration. Both P/D phase plugins must return the request throughput as a positive `throughput` value in req/s. The core orchestrator uses this value directly as the QPS of each phase and does not fall back to formulas based on TTFT, TPOT, or the concurrency level.

`pd_disagg` calls this extension point only to search for Prefill and Decode service parameters and recommend a ratio. After you start the complete PD service, use the regular
`standard` mode to set `skip_pso=true` and `manage_simulator_lifecycle=false` to adjust the benchmark-side concurrency and request rate.
This step does not call the PD simulator plugin. The TTFT/TPOT values output by the plugin must be in seconds, and throughput must be in req/s.
