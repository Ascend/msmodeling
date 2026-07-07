# Custom Plugin Developer Guide

## Overview

Service Parameter Optimizer supports custom plugins. You can develop your own plugins to customize search parameters, service frameworks, and performance benchmark tools.

The process of developing a custom plugin is as follows:

1. Create a Python project as a plugin.
2. Develop the custom plugin.

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

1. Inherit from `optix.optimizer.simulator.SimulatorInterface`, and implement the `base_url` and `data_field` properties and the `update_command` method. Example:

    ```python
    class optix.optimizer.simulator.SimulatorInterface()
        Bases: ABC
        #Operations on the service framework. This class manages service-related functions.
        abstract property data_field: Tuple[OptimizerConfigField] | None
            #Obtain the data field attribute.
            Returns: Optional[Tuple[OptimizerConfigField]]
        abstract property setter data_field: Tuple[OptimizerConfigField] | None
            #Set the data field attribute.
            Returns: None
        abstract update_command() → None
            #Update the service startup command based on data_field before service startup. Update the self.command attribute.
            Returns: None
        update_config(params: Tuple[OptimizerConfigField] | None = None) → bool
            #Update the service configuration file or other configurations based on the input parameter values, to apply new parameter values to the configuration.
            Args:
                #params: Tuple of tuning parameters, each defined by its value and config_position.
            Returns: bool, indicating update success or failure.
        abstract stop()
            #Perform any other necessary preparation during runtime.
            Returns: None
    ```

2. Register the service framework in the `__init__.py` file. Example:

    ```python
    from optix.optimizer.register import register_simulator
    register_simulator("vllm_infer", VllmSimulator)
    ```

    If your engine depends on a CLI on `PATH` (e.g. `vllm`), set `required_executable = "vllm"` on the class; OptiX validates before constructing the plugin for `-e`. Omit for fixed install paths (e.g. MindIE daemon).

### Customizing a Performance Benchmark Tool

1. Inherit from `optix.optimizer.benchmark.BenchmarkInterface` and implement the `data_field property` and `get_performance_index` methods.
Example:

    ```python
    class optix.optimizer.benchmark.BenchmarkInterface():
        Bases: ABC
        property num_prompts: Tuple[OptimizerConfigField] | None
            #Obtain the number of data retrieval requests.
            Returns: Optional[Tuple[OptimizerConfigField]]
            
        property setter num_prompts: Tuple[OptimizerConfigField] | None
            #Set the number of data retrieval requests.
            Returns: None
            
        property data_field: Tuple[OptimizerConfigField] | None
            #Obtain the data field attribute.
            Returns: Optional[Tuple[OptimizerConfigField]]
            
        abstract property setter data_field: Tuple[OptimizerConfigField] | None
            #Set the data field attribute.
            Returns: None
        
        abstract get_performance_index() → PerformanceIndex
            #Retrieve performance metrics.
            #Returns: metric data
        
        abstract stop()
            #Perform any other necessary preparation during runtime.
            Returns: None
        
        abstract update_command() → None
            #Update the service startup command based on data_field before service startup. Update the self.command attribute.
            Returns: None
    ```

2. Register the benchmark tool in the `__init__.py` file. Example:

    ```python
    from optix.optimizer.register import register_benchmarks
    register_benchmarks("vllm_infer_benchmark", VllmBenchMark)
    ```

    If the benchmark depends on a CLI on `PATH`, set `required_executable = "your_tool"` on the class; OptiX validates before constructing the plugin for `-b`.

3. Set the plugin entry point.

    Add your custom registration function to the entry point group 'optix.plugins'.
    For example, to register the `register` function from the `vllm_inference_optimization` module:

    ```toml
    [project.entry-points.'optix.plugins']
    vllm_inference_optimization = "vllm_inference_optimization:register"
    ```

4. Installing the Plugin
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

    You can specify which plugin modules to use via the Service Parameter Optimizer's command-line arguments.
    For example, after registering the service framework `service framework vllm_infer` and the benchmark tool `vllm_infer_benchmark`, check whether they appear in the supported services and benchmark tools: Example:

    ```bash
    msmodeling optix -h
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

    Run optimization using the specified plugin module:

    ```bash
    msmodeling optix -e vllm_infer -b vllm_infer_benchmark
    ```
