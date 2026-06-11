# 自定义插件开发指导

## 简介

寻优工具支持自定义插件，用户可通过开发自定义插件实现：自定义搜索参数配置、自定义服务框架，以及自定义性能测试工具。

## 适用对象与前置条件

本文适用于需要扩展 OptiX 寻优能力的开发者。开始前请确认：

- 已完成 msModeling 与 OptiX 的安装，可正常执行 `msmodeling optix -h`。
- 熟悉 Python 包结构、`pyproject.toml` 与 entry points 机制。
- 已明确需要扩展的对象：搜索参数配置、服务框架或性能测试工具。

> [!NOTE]
> 以下示例中的 `optix.*` 模块路径以已安装的 OptiX 版本为准；如果当前仓库分支未包含 OptiX 源码，请先切换到包含 OptiX 代码的发布分支或安装对应发布包。

自定义插件开发流程如下：

1. 创建自己的Python项目作为插件。
2. 自定义插件开发。

## 插件接入流程概览

| 步骤 | 目标 | 关键操作 |
| --- | --- | --- |
| 1 | 创建插件工程 | 创建独立 Python 项目，并准备 `pyproject.toml`。 |
| 2 | 实现扩展类 | 按需继承 `Settings`、`SimulatorInterface` 或 `BenchmarkInterface`。 |
| 3 | 注册扩展 | 在插件包中实现 `register()`，调用 OptiX 的注册函数。 |
| 4 | 声明入口点 | 在 `pyproject.toml` 中添加 `[project.entry-points.'optix.plugins']`。 |
| 5 | 安装并验证 | 执行 `pip install -e .`，通过 `msmodeling optix -h` 确认扩展已加载。 |

## 自定义插件开发操作步骤

### 自定义参数搜索配置

1. 继承Settings类

    settings是通过pydantic-settings实现的，可在类里面添加删除属性。例如：

    ```python
    from optix.config.config import Settings
    class CusSettings(Settings):
        name: str = "vllm-inference-optimization"
    ```

2. 注册settings初始化函数

    在自己的Python项目里面添加注册函数，实现注册Settings初始化，例如：

    ```python
    def register():
        from vllm_inference_optimization.settings import CusSettings
        from optix.config.config import register_settings
        register_settings(lambda : CusSettings())
    ```

3. 使用settings

    使用时导入get_settings 来获取自定义的settings，例如：

    ```python
    from optix.config.config import get_settings
    settings = get_settings()
    ```

### 自定义服务框架

1. 继承optix.optimizer.simulator.SimulatorInterface，实现base_url和data_field property，实现update_command方法等。例如：

    ```python
    class optix.optimizer.simulator.SimulatorInterface()
        Bases: ABC
        #操作服务框架。用于操作服务相关功能。
        abstract property data_field: Tuple[OptimizerConfigField] | None
            #获取data field 属性 
            Returns: Optional[Tuple[OptimizerConfigField]]
        abstract property setter data_field: Tuple[OptimizerConfigField] | None
            #设置data field 属性 
            Returns: None
        abstract update_command() → None
            #服务启动前根据data_field更新服务启动命令。更新self.command属性。
            Returns: None
        update_config(params: Tuple[OptimizerConfigField] | None = None) → bool
            #根据参数更新服务的配置文件，或者其他配置，服务启动前根据传递的参数值 修改配置文件。使得新的配置生效。
            Args:
                #params: 调优参数列表，是一个元组，根据其中每一个元素的value和config position进行定义
            Returns: bool，返回更新成功或者失败。
        abstract stop()
            #运行时，其他的准备工作。 
            Returns: None
    ```

2. 并在init文件中注册服务框架，例如：

    ```python
    from optix.optimizer.register import register_simulator
    register_simulator("vllm_infer", VllmSimulator)
    ```

### 自定义性能测试工具

1. 继承optix.optimizer.benchmark.BenchmarkInterface，实现data_field property，get_performance_index方法等。
例如：

    ```python
    class optix.optimizer.benchmark.BenchmarkInterface():
        Bases: ABC
        property num_prompts: Tuple[OptimizerConfigField] | None
            #获取数据的请求数
            Returns: Optional[Tuple[OptimizerConfigField]]
            
        property setter num_prompts: Tuple[OptimizerConfigField] | None
            #设置获取数据的请求数
            Returns: None
            
        property data_field: Tuple[OptimizerConfigField] | None
            #获取data field属性 
            Returns: Optional[Tuple[OptimizerConfigField]]
            
        abstract property setter data_field: Tuple[OptimizerConfigField] | None
            #设置data field属性 
            Returns: None
        
        abstract get_performance_index() → PerformanceIndex
            #获取性能指标 
            #Returns: 指标数据类
        
        abstract stop()
            #运行时，其他的准备工作。 
            Returns: None
        
        abstract update_command() → None
            #服务启动前根据data_field更新服务启动命令。更新self.command属性。 
            Returns: None
    ```

2. 并在init文件中注册benchmark，例如：

    ```python
    from optix.optimizer.register import register_benchmarks
    register_benchmarks("vllm_infer_benchmark", VllmBenchMark)
    ```

3. 设置插件入口点

    将自定义的内容的注册函数添加到入口组'optix.plugins'即可。
    例如通过调用vllm_inference_optimization模块的register来注册，例如：

    ```python
    [project.entry-points.'optix.plugins']
    vllm_inference_optimization = "vllm_inference_optimization:register"
    ```

4. 安装插件
    入口设置为optix.plugins，例如：

    ```python
    [project.entry-points.'optix.plugins']
    vllm_inference_optimization="vllm_inference_optimization:register"
    ```

    使用插件模式前需要先在插件目录中（确保当前路径下包含pyproject.toml）对插件进行安装，例如：

    ```bash
    pip install -e .
    ```

5. 使用插件

    可以通过寻优工具的调用参数来指定插件实现的模块。
    例如，新注册了服务框架vllm_infer和性能测试客户端vllm_infer_benchmark，先查看支持的服务和benchmark工具，是否包含刚刚注册的vllm_infer和vllm_infer_benchmark。例如：

    ```bash
    msmodeling optix -h
    ```

    ```ColdFusion
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

    使用指定插件的实现进行寻优，例如：

    ```bash
    msmodeling optix -e vllm_infer -b vllm_infer_benchmark
    ```
