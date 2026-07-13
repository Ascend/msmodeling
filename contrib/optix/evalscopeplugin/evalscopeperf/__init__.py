"""evalscopeperf 插件：封装 evalscope 的 perf 命令作为 optix 寻优器的benchmark。"""
def register():
    """向 optix 注册本插件的 benchmark 与配置。"""
    # 导入当前包内的模块
    from evalscopeperf.benchmark import EvalscopePerfBenchMark
    from evalscopeperf.settings import CusSettings
    from optix.optimizer.register import register_benchmarks
    from optix.config.config import register_settings
    # 注册插件
    register_benchmarks("evalscopeperf", EvalscopePerfBenchMark)
    register_settings(lambda : CusSettings())