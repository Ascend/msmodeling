def register():
    from multihost_inference_optimization.simulator import MultiHostSimulator
    from multihost_inference_optimization.settings import CusSettings
    from optix.optimizer.register import register_simulator
    from optix.config.config import register_settings

    register_simulator("multihost_infer", MultiHostSimulator)
    register_settings(lambda: CusSettings())
