import torch

from tensor_cast.transformers.utils import init_on_device_without_buffers


def test_init_on_device_without_buffers_moves_parameters_and_restores_factories() -> None:
    original_empty = torch.empty

    class TinyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("scale", torch.tensor([1.0]))

    with init_on_device_without_buffers("meta"):
        module = TinyModule()
        tensor = torch.zeros(1)

    assert module.weight.device.type == "meta"
    assert module.scale.device.type == "cpu"
    assert tensor.device.type == "meta"
    assert torch.empty is original_empty
    assert torch.empty(1).device.type == "cpu"
