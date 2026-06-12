import torch
from torch._dynamo.utils import detect_fake_mode
from torch.fx.passes.shape_prop import ShapeProp


def shape_propagation(gm: torch.fx.GraphModule, inputs) -> torch.fx.GraphModule:
    ShapeProp(
        gm=gm,
        fake_mode=detect_fake_mode(inputs),
    ).propagate(*tuple(inputs))
    return gm
