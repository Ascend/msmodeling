import torch

from tensor_cast.layers.internal import CopyLayerWrapper, RegionMarkerWrapper
from tensor_cast.transformers.model import TransformerModel


class LayerWithMetadata(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(2, 3, device="meta"))
        self.register_buffer("scale", torch.empty(3, device="meta"))
        self.attention_type = "self"
        self.layer_type = "decoder"

    def forward(self, hidden_states):
        return hidden_states


def test_representative_layer_wrapper_metadata_and_weight_size():
    layer = LayerWithMetadata()
    representative = RegionMarkerWrapper(
        region_id=11,
        layer=layer,
        repeat_count=4,
    )
    copy_layer = CopyLayerWrapper(
        region_id=11,
        layer=layer,
        representative=representative,
    )
    container = torch.nn.Sequential(representative, copy_layer)

    assert representative.region_id == 11
    assert representative.repeat_count == 4
    assert copy_layer.region_id == 11
    assert copy_layer.representative == representative
    assert copy_layer.attention_type == layer.attention_type
    assert copy_layer.layer_type == layer.layer_type
    assert list(copy_layer.children()) == []
    assert list(copy_layer.named_children()) == []

    single_layer_weight_size = TransformerModel.get_weight_size_nested([layer])
    extra_weight_size = TransformerModel.get_represented_extra_weight_size(container)
    assert extra_weight_size == 3 * single_layer_weight_size
