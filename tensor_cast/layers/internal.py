import torch

from .. import ops  # noqa: F401
from .utils import ModelWrapperBase


class RegionMarkerWrapper(ModelWrapperBase):
    def __init__(
        self,
        region_id: int,
        layer: torch.nn.Module,
        repeat_count: int = 1,
    ):
        """
        Wrap a layer with region markers.
        Args:
            region_id: The id of the region to mark.
            layer: Original layer instance to wrap
            repeat_count: Number of structurally equivalent layers represented by this
                wrapper. The first occurrence runs the real layer, the rest replay the
                marked region.
        """
        super().__init__(layer)
        self.region_id = region_id
        self.repeat_count = repeat_count
        self.returns_tuple = True
        self.return_length = 1

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        hidden_states = torch.ops.tensor_cast._internal_mark_region_begin(
            hidden_states,
            self.region_id,
        )
        # Keep the begin marker in the dataflow so compile/DCE cannot drop it
        # while preserving the paired end marker.
        args = (hidden_states,) + args[1:]
        result = self._inner.forward(*args, **kwargs)

        # Handle both single tensor and tuple returns
        if isinstance(result, tuple):
            self.returns_tuple = True
            self.return_length = len(result)
            # Extract the first element (hidden_states) from tuple
            hidden_states = result[0]
            hidden_states = torch.ops.tensor_cast._internal_mark_region_end(
                hidden_states,
                self.region_id,
            )
            # Return tuple with marked hidden_states and other elements
            return (hidden_states,) + result[1:]
        else:
            self.returns_tuple = False
            # Single tensor return
            hidden_states = result
            hidden_states = torch.ops.tensor_cast._internal_mark_region_end(
                hidden_states,
                self.region_id,
            )
            return hidden_states


class CopyLayerWrapper(torch.nn.Module):
    def __init__(
        self,
        region_id: int,
        layer: torch.nn.Module,
        representative: RegionMarkerWrapper,
    ):
        """
        Wrap a layer with a copy operation that copies a previously marked region.
        Args:
            region_id: The id of the range to repeat.
            layer: Original layer instance used only to copy lightweight metadata.
            representative: The representative layer whose return format should be mirrored.
        """
        super().__init__()
        self.region_id = region_id
        object.__setattr__(self, "representative", representative)
        for attr_name in ("attention_type", "layer_type"):
            if hasattr(layer, attr_name):
                setattr(self, attr_name, getattr(layer, attr_name))

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        # The following copy operation would be equivalent to:
        # result = self._inner.forward(*args, **kwargs)
        hidden_states = torch.ops.tensor_cast._internal_copy_region(
            hidden_states,
            self.region_id,
        )

        # For CopyLayerWrapper, we need to return the same format as the original layer.
        # Since we're copying a decoder layer, we need to return a tuple.
        # The decoder layer always returns at least (hidden_states,)
        # We'll construct a minimal tuple with just hidden_states and None for other outputs.

        # Check kwargs to determine what outputs are expected
        output_attentions = kwargs.get("output_attentions", False)
        use_cache = kwargs.get("use_cache", False)
        output_router_logits = kwargs.get("output_router_logits", False)

        if self.representative.returns_tuple:
            outputs = (hidden_states,)
            return_length = getattr(self.representative, "return_length", 1)

            if output_attentions:
                outputs += (None,)  # self_attn_weights

            if use_cache:
                outputs += (None,)  # present_key_value

            if output_router_logits:
                outputs += (None,)  # router_logits

            while len(outputs) < return_length:
                outputs += (None,)
        else:
            outputs = hidden_states

        return outputs
