import torch

from .. import ops  # noqa: F401
from .utils import ModelWrapperBase


is_tuple = False


class RegionMarkerWrapper(ModelWrapperBase):
    def __init__(
        self,
        region_id: int,
        layer: torch.nn.Module,
    ):
        """
        Wrap a layer with region markers.
        Args:
            region_id: The id of the region to mark.
            layer: Original layer instance to wrap
        """
        super().__init__(layer)
        self.region_id = region_id

    def forward(self, *args, **kwargs):
        global is_tuple
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
            is_tuple = True
            # Extract the first element (hidden_states) from tuple
            hidden_states = result[0]
            hidden_states = torch.ops.tensor_cast._internal_mark_region_end(
                hidden_states,
                self.region_id,
            )
            # Return tuple with marked hidden_states and other elements
            return (hidden_states,) + result[1:]
        else:
            is_tuple = False
            # Single tensor return
            hidden_states = result
            hidden_states = torch.ops.tensor_cast._internal_mark_region_end(
                hidden_states,
                self.region_id,
            )
            return hidden_states


class CopyLayerWrapper(ModelWrapperBase):
    def __init__(
        self,
        region_id: int,
        layer: torch.nn.Module,
    ):
        """
        Wrap a layer with a copy operation that copies a previously marked region.
        Args:
            region_id: The id of the range to repeat.
            layer: Original layer instance to repeat from
        """
        super().__init__(layer)
        self.region_id = region_id

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

        if is_tuple:
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (None,)  # self_attn_weights

            if use_cache:
                outputs += (None,)  # present_key_value

            if output_router_logits:
                outputs += (None,)  # router_logits
        else:
            outputs = hidden_states

        return outputs
