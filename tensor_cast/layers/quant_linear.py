from typing import Optional

import torch

from .. import ops  # noqa: F401
from ..model_config import LinearQuantConfig
from ..quantize_utils import (
    LinearQuantType,
    quant_type_to_dynamic_quant_dtype,
    quant_type_to_weight_dtype,
    QuantGranularity,
    QuantScheme,
)
from ..utils import DTYPE_FP4, DTYPE_FP8


class QuantLinearBase(torch.nn.Module):
    """
    A quantized linear layer that replaces a standard torch.nn.Linear layer.
    It handles different quantization types for weights and activations.
    """

    def __init__(self, linear_layer: torch.nn.Linear, quant_config: LinearQuantConfig):
        super().__init__()
        # TODO(jgong5): we should use `linear_layer.in_features` and `linear_layer.out_features`
        #               instead but they are not set correctly when the linear layer is sharded.
        self.in_features = linear_layer.weight.shape[1]
        self.out_features = linear_layer.weight.shape[0]
        self.quant_config = quant_config
        self.dynamic_quant_dtype = quant_type_to_dynamic_quant_dtype(quant_config.quant_type)

        # Store bias if it exists
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.clone())
        else:
            self.register_buffer("bias", None)

        q_weight_dtype = quant_type_to_weight_dtype(quant_config.quant_type)

        weight_scale = quant_config.weight_scale
        weight_offset = quant_config.weight_offset
        if quant_config.weight_scale is None:
            weight_scale, weight_offset = self._calculate_dynamic_qparams(
                linear_layer.weight.data,
                q_weight_dtype,
                quant_config.weight_quant_granularity,
                quant_config.weight_quant_scheme,
                quant_config.weight_group_size,
            )
        # Register scales and offsets
        self.register_buffer("weight_scale", weight_scale)
        if weight_offset is not None:
            self.register_buffer("weight_offset", weight_offset)
        else:
            self.register_buffer("weight_offset", None)

        # Store group_size for MXFP4
        if quant_config.quant_type == LinearQuantType.MXFP4:
            # Calculate group_size from weight_scale shape
            # weight_scale shape is (K_group,)
            K_group = self.weight_scale.shape[0]
            self.group_size = (self.in_features + K_group - 1) // K_group
        else:
            self.group_size = None

        # Quantize the weight
        quantized_weight = self.quantize_weight(
            linear_layer.weight.data,
            self.weight_scale,
            self.weight_offset,
            q_weight_dtype,
        )

        if self.quant_config.quant_type == LinearQuantType.W4A8:
            # Pack two 4-bit values into a single int8
            quantized_weight = self.pack_int4(quantized_weight)

        quantized_weight = quantized_weight.transpose(0, 1).contiguous().transpose(0, 1)
        self.register_buffer("qweight", quantized_weight)

        if quant_config.activation_scale is not None:
            self.register_buffer("activation_scale", quant_config.activation_scale)
        else:
            self.register_buffer("activation_scale", None)

        if quant_config.activation_offset is not None:
            self.register_buffer("activation_offset", quant_config.activation_offset)
        else:
            self.register_buffer("activation_offset", None)

    def quantize_weight(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        offset: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generic quantization function."""
        if offset is not None:
            # Asymmetric quantization
            tensor = tensor / scale + offset
        else:
            # Symmetric quantization
            # Special handling for group-wise quantization (used by MXFP4)
            if self.quant_config.quant_type == LinearQuantType.MXFP4:
                assert self.group_size is not None
                out_features, in_features = tensor.shape
                num_groups = (in_features + self.group_size - 1) // self.group_size

                padded_in_features = num_groups * self.group_size
                if in_features < padded_in_features:
                    pad_size = padded_in_features - in_features
                    tensor = torch.nn.functional.pad(tensor, (0, pad_size))

                tensor_grouped = tensor.reshape(out_features, num_groups, self.group_size)

                if scale.ndim == 1:
                    scale = scale.reshape(1, -1, 1)
                else:
                    raise ValueError(f"Unsupported scale shape for MXFP4: {scale.shape}")

                quantized_grouped = tensor_grouped / scale
                tensor = quantized_grouped.reshape(out_features, padded_in_features)

                if in_features < padded_in_features:
                    tensor = tensor[:, :in_features]
            else:
                tensor = tensor / scale

        # For FP8, we don't round, just cast. For integer formats, we do.
        if dtype == DTYPE_FP8:
            return tensor.to(dtype)

        # TODO(jgong5): we should use a special conversion for mxfp4 while we treat
        #               it similarly as integer quantization now since we always use
        #               meta tensor for mxfp4 simulation.
        return tensor.round().to(dtype)

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantizes the weight tensor."""
        if self.quant_config.quant_type == LinearQuantType.W4A8:
            unpacked_qweight = self.unpack_int4(self.qweight)
        else:
            unpacked_qweight = self.qweight

        weight = unpacked_qweight.to(self.weight_scale.dtype)
        if self.weight_offset is not None:
            return (weight - self.weight_offset) * self.weight_scale
        return weight * self.weight_scale

    def pack_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """Packs a tensor of int8 values (where each value is in [-8, 7]) into an int8 tensor."""
        # Ensure values are in the correct range for int4
        tensor = tensor.clamp(-8, 7)
        # Shift to be non-negative for bitwise operations
        high_bits = (tensor[:, ::2] + 8).to(torch.uint8)
        low_bits = (tensor[:, 1::2] + 8).to(torch.uint8)
        return (high_bits << 4) | low_bits

    def unpack_int4(self, packed_tensor: torch.Tensor) -> torch.Tensor:
        """Unpacks an int8 tensor into two int4 tensors (represented as int8)."""
        high_bits = (packed_tensor >> 4).to(torch.int8) - 8
        low_bits = (packed_tensor & 0x0F).to(torch.int8) - 8
        # Interleave the unpacked values
        unpacked = torch.empty(
            self.out_features,
            self.in_features,
            dtype=torch.int8,
            device=packed_tensor.device,
        )
        unpacked[:, ::2] = high_bits
        unpacked[:, 1::2] = low_bits
        return unpacked

    def _calculate_dynamic_qparams(
        self,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        quant_granularity: QuantGranularity,
        quant_scheme: QuantScheme,
        group_size: int = 0,
        group_dim: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates scale and offset for dynamic activation quantization."""
        if self.quant_config.quant_type == LinearQuantType.MXFP4 and not x.is_meta:
            raise ValueError(
                "MXFP4 simulation is not supported in QuantLinearBase for non-meta tensors, "
                "use TensorCastQuantLinear instead."
            )
        x = x.reshape(-1, x.shape[-1])

        if quant_dtype == torch.int8:
            q_min, q_max = -128.0, 127.0
        elif quant_dtype == DTYPE_FP8:
            # FP8 uses the full representable range
            q_max = torch.finfo(DTYPE_FP8).max
            q_min = -q_max  # Symmetric range for FP8
        elif quant_dtype == DTYPE_FP4:
            # simulate FP4 with int4, we always use meta tensor for DTYPE_FP4
            # so how to get the scales is not important
            q_max = 7.0
            q_min = -8.0
        else:
            raise ValueError(f"Unsupported dynamic quant dtype: {quant_dtype}")

        # Determine reduction dimensions and calculate min/max based on granularity
        if quant_granularity == QuantGranularity.PER_TENSOR:
            min_val = torch.amin(x)
            max_val = torch.amax(x)
        elif quant_granularity == QuantGranularity.PER_SAMPLE:
            dim = tuple(range(1, x.ndim))  # Reduce over all dims except batch
            min_val = torch.amin(x, dim=dim)
            max_val = torch.amax(x, dim=dim)
        elif quant_granularity == QuantGranularity.PER_GROUP:
            # Group-wise quantization along the specified dimension
            assert group_size > 0, "group_size must be greater than 0 for PER_CHANNEL_GROUP"
            group_dim = x.ndim - 1

            # Get the size of the dimension to group
            dim_size = x.shape[group_dim]
            num_groups = (dim_size + group_size - 1) // group_size

            # Pad the tensor if necessary to make it divisible by group_size
            padded_dim_size = num_groups * group_size
            if dim_size < padded_dim_size:
                pad_size = padded_dim_size - dim_size
                x = torch.nn.functional.pad(x, (0, pad_size))

            # Reshape to expose groups
            x_grouped = x.reshape(*x.shape[:-1], num_groups, group_size)

            # reduction dim is all dims except for num_groups
            reduce_dim = tuple(range(0, x_grouped.ndim - 2)) + (x_grouped.ndim - 1,)
            min_val = torch.amin(x_grouped, dim=reduce_dim)
            max_val = torch.amax(x_grouped, dim=reduce_dim)
        else:
            raise ValueError(f"Unknown granularity: {quant_granularity}")

        if quant_scheme == QuantScheme.SYMMETRIC:
            max_abs = torch.maximum(torch.abs(min_val), torch.abs(max_val))
            scale = max_abs / q_max
            offset = None
        elif quant_scheme == QuantScheme.ASYMMETRIC:
            assert self.dynamic_quant_dtype != DTYPE_FP8, "FP8 only supports symmetric quantization"
            scale = (max_val - min_val) / (q_max - q_min)
            offset = q_min - min_val / scale
        else:
            raise ValueError(f"Unknown scheme: {quant_scheme}")

        scale = torch.where(scale == 0, 1.0, scale)  # Avoid division by zero
        return scale, offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the quantized linear layer.
        """
        if self.quant_config.quant_type == LinearQuantType.MXFP4:
            raise ValueError("MXFP4 simulation is not supported in QuantLinearBase, use TensorCastQuantLinear instead.")
        # Dequantize weights
        dequantized_weight = self.dequantize_weight().to(x.dtype)

        # Only quantize activations for W8A8 or W4A8 or FP8 types
        if self.quant_config.quant_type in [
            LinearQuantType.W8A8,
            LinearQuantType.W4A8,
            LinearQuantType.FP8,
        ]:
            # Use pre-computed static parameters if available
            if self.activation_scale is not None:
                act_scale = self.activation_scale
                act_offset = self.activation_offset
            # Otherwise, compute parameters dynamically
            else:
                assert self.quant_config.dynamic_quant_granularity is not None
                assert self.quant_config.dynamic_quant_scheme is not None
                act_scale, act_offset = self._calculate_dynamic_qparams(
                    x,
                    self.dynamic_quant_dtype,
                    self.quant_config.dynamic_quant_granularity,
                    self.quant_config.dynamic_quant_scheme,
                )
            if act_offset is None:
                act_offset = 0.0

            if self.quant_config.quant_type == LinearQuantType.FP8:
                # For FP8, we directly convert to FP8 format without rounding
                x = (x / act_scale).to(DTYPE_FP8)
                # Scale it back for computation
                x = x.to(dequantized_weight.dtype) * act_scale
            else:
                # For integer quantization (W8A8, W4A8)
                q_x = torch.round(x / act_scale + act_offset)
                x = (q_x - act_offset) * act_scale

        # Perform linear operation
        output = torch.nn.functional.linear(x, dequantized_weight, self.bias)

        out_dtype = self.quant_config.out_dtype if self.quant_config is not None else x.dtype
        return output.to(out_dtype)

    def __repr__(self):
        return (
            f"QuantLinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_type={self.quant_config.quant_type})"
        )


class TensorCastQuantLinear(QuantLinearBase):
    def __init__(self, linear_layer: torch.nn.Linear, quant_config: LinearQuantConfig):
        super().__init__(linear_layer, quant_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the quantized linear operation using custom tensor_cast ops.

        This method selects the appropriate custom operator based on the
        availability of static activation quantization parameters.
        """
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        qweight = self.qweight.transpose(0, 1)
        out_dtype = self.quant_config.out_dtype if self.quant_config.out_dtype is not None else x.dtype
        if self.activation_scale is None:
            # Dynamic quantization path
            if self.quant_config.dynamic_quant_granularity == QuantGranularity.PER_TENSOR:
                dims = []
            elif self.quant_config.dynamic_quant_granularity == QuantGranularity.PER_SAMPLE:
                dims = [-1]
            else:
                dims = []

            if self.quant_config.quant_type == LinearQuantType.MXFP4:
                # MXFP4 dynamic quantization
                x, activation_scale = torch.ops.tensor_cast.dynamic_quantize_mxfp4(
                    x,
                    group_size=self.group_size,
                )
                activation_offset = None
            elif self.quant_config.dynamic_quant_scheme == QuantScheme.SYMMETRIC:
                x, activation_scale = torch.ops.tensor_cast.dynamic_quantize_symmetric(
                    x,
                    dims=dims,
                    scale_dtype=self.weight_scale.dtype,
                    out_dtype=torch.int8,
                )
                activation_offset = None
            else:
                assert self.quant_config.dynamic_quant_scheme == QuantScheme.ASYMMETRIC
                x, activation_scale, activation_offset = torch.ops.tensor_cast.dynamic_quantize_asymmetric(
                    x,
                    dims=dims,
                    scale_dtype=self.weight_scale.dtype,
                    out_dtype=torch.int8,
                )
        else:
            # Static quantization path
            assert self.quant_config.quant_type != LinearQuantType.FP8, (
                "FP8 does not support static activation quantization"
            )
            activation_scale = self.activation_scale
            activation_offset = self.activation_offset
            x = torch.ops.tensor_cast.quantize(
                x,
                activation_scale,
                activation_offset,
                out_dtype=torch.int8,
            )

        if self.quant_config.quant_type == LinearQuantType.FP8:
            # Use FP8 linear operation
            out = torch.ops.tensor_cast.fp8_linear(
                x,
                qweight,
                x_scale=activation_scale,
                w_scale=self.weight_scale,
                bias=self.bias,
                out_dtype=out_dtype,
            )
        elif self.quant_config.quant_type == LinearQuantType.MXFP4:
            # Use MXFP4 linear operation
            out = torch.ops.tensor_cast.mxfp4_linear(
                x,
                qweight,
                x_scale=activation_scale,
                w_scale=self.weight_scale,
                bias=self.bias,
                out_dtype=out_dtype,
            )
        else:
            # Use integer quantization operations
            op = (
                torch.ops.tensor_cast.static_quant_linear_int4
                if self.quant_config.quant_type == LinearQuantType.W4A8
                else torch.ops.tensor_cast.static_quant_linear
            )
            out = op(
                x,
                qweight,
                self.weight_scale,
                w_offset=self.weight_offset,
                x_scale=activation_scale,
                x_offset=activation_offset,
                bias=self.bias,
                out_dtype=out_dtype,
            )
        return out.reshape(*x_shape[:-1], out.shape[-1]).to(out_dtype)
