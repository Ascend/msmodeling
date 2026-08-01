import torch
import torch._prims as prims
import torch.nn.functional as F

from ... import config

_RMS_NORM_DTYPE_LIST = [torch.float16, torch.bfloat16]
_LAYER_NORM_DTYPE_LIST = [torch.float16, torch.bfloat16, torch.float32]
_DEFAULT_EPS_SCALAR_WORKAROUND = {"eps": 1e-6}


def _create_pattern_result(pattern, replacement, example_inputs):
    return pattern, replacement, example_inputs, _DEFAULT_EPS_SCALAR_WORKAROUND


class TorchRMSNormPattern:
    """Match direct torch.rms_norm calls."""

    @staticmethod
    def create(shape=(2, 4)):
        def get_inputs():
            hidden_states = torch.empty(*shape, dtype=torch.bfloat16, device="meta")
            weight = torch.empty(shape[-1], dtype=torch.float32, device="meta")
            return [hidden_states, weight]

        def pattern(hidden_states, weight, eps):
            return torch.rms_norm(hidden_states, (shape[-1],), weight, eps)

        def replacement(hidden_states, weight, eps):
            return torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)

        return _create_pattern_result(pattern, replacement, get_inputs())


class LayerNormPattern:
    """Match functional layer normalization with optional affine parameters."""

    @staticmethod
    def create(dtype, affine: bool, shape=(2, 4)):
        def get_inputs():
            hidden_states = torch.empty(*shape, dtype=dtype, device="meta")
            if affine:
                weight = torch.empty(shape[-1], dtype=torch.float32, device="meta")
                bias = torch.empty(shape[-1], dtype=torch.float32, device="meta")
                return [hidden_states, weight, bias]
            return [hidden_states]

        def pattern_affine(hidden_states, weight, bias, eps):
            out = F.layer_norm(
                prims.convert_element_type(hidden_states, torch.float32),
                (shape[-1],),
                prims.convert_element_type(weight, torch.float32),
                prims.convert_element_type(bias, torch.float32),
                eps,
            )
            return prims.convert_element_type(out, dtype)

        def replacement_affine(hidden_states, weight, bias, eps):
            return torch.ops.tensor_cast.layer_norm(hidden_states, weight, bias, eps)

        def pattern_no_affine(hidden_states, eps):
            out = F.layer_norm(
                prims.convert_element_type(hidden_states, torch.float32),
                (shape[-1],),
                None,
                None,
                eps,
            )
            return prims.convert_element_type(out, dtype)

        def replacement_no_affine(hidden_states, eps):
            return torch.ops.tensor_cast.layer_norm(hidden_states, None, None, eps)

        if affine:
            return _create_pattern_result(pattern_affine, replacement_affine, get_inputs())
        return _create_pattern_result(pattern_no_affine, replacement_no_affine, get_inputs())


class LayerNormDecomposedPattern:
    """Match decomposed layer normalization graphs."""

    @staticmethod
    def create(dtype, affine: bool, shape=(2, 4), output_cast: bool = True):
        def get_inputs():
            hidden_states = torch.empty(*shape, dtype=dtype, device="meta")
            if affine:
                weight = torch.empty(shape[-1], dtype=torch.float32, device="meta")
                bias = torch.empty(shape[-1], dtype=torch.float32, device="meta")
                return [hidden_states, weight, bias]
            return [hidden_states]

        def _layer_norm_core(hidden_states, eps):
            hidden_states_fp32 = prims.convert_element_type(hidden_states, torch.float32)
            reduction_dim = hidden_states.ndim - 1
            variance, mean = torch.ops.aten.var_mean.correction(
                hidden_states_fp32,
                [reduction_dim],
                correction=0,
                keepdim=True,
            )
            centered = torch.ops.aten.sub.Tensor(hidden_states_fp32, mean)
            rstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, eps))
            return torch.ops.aten.mul.Tensor(centered, rstd)

        def pattern_affine(hidden_states, weight, bias, eps):
            out = _layer_norm_core(hidden_states, eps)
            out = torch.ops.aten.mul.Tensor(out, weight)
            out = torch.ops.aten.add.Tensor(out, bias)
            if output_cast:
                return prims.convert_element_type(out, dtype)
            return out

        def replacement_affine(hidden_states, weight, bias, eps):
            return torch.ops.tensor_cast.layer_norm(hidden_states, weight, bias, eps)

        def pattern_no_affine(hidden_states, eps):
            out = _layer_norm_core(hidden_states, eps)
            if output_cast:
                return prims.convert_element_type(out, dtype)
            return out

        def replacement_no_affine(hidden_states, eps):
            return torch.ops.tensor_cast.layer_norm(hidden_states, None, None, eps)

        if affine:
            return _create_pattern_result(pattern_affine, replacement_affine, get_inputs())
        return _create_pattern_result(pattern_no_affine, replacement_no_affine, get_inputs())


class TorchRMSNormDecomposedPattern:
    """Match decomposed torch RMS normalization graphs."""

    @staticmethod
    def create(dtype, shape=(2, 4)):
        def get_inputs():
            hidden_states = torch.empty(*shape, dtype=dtype, device="meta")
            weight = torch.empty(shape[-1], dtype=torch.float32, device="meta")
            return [hidden_states, weight]

        def pattern(hidden_states, weight, eps):
            hidden_states_fp32 = prims.convert_element_type(hidden_states, torch.float32)
            reduction_dim = hidden_states.ndim - 1
            variance = torch.ops.aten.mean.dim(
                torch.ops.aten.pow.Tensor_Scalar(hidden_states_fp32, 2),
                [reduction_dim],
                True,
            )
            hidden_states_normed = torch.ops.aten.mul.Tensor(
                hidden_states_fp32,
                torch.ops.aten.rsqrt.default(torch.ops.aten.add.Scalar(variance, eps)),
            )
            out = torch.ops.aten.mul.Tensor(hidden_states_normed, weight)
            return prims.convert_element_type(out, dtype)

        def replacement(hidden_states, weight, eps):
            return torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)

        return _create_pattern_result(pattern, replacement, get_inputs())


class ModulatedLayerNormPattern:
    """Match layer normalization followed by scale and shift modulation."""

    @staticmethod
    def create(affine: bool):
        def get_inputs():
            hidden_states = torch.empty(2, 3, 4, dtype=torch.bfloat16, device="meta")
            scale = torch.empty(2, 3, 4, dtype=torch.bfloat16, device="meta")
            shift = torch.empty(2, 3, 4, dtype=torch.bfloat16, device="meta")
            if affine:
                weight = torch.empty(4, dtype=torch.float32, device="meta")
                bias = torch.empty(4, dtype=torch.float32, device="meta")
                return [hidden_states, weight, bias, scale, shift]
            return [hidden_states, scale, shift]

        def pattern_affine(hidden_states, weight, bias, scale, shift, eps):
            normed = torch.ops.tensor_cast.layer_norm(hidden_states, weight, bias, eps)
            return torch.ops.aten.add.Tensor(
                torch.ops.aten.mul.Tensor(normed, torch.ops.aten.add.Tensor(scale, 1.0)), shift
            )

        def replacement_affine(hidden_states, weight, bias, scale, shift, eps):
            return torch.ops.tensor_cast.modulated_layer_norm(hidden_states, weight, bias, scale, shift, eps)

        def pattern_no_affine(hidden_states, scale, shift, eps):
            normed = torch.ops.tensor_cast.layer_norm(hidden_states, None, None, eps)
            return torch.ops.aten.add.Tensor(
                torch.ops.aten.mul.Tensor(normed, torch.ops.aten.add.Tensor(scale, 1.0)), shift
            )

        def replacement_no_affine(hidden_states, scale, shift, eps):
            return torch.ops.tensor_cast.modulated_layer_norm(hidden_states, None, None, scale, shift, eps)

        if affine:
            return _create_pattern_result(pattern_affine, replacement_affine, get_inputs())
        return _create_pattern_result(pattern_no_affine, replacement_no_affine, get_inputs())


class RMSNormPattern:
    """
    Pattern for RMS normalization.
    This pattern computes the RMS normalization of the input tensor.
    """

    @staticmethod
    def create(dtype):
        def get_inputs():
            hidden_states = torch.empty(2, 4, dtype=dtype, device="meta")
            weight = torch.empty(4, dtype=dtype, device="meta")
            return [hidden_states, weight]

        def pattern(hidden_states, weight, eps):
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + eps)
            out = weight * hidden_states.to(dtype)
            return out

        def replacement(hidden_states, weight, eps):
            out = torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)
            return out

        return _create_pattern_result(pattern, replacement, get_inputs())


class DiffusersRMSNormPattern:
    """Match the same-low-precision path in diffusers.models.normalization.RMSNorm."""

    @staticmethod
    def create(dtype, shape=(2, 4)):
        def get_inputs():
            hidden_states = torch.empty(*shape, dtype=dtype, device="meta")
            weight = torch.empty(shape[-1], dtype=dtype, device="meta")
            return [hidden_states, weight]

        def pattern(hidden_states, weight, eps):
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + eps)
            hidden_states = hidden_states.to(dtype)
            return hidden_states * weight

        def replacement(hidden_states, weight, eps):
            return torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)

        return _create_pattern_result(pattern, replacement, get_inputs())


class AddRMSNormPattern:
    @staticmethod
    def create():
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, residual, weight]

        def pattern(hidden_states, residual, weight, eps):
            out = torch.ops.tensor_cast.rms_norm(hidden_states + residual, weight, eps)
            return out

        def replacement(hidden_states, residual, weight, eps):
            out = torch.ops.tensor_cast.add_rms_norm(hidden_states, residual, weight, eps)
            return out

        return _create_pattern_result(pattern, replacement, get_inputs())


class AddRMSNorm2Pattern:
    """AddRMSNorm2 pattern that produces both the output and the residual."""

    @staticmethod
    def create():
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, residual, weight]

        def pattern(hidden_states, residual, weight, eps):
            residual = hidden_states + residual
            out = torch.ops.tensor_cast.rms_norm(residual, weight, eps)
            return out, residual

        def replacement(hidden_states, residual, weight, eps):
            out, residual = torch.ops.tensor_cast.add_rms_norm2(hidden_states, residual, weight, eps)
            return out, residual

        return _create_pattern_result(pattern, replacement, get_inputs())


class RMSNormQuantPattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            scale = torch.empty(1, device="meta")
            offset = torch.empty(1, device="meta")
            return [hidden_states, weight, scale, offset]

        def pattern(hidden_states, weight, scale, offset):
            out = torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)
            out = torch.ops.tensor_cast.quantize(out, scale, offset)
            return out

        def replacement(hidden_states, weight, scale, offset):
            out = torch.ops.tensor_cast.rms_norm_quant(hidden_states, weight, scale, offset, eps)
            return out

        return (pattern, replacement, get_inputs())


class AddRMSNormQuantPattern:
    @staticmethod
    def create(eps: float = 1e-6):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            scale = torch.empty(1, device="meta")
            offset = torch.empty(1, device="meta")
            return [hidden_states, residual, weight, scale, offset]

        def pattern(hidden_states, residual, weight, scale, offset):
            out = torch.ops.tensor_cast.rms_norm_quant(hidden_states + residual, weight, scale, offset, eps)
            return out

        def replacement(hidden_states, residual, weight, scale, offset):
            out = torch.ops.tensor_cast.add_rms_norm_quant(hidden_states, residual, weight, scale, offset, eps)
            return out

        return (pattern, replacement, get_inputs())


class AddRMSNormQuant2Pattern:
    """AddRMSNormQuant2 pattern that produces both the output and the residual."""

    @staticmethod
    def create(eps: float = 1e-6):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            scale = torch.empty(4, device="meta")
            offset = torch.empty(4, device="meta")
            return [hidden_states, residual, weight, scale, offset]

        def pattern(hidden_states, residual, weight, scale, offset):
            residual = hidden_states + residual
            out = torch.ops.tensor_cast.rms_norm_quant(residual, weight, scale, offset, eps)
            return out, residual

        def replacement(hidden_states, residual, weight, scale, offset):
            out, residual = torch.ops.tensor_cast.add_rms_norm_quant2(
                hidden_states, residual, weight, scale, offset, eps
            )
            return out, residual

        return (pattern, replacement, get_inputs())


class RMSNormDynamicQuantPattern:
    """Pattern for RMS norm followed by dynamic quantization (symmetric or asymmetric)."""

    @staticmethod
    def create(
        eps: float = 1e-6,
        symmetric: bool = True,
        per_sample: bool = False,
        scale_dtype: torch.dtype = torch.float32,
        out_dtype: torch.dtype = torch.int8,
    ):
        dims = [-1] if per_sample else []

        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, weight]

        def pattern(hidden_states, weight):
            out = torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)
            if symmetric:
                result = torch.ops.tensor_cast.dynamic_quantize_symmetric(
                    out, dims, scale_dtype=scale_dtype, out_dtype=out_dtype
                )
                return result
            else:
                result = torch.ops.tensor_cast.dynamic_quantize_asymmetric(
                    out, dims, scale_dtype=scale_dtype, out_dtype=out_dtype
                )
                return result

        def replacement(hidden_states, weight):
            if symmetric:
                result = torch.ops.tensor_cast.rms_norm_dynamic_quant_symmetric(
                    hidden_states,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return result
            else:
                result = torch.ops.tensor_cast.rms_norm_dynamic_quant_asymmetric(
                    hidden_states,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return result

        return (pattern, replacement, get_inputs())


class AddRMSNormDynamicQuantPattern:
    """Pattern for add RMS norm followed by dynamic quantization (symmetric or asymmetric)."""

    @staticmethod
    def create(
        eps: float = 1e-6,
        symmetric: bool = True,
        per_sample: bool = False,
        scale_dtype: torch.dtype = torch.float32,
        out_dtype: torch.dtype = torch.int8,
    ):
        dims = [-1] if per_sample else []

        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, residual, weight]

        def pattern(hidden_states, residual, weight):
            if symmetric:
                result = torch.ops.tensor_cast.rms_norm_dynamic_quant_symmetric(
                    hidden_states + residual,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return result
            else:
                result = torch.ops.tensor_cast.rms_norm_dynamic_quant_asymmetric(
                    hidden_states + residual,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return result

        def replacement(hidden_states, residual, weight):
            if symmetric:
                result = torch.ops.tensor_cast.add_rms_norm_dynamic_quant_symmetric(
                    hidden_states,
                    residual,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return result
            else:
                result = torch.ops.tensor_cast.add_rms_norm_dynamic_quant_asymmetric(
                    hidden_states,
                    residual,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return result

        return (pattern, replacement, get_inputs())


class AddRMSNormDynamicQuant2Pattern:
    """Pattern for add RMS norm2 followed by dynamic quantization (symmetric or asymmetric)."""

    @staticmethod
    def create(
        eps: float = 1e-6,
        symmetric: bool = True,
        per_sample: bool = False,
        scale_dtype: torch.dtype = torch.float32,
        out_dtype: torch.dtype = torch.int8,
    ):
        dims = [-1] if per_sample else []

        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, residual, weight]

        def pattern(hidden_states, residual, weight):
            residual = hidden_states + residual
            if symmetric:
                result = torch.ops.tensor_cast.rms_norm_dynamic_quant_symmetric(
                    residual,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return *result, residual
            else:
                result = torch.ops.tensor_cast.rms_norm_dynamic_quant_asymmetric(
                    residual,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return *result, residual

        def replacement(hidden_states, residual, weight):
            if symmetric:
                out, scale, residual = torch.ops.tensor_cast.add_rms_norm_dynamic_quant2_symmetric(
                    hidden_states,
                    residual,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return out, scale, residual
            else:
                out, scale, offset, residual = torch.ops.tensor_cast.add_rms_norm_dynamic_quant2_asymmetric(
                    hidden_states,
                    residual,
                    weight,
                    eps,
                    dims,
                    scale_dtype=scale_dtype,
                    out_dtype=out_dtype,
                )
                return out, scale, offset, residual

        return (pattern, replacement, get_inputs())


class RMSNormDynamicQuantMXFP4Pattern:
    """Pattern for RMS norm followed by MXFP4 dynamic quantization."""

    @staticmethod
    def create(eps: float = 1e-6, group_size: int = 32):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, weight]

        def pattern(hidden_states, weight):
            out = torch.ops.tensor_cast.rms_norm(hidden_states, weight, eps)
            return torch.ops.tensor_cast.dynamic_quantize_mxfp4(out, group_size=group_size)

        def replacement(hidden_states, weight):
            return torch.ops.tensor_cast.rms_norm_dynamic_quant_mxfp4(
                hidden_states,
                weight,
                eps,
                group_size,
            )

        return (pattern, replacement, get_inputs())


class AddRMSNormDynamicQuantMXFP4Pattern:
    """Pattern for add RMS norm followed by MXFP4 dynamic quantization."""

    @staticmethod
    def create(eps: float = 1e-6, group_size: int = 64):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, residual, weight]

        def pattern(hidden_states, residual, weight):
            return torch.ops.tensor_cast.rms_norm_dynamic_quant_mxfp4(
                hidden_states + residual,
                weight,
                eps,
                group_size,
            )

        def replacement(hidden_states, residual, weight):
            return torch.ops.tensor_cast.add_rms_norm_dynamic_quant_mxfp4(
                hidden_states,
                residual,
                weight,
                eps,
                group_size,
            )

        return (pattern, replacement, get_inputs())


class AddRMSNormDynamicQuant2MXFP4Pattern:
    """Pattern for add RMS norm2 followed by MXFP4 dynamic quantization."""

    @staticmethod
    def create(eps: float = 1e-6, group_size: int = 64):
        def get_inputs():
            hidden_states = torch.empty(2, 4, device="meta")
            residual = torch.empty(2, 4, device="meta")
            weight = torch.empty(4, device="meta")
            return [hidden_states, residual, weight]

        def pattern(hidden_states, residual, weight):
            residual = hidden_states + residual
            result = torch.ops.tensor_cast.rms_norm_dynamic_quant_mxfp4(
                residual,
                weight,
                eps,
                group_size,
            )
            return *result, residual

        def replacement(hidden_states, residual, weight):
            out, scale, residual = torch.ops.tensor_cast.add_rms_norm_dynamic_quant2_mxfp4(
                hidden_states,
                residual,
                weight,
                eps,
                group_size,
            )
            return out, scale, residual

        return (pattern, replacement, get_inputs())


def register_all_patterns():
    from . import register_pattern

    norm_pattern_shapes = [(2, 4), (2, 3, 4), (2, 3, 5, 4)]

    if config.compilation.fusion_patterns.enable_rms_norm:
        for shape in norm_pattern_shapes:
            pattern, replacement, example_inputs, scalar_workaround = TorchRMSNormPattern.create(shape)
            register_pattern(
                f"torch_rms_norm_pattern_rank_{len(shape)}",
                pattern,
                replacement,
                example_inputs,
                scalar_workaround=scalar_workaround,
                level=0,
            )
        for dtype in _RMS_NORM_DTYPE_LIST:
            for shape in norm_pattern_shapes:
                pattern, replacement, example_inputs, scalar_workaround = TorchRMSNormDecomposedPattern.create(
                    dtype, shape
                )
                register_pattern(
                    f"torch_rms_norm_decomposed_pattern_{dtype}_rank_{len(shape)}",
                    pattern,
                    replacement,
                    example_inputs,
                    scalar_workaround=scalar_workaround,
                    level=0,
                )
                pattern, replacement, example_inputs, scalar_workaround = DiffusersRMSNormPattern.create(dtype, shape)
                register_pattern(
                    f"diffusers_rms_norm_pattern_{dtype}_rank_{len(shape)}",
                    pattern,
                    replacement,
                    example_inputs,
                    scalar_workaround=scalar_workaround,
                    level=0,
                )
            pattern, replacement, example_inputs, scalar_workaround = RMSNormPattern.create(dtype)
            register_pattern(
                f"rms_norm_pattern_{dtype}",
                pattern,
                replacement,
                example_inputs,
                scalar_workaround=scalar_workaround,
                level=0,
            )

    if config.compilation.fusion_patterns.enable_layer_norm:
        for dtype in _LAYER_NORM_DTYPE_LIST:
            for shape in norm_pattern_shapes:
                for affine in (False, True):
                    if dtype in _RMS_NORM_DTYPE_LIST:
                        pattern, replacement, example_inputs, scalar_workaround = LayerNormPattern.create(
                            dtype, affine, shape
                        )
                        register_pattern(
                            f"layer_norm_pattern_{dtype}_rank_{len(shape)}_affine_{affine}",
                            pattern,
                            replacement,
                            example_inputs,
                            scalar_workaround=scalar_workaround,
                            level=0,
                        )
                    pattern, replacement, example_inputs, scalar_workaround = LayerNormDecomposedPattern.create(
                        dtype,
                        affine,
                        shape,
                        output_cast=dtype in _RMS_NORM_DTYPE_LIST,
                    )
                    register_pattern(
                        f"layer_norm_decomposed_pattern_{dtype}_rank_{len(shape)}_affine_{affine}",
                        pattern,
                        replacement,
                        example_inputs,
                        scalar_workaround=scalar_workaround,
                        level=0,
                    )

    if config.compilation.fusion_patterns.enable_modulated_layer_norm:
        for affine in (False, True):
            pattern, replacement, example_inputs, scalar_workaround = ModulatedLayerNormPattern.create(affine)
            register_pattern(
                f"modulated_layer_norm_pattern_affine_{affine}",
                pattern,
                replacement,
                example_inputs,
                scalar_workaround=scalar_workaround,
                level=1,
            )

    if config.compilation.fusion_patterns.enable_rms_norm_quant:
        register_pattern(
            "rms_norm_quant_pattern",
            *RMSNormQuantPattern.create(),
        )

    if config.compilation.fusion_patterns.enable_add_rms_norm:
        pattern, replacement, example_inputs, scalar_workaround = AddRMSNormPattern.create()
        register_pattern(
            "add_rms_norm_pattern",
            pattern,
            replacement,
            example_inputs,
            scalar_workaround=scalar_workaround,
            level=1,  # make sure RMSNorm+Quant is fused before it.
        )
        pattern, replacement, example_inputs, scalar_workaround = AddRMSNorm2Pattern.create()
        register_pattern(
            "add_rms_norm2_pattern",
            pattern,
            replacement,
            example_inputs,
            scalar_workaround=scalar_workaround,
            level=1,  # make sure RMSNorm+Quant is fused before it.
        )
        if config.compilation.fusion_patterns.enable_rms_norm_quant:
            register_pattern(
                "add_rms_norm_quant_pattern",
                *AddRMSNormQuantPattern.create(),
            )

            register_pattern(
                "add_rms_norm_quant2_pattern",
                *AddRMSNormQuant2Pattern.create(),
            )

    # Register dynamic quantization patterns
    if config.compilation.fusion_patterns.enable_rms_norm_quant:
        # Register variants for each pattern
        for symmetric in [True, False]:
            for per_sample in [True, False]:
                variant_name = "symmetric" if symmetric else "asymmetric"
                variant_name += "_per_sample" if per_sample else "_per_tensor"

                # RMS norm dynamic quantization pattern
                register_pattern(
                    f"rms_norm_dynamic_quant_{variant_name}_pattern",
                    *RMSNormDynamicQuantPattern.create(symmetric=symmetric, per_sample=per_sample),
                )

                if config.compilation.fusion_patterns.enable_add_rms_norm:
                    # Add RMS norm dynamic quantization pattern
                    register_pattern(
                        f"add_rms_norm_dynamic_quant_{variant_name}_pattern",
                        *AddRMSNormDynamicQuantPattern.create(symmetric=symmetric, per_sample=per_sample),
                    )

                    # Add RMS norm2 dynamic quantization pattern
                    register_pattern(
                        f"add_rms_norm_dynamic_quant2_{variant_name}_pattern",
                        *AddRMSNormDynamicQuant2Pattern.create(symmetric=symmetric, per_sample=per_sample),
                    )

        # Register MXFP4 patterns
        for group_size in [32, 64]:
            register_pattern(
                f"rms_norm_dynamic_quant_mxfp4_g{group_size}_pattern",
                *RMSNormDynamicQuantMXFP4Pattern.create(group_size=group_size),
            )

            if config.compilation.fusion_patterns.enable_add_rms_norm:
                register_pattern(
                    f"add_rms_norm_dynamic_quant_mxfp4_g{group_size}_pattern",
                    *AddRMSNormDynamicQuantMXFP4Pattern.create(group_size=group_size),
                )

                register_pattern(
                    f"add_rms_norm_dynamic_quant2_mxfp4_g{group_size}_pattern",
                    *AddRMSNormDynamicQuant2MXFP4Pattern.create(group_size=group_size),
                )
