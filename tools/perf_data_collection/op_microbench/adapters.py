# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Allow-listed replay adapters for standalone NPU microbenchmarks.

Every public request uses the CSV-row contract consumed by the existing
``op_replay/*_run.py`` scripts. This module selects an allow-listed script and
exposes its ``build_case``/``run_case`` path inside the profiled worker; it
does not define a second operator argument schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

if __package__ and "." in __package__:
    from ..op_replay.operator_metadata import get_operator_metadata

    _REPLAY_PACKAGE = f"{__package__.rsplit('.', maxsplit=1)[0]}.op_replay"
else:
    from op_replay.operator_metadata import get_operator_metadata

    _REPLAY_PACKAGE = "op_replay"


REPLAY_REQUIRED_COLUMNS = (
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
)


@dataclass(frozen=True)
class AdapterSpec:
    """Static contract for one supported logical operator."""

    name: str
    api_path: str
    target_op_types: tuple[str, ...]
    replay_module: str
    is_composite: bool = False
    api_output_dtype_compatibility: tuple[tuple[int, str, str], ...] = ()


class AdapterValidationError(ValueError):
    """Raised when a registered adapter cannot execute a request."""

    def __init__(self, message: str, message_zh: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.message_zh = message_zh
        self.details = details


def _replay_spec(
    name: str,
    api_path: str,
    *,
    target_op_types: tuple[str, ...] | None = None,
    is_composite: bool = False,
    api_output_dtype_compatibility: tuple[tuple[int, str, str], ...] = (),
) -> AdapterSpec:
    declared_targets = target_op_types or ()
    metadata_targets = get_operator_metadata(name).all_names
    return AdapterSpec(
        name=name,
        api_path=api_path,
        target_op_types=tuple(dict.fromkeys((*declared_targets, *metadata_targets))),
        replay_module=f"{_REPLAY_PACKAGE}.{name}_run",
        is_composite=is_composite,
        api_output_dtype_compatibility=api_output_dtype_compatibility,
    )


ADAPTERS = {
    "MatMulV2": _replay_spec("MatMulV2", "torch.mm", target_op_types=("MatMulV2", "MatMulV3", "MatMulCommon")),
    "MatMulV3": _replay_spec("MatMulV3", "torch.mm", target_op_types=("MatMulV3", "MatMulV2", "MatMulCommon")),
    "BatchMatMulV2": _replay_spec(
        "BatchMatMulV2",
        "torch.matmul",
        target_op_types=(
            "BatchMatMulV2",
            "BatchMatMul",
            "BatchMatMulNd",
            "MatMulV2",
            "MatMulV3",
            "MatMulCommon",
            "Mul",
        ),
    ),
    "Add": _replay_spec("Add", "torch.add"),
    "Cast": _replay_spec("Cast", "torch.Tensor.to"),
    "CastAiCore": _replay_spec("CastAiCore", "torch.Tensor.to"),
    "Fill": _replay_spec("Fill", "torch.full"),
    "LayerNormV3": _replay_spec("LayerNormV3", "torch.native_layer_norm"),
    "MoeGatingTopK": _replay_spec("MoeGatingTopK", "torch_npu.npu_moe_gating_top_k"),
    "Mul": _replay_spec("Mul", "torch.mul"),
    "RmsNorm": _replay_spec("RmsNorm", "torch_npu.npu_rms_norm"),
    "SwiGlu": _replay_spec("SwiGlu", "torch_npu.npu_swiglu"),
    "Transpose": _replay_spec("Transpose", "torch.transpose(...).contiguous"),
    "AddRmsNormBias": _replay_spec(
        "AddRmsNormBias",
        "torch.ops._C_ascend.npu_add_rms_norm_bias",
        target_op_types=("AddRmsNormBias", "AddRmsNorm"),
        is_composite=True,
    ),
    "ArgMaxV2": _replay_spec(
        "ArgMaxV2",
        "torch.argmax",
        api_output_dtype_compatibility=((0, "DT_INT32", "DT_INT64"),),
    ),
    "AscendQuantV2": _replay_spec(
        "AscendQuantV2", "torch_npu.npu_quantize", target_op_types=("AscendQuantV2", "AscendQuant")
    ),
    "DynamicQuant": _replay_spec(
        "DynamicQuant", "torch_npu.npu_dynamic_quant", target_op_types=("DynamicQuant", "DynamicQuantV2")
    ),
    "FusedInferAttentionScore": _replay_spec(
        "FusedInferAttentionScore",
        "torch_npu.npu_fused_infer_attention_score",
        target_op_types=("FusedInferAttentionScore", "FusedInferAttentionScoreV2", "FusedInferAttentionScoreV3"),
    ),
    "GatherV2": _replay_spec("GatherV2", "torch.nn.functional.embedding"),
    "GroupedMatmul": _replay_spec("GroupedMatmul", "torch_npu.npu_grouped_matmul"),
    "GroupedMatmulSwigluQuant": _replay_spec("GroupedMatmulSwigluQuant", "torch_npu.npu_grouped_matmul_swiglu_quant"),
    "Index": _replay_spec("Index", "torch.Tensor.__getitem__"),
    "InterleaveRope": _replay_spec("InterleaveRope", "torch_npu.npu_interleave_rope"),
    "KvRmsNormRopeCache": _replay_spec(
        "KvRmsNormRopeCache",
        "torch_npu.npu_kv_rmsnorm_rope_cache",
        target_op_types=("KvRmsNormRopeCache", "KvRmsNormRopeCacheV2"),
    ),
    "LightningIndexer": _replay_spec("LightningIndexer", "torch_npu.npu_lightning_indexer"),
    "MaskedFill": _replay_spec("MaskedFill", "torch.Tensor.masked_fill_"),
    "MatMulCommon": _replay_spec("MatMulCommon", "torch.mm", target_op_types=("MatMulCommon", "MatMulV2", "MatMulV3")),
    "MoeTokenPermute": _replay_spec("MoeTokenPermute", "torch_npu.npu_moe_token_permute"),
    "MoeTokenUnpermute": _replay_spec("MoeTokenUnpermute", "torch_npu.npu_moe_token_unpermute"),
    "PadV3": _replay_spec("PadV3", "torch.nn.functional.pad"),
    "QuantBatchMatmulV3": _replay_spec(
        "QuantBatchMatmulV3",
        "torch_npu.npu_quant_matmul",
        target_op_types=("QuantBatchMatmulV3", "WeightQuantBatchMatmulV2", "WeightQuantBatchMatmulV3"),
    ),
    "ReshapeAndCacheNdKernel": _replay_spec(
        "ReshapeAndCacheNdKernel",
        "torch_npu._npu_reshape_and_cache",
        target_op_types=("ReshapeAndCacheNdKernel", "ReshapeAndCacheNd"),
    ),
    "RINGMLAPrefillBF16Kernel": _replay_spec("RINGMLAPrefillBF16Kernel", "torch_npu.atb.npu_ring_mla"),
    "ScatterNdUpdate": _replay_spec("ScatterNdUpdate", "torch_npu.npu_scatter_nd_update"),
    "ScatterNdUpdateAiCore": _replay_spec("ScatterNdUpdateAiCore", "torch_npu.npu_scatter_nd_update"),
    "Slice": _replay_spec("Slice", "torch_npu.npu_slice"),
    "SliceAiCore": _replay_spec("SliceAiCore", "torch_npu.npu_slice"),
    "SoftmaxV2": _replay_spec("SoftmaxV2", "torch.nn.functional.softmax"),
    "Sort": _replay_spec(
        "Sort",
        "torch.sort",
        api_output_dtype_compatibility=((1, "DT_INT32", "DT_INT64"),),
    ),
    "SparseFlashAttention": _replay_spec("SparseFlashAttention", "torch.ops._C_ascend.npu_sparse_flash_attention"),
    "split_qkv_rmsnorm_rope_kernel": _replay_spec("split_qkv_rmsnorm_rope_kernel", "torch.ops.vllm.qkv_rmsnorm_rope"),
    "TensorMove": _replay_spec("TensorMove", "torch.Tensor.copy_"),
    "TransposeBatchMatMul": _replay_spec(
        "TransposeBatchMatMul",
        "torch_npu.npu_transpose_batchmatmul",
        target_op_types=("TransposeBatchMatMul", "BatchMatMulV2", "BatchMatMul"),
    ),
    "_triton_rope_siso": _replay_spec("_triton_rope_siso", "vllm_ascend.ops.triton.rope.rope_forward_triton_siso"),
    "mla_preprocess_0_mix_aic": _replay_spec("mla_preprocess_0_mix_aic", "torch.ops._C_ascend.mla_preprocess"),
}


def get_adapter(kernel_type: str) -> AdapterSpec:
    """Return a registered adapter or fail without executing arbitrary code."""
    try:
        return ADAPTERS[kernel_type]
    except KeyError as exc:
        raise AdapterValidationError(
            f"No standalone microbenchmark adapter is registered for {kernel_type}.",
            f"尚未为 {kernel_type} 注册独立单算子 microbenchmark 适配器。",
            kernel_type=kernel_type,
            supported_operators=sorted(ADAPTERS),
        ) from exc


def describe_adapter(kernel_type: str) -> dict[str, Any]:
    """Return the uniform CSV-row request contract for one adapter."""
    spec = get_adapter(kernel_type)
    return {
        "kernel_type": spec.name,
        "api_path": spec.api_path,
        "request_mode": "replay_row",
        "required_columns": list(REPLAY_REQUIRED_COLUMNS),
        "runtime_columns": "Preserve every Runtime ... column required by the source replay row.",
        "description_zh": "请求字段与 replay CSV 行一致；保留六个基础描述列、空槽以及该算子需要的 Runtime ... 列。",
    }


def _build_fia_case(module: Any, row: dict[str, str], runtime_torch_npu: Any) -> tuple[dict[str, Any], Any]:
    case = module.build_row_case(row)
    validation_error = module.validate_case_for_replay(case, row)
    if validation_error is not None:
        raise ValueError(validation_error)

    def invoke():
        return runtime_torch_npu.npu_fused_infer_attention_score(
            case["query"],
            case["key"],
            case["value"],
            atten_mask=case["atten_mask"],
            actual_seq_lengths=case["actual_seq_lengths"],
            actual_seq_lengths_kv=case["actual_seq_lengths_kv"],
            block_table=case["block_table"],
            query_rope=case["query_rope"],
            key_rope=case["key_rope"],
            num_heads=case["num_heads"],
            scale=case["scale"],
            input_layout=case["input_layout"],
            num_key_value_heads=case["num_key_value_heads"],
            sparse_mode=case["sparse_mode"],
            block_size=case["block_size"],
            softmax_lse_flag=case["softmax_lse_flag"],
        )

    return case, invoke


def _build_quant_batch_matmul_case(module: Any, row: dict[str, str]) -> tuple[dict[str, Any], Any]:
    case = module.build_row_tensors(row)
    use_graph_mode = case["weight_format"] == "FRACTAL_NZ"

    def invoke():
        return module.run_quant_matmul(
            case["x_tensor"],
            case["weight_tensor"],
            case["scale_tensor"],
            case["bias_tensor"],
            case["offset_tensor"],
            case["pertoken_scale_tensor"],
            case["output_dtype_name"],
            use_graph_mode,
        )

    return case, invoke


def _build_ring_mla_case(module: Any, row: dict[str, str], runtime_torch_npu: Any) -> tuple[dict[str, Any], Any]:
    case = module.build_row_case(row)
    seqlen = case["seqlen_candidates"][0]
    mask_type = "no_mask" if case["has_prefix_state"] else case["mask_type"]
    output = case["pre_out"] if case["has_prefix_state"] else case["output"]
    softmax_lse = case["prev_lse"] if case["has_prefix_state"] else case["softmax_lse"]

    def invoke():
        return runtime_torch_npu.atb.npu_ring_mla(
            q_nope=case["q_nope"],
            q_rope=case["q_rope"],
            k_nope=case["k_nope"],
            k_rope=case["k_rope"],
            value=case["value"],
            mask=case["mask"],
            seqlen=seqlen,
            head_num=case["head_num"],
            kv_head_num=case["kv_head_num"],
            pre_out=case["pre_out"],
            prev_lse=case["prev_lse"],
            qk_scale=case["qk_scale"],
            kernel_type="kernel_type_high_precision",
            mask_type=mask_type,
            input_layout="type_bsnd",
            calc_type=case["calc_type"],
            output=output,
            softmax_lse=softmax_lse,
        )

    return case, invoke


def _build_mla_preprocess_case(module: Any, row: dict[str, str]) -> tuple[dict[str, Any], Any]:
    case = module.build_case(row)

    def invoke():
        result = module.run_case(case)
        return tuple(case["outputs"]) if result is None else result

    return case, invoke


def _build_replay_runtime_case(
    request: dict[str, Any],
    spec: AdapterSpec,
    runtime_torch_npu: Any,
) -> dict[str, Any]:
    replay_common = importlib.import_module(f"{_REPLAY_PACKAGE}.common")
    replay_common.init_runtime()
    module = importlib.import_module(spec.replay_module)
    row = dict(request["replay_row"])
    if spec.name == "FusedInferAttentionScore":
        replay_case, invoke = _build_fia_case(module, row, runtime_torch_npu)
    elif spec.name == "QuantBatchMatmulV3":
        replay_case, invoke = _build_quant_batch_matmul_case(module, row)
    elif spec.name == "RINGMLAPrefillBF16Kernel":
        replay_case, invoke = _build_ring_mla_case(module, row, runtime_torch_npu)
    elif spec.name == "mla_preprocess_0_mix_aic":
        replay_case, invoke = _build_mla_preprocess_case(module, row)
    else:
        replay = module.op
        replay.prepare()
        replay_case = replay.build_case(row)

        def invoke():
            result = replay.run_case(replay_case)
            if result is None and spec.name == "ReshapeAndCacheNdKernel":
                return tuple(replay_case["inputs"][2:4])
            return result

    return {
        "spec": spec,
        "invoke": invoke,
    }


def build_runtime_case(request: dict[str, Any]) -> dict[str, Any]:
    """Build one CSV-row replay case inside the NPU worker."""
    import torch_npu

    spec = get_adapter(request["kernel_type"])
    return _build_replay_runtime_case(request, spec, torch_npu)
