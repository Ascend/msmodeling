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

"""Supply generated graphs to the official ModelRunner inference pipeline.

Template handling ends at graph acquisition and metadata binding. Runtime,
performance, memory, traces and observers belong to ModelRunner.run_inference.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional


logger = logging.getLogger(__name__)


def log_template_route(inner, requests, status, *, reason='', op_count=None):
    """Report request routing; runner-only decisions have no concrete B/S yet."""
    if not logger.isEnabledFor(logging.INFO):
        return
    user = getattr(inner, 'user_input', None)
    parallel = getattr(getattr(getattr(inner, 'model', None), 'model_config', None), 'parallel_config', None)
    for request in requests or [None]:
        decode = bool(getattr(request, 'is_decode', False))
        logger.info(
            '[template %s] scope=%s model=%s phase=%s B=%s S=%s TP=%s EP=%s MOE_DP=%s DCP=%s '
            'compile=%s MTP=%s DFC=%s linear=%s attention=%s ops=%s reason=%s',
            status,
            'request' if request is not None else 'runner',
            getattr(user, 'model_id', None),
            ('decode' if decode else 'prefill') if request is not None else 'unknown',
            getattr(request, 'concurrency', None),
            getattr(request, 'seq_len' if decode else 'query_len', None),
            getattr(parallel, 'tensor_parallel_size', getattr(user, 'tp_size', None)),
            getattr(parallel, 'expert_parallel_size', getattr(user, 'ep_size', None)),
            getattr(parallel, 'moe_data_parallel_size', getattr(user, 'moe_dp_size', None)),
            getattr(parallel, 'decode_context_parallel_size', getattr(user, 'dcp_size', None)),
            getattr(user, 'do_compile', None),
            getattr(user, 'num_mtp_tokens', 0),
            getattr(user, 'enable_dispatch_ffn_combine', False),
            getattr(user, 'quantize_linear_action', None),
            getattr(user, 'quantize_attention_action', None),
            op_count,
            str(reason).replace('\n', ' ')[:300],
        )


def create_runner(user_input, *, enabled=True, incompatible_reason=''):
    """Keep compile provenance while deferring compilation for template candidates."""
    from tensor_cast.core.model_runner import ModelRunner
    from template_engine.registry import template_path_for

    if (
        enabled
        and not incompatible_reason
        and user_input.do_compile
        and user_input.pp_size == 1
        and template_path_for(
            user_input.model_id,
            num_hidden_layers=user_input.num_hidden_layers_override or None,
            do_compile=True,
        )
    ):
        return ModelRunner(user_input, defer_compile=True)
    return ModelRunner(user_input)


def wrap_registered_runner(runner, *, enabled=True, incompatible_reason=''):
    """Shared discovery policy for text_generate and the throughput optimizer."""
    from template_engine.registry import template_path_for
    from tensor_cast.core.user_config import UserInputConfig
    from dataclasses import fields

    if not enabled or incompatible_reason:
        reason = 'template_optimization_disabled' if not enabled else incompatible_reason
        log_template_route(runner, None, 'FALLBACK', reason=reason)
        return runner

    user = runner.user_input
    defaults = {field.name: field.default for field in fields(UserInputConfig)}
    # These graph modifiers are absent from capture's template identity.
    # Compare official defaults rather than silently serving another graph.
    for name in (
        'quantize_non_expert_linear_action',
        'quantize_lmhead',
        'mxfp4_group_size',
        'allow_graph_break',
        'dynamic_shapes',
        'enable_multistream',
        'multistream_overlap_shared_expert',
        'enable_sequence_parallel',
        'enable_matmul_allreduce',
        'enable_mlapo',
        'prefix_cache_hit_rate',
        'enable_redundant_experts',
        'enable_shared_expert_tp',
        'enable_external_shared_experts',
        'host_external_shared_experts',
        'word_embedding_tp',
        'block_size',
        'o_proj_tp_size',
        'o_proj_dp_size',
        'mlp_tp_size',
        'mlp_dp_size',
        'lmhead_tp_size',
        'lmhead_dp_size',
        'vision_tp_size',
        'image_batch_size',
        'enable_decode_kv_offload',
        'enable_prefill_kv_offload',
        'graph_log_url',
    ):
        default = defaults[name]
        if getattr(user, name, default) != default:
            incompatible_reason = f'unsupported_template_option:{name}'
            break
    if getattr(user, 'speculative_method', None) not in (None, 'mtp'):
        incompatible_reason = incompatible_reason or 'unsupported_speculative_method'
    if incompatible_reason:
        log_template_route(runner, None, 'FALLBACK', reason=incompatible_reason)
        return runner
    path = template_path_for(
        user.model_id,
        num_hidden_layers=getattr(runner.model, 'num_hidden_layers', None),
        do_compile=getattr(user, 'do_compile', None),
    )
    if not path:
        log_template_route(runner, None, 'FALLBACK', reason='no_matching_registered_template')
        return runner
    from template_engine.structure_buckets import parallel_geometry

    parallel = runner.model.model_config.parallel_config
    canonical_dp, _ = parallel_geometry(
        {
            'TP': parallel.tensor_parallel_size,
            'EP': parallel.expert_parallel_size,
            'MOE_DP': parallel.moe_data_parallel_size,
            'DCP': parallel.decode_context_parallel_size,
            'B': 1,
        }
    )
    if parallel.data_parallel_size != canonical_dp:
        log_template_route(runner, None, 'FALLBACK', reason='parallel_layout_not_encoded')
        return runner
    return TemplateBackedModelRunner(runner, path)


class TemplateBackedModelRunner:
    """Delegates everything and intercepts ``run_inference``; covered
    configs replay template ops through the wrapped runner's own
    performance backend, everything else falls back per call.
    """

    def __init__(
        self,
        inner: Any,
        template_path: str = "",
        *,
        source_tag: str = "template_graph",
        do_compile: bool | None = None,
    ) -> None:
        self._inner = inner
        self._template_path = template_path
        self._graph = None
        self._graph_loaded = False
        self.source_tag = source_tag
        self._do_compile = do_compile

    def __getattr__(self, name: str) -> Any:  # noqa: D105 - delegation
        return getattr(self._inner, name)

    # -- template resolution -------------------------------------------------

    def _get_graph(self):
        if not self._graph_loaded:
            from template_engine.registry import load_registered_template

            self._graph = load_registered_template(self._template_path)
            self._graph_loaded = True
        return self._graph

    def _template_config(self, request: Any) -> dict[str, Any]:
        parallel = self.model.model_config.parallel_config
        user_input = self.user_input
        is_decode = bool(getattr(request, "is_decode", False))
        return {
            "model_id": getattr(user_input, "model_id", None),
            "do_compile": self._do_compile if self._do_compile is not None else bool(user_input.do_compile),
            "B": int(request.concurrency),
            "S": int(request.seq_len if is_decode else request.query_len),
            "TP": int(parallel.tensor_parallel_size),
            "EP": int(parallel.expert_parallel_size),
            "MOE_DP": int(parallel.moe_data_parallel_size),
            "DCP": int(getattr(parallel, "decode_context_parallel_size", 1) or 1),
            "num_mtp_tokens": int(user_input.num_mtp_tokens or 0),
            "enable_dispatch_ffn_combine": bool(getattr(user_input, "enable_dispatch_ffn_combine", False)),
            "is_decode": is_decode,
            "quantize_linear_action": str(user_input.quantize_linear_action or "DISABLED"),
            "quantize_attention_action": str(user_input.quantize_attention_action or "DISABLED"),
        }

    # -- intercepted entry ---------------------------------------------------

    def run_inference(
        self,
        requests: Optional[List[Any]] = None,
        generate_inputs_func: Callable = None,
        with_sampler: bool = False,
        runtime_observer: Optional[Callable[[Any], None]] = None,
    ) -> Any:
        return self._run_inference_wrapped(requests, generate_inputs_func, with_sampler, runtime_observer)

    def _run_inference_wrapped(
        self,
        requests: Optional[List[Any]] = None,
        generate_inputs_func: Callable = None,
        with_sampler: bool = False,
        runtime_observer: Optional[Callable[[Any], None]] = None,
    ) -> Any:
        inner = self._inner
        from tensor_cast.core.model_runner import PipelineModel  # noqa: PLC0415

        if with_sampler or isinstance(inner.model, PipelineModel):
            log_template_route(
                inner,
                requests if requests is not None else inner.request_info_default,
                'FALLBACK',
                reason='sampler' if with_sampler else 'pipeline_model',
            )
            return inner.run_inference(requests, generate_inputs_func, with_sampler, runtime_observer)

        if requests is None:
            requests = inner.request_info_default
        if not requests or len(requests) != 1:
            log_template_route(inner, requests, 'FALLBACK', reason='unsupported_request_count')
            return inner.run_inference(requests, generate_inputs_func, with_sampler, runtime_observer)
        request = requests[0]

        # Serving policy (see module docstring):
        # - decode: served through coverage only on templates stamped
        #   decode_query_semantics=mtp_plus_1 (optimizer-aligned decode
        #   anchors); legacy-convention decode anchors (query_len=S vs the
        #   optimizer's mtp+1) would be wrong by orders of magnitude, so
        #   unstamped templates always fall back.
        # - chunked prefill (seq_len > query_len, not decode): the anchor
        #   models context=S while the chunk request carries its own
        #   accumulated context whose ratio to the chunk is arbitrary - fall
        #   back.
        # - full prefill (seq_len == query_len): served from the template.
        is_decode = bool(getattr(request, "is_decode", False))
        decode_stamped = (self._get_graph().get("meta") or {}).get("decode_query_semantics") == "mtp_plus_1"
        if (is_decode and not decode_stamped) or (not is_decode and int(request.seq_len) != int(request.query_len)):
            log_template_route(
                inner,
                requests,
                'FALLBACK',
                reason='decode_semantics_missing' if is_decode else 'chunked_prefill',
            )
            return inner.run_inference(requests, generate_inputs_func, with_sampler, runtime_observer)

        if is_decode and int(request.query_len) != int(getattr(inner.user_input, 'num_mtp_tokens', 0) or 0) + 1:
            log_template_route(inner, requests, 'FALLBACK', reason='decode_query_length_not_covered')
            return inner.run_inference(requests, generate_inputs_func, with_sampler, runtime_observer)

        try:
            from template_engine.template_graph import generate, TemplateGraphRefusal

            config = self._template_config(request)
            out = generate(self._get_graph(), config, return_ops=True)
            ops = out["ops"]
            source = out["source"]
        except TemplateGraphRefusal as exc:
            log_template_route(inner, requests, 'FALLBACK', reason=getattr(exc, 'blocker', str(exc)))
            return inner.run_inference(requests, generate_inputs_func, with_sampler, runtime_observer)

        from template_engine.runtime_bridge import request_op_invoke_infos

        def graph_provider(input_kwargs):
            return request_op_invoke_infos(
                ops,
                input_kwargs['attention_meta'],
                dcp_size=inner.model.model_config.parallel_config.decode_context_parallel_size,
            )

        metrics = inner.run_inference(
            requests,
            generate_inputs_func,
            with_sampler,
            runtime_observer,
            graph_provider=graph_provider,
        )
        metrics.source_tag = source
        log_template_route(inner, requests, 'HIT', reason='template_graph', op_count=len(ops))
        return metrics
