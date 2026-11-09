"""Translate template B/S coordinates to the official MS request configuration.

This adapter does not resolve parallel or model architecture rules. Those remain
owned by UserInputConfig, ConfigResolver and the model's own configuration.
"""

from argparse import Namespace
from typing import Any, Mapping

from tensor_cast.core.quantization.datatypes import QuantizeAttentionAction, QuantizeLinearAction


def user_input_for(config: Mapping[str, Any], *, model_id: str | None = None, layers: int = 0, compilation_config=None):
    from tensor_cast.core.user_config import UserInputConfig

    identity = model_id or config.get('model_id')
    if not identity:
        raise ValueError('model_id is required for template capture or performance evaluation')
    batch, sequence = int(config['B']), int(config['S'])
    mtp = int(config.get('num_mtp_tokens', 0) or 0)
    decode = bool(config.get('is_decode', False))
    query = mtp + 1 if decode else sequence
    if batch < 1 or query < 1 or sequence < query or mtp < 0:
        raise ValueError('B and S must be positive and decode S must include MTP+1 query tokens')
    tp, ep, md, dcp = (int(config.get(k, 1)) for k in ('TP', 'EP', 'MOE_DP', 'DCP'))
    mode = config.get('do_compile')
    if not isinstance(mode, bool):
        raise ValueError('Explicit compile provenance is required for execution')
    options = list(compilation_config or [])
    if config.get('enable_dispatch_ffn_combine') and 'enable_dispatch_ffn_combine' not in options:
        options.append('enable_dispatch_ffn_combine')
    return UserInputConfig.from_args(
        Namespace(
            model_id=str(identity),
            device=str(config.get('device', 'TEST_DEVICE')),
            num_queries=batch,
            query_length=query,
            context_length=sequence - query,
            num_devices=max(tp, ep, dcp) * md,
            tp_size=tp,
            ep_size=ep,
            moe_dp_size=md,
            dcp_size=dcp,
            compile=mode,
            decode=decode,
            num_mtp_tokens=mtp,
            quantize_linear_action=QuantizeLinearAction(config.get('quantize_linear_action', 'DISABLED')),
            quantize_attention_action=QuantizeAttentionAction(config.get('quantize_attention_action', 'DISABLED')),
            num_hidden_layers_override=layers,
            compilation_config=options,
        )
    )


def config_from_args(args):
    """Convert official query/context lengths into template coordinates."""
    options = set(args.compilation_config or [])
    unsupported = options - {'enable_dispatch_ffn_combine'}
    if unsupported:
        raise ValueError('Template identity does not yet encode compilation options: ' + ', '.join(sorted(unsupported)))
    query, context = int(args.query_length), int(args.context_length)
    mtp = int(args.num_mtp_tokens)
    if args.decode and query != mtp + 1:
        raise ValueError('Template decode requires query-length = num-mtp-tokens + 1')
    if not args.decode and context != 0:
        raise ValueError('Template prefill currently supports context-length=0 only')
    return {
        'model_id': args.model_id,
        'device': args.device,
        'B': args.num_queries,
        'S': query + context,
        'TP': args.tp_size,
        'EP': args.ep_size,
        'MOE_DP': args.moe_dp_size,
        'DCP': args.dcp_size,
        'is_decode': args.decode,
        'do_compile': args.compile,
        'num_mtp_tokens': mtp,
        'quantize_linear_action': str(args.quantize_linear_action),
        'quantize_attention_action': str(args.quantize_attention_action),
        'enable_dispatch_ffn_combine': 'enable_dispatch_ffn_combine' in (args.compilation_config or []),
    }


def capture_cli_args(config):
    """Serialize the official request fields for an isolated capture subprocess."""
    user = user_input_for(config)
    fields = [
        ('--model-id', user.model_id),
        ('--device', user.device),
        ('--num-queries', user.num_queries),
        ('--query-length', user.query_len),
        ('--context-length', user.context_length),
        ('--num-devices', user.world_size),
        ('--tp-size', user.tp_size),
        ('--ep-size', user.ep_size),
        ('--moe-dp-size', user.moe_dp_size),
        ('--dcp-size', user.dcp_size),
        ('--num-mtp-tokens', user.num_mtp_tokens),
        ('--quantize-linear-action', user.quantize_linear_action),
        ('--quantize-attention-action', user.quantize_attention_action),
    ]
    argv = [item for flag, value in fields for item in (flag, str(value))]
    if user.do_compile:
        argv.append('--compile')
    if user.decode:
        argv.append('--decode')
    if user.enable_dispatch_ffn_combine:
        argv.extend(['--compilation-config', 'enable_dispatch_ffn_combine'])
    return argv
