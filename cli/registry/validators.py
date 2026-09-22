"""Cross-field validators for CLI/WebUI parameter registry.

Validators check relationships between multiple parameters.
Return None on success, str (error message) on failure.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def prefill_decode_mutex(params: dict[str, Any], provided: set[str]) -> str | None:
    """--prefill and --decode are mutually exclusive.

    When neither is set, prefill is assumed (backward compatible).
    An info-level log is emitted to encourage explicit phase specification,
    but this is NOT an error — it's the documented default path.

    Note: CLI uses store_true flags (presence in ``provided`` is sufficient).
    Web UI may have default values (e.g., prefill=True by default), so we also
    check actual param values to catch cases where both end up True even if
    only one was explicitly touched (prevents validation bypass in Web UI).

    Args:
        params: Dict keyed by kebab-case param names.
        provided: Set of explicitly passed kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    # Check actual values (catches Web UI defaults + explicit touches)
    prefill_val = params.get("prefill")
    decode_val = params.get("decode")
    prefill_on = prefill_val is True or ("prefill" in provided and bool(prefill_val))
    decode_on = decode_val is True or ("decode" in provided and bool(decode_val))

    if prefill_on and decode_on:
        return "--prefill and --decode are mutually exclusive; pass exactly one"
    if not prefill_on and not decode_on:
        # Info-level: this is the backward-compatible default path, not an error.
        # All existing scripts/CI that omit both flags will see this message.
        logger.warning(
            "Neither --prefill nor --decode specified; defaulting to prefill phase. "
            "Consider passing --prefill or --decode explicitly for clarity."
        )
    return None


def product_eq_num_devices(params: dict[str, Any]) -> str | None:
    """tp_size × dp_size × pp_size == num_devices.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    num_devices = params.get("num-devices")
    if not num_devices or not isinstance(num_devices, (int, float)):
        return None

    tp = params.get("tp-size")
    if not tp or not isinstance(tp, (int, float)):
        return None

    pp = params.get("pp-size") or 1
    dp = params.get("dp-size")

    # dp is optional: if not provided, derive it
    if dp is None or dp == "":
        dp = num_devices // (tp * pp)

    if tp * dp * pp == num_devices:
        return None
    return f"tp_size({tp}) × dp_size({dp}) × pp_size({pp}) must equal num_devices({num_devices})"


def moe_product_eq_num_devices(params: dict[str, Any]) -> str | None:
    """text_generate: moe_tp_size × moe_dp_size × ep_size == num_devices."""
    num_devices = params.get("num-devices")
    if not num_devices or not isinstance(num_devices, (int, float)):
        return None

    ep = params.get("ep-size") or 1
    moe_dp = params.get("moe-dp-size")
    moe_tp = params.get("moe-tp-size")

    # moe_dp is optional: if not provided, default to 1
    if moe_dp is None or moe_dp == "":
        moe_dp = 1

    # moe_tp is optional: if not provided, derive it
    if moe_tp is None or moe_tp == "":
        moe_tp = num_devices // (ep * moe_dp)

    if moe_tp * moe_dp * ep == num_devices:
        return None
    return f"moe_tp({moe_tp}) × moe_dp({moe_dp}) × ep({ep}) must equal num_devices({num_devices})"


def per_layer_product_eq_num_devices(params: dict[str, Any]) -> str | None:
    """Per-layer TP×DP==num_devices for o_proj/mlp/lmhead.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    num_devices = params.get("num-devices")
    if not num_devices or not isinstance(num_devices, (int, float)):
        return None

    groups = [
        ("o-proj-tp-size", "o-proj-dp-size", "o_proj"),
        ("mlp-tp-size", "mlp-dp-size", "mlp"),
        ("lmhead-tp-size", "lmhead-dp-size", "lmhead"),
    ]

    for tp_field, dp_field, name in groups:
        tp_raw = params.get(tp_field)
        dp_raw = params.get(dp_field)

        # Skip if both are null/empty
        if (tp_raw is None or tp_raw == "") and (dp_raw is None or dp_raw == ""):
            continue

        tp = 1 if (tp_raw is None or tp_raw == "") else tp_raw
        dp = (num_devices // tp) if (dp_raw is None or dp_raw == "") else dp_raw

        if tp * dp != num_devices:
            return f"{name}: {tp_field}({tp}) × {dp_field}({dp}) must equal num_devices({num_devices})"

    return None


def effective_len_ge1(params: dict[str, Any]) -> str | None:
    """floor(len × (1 - prefix_cache_hit_rate)) ≥ 1.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    # Try query_length first (text_generate), then input_length (throughput)
    # Use explicit None check to handle falsy values (0, 0.0) correctly
    length = params.get("query-length")
    if length is None:
        length = params.get("input-length")
    if not length or not isinstance(length, (int, float)):
        return None

    rate = params.get("prefix-cache-hit-rate")
    if rate is None:
        rate = 0
    effective = int(length * (1 - rate))

    if effective >= 1:
        return None
    return f"effective length must be ≥ 1 (got {effective}); lower prefix_cache_hit_rate or raise length"


def shared_expert_mutex(params: dict[str, Any]) -> str | None:
    """enable_shared_expert_tp XOR host_external_shared_experts; EP>1 if shared_expert_tp.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    shared_expert_tp = params.get("enable-shared-expert-tp")
    host_external = params.get("host-external-shared-experts")
    ep_size = params.get("ep-size")

    # Constraint 1: XOR
    if shared_expert_tp is True and host_external is True:
        return "enable_shared_expert_tp and host_external_shared_experts are mutually exclusive"

    # Constraint 2: ep_size > 1 if shared_expert_tp
    if shared_expert_tp is True and (not ep_size or ep_size <= 1):
        return "enable_shared_expert_tp requires ep_size > 1"

    return None


def lte_num_devices(params: dict[str, Any], field_name: str) -> str | None:
    """Field-level: value ≤ num_devices.

    Args:
        params: Dict keyed by kebab-case param names.
        field_name: The field to check.

    Returns:
        None if valid, error message string if invalid.
    """
    num_devices = params.get("num-devices") or 0
    value = params.get(field_name)

    if value is None or value == "":
        return None

    # Normalize to list
    if isinstance(value, (int, float)):
        values = [value]
    elif isinstance(value, str):
        values = [v for v in value.replace(",", " ").split() if v]
    else:
        values = value

    # Check all values ≤ num_devices
    for v in values:
        if float(v) > num_devices:
            return f"{field_name} must be ≤ num_devices({num_devices}), got {v}"

    return None


def divides_num_devices(params: dict[str, Any], field_name: str) -> str | None:
    """Field-level: num_devices % value == 0.

    Args:
        params: Dict keyed by kebab-case param names.
        field_name: The field to check.

    Returns:
        None if valid, error message string if invalid.
    """
    num_devices = params.get("num-devices")
    value = params.get(field_name)

    if not num_devices or not value:
        return None

    if float(value) == 0:
        return f"{field_name} cannot be 0"

    if num_devices % float(value) != 0:
        return f"num_devices({num_devices}) must be divisible by {field_name}({value})"

    return None


def positive_integer_if_provided(params: dict[str, Any], field_name: str) -> str | None:
    """Field-level: null/empty allowed, else a positive integer.

    Args:
        params: Dict keyed by kebab-case param names.
        field_name: The field to check.

    Returns:
        None if valid, error message string if invalid.
    """
    value = params.get(field_name)

    if value is None or value == "":
        return None

    try:
        n = int(value)
        if n <= 0:
            return f"{field_name} must be a positive integer, got {value}"
    except (ValueError, TypeError):
        return f"{field_name} must be a positive integer, got {value}"

    return None


# ── throughput_optimizer validators ───────────────────────────────────


def valid_parallel_combo(params: dict[str, Any]) -> str | None:
    """At least one (tp, ep, moe_dp) combo valid under num_devices.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    num_devices = params.get("num-devices")
    if not num_devices or num_devices <= 0:
        return None

    def _as_list(v: Any) -> list:
        if v is None or v == "":
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, str):
            return [x for x in v.replace(",", " ").split() if x]
        return [v]

    tp_sizes = _as_list(params.get("tp-sizes"))
    ep_sizes = _as_list(params.get("ep-sizes"))
    moe_dp_sizes = _as_list(params.get("moe-dp-sizes"))

    if not tp_sizes and not ep_sizes and not moe_dp_sizes:
        return None

    for tp in tp_sizes or [1]:
        for ep in ep_sizes or [1]:
            for moe_dp in moe_dp_sizes or [1]:
                try:
                    num_tp = int(tp)
                    num_ep = int(ep)
                    num_moe_dp = int(moe_dp)
                except (ValueError, TypeError):
                    continue
                if num_tp == 0 or num_ep == 0:
                    continue
                valid_tp = num_devices % num_tp == 0
                valid_ep = num_devices % num_ep == 0
                valid_moe_dp = num_moe_dp == 0 or num_devices % (num_ep * num_moe_dp) == 0
                if valid_tp and valid_ep and valid_moe_dp:
                    return None
    return (
        f"no valid parallel combination under num_devices({num_devices}): "
        "every (tp, ep, moe_dp) candidate is invalid; adjust tp_sizes/ep_sizes/"
        "moe_dp_sizes or num_devices"
    )


def effective_len_ge1_to(params: dict[str, Any]) -> str | None:
    """floor(input_length × (1 - prefix_cache_hit_rate)) ≥ 1 (throughput variant).

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    length = params.get("input-length")
    if not length:
        return None
    # Use explicit None check to handle falsy values (0.0) correctly
    rate = params.get("prefix-cache-hit-rate")
    if rate is None:
        rate = 0
    try:
        effective = int(int(length) * (1 - float(rate)))
    except (ValueError, TypeError):
        return None
    if effective >= 1:
        return None
    return f"effective input length must be ≥ 1 (got {effective}); lower prefix_cache_hit_rate or raise input_length"


def pd_ratio_mutex_disagg(params: dict[str, Any]) -> str | None:
    """enable_optimize_prefill_decode_ratio XOR disagg.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    if params.get("enable-optimize-prefill-decode-ratio") is True and params.get("disagg") is True:
        return "PD-ratio optimization cannot be used together with disagg"
    return None


def mtp_tokens_vs_acceptance_rate(params: dict[str, Any]) -> str | None:
    """num_mtp_tokens ≤ len(mtp_acceptance_rate) + 1.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    n = params.get("num-mtp-tokens")
    if not n:
        return None
    # Normalize to list (argparse nargs="+" always gives a list)
    if isinstance(n, (list, tuple)):
        values = [int(x) for x in n if x]
    else:
        values = [int(n)]
    if not values:
        return None
    raw = params.get("mtp-acceptance-rates")
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    elif isinstance(raw, str):
        items = [x for x in raw.replace(",", " ").split() if x]
    else:
        items = []
    max_n = max(values)
    if max_n <= len(items) + 1:
        return None
    return f"num_mtp_tokens({max_n}) must be ≤ len(mtp_acceptance_rate)+1 ({len(items) + 1})"


# ── video_generate validators ──────────────────────────────────────────


def cfg_parallel_requires_world_size_2(params: dict[str, Any]) -> str | None:
    """use_cfg && cfg_parallel requires world_size ≥ 2.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    use_cfg = params.get("use-cfg") is True
    cfg_parallel = params.get("cfg-parallel") is True
    world_size = params.get("num-devices")
    if use_cfg and cfg_parallel and (not world_size or int(world_size) < 2):
        return "cfg_parallel requires world_size ≥ 2"
    return None


def cfg_parallel_requires_use_cfg(params: dict[str, Any]) -> str | None:
    """cfg_parallel=True requires use_cfg=True.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    cfg_parallel = params.get("cfg-parallel") is True
    use_cfg = params.get("use-cfg") is True
    if cfg_parallel and not use_cfg:
        return "cfg-parallel requires use-cfg to be enabled"
    return None


def num_devices_matches_ulysses_size(params: dict[str, Any]) -> str | None:
    """num_devices == ulysses_size, or 2 * ulysses_size with cfg_parallel.

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    num_devices = params.get("num-devices")
    ulysses_size = params.get("ulysses-size")
    cfg_parallel = params.get("cfg-parallel") is True
    if not num_devices or not ulysses_size:
        return None
    n, u = int(num_devices), int(ulysses_size)
    if cfg_parallel:
        if n != 2 * u:
            return f"num-devices ({n}) must equal 2 * ulysses-size ({u}) = {2 * u} when cfg-parallel is enabled"
    else:
        if n != u:
            return f"num-devices ({n}) must equal ulysses-size ({u})"
    return None


# ── Unified speculative decoding G2/G3 validators ──────────────────────
# wants_provided=True: distinguish "explicitly passed default" from "not passed"

_DRAFT_METHODS = ("dflash", "dspark")

#: Draft-dependent options requiring --speculative-method (G3).
_DRAFT_DEPENDENT_FIELDS = (
    "num-speculative-tokens",
    "acceptance-length",
    "num-draft-layers",
    "draft-model-config-path",
)

#: dspark-only options requiring method == dspark (G3).
_DSPARK_ONLY_FIELDS = (
    "dspark-markov-rank",
    "dspark-markov-head",
)


def draft_dependents_require_method(params: dict[str, Any], provided: set[str]) -> str | None:
    """G3: draft-dependent options require --speculative-method.

    Args:
        params: Dict keyed by kebab-case param names.
        provided: Set of explicitly passed kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    method = params.get("speculative-method")
    if method is None:
        passed = [f for f in _DRAFT_DEPENDENT_FIELDS if f in provided]
        if passed:
            return (
                "--num-speculative-tokens / --acceptance-length / "
                "--num-draft-layers / --draft-model-config-path require --speculative-method"
            )
    if method != "dspark":
        passed = [f for f in _DSPARK_ONLY_FIELDS if f in provided]
        if passed:
            return "--dspark-markov-rank / --dspark-markov-head require --speculative-method dspark"
    return None


def draft_mtp_mutex(params: dict[str, Any], provided: set[str]) -> str | None:
    """G2: draft speculative decoding and MTP are mutually exclusive.

    Args:
        params: Dict keyed by kebab-case param names.
        provided: Set of explicitly passed kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    method = params.get("speculative-method")
    if method not in _DRAFT_METHODS:
        return None
    n = params.get("num-mtp-tokens")
    if isinstance(n, (list, tuple)):
        mtp_on = any(int(x) > 0 for x in n if x)
    else:
        mtp_on = bool(n) and int(n) > 0
    if not mtp_on:
        sizes = params.get("num-mtp-token-sizes")
        if isinstance(sizes, (list, tuple)):
            mtp_on = any(int(x) > 0 for x in sizes if x)
    if mtp_on:
        label = "DSpark" if method == "dspark" else "Dflash"
        return f"{label} and MTP are mutually exclusive"
    return None


def draft_legacy_mtp_mutex(params: dict[str, Any], provided: set[str]) -> str | None:
    """Unified speculative decoding: cannot mix with legacy --num-mtp-tokens.

    When using --speculative-method (dflash/dspark/mtp), the legacy --num-mtp-tokens
    flag cannot be used, even if set to 0.

    Args:
        params: Dict keyed by kebab-case param names.
        provided: Set of explicitly passed kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    method = params.get("speculative-method")
    if method is None:
        return None
    if "num-mtp-tokens" in provided:
        return f"--speculative-method={method} cannot be used with legacy --num-mtp-tokens"
    return None


def mtp_requires_num_speculative_tokens(params: dict[str, Any], provided: set[str]) -> str | None:
    """MTP method requires --num-speculative-tokens.

    Args:
        params: Dict keyed by kebab-case param names.
        provided: Set of explicitly passed kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    method = params.get("speculative-method")
    if method != "mtp":
        return None
    n = params.get("num-speculative-tokens")
    if n is None or (isinstance(n, (list, tuple)) and not n):
        return "--speculative-method=mtp requires --num-speculative-tokens"
    return None


def mtp_no_draft_layers(params: dict[str, Any]) -> str | None:
    """MTP method cannot use draft layers (draft-specific option).

    Args:
        params: Dict keyed by kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    method = params.get("speculative-method")
    if method != "mtp":
        return None
    draft_layers = params.get("num-draft-layers")
    if draft_layers is not None and draft_layers != "" and draft_layers != 0:
        return "--speculative-method=mtp cannot use --num-draft-layers (draft-specific option)"
    return None


def speculative_tokens_positive(params: dict[str, Any], provided: set[str]) -> str | None:
    """When using --speculative-method, --num-speculative-tokens must be > 0.

    Args:
        params: Dict keyed by kebab-case param names.
        provided: Set of explicitly passed kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    method = params.get("speculative-method")
    if method is None:
        return None
    if "num-speculative-tokens" not in provided:
        return None
    n = params.get("num-speculative-tokens")
    if n is None:
        return None
    # Check if all values are 0
    if isinstance(n, (list, tuple)):
        all_zero = all(int(x) == 0 for x in n if x)
    else:
        all_zero = int(n) == 0
    if all_zero:
        return f"--speculative-method={method} requires --num-speculative-tokens > 0"
    return None


def mtp_no_legacy_acceptance_rate(params: dict[str, Any], provided: set[str]) -> str | None:
    """MTP method cannot use legacy --mtp-acceptance-rate(s).

    Args:
        params: Dict keyed by kebab-case param names.
        provided: Set of explicitly passed kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    method = params.get("speculative-method")
    if method != "mtp":
        return None
    if "mtp-acceptance-rates" in provided:
        return "--speculative-method=mtp cannot use legacy --mtp-acceptance-rate(s)"
    return None


def legacy_mtp_no_acceptance_length(params: dict[str, Any], provided: set[str]) -> str | None:
    """Legacy --num-mtp-tokens cannot use --acceptance-length.

    Args:
        params: Dict keyed by kebab-case param names.
        provided: Set of explicitly passed kebab-case param names.

    Returns:
        None if valid, error message string if invalid.
    """
    # Only applies when NOT using --speculative-method (legacy mode)
    method = params.get("speculative-method")
    if method is not None:
        return None
    if "num-mtp-tokens" in provided and "acceptance-length" in provided:
        return "legacy --num-mtp-tokens cannot use --acceptance-length"
    return None
