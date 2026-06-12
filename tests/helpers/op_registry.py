"""Shared op metadata builder for regression fixtures."""


def build_op_registry(cfg_registry: dict) -> dict:
    """Build a lightweight op registry from shared hf config cache."""
    per_model_ops = {}
    for model_id, hf_config in cfg_registry.items():
        per_model_ops[model_id] = {
            "model_type": getattr(hf_config, "model_type", None),
            "num_hidden_layers": getattr(hf_config, "num_hidden_layers", None),
        }
    return per_model_ops
