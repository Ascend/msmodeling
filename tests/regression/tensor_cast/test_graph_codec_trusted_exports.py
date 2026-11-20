"""The graph codec trusts exact exports only; MLA must be reachable by name."""

import pytest

from template_engine.graph_codec import _TRUSTED_EXPORTS, GraphCodecError, _resolve_torch_path


def test_mla_export_is_frozen_in_the_trusted_registry():
    """Kimi-K3 captures reference the MLA implementation by exact module path."""
    assert "MultiheadLatentAttentionTensorCast" in _TRUSTED_EXPORTS["tensor_cast.layers.mla"]


def test_mla_export_resolves_to_the_implementation_class():
    resolved = _resolve_torch_path("tensor_cast.layers.mla.MultiheadLatentAttentionTensorCast")
    from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast

    assert resolved is MultiheadLatentAttentionTensorCast


@pytest.mark.parametrize(
    "name",
    [
        "tensor_cast.layers.mla.NotAnApprovedName",
        "tensor_cast.layers.mla",
        "tensor_cast.layers.mla._private_helper",
    ],
)
def test_registry_still_rejects_unapproved_references(name):
    with pytest.raises(GraphCodecError, match="not an approved export"):
        _resolve_torch_path(name)
