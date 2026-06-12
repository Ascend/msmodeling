import pytest


def test_cfg_registry_unknown_alias_has_clear_error(cfg_registry):
    with pytest.raises(KeyError, match="Unknown model alias 'not_exists_alias'"):
        _ = cfg_registry["not_exists_alias"]
