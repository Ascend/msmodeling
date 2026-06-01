import pytest
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.model_config import ParallelConfig


def _get_parallel_config(parallel_configuration: tuple):
    parallel_config = ParallelConfig(
        world_size=parallel_configuration[0],
        tensor_parallel_size=parallel_configuration[1],
        o_proj_tensor_parallel_size=parallel_configuration[2],
        mlp_tensor_parallel_size=parallel_configuration[3],
        lmhead_tensor_parallel_size=parallel_configuration[4],
    )
    if len(parallel_configuration) > 5:
        parallel_config.embedding_parallel = parallel_configuration[5]
    return parallel_config


def _has_dp_transform(parallel_config: ParallelConfig):
    return (
        parallel_config.data_parallel_size != parallel_config.mlp_data_parallel_size
        or parallel_config.data_parallel_size != parallel_config.o_proj_data_parallel_size
        or parallel_config.data_parallel_size != parallel_config.lmhead_data_parallel_size
    )


def test_parallel_config_layer_split_flags():
    cfg = ParallelConfig(
        world_size=8,
        tensor_parallel_size=2,
        data_parallel_size=4,
        mlp_tensor_parallel_size=4,
        lmhead_tensor_parallel_size=1,
    )
    assert cfg.has_attn_tp()
    assert cfg.has_mlp_tp()
    assert not cfg.has_lmhead_tp()


@pytest.mark.parametrize(
    "parallel_configuration, expected_dp_transform",
    [
        ((16, 1, 1, 1, 1), False),
        ((16, 4, 2, 8, 16), True),
        ((16, 4, 2, 8, 16, True), True),
    ],
)
def test_parallel_config_topology_flags(parallel_configuration, expected_dp_transform):
    cfg = _get_parallel_config(parallel_configuration)
    assert cfg.world_size == parallel_configuration[0]
    assert cfg.tensor_parallel_size == parallel_configuration[1]
    assert _has_dp_transform(cfg) is expected_dp_transform


def test_mtp_ep_user_config_parallel_fields():
    user_config = UserInputConfig(
        model_id="deepseek-ai/DeepSeek-V3.1",
        num_mtp_tokens=2,
        world_size=16,
        ep_size=16,
        moe_dp_size=1,
        moe_tp_size=1,
    )
    assert user_config.num_mtp_tokens == 2
    assert user_config.world_size == 16
    assert user_config.ep_size == 16
