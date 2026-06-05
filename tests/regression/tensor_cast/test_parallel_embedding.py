import torch
from torch import nn

from tensor_cast.layers.parallel_embedding import ParallelEmbedding
from tensor_cast.model_config import WordEmbeddingTPMode


class _FakeParallelGroup:
    def __init__(self, world_size, rank_in_group=0):
        self.world_size = world_size
        self.rank_in_group = rank_in_group

    def all_reduce(self, input_):
        return input_

    def all_gather(self, input_, dim=-1):
        return torch.cat([input_] * self.world_size, dim=dim)


def _make_embedding(vocab_size: int, hidden_size: int, padding_idx):
    embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
    return embedding


def test_row_shard_padding_idx_inside_local_range_is_relativized():
    """Padding idx falling in the owning rank's local range must be rebased."""
    vocab_size, hidden_size, tp_size = 32, 8, 4
    block_size = vocab_size // tp_size  # 8
    # padding_idx = 30 -> rank 3 owns [24, 32), local index 30 - 24 = 6
    padding_idx = 30
    owning_rank = padding_idx // block_size
    embedding = _make_embedding(vocab_size, hidden_size, padding_idx)
    parallel = ParallelEmbedding(
        embedding=embedding,
        tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=owning_rank),
        shard_mode=WordEmbeddingTPMode.row,
    )
    assert parallel._inner.weight.shape == (block_size, hidden_size)
    assert parallel._inner.padding_idx == padding_idx - owning_rank * block_size
    assert parallel._row_start == owning_rank * block_size
    assert parallel._row_end == owning_rank * block_size + block_size


def test_row_shard_padding_idx_outside_local_range_becomes_none():
    """Ranks that do not own the padding row must have padding_idx cleared."""
    vocab_size, hidden_size, tp_size = 32, 8, 4
    block_size = vocab_size // tp_size  # 8
    padding_idx = 30  # owned by rank 3
    for rank in range(tp_size):
        if rank == padding_idx // block_size:
            continue
        embedding = _make_embedding(vocab_size, hidden_size, padding_idx)
        parallel = ParallelEmbedding(
            embedding=embedding,
            tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=rank),
            shard_mode=WordEmbeddingTPMode.row,
        )
        assert parallel._inner.padding_idx is None, f"rank {rank} should have padding_idx=None"


def test_row_shard_no_padding_idx_stays_none():
    """An embedding without padding_idx must keep padding_idx=None after sharding."""
    vocab_size, hidden_size, tp_size = 32, 8, 4
    for rank in range(tp_size):
        parallel = ParallelEmbedding(
            embedding=_make_embedding(vocab_size, hidden_size, padding_idx=None),
            tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=rank),
            shard_mode=WordEmbeddingTPMode.row,
        )
        assert parallel._inner.padding_idx is None


def test_col_shard_preserves_padding_idx():
    """Column sharding does not touch the vocab dim, so padding_idx must be unchanged."""
    vocab_size, hidden_size, tp_size = 32, 8, 4
    padding_idx = 30
    for rank in range(tp_size):
        embedding = _make_embedding(vocab_size, hidden_size, padding_idx)
        parallel = ParallelEmbedding(
            embedding=embedding,
            tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=rank),
            shard_mode=WordEmbeddingTPMode.col,
        )
        assert parallel._inner.weight.shape == (vocab_size, hidden_size // tp_size)
        assert parallel._inner.padding_idx == padding_idx


def test_tp_size_one_is_noop():
    """With tp_size=1 the layer must not touch weight or padding_idx."""
    vocab_size, hidden_size, padding_idx = 32, 8, 30
    embedding = _make_embedding(vocab_size, hidden_size, padding_idx)
    parallel = ParallelEmbedding(
        embedding=embedding,
        tp_group=_FakeParallelGroup(world_size=1, rank_in_group=0),
        shard_mode=WordEmbeddingTPMode.row,
    )
    assert parallel._inner.weight.shape == (vocab_size, hidden_size)
    assert parallel._inner.padding_idx == padding_idx


def test_row_shard_negative_padding_idx_is_normalized():
    """nn.Embedding normalizes negative padding_idx at __init__, but a caller may set it
    after construction. ParallelEmbedding must still place it on the correct rank.
    """
    vocab_size, hidden_size, tp_size = 32, 8, 4
    block_size = vocab_size // tp_size
    # Manually bypass nn.Embedding's normalization to inject a negative value.
    raw_negative = -1
    expected_positive = vocab_size + raw_negative  # 31
    owning_rank = expected_positive // block_size  # 3
    for rank in range(tp_size):
        embedding = _make_embedding(vocab_size, hidden_size, padding_idx=None)
        embedding.padding_idx = raw_negative
        parallel = ParallelEmbedding(
            embedding=embedding,
            tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=rank),
            shard_mode=WordEmbeddingTPMode.row,
        )
        if rank == owning_rank:
            assert parallel._inner.padding_idx == expected_positive - rank * block_size
        else:
            assert parallel._inner.padding_idx is None


def test_row_shard_padding_idx_assertion_does_not_trigger():
    """Regression: F.embedding's padding_idx < weight.size(0) assertion must hold
    on every rank after sharding (the failure mode that surfaced under torch.compile).
    """
    vocab_size, hidden_size, tp_size = 32, 8, 4
    padding_idx = 30
    for rank in range(tp_size):
        embedding = _make_embedding(vocab_size, hidden_size, padding_idx)
        parallel = ParallelEmbedding(
            embedding=embedding,
            tp_group=_FakeParallelGroup(world_size=tp_size, rank_in_group=rank),
            shard_mode=WordEmbeddingTPMode.row,
        )
        pad = parallel._inner.padding_idx
        if pad is not None:
            assert pad < parallel._inner.weight.size(0), (
                f"rank {rank}: padding_idx {pad} must be < num_embeddings {parallel._inner.weight.size(0)}"
            )
        # Direct call into _inner with a local-safe index must not raise.
        local_idx = torch.zeros((1, 2), dtype=torch.long)
        out = parallel._inner(local_idx)
        assert out.shape == (1, 2, hidden_size)
