import torch
from torch import nn

from ..model_config import WordEmbeddingTPMode
from ..parallel_group import ParallelGroup
from .utils import get_partial_sharded, ModelWrapperBase


class ParallelEmbedding(ModelWrapperBase):
    """
    A parallel embedding layer that replaces a standard torch.nn.Embedding layer.
    """

    def __init__(
        self,
        embedding: torch.nn.Embedding,
        tp_group: ParallelGroup,
        shard_mode: WordEmbeddingTPMode = WordEmbeddingTPMode.col,
    ):
        super().__init__(embedding)
        self.tp_group = tp_group
        self.tp_size = tp_group.world_size
        self.tp_rank = tp_group.rank_in_group
        try:
            self.shard_mode = WordEmbeddingTPMode(shard_mode)
        except ValueError as err:
            raise ValueError(f"word embedding tp mode must be 'col' or 'row', got {shard_mode!r}.") from err
        self._vocab_size = self.num_embeddings
        self._row_start = 0
        self._row_end = self._vocab_size
        self._orig_padding_idx = embedding.padding_idx
        self.create_weights()

    def create_weights(self):
        if not self.tp_size > 1:
            return
        shard_dim = 1 if self.shard_mode == WordEmbeddingTPMode.col else 0
        shard_weight = get_partial_sharded(self._inner.weight, self.tp_size, self.tp_rank, dim=shard_dim)
        self._inner.weight = nn.Parameter(shard_weight.contiguous())
        if self.shard_mode == WordEmbeddingTPMode.row:
            block_size = self._inner.weight.shape[0]
            self._row_start = self.tp_rank * block_size
            self._row_end = min(self._row_start + block_size, self._vocab_size)
            orig_padding_idx = self._orig_padding_idx
            if orig_padding_idx is not None and orig_padding_idx < 0:
                orig_padding_idx = self._vocab_size + orig_padding_idx
            if orig_padding_idx is not None and self._row_start <= orig_padding_idx < self._row_end:
                self._inner.padding_idx = orig_padding_idx - self._row_start
            else:
                self._inner.padding_idx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return self._inner(x)
        if self.shard_mode == WordEmbeddingTPMode.row:
            in_local_vocab = (x >= self._row_start) & (x < self._row_end)
            safe_local_indices = torch.where(in_local_vocab, x - self._row_start, torch.zeros_like(x))
            x = self._inner(safe_local_indices)
            x = x.masked_fill_(~in_local_vocab.unsqueeze(-1), 0)
            return self.tp_group.all_reduce(x)
        x = self._inner(x)
        x = self.tp_group.all_gather(x, dim=-1)
        x = x[..., : self.embedding_dim]
        return x
