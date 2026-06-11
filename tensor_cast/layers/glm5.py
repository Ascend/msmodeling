from .mla import DeepseekSparseAttention


class Glm5SparseAttention(DeepseekSparseAttention):
    def forward(self, *args, **kwargs):
        attn_output, attn_weights = super().forward(*args, **kwargs)
        return attn_output, attn_weights, None
