import random

import torch

from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.layers.attention import AttentionMetadataTensorCast
from tensor_cast.model_config import LinearQuantConfig, ModelConfig, QuantConfig
from tensor_cast.quantize_utils import LinearQuantType
from tensor_cast.transformers.utils import get_attention_quant_config, strip_module_name
from tensor_cast.utils import exact_division


def assert_close(self, value1, value2, rtol=0.01):
    self.assertLessEqual(abs(value1 - value2) / value1, rtol, f"{value1} vs. {value2}, rtol={rtol}")


def count_events(runtime, op):
    return sum(1 for event in runtime.event_list if event.op_invoke_info.func == op)


def create_attn_metadata_and_kv_cache(model, model_config: ModelConfig):
    batch_size = 2
    query_len_1 = 55
    query_len_2 = 45
    seq_len_1 = 2000
    seq_len_2 = 1500
    num_blocks = 10000
    block_size = 128
    max_seq_len = max(seq_len_1, seq_len_2)
    query_start_loc = torch.tensor([0, query_len_1, query_len_1 + query_len_2], dtype=torch.long)
    seq_lens = torch.tensor([seq_len_1, seq_len_2], dtype=torch.long)
    query_lens = torch.tensor([query_len_1, query_len_2], dtype=torch.long)
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(batch_size):
        block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)]
        block_tables.append(block_table)
    block_table_tensor = torch.tensor(block_tables, dtype=torch.long)
    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_lens=query_lens,
        block_table_tensor=block_table_tensor,
    )

    num_tokens = query_len_1 + query_len_2
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        kvcache_dtype = model_config.dtype
        if (attention_config := get_attention_quant_config(model, i)) is not None:
            kvcache_dtype = attention_config.get_quant_dtype()

        if model.text_config.num_key_value_heads >= model_config.parallel_config.tensor_parallel_size:
            kv_heads = exact_division(
                model.text_config.num_key_value_heads,
                model_config.parallel_config.tensor_parallel_size,
            )
        else:
            assert model_config.parallel_config.tensor_parallel_size % model.text_config.num_key_value_heads == 0
            kv_heads = 1
        kv_cache_by_layers[i] = torch.empty(
            [
                2,
                num_blocks,
                block_size,
                kv_heads,
                model.text_config.head_dim,
            ],
            dtype=kvcache_dtype,
            device="meta",
        )
    return attn_meta, kv_cache_by_layers, num_tokens


def create_mla_metadata_and_kv_cache(model, model_config: ModelConfig, query_len_1=55, query_len_2=45):
    batch_size = 2
    seq_len_1 = 2000
    seq_len_2 = 1500
    num_blocks = 10000
    block_size = 128
    max_seq_len = max(seq_len_1, seq_len_2)
    query_start_loc = torch.tensor([0, query_len_1, query_len_1 + query_len_2], dtype=torch.long)
    seq_lens = torch.tensor([seq_len_1, seq_len_2], dtype=torch.long)
    query_lens = torch.tensor([query_len_1, query_len_2], dtype=torch.long)
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(batch_size):
        block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)]
        block_tables.append(block_table)
    block_table_tensor = torch.tensor(block_tables, dtype=torch.long)
    attn_meta = AttentionMetadataTensorCast(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        query_lens=query_lens,
        block_table_tensor=block_table_tensor,
    )

    num_tokens = query_len_1 + query_len_2
    kv_cache_by_layers = {}
    for i in range(model.num_hidden_layers):
        kvcache_dtype = model_config.dtype
        if (attention_config := get_attention_quant_config(model, i)) is not None:
            kvcache_dtype = attention_config.get_quant_dtype()
        kv_cache_by_layers[i] = torch.empty(
            [
                num_blocks,
                block_size,
                model.text_config.kv_lora_rank + model.text_config.qk_rope_head_dim,
            ],
            dtype=kvcache_dtype,
            device="meta",
        )
    return attn_meta, kv_cache_by_layers, num_tokens


def has_submodule_with_cls_name(module, cls_name):
    return any(type(sub_module).__name__ == cls_name for _, sub_module in module.named_modules())


def get_linear_quant_config(quant_type, weight=None, **kwargs):
    """Helper to create a default symmetric per-tensor weight quant config.
    Can be customized via kwargs
    """
    config_args = {
        "quant_type": quant_type,
    }
    if "weight_scale" not in kwargs and weight is not None:
        w_scale = torch.max(torch.abs(weight)) / 127.0
        config_args.update({"weight_scale": w_scale})
    config_args.update(kwargs)
    return LinearQuantConfig(**config_args)


def get_quant_config(model=None, quant_type=LinearQuantType.W4A8, **kwargs):
    quant_config = QuantConfig()
    if model is None:
        quant_config.linear_configs["*"] = get_linear_quant_config(
            quant_type,
            torch.randn(1) if "weight_group_size" not in kwargs else None,
            **kwargs,
        )
        return quant_config
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            strip_name = strip_module_name(name)
            quant_config.linear_configs[strip_name] = get_linear_quant_config(
                quant_type,
                module.weight.data,
                **kwargs,
            )
    return quant_config


def update_parallel_parameter(user_input: UserInputConfig, world_size=1, tp_size=1, ep=False):
    user_input.world_size = world_size
    user_input.tp_size = tp_size
    user_input.ep_size = world_size if ep else 1
