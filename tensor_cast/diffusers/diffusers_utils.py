import importlib
import json
from typing import Union

import torch


def get_diffusers_transformer_module(config_or_json_path: Union[str, dict]):
    if isinstance(config_or_json_path, str):
        with open(config_or_json_path, encoding="utf-8") as f:
            config = json.load(f)
    elif isinstance(config_or_json_path, dict):
        config = config_or_json_path
    else:
        raise TypeError(
            f"config_or_json_path must be a dict or a path to config.json, but got {type(config_or_json_path)}."
        )

    try:
        module = importlib.import_module("diffusers.models")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Import models from diffusers failed.") from e

    class_name = config.get("_class_name")
    if class_name is None or not isinstance(class_name, str):
        raise ValueError(
            "Unable to find _class_name attribute or _class_name not a str from the diffusers transformer config.json."
        )
    if class_name not in dir(module):
        raise KeyError(f"The class {class_name} is not supported by diffusers.")
    model_class = getattr(module, class_name)
    return model_class


# 4 = temporal_compression_ratio, 8 = spatial_compression_ratio
_model_class_to_vae_stride = {
    "WanTransformer3DModel": (4, 8),
    "HunyuanVideoTransformer3DModel": (4, 8),
    "HunyuanVideo15Transformer3DModel": (4, 16),
    "Default": (4, 8),
}


def model_class_to_vae_stride(model_class: str) -> tuple:
    if model_class not in _model_class_to_vae_stride.keys():
        model_class = "Default"
    return _model_class_to_vae_stride.get(model_class)


# TODO only wan and hunyuanvideo is supported by now
def generate_hunyuanvideo_input(**kwargs):
    batch_size = kwargs.get("batch_size")
    assert isinstance(batch_size, int)

    seq_lens = kwargs.get("seq_lens")
    assert isinstance(seq_lens, int)

    dtype = kwargs.get("dtype")

    attention_mask = torch.zeros(
        [batch_size, seq_lens],
        device=torch.device("meta"),
        dtype=dtype,
    )
    return {
        "encoder_attention_mask": attention_mask,
    }


def generate_hunyuanvideo15_input(**kwargs):
    res = {}
    batch_size = kwargs.get("batch_size")
    assert isinstance(batch_size, int)

    seq_lens = kwargs.get("seq_lens")
    assert isinstance(seq_lens, int)

    dtype = kwargs.get("dtype")

    attention_mask = torch.zeros(
        [batch_size, seq_lens],
        device=torch.device("meta"),
        dtype=dtype,
    )
    res["encoder_attention_mask"] = attention_mask
    res["encoder_attention_mask_2"] = attention_mask
    text_embed_2_dim = kwargs.get("text_embed_2_dim")
    if text_embed_2_dim is not None:
        res["encoder_hidden_states_2"] = torch.zeros(
            [batch_size, seq_lens, text_embed_2_dim],
            device=torch.device("meta"),
            dtype=dtype,
        )
    image_embed_dim = kwargs.get("image_embed_dim")
    if image_embed_dim is not None:
        res["image_embeds"] = SafeMetaTensor((image_embed_dim, image_embed_dim, image_embed_dim), dtype=dtype)
    return res


_model_class_input = {
    "HunyuanVideoTransformer3DModel": generate_hunyuanvideo_input,
    "HunyuanVideo15Transformer3DModel": generate_hunyuanvideo15_input,
}


def model_class_to_input(model_class):
    return _model_class_input.get(model_class, lambda **kwargs: {})


def get_ulysses_split_dim(hidden_states: torch.Tensor, ulysses_size: int) -> int:
    if hidden_states is None:
        raise ValueError("hidden_states is None")
    split_dim = -1
    if hidden_states.shape[-2] // 2 % ulysses_size == 0:
        split_dim = -2
    return split_dim


class SafeMetaTensor(torch.Tensor):
    """
    A meta-device tensor subclass that enables safe boolean indexing (e.g., `x[mask]`)
    during model tracing or initialization.

    Plain meta tensors crash when used in boolean indexing because PyTorch cannot
    determine output shapes without real data. This class sidesteps the issue by
    returning a valid-shaped meta tensor directly in `__getitem__`.

    Use this class for all meta tensors involved in indexing—especially boolean masks
    and the tensors they index into. Avoid operations that convert it back to plain tensors
    (e.g., `.bool()`, `.float()`), as they break the subclass protection.
    """

    @staticmethod
    def __new__(cls, shape, dtype=None, device=None, requires_grad=False):
        if device is not None and device != torch.device("meta"):
            raise ValueError("SafeMetaTensor only supports 'meta' device.")
        return torch.empty(shape, dtype=dtype, device="meta", requires_grad=requires_grad).as_subclass(cls)

    def __bool__(self):
        return True

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor) and idx.dtype == torch.bool and idx.device.type == "meta":
            return SafeMetaTensor((self.shape[0],) + self.shape[1:], dtype=self.dtype)

        return super().__getitem__(idx)
