# Third-Party License Texts

License texts for code in this repository that originates upstream and stays
under its original terms. [`LICENSE`](LICENSE) is the pre-existing composite
file; the other files here were added for the notices listed below.

## Available texts

| File | License(s) | Upstream |
|---|---|---|
| [`LICENSE`](LICENSE) | Apache-2.0, plus the TensorFlow/Caffe notice and AGPL-3.0 | huggingface/transformers; `tensor_cast/**/deepseek_v4.py` (HuggingFace Inc., sgl-project) |
| [`LICENSE-DeepSeek-MIT`](LICENSE-DeepSeek-MIT) | MIT | `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3.1` |
| [`LICENSE-Kimi-K2.5-Modified-MIT`](LICENSE-Kimi-K2.5-Modified-MIT) | Modified MIT | `moonshotai/Kimi-K2.5` |
| [`LICENSE-inclusionAI-Ling-MIT`](LICENSE-inclusionAI-Ling-MIT) | MIT | `inclusionAI/Ling-flash-2.0` |

`LICENSE` is one file holding several sections, in this order: the Apache-2.0
text, the TensorFlow/Caffe notice, the `DeepSeek-V4 related files (AGPL-3.0)`
mapping, and then the AGPL-3.0 text itself.

The DeepSeek-V3 `LICENSE-CODE` and the DeepSeek-V3.1 `LICENSE` carry the same
MIT terms and the same copyright holder, so one text covers both.

## Files carrying these notices

| File | Upstream | Change |
|---|---|---|
| `tests/assets/model_config/kimi_k2_5/configuration_deepseek.py` | `deepseek-ai/DeepSeek-V3`, plus the `pretraining_tp` handling of huggingface/transformers | reformatted, `pretraining_tp` added |
| `tests/assets/model_config/kimi_k2_thinking/configuration_deepseek.py` | same as above | reformatted, `pretraining_tp` added |
| `tests/assets/model_config/deepseekv3.1_remote/configuration_deepseek.py` | `deepseek-ai/DeepSeek-V3.1` | reformatted only |
| `tests/assets/model_config/kimi_k2_5/configuration_kimi_k25.py` | `moonshotai/Kimi-K2.5` | reformatted only |
| `tests/assets/model_config/ling_flash_2_0/configuration_bailing_moe_v2.py` | `inclusionAI/Ling-flash-2.0` | unmodified copy |
| `tensor_cast/transformers/builtin_model/bailing_moe_hf/configuration_bailing_moe_v2.py` | `inclusionAI/Ling-flash-2.0` | `model_type` added, `super().__init__` rewritten |

## Adding a new text

1. Keep the upstream text byte-for-byte, including its copyright line.
2. Name the file `LICENSE-<upstream>-<license>` so the mapping stays readable.
3. Add a row to both tables above and point the code header at the file.
