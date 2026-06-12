# RFC: Support for Adapting the Qwen3-VL Multimodal Model

## Metadata

| Item              | Content                                 |
| :---------------- | :-------------------------------------- |
| **Status**        | Approved                                |
| **Author**        | weixin_43368449                         |
| **Creation Date** | 2025-12-25                              |
| **Related Links** | [Adaptation of the Qwen3-VL Base Model](https://gitcode.com/Ascend/msit/pull/4912) |

---

## 1. Overview

This proposal aims to address support for the Qwen3-VL multimodal large language model within the project, enabling it to be correctly loaded and executed under the `tensor_cast` framework.

---

## 2. Detailed Design

**Keep the existing architecture unchanged**: Continue using the current logic while introducing additional adaptation logic specifically for multimodal models.

* To ensure single responsibility, an independent `VLModelWrapper` class is designed to implement the forward-pass logic for multimodal models.
* Image input support is added to the inference scripts.
* The processing logic for the vision and text components is adapted and modified, including but not limited to attention mechanisms and rotary position embeddings.

### 2.1 Implementation Plan

#### 2.1.1 Inference Script Enhancements

In the inference scripts, support for image inputs is added:

1. Introduced the `generate_image_inputs` function to generate image-related input tensors.
2. Added command-line arguments: `--image-batch-size`, `--image-height`, and `--image-width`.
3. Implemented image resizing to ensure compliance with model requirements.

#### 2.1.2 Introduction of the VLModelWrapper Class

To support vision-language models such as Qwen3-VL, a new `VLModelWrapper` class is introduced to handle the forward-pass logic for multimodal models. This class inherits from `ModelWrapperBase` and is capable of processing image inputs, text inputs, and their fusion.

Key features include:

* Support for image-related input parameters such as `pixel_values` and `image_grid_thw`
* Initialization logic for attention mechanisms in vision layers
* Adaptation of the vision-language feature fusion process
* A unified multimodal forward interface

#### 2.1.3 Processing Logic Adaptation

**Vision Attention Mechanism Adaptation**

In the `flash_attention_forward` function, additional logic is introduced to distinguish visual attention from textual attention by checking the presence of the `attention_by_layers` parameter. For visual attention, different indexing mechanisms and processing logic are applied, including:

* Specialized attention computation methods for visual features
* Adaptation of interactions between visual tokens and textual tokens

**Rotary Position Embedding Adaptation**

In the `CachingRotaryEmb` class, We handle it uniformly by judging the dimension of 'position_ids' and converting it to text_position_ids if it is VL

---

### 2.2 Alternative Solutions

*(None specified)*

---

### 2.3 Design Analysis

**Reasons for Choosing the Current Approach:**

1. **Architectural Clarity**: By introducing the `VLModelWrapper` class, the existing architecture remains clear and adheres to the single-responsibility principle, avoiding large-scale modifications to the current codebase.

2. **Maintainability**: Multimodal-specific logic is encapsulated within an independent class, making future maintenance and extension easier.

3. **Compatibility**: The existing large language model processing flow is unaffected, ensuring backward compatibility.

---

## 3. Implementation Plan

### Completed Features

* [x] Completed prefill and decode simulation for Qwen3-VL on a single GPU

### Future Optimizations

* [ ] Parallelization strategies for the vision component in multi-GPU setups
* [ ] prefill stage compile support
* [ ] Support for MoE and corresponding parallelization strategies
* [ ] Add support for Qwen2.5-VL
* [ ] Add support for video inputs

---
