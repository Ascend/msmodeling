# RFC: Modeling Support for Hunyuanvideo Model

## Metadata

| Item | Details |
|:-----|:--------|
| **Status** | Approved |
| **Author** | wqh17101 |
| **Creation Date** | 2025-12-19 |
| **Related Links** |  https://gitcode.com/Ascend/msit/pull/4911 |

---

## 1. Oversview

This proposal aims to address the adaptation issues of the Hunyuanvideo multimodal model, enabling it to run correctly under the tensor_cast framework.

## 2. Detailed Design

### Model Input Construction Adaptation

To support model structure parsing and graph construction under the tensor_cast framework, placeholder inputs conforming to the forward interface of HunyuanVideo series models need to be provided, using tensors with device="meta".Input generation logic has been implemented for the following two types of models:

- HunyuanVideoTransformer3DModel: Constructs the basic encoder_attention_mask;

## 3. Implementation Plan

### Completed

- [x] Completed DIT modeling for hunyuanvideo
- [x] Completed HunyuanVideo1.5 T2V normal-run modeling and Diffusers pipeline loading.
  - Supported model IDs must resolve to a canonical Diffusers checkpoint with
    `HunyuanVideo15Transformer3DModel` and `HunyuanVideo15Pipeline`.
  - Raw Tencent checkpoints using `HunyuanVideo_1_5_DiffusionTransformer` are rejected;
    use an upstream or community Diffusers conversion instead.
  - Image-to-video variants are rejected. The modeled T2V vision input is an all-zero
    `[batch_size, 729, image_embed_dim]` tensor, where `image_embed_dim` comes from
    the selected canonical Transformer configuration.

### Follow-up Development

- [ ] Support for image input
- [ ] VAE modeling
- [ ]Support for various parallelization features
