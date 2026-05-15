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

### Follow-up Development

- [ ] Completed DIT modeling for hunyuanvideo1.5
- [ ] Support for image input
- [ ] VAE modeling
- [ ] Diffusers pipeline loading support
- [ ]Support for various parallelization features
