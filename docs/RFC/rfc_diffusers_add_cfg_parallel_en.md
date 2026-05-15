# RFC: Support Classifier-free Guidance Parallel for Diffusers Model Proposal

## Metadata

| Item | Details |
|:-----|:--------|
| **Status** | Approved |
| **Author** | zhengxinqian |
| **Creation Date** | 2025-12-26 |
| **Related Links** | [【feature】support classifier-free guidance parallel](https://gitcode.com/Ascend/msit/pull/4914) |

---

## 1. Overview

This proposal aims to support classifier-free guidance (CFG) feature and CFG parallel in diffusion model.

## 2. Background

Classifier-free guidance (CFG) is a feature in diffusion model inference pipeline. It enables conditional control over outputs without relying on pre-trained classifiers, simplifying the inference workflow effectively.
CFG parallel dispatch conditional control and unconditional control to two DiT model instances as input, and gather the output of DiT model.
CFG is a feature of diffusion model inference pipeline but not a part of DiT model, therefore this feature will be built in video_generate.py.

## 3. Detailed Design

- case 1 (use cfg, cfg world size == 1)

    Do DiT model inference twice

- case 2 (use cfg, cfg world size == 2)

    Do DiT model inference once, and all-gather the output.
