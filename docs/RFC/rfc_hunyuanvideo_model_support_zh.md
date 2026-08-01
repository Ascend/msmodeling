# RFC: Hunyuanvideo模型建模支持

## 元数据

| 项目 | 内容                                        |
| :--- |:------------------------------------------|
| **状态** | 已批准                                       |
| **作者** | genius52                                  |
| **创建日期** | 2025-12-25                                |
| **相关链接** | https://gitcode.com/Ascend/msit/pull/4911 |

---

## 1. 概述

本提案旨在解决hunyuanvideo多模态模型适配问题，使其能够在tensor_cast框架下正确运行。

## 2. 方案设计

### 模型输入构造适配

为支持tensor_cast框架下的模型结构解析与图构建，需为 HunyuanVideo系列模型提供符合其前向接口的占位输入，这些输入使用 device="meta" 的张量。
当前已实现对以下两类模型的输入生成逻辑：

- HunyuanVideoTransformer3DModel：构造基础的 encoder_attention_mask；

## 3. 实施计划

### 已完成

- [x] 已完成hunyuanvideo的dit建模
- [x] 已完成HunyuanVideo1.5 T2V normal-run建模和Diffusers pipeline加载。
  - canonical Diffusers checkpoint 使用`HunyuanVideo15Transformer3DModel`和
    `HunyuanVideo15Pipeline`。
  - 官方远端Tencent仓库也支持显式的原始T2V `transformer/<variant>` selector；
    TensorCast会校验其原生pipeline和Transformer JSON契约，并将配置转换为内置
    Diffusers模型配置。本地原始目录仍不受支持；TensorCast不会执行Tencent
    `hyvideo`代码或转换checkpoint权重。
  - 明确拒绝I2V和超分variant。建模的T2V视觉输入是全零
    `[batch_size, 729, image_embed_dim]`张量，其中`image_embed_dim`来自已校验的
    Transformer配置。

### 后续开发

- [ ] image输入的支持
- [ ] vae建模
- [ ] 各种并行特性支持
