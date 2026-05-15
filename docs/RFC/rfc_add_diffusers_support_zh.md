# RFC: 增加diffusers模型建模支持

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | Accepted (已批准) |
| **作者** | HongMaoShuiGuai |
| **创建日期** | 2025-12-24 |
| **相关链接** | [Issue/PR/Commit URL(s)] |

---

## 1. 概述

在tensor_cast中增加了diffusers框架下模型的建模，当前只适配了Wan2.1、Wan2.2、HunyuanVideo模型的dit部分建模，增加了对应的性能建模入口脚本。当前版本只提供了基础建模能力，缺少量化、并行等其余优化特性的支持。

## 2. 方案设计

### 2.1 推荐方案

TODO

### 2.2 替代方案

工程初衷是实现生成式多模态模型的推理性能建模，在前期调研期间考虑过如下几种方案

1. **基于魔乐社区工程**
    - **缺点**：每个模型有对应的工程文件，缺乏统一的架构
2. **基于cachedit**
    - **缺点**： cachedit中模型的定义等代码是复用的diffusers，项目实际提供的是并行和cache的特性支持

### 2.3 方案分析

TODO

## 3. 实施计划

- [x] 已完成hunyuanvideo的dit建模
- [x] 已完成hunyuanvideo1.5的dit建模
- [x] 已完成量化特性适配
- [x] 已完成dit attention的性能建模

## TODO List

1. 模型初始化时使用from_config接口，diffusers会提示warning这个接口将在diffusers的v1.0版本（当前为0.36）被替换为load_config，但是当前版本两个接口的返回值不同，from_config能够基于config.json文件路径返回随机初始化的模型实例，load_config接口类似json.load，返回值是dict。正式版发布后需要确认一下。并且该接口使用和当前项目中transformer模型的初始化代码不统一。
2. tensor_cast中原本支持的transformer模型仿真建模不需要提前准备模型的config.json文件，可以直接从hugging face上拉取模型配置文件。diffusers模型的初始化接口未确认是否支持类似能力，当前使用时需要手动准备模型配置文件。
3. diffusers框架针对attention算子的多种实现做过一定的适配，可以使用diffusers.models.attention_dispatch._AttentionBackendRegistry注册新的attention算子并指定模型使用该算子。但是存在两个问题，一是框架中并非所有模型都支持，hunyuanvideo中使用的是diffusers.models.attention_processor；二是接口_AttentionBackendRegistry在当前diffusers版本上并未直接开放，使用该接口注册新的attention_backend后，还需要手动在枚举类AttentionBackendName中增加对应的枚举。
4. 模型层间尚未使用repeat layers处理冗余层，仿真效率存在优化空间。
5. 模型的输入存在不同。不同的模型的输入要求不一样，例如WanTransformer3DModel的forward必要输入hidden_states、timestep、encoder_hidden_states，可选输入encoder_hidden_states_image、return_dict、attention_kwargs；HunyuanVideoTransformer3DModel的forward必要输入为hidden_states、time_stamp、encoder_hidden_states、encoder_attention_mask、pooled_projections，可选输入guidance、attention_kwargs、return_dict。而且因为diffusers中的模型的forward不写**kwargs，多余的参数传入会直接报错。当前只支持了这两种模型，后续新增模型需要考虑更加泛化的设计。
6. 量化特性适配
7. 并行特性适配，cfg并行、ulysses并行、ring-attention、cp
8. attention_cache和dit_cache特性适配
9. 仿真的性能和实测结果的对齐
10. 完整RFC
11. 当前只建模了dit部分，补齐完整的diffusers pipe建模
