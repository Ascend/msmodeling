# RFC: Support for DeepseekV32 Model

## Metadata

| Item              | Content                                 |
| :---------------- | :-------------------------------------- |
| **Status**        | Approved                                |
| **Author**        | HongMaoShuiGuai                         |
| **Creation Date** | 2026-1-30                              |
| **Related Links** | [Adaptation of the DeepseekV32 Model](https://gitcode.com/Ascend/msmodeling/merge_requests/65) |

---

## 1. Overview

This proposal aims to add support for DeepseekV32 in the project and incorporate DeepseekSparseAttention, enabling proper loading and execution within the `tensor_cast` framework.

---

## 2. Detailed Design

### 2.1 Implementation Plan

#### 2.1.1 Add Model Loading Interface

The current tool already supports two model loading methods: either by passing in a model repository ID, then automatically pulling the model's `config.json` file from Hugging Face via the `AutoModel` from transformers to complete model loading; or by providing a local model path, and loading the model via the remote_code method of the `AutoModel` interface in transformers, using the `configuration.py` and `modeling.py` files located in the specified path.

However, the current deepseekv32 version is not yet supported in transformers, and the weight files provided on Hugging Face do not include configuration or modeling files. To address this scenario, a new model loading interface has been added. Within tensor_cast, we have implemented custom `configuration_deepseekv32` and `modeling_deepseekv32`, and then registered the model architecture and configuration files into `AutoConfig` and `AutoModel` via their respective `register` interfaces. Subsequently, by using `AutoModel`, the model's config.json file is automatically fetched from Hugging Face to complete the model loading.

#### 2.1.2 Adaptation of DeepseekV32 Model Architecture

DeepseekV32 introduces DeepseekSparseAttention, which builds upon the original MLA structure by incorporating sparsity and an Indexer component for selecting critical key values. The rest of its architecture remains identical to DeepseekV3. Therefore, `configuration_deepseekv32.py` and `modeling_deepseekv32.py` only need to inherit from the DeepseekV3 versions. As for the newly added `DeepseekSparseAttention` and `Indexer`, it is sufficient to implement only the initialization of their model structures, while their `forward` methods can be omitted as they will need to be implemented within tensor_cast.

#### 2.1.3 Adaptation of DeepseekSparseAttention Architecture

##### Sparse Calculation of MLA Part

In the original deepseekv32 code, after obtaining the score via q*kT, the top k tokens are filtered using an indexer. A tensor with the same shape as the score is then created, where the top k positions are set to 0 and the rest to -inf, and this tensor is added to the score. This computational approach is equivalent to directly sparsifying the k matrix before calculating the score. Therefore, in the newly added code, an input for the top k indices has been included in the mla operator's inputs. When evaluating the performance of the simulation operator, the minimum value between k's seqlen dimension and the top k is used to assess computational load.

Additionally, during the implementation of the mla module, the mla operator directly calls the mla op in tensor_cast. The forward function inherited by `DeepseekSparseAttention` cannot directly modify the operator's inputs. As a solution, a partial function is used to encapsulate the operator call as the `attention_backend` of the mla module. This implementation may be adapted to transformers' `attention_backend` approach in the future, depending on actual requirements.

##### Indexer Part

In the mla module of the layer, an indexer module has been added, implementing the indexer computation logic based on the original source code. The indexer computation requires the output from `q_a_layernorm` within the mla module. While retrieving this output via the return value of the forward function would involve significant code changes, the current implementation uses a `forward_hook` to capture the output. This approach needs future evaluation regarding whether modifications are necessary, and if the mlapo operator is integrated, how to obtain this value will require redesign.

The indexer component incorporates a caching mechanism, similar to kv_cache, where an indexer_cache is created at the time of model input.

The current implementation has certain limitations and areas for improvement:

1. The original deepseekv32 source code does not implement parallelism for the indexer. The specific parallelization strategy and feasibility for the indexer are still unclear.
2. In the original deepseekv32 source code, the indexer uses fp8 quantization, while the current simulation uses a half-precision version.
3. The source code uses a custom `fp8_index` operator. A half-precision simulation operator was implemented based on this operator. The simulation currently employs the default memory-bandwidth-bound evaluation method, and there are still some issues with the input tensors. This operator is expected to account for a relatively small portion of the overall inference time, thus it is assigned a lower priority.
4. The combination of mla's cp parallelization and dsa is not currently supported.
5. The cache implementation requires verification to ensure consistency with the actual deployment.

---

### 2.2 Alternative Solutions

---

### 2.3 Design Analysis

**Reasons for Choosing the Current Approach:**

---

## 3. Implementation Plan

### Completed Features

* [x] Support performance simulation modeling for deepseekv32
* [x] Support performance simulation modeling for DeepseekSparseAttention

### Future Optimizations

* [ ] Future versions after transformers 5.0 may provide support for deepseekv32, and subsequent implementations might need to switch to loading the model from transformers.
* [ ] Refinement of the indexer component.
* [ ] When using kvcache blocks, how does the actual DSA load kv cache according to the tokens filtered by the indexer?

---
