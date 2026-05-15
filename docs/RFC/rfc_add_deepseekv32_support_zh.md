# RFC: DeepseekV32模型适配支持

## 元数据

| 项目 | 内容                 |
| :--- |:-------------------|
| **状态** | 已批准                |
| **作者** | HongMaoShuiGuai    |
| **创建日期** | 2026-1-30         |
| **相关链接** | [适配DeepseekV32模型](https://gitcode.com/Ascend/msmodeling/merge_requests/65) |

---

## 1. 概述

本提案旨在在项目中增加DeepseekV32的支持，并增加DeepseekSparseAttention的支持，使其能够在tensor_cast框架下正确加载和运行。

## 2. 详细设计

### 2.1 实现方案

#### 2.1.1 增加模型加载接口

当前工具已支持两种模型加载方式：传入模型repo id，然后通过transformers的AutoModel自动从huggingface上拉取模型的`config.json`文件，完成模型加载；或者传入本地的模型路径，通过transformers的`AutoModel`接口中的remote code的方式，从传入路径下的`configuration.py`和`modeling.py`文件加载模型。
而当前deepseekv32版本既没有在transformers上支持，huggingface上提供的权重文件中也没有configuration文件和modeling文件，针对这种场景，增加了新的模型加载接口。在tensor_cast中自定义实现`configuration_deepseekv32.py`和`modeling_deepseekv32.py`，再通过AutoConfig和AutoModel的register接口将模型结构和配置文件注册到`AutoConfig`和`AutoModel`中。再通过`AutoModel`，自动从huggingface上拉去模型的`config.json`文件，完成模型加载。

#### 2.1.2 DeepseekV32模型结构适配

DeepseekV32新增了DeepseekSparseAttention，基于原本的mla结构，增加了稀疏性，以及一个Indexer结构用于筛选关键key值，其余结构和DeepseekV3相同。因此`configuration_deepseekv32.py`和`modeling_deepseekv32.py`只需要继承deepseekv3的版本，而新增的`DeepseekSparseAttention`和`Indexer`可以只实现模型结构的初始化，`forward`方法因为需要在tensor_cast中实现，可以略过。

#### 2.1.3 DeepseekSparseAttention结构适配

当前tensor_cast中mla结构只支持将对应模型的attention模块替换成tensor_cast的mla模块，考虑到后续算法演进，可能会出现多种类mla结构，因此在模型初始化时，增加了根据`model_type`选择`mla_cls`的能力，后续根据实际模型的发展情况，可能需要提供根据用户输入配置`mla_cls`的能力。
在layer层中，新增加了mla_cls，DeepseekSparseAttention，继承`MultiheadLatentAttentionTensorCast`。相比于原本的mla结构，dsa新增两部分能力，mla稀疏计算和indexer的topk选择。mla稀疏计算大部分和原本的mla计算相同，差异部分通过封装加重定义的方式进行适配。原本代码中indexer部分计算包含在mla中，而tensor_cast进行仿真时并不需要indexer的返回值，考虑到相同代码复用，indexer部分的计算仿真放置在原本mla的forward之后，与实际计算的顺序存在差别。

##### mla部分稀疏计算

deepseekv32原本代码中，在q*kT得到score后，通过indexer筛选出index_topk个token，然后创建一个shape和score一致的，index_topk部分为0，其余为-inf的tensor，累加至score上。这种计算方式等价于在计算score前，直接对k矩阵进行稀疏。因此新增代码中，在mla算子输入中增加了index_topk的输入，并在计算仿真算子的性能时，选择k的seqlen维度和index_topk的最小值进行计算量的评估。
另外，mla模块的实现过程中，mla算子是直接调用的tensor_cast中的mla op，`DeepseekSparseAttention`继承后的`forward`函数，无法直接修改算子的输入，因此使用偏函数，将算子的调用封装成mla模块的`attention_backend`，后续视实际需要可以考虑改为transformers的`attention_backend`的实现方式。

##### indexer部分实现

在layer层的mla中，增加indexer模块，仿照源码实现indexer计算逻辑。indexer的计算需要mla模块中的`q_a_layernorm`的输出，通过`forward`的返回值获取的化对代码改动较大，当前采用`forward_hook`的方式获取输出，后续需要评估是否需要修改，并且如果接入mlapo算子后，该值该如何获取也需要重新设计。
indexer部分存在cache机制，仿照kvcache，在模型输入的时候创建了一个indexer_cache。
当前实现版本存在不足，待完善点：

1. deepseekv32的源码中未实现indexer的并行，indexer的并行和并行方案暂不明确
2. deepseekv32的源码中indexer部分使用的是fp8量化，当前仿真仅改为半精度版本
3. 源码中使用了一个`fp8_index`自定义算子，基于该算子实现了半精度的仿真算子，仿真使用的默认memory-bandwidth bound的评估方式，且输入的tensor还存在一定问题。预计该算子在整体的推理中耗时占用较小，优先级靠后
4. mla的cp并行+dsa暂不支持
5. cache部分需要确认和实际部署是否一致

### 2.2 替代方案

### 2.3 方案分析

**选择当前方案的原因：**

## 3. 实施计划

### 已完成功能开发

- [x] 支持deepseekv32性能仿真建模
- [x] 支持DeepseekSparseAttention性能仿真建模

### 后续优化

- [ ] transformers5.0之后的版本可能会提供deepseekv32的支持，后续可能需要改为从transformers加载模型
- [ ] indexer部分完善
- [ ] 使用kvcache block时，实际的dsa如何按照indexer筛选出的token加载kv cache
