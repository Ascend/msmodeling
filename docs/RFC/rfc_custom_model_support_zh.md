# RFC: 支持用户自定义模型

---

## 元数据

| 项目 | 内容                                             |
| :--- |:-----------------------------------------------|
| **状态** | 已批准                                            |
| **作者** | genius52                                       |
| **创建日期** | 2026-1-28                                      |
| **更新日期** | 2026-3-11                                      |
| **相关链接** | <https://gitcode.com/Ascend/msmodeling/pull/105> |

---

## 1. 插件化系统执行工作流程

### 1.1 模型加载过程

系统在初始化期间从两个目录加载模型定义，`tensor_cast/transformers/builtin_model` 和 `tensor_cast/custom_model`：
**内置模型注册**：

- **位置**：`tensor_cast/transformers/builtin_model/` 目录
- **执行注册**：在 `__init__.py` 中自动模块导入

**用户自定义模型注册**：
      - **位置**：`tensor_cast/custom_model/` 目录（仅在存在时加载）
      - **执行注册**：`custom_model_registry.py:336` 中的 `import_custom_model_modules()`
**注册机制**：

- `@register_custom_model(model_type)`：用户自定义模型注册

### 1.2 模型初始化

`TransformerModel.__init__()` 中，系统会检查模型类型是否已注册自定义模型：

```python
with self.set_default_dtype():
    # 检查是否有自定义模型实现
    custom_fn = get_custom_model(self.hf_config.model_type)
    if custom_fn:
        custom_fn(self)
    else:
        # 执行默认逻辑
        wrap_model(self)
        maybe_enable_mtp(self)
        maybe_reuse_layers(self)
        patch_rotary_emb(self)
        patch_attention(self)
        patch_mla(self)
        patch_moe(self)
        quantize_model(self)
        shard_model(self)
```

### 1.3 转换函数调用

所有转换函数都已抽取到独立的 `transformations.py` 文件中：

```python
# transformtions.py 中的独立转换函数
def wrap_model(model) -> None:
    # 标准化前向接口包装
    
def maybe_enable_mtp(model) -> None:
    # MTP (Multi-Stage Training) 多阶段训练机制
    # 当模型配置启用 MTP 时，将后续层训练模块替换为 MTP 训练模块，支持多阶段训练架构
    
def maybe_reuse_layers(model) -> None:
    # 层重用优化
    
def patch_rotary_emb(model) -> None:
    # 旋转位置编码适配
    
def patch_attention(model) -> None:
    # 注意力模块处理
    
def patch_mla(model) -> None:
    # 多头潜在注意力处理
    
def patch_moe(model) -> None:
    # 专家混合模块处理
    
def quantize_model(model) -> None:
    # 模型量化处理
    
def shard_model(model) -> None:
    # 模型分片处理
```

### 1.4 执行逻辑

系统采用简单的条件检查机制：

- **条件检查**：如果注册了自定义模型函数，则执行自定义逻辑
- **默认执行**：如果没有自定义模型注册，则按固定顺序执行9个标准转换函数

### 1.5 组件职责

**注册系统** (`model_registry.py`)：

- `register_custom_model(model_type)`：存储用户自定义模型函数
- 模式匹配支持精确匹配和通配符（如"bailing_moe"、"deepseek_*"）
- 注册表为单一的全局字典 `_CUSTOM_MODEL_REGISTRY`

**转换函数库** (`transformations.py`)：

- 9个独立的转换函数，每个函数负责特定的模型处理操作
- 按固定顺序执行：wrap_model, maybe_enable_mtp, maybe_reuse_layers, patch_rotary_emb, patch_attention, patch_mla, patch_moe, quantize_model, shard_model
  - **maybe_enable_mtp**：MTP 多阶段训练机制
- 每个函数接收 `TransformerModel` 实例并返回处理后的模型

**模型初始化器** (`TransformerModel.__init__()`)：

- 检查是否注册了自定义模型函数
- 决定执行自定义逻辑还是默认转换流程
- 确保在正确的上下文中执行转换操作

---

## 2. 插件化系统工作原理

基于实际代码实现的分析，插件系统工作原理如下：

### 2.1 插件系统工作流程

**步骤1：模型类型检测**

- 当模型初始化时，系统从`self.hf_config.model_type`检查其`model_type`
- 这决定了应该应用哪种定制化逻辑

**步骤2：注册表加载**

- 系统自动加载内置模型（`builtin_model/`）和用户模型（`custom_model/`）
- 内置模型通过模块导入在初始化时加载
- 用户模型在 `tensor_cast/custom_model/` 目录存在时才加载

**步骤3：执行逻辑**

- **检查自定义模型**：调用 `get_custom_model()` 检查是否注册了自定义模型函数
- **自定义执行**：如果找到自定义函数，直接执行自定义逻辑
- **默认执行**：如果没有自定义函数，按固定顺序执行9个标准转换函数

### 2.2 核心组件

**注册系统** (`model_registry.py`)：

- `_CUSTOM_MODEL_REGISTRY`：存储用户自定义模型函数
- 模式匹配支持精确匹配和通配符（如"bailing_moe"、"deepseek_*"）
- 注册装饰器为单一的 `@register_custom_model`

**转换函数库** (`transformations.py`)：

- 9个独立的转换函数，每个函数处理特定的模型转换任务
- 函数按固定顺序独立调用，不存在复杂的选择逻辑
- 每个函数直接接收和操作 `TransformerModel` 实例

**模型初始化器** (`model.py`)：

- 实现 `get_custom_model()` 的条件检查逻辑
- 根据检查结果选择执行路径
- 确保在正确的上下文中执行转换操作

---

## 3. 实现状态

### 3.1 已完成功能

- **步骤处理机制**：成功将单一的`patch_model`替换为多步骤处理（9个标准转换步骤）
- **注册系统**：实现了简单的条件检查机制，支持用户自定义模型注册
- **转换函数抽取**：所有转换函数已抽取到独立的 `transformations.py` 文件中

### 3.2 当前注册机制

#### 3.2.1 单一注册装饰器

```python
@register_custom_model("model_type")
def _(model: TransformerModel):
    """用户自定义模型处理函数"""
    # 实现自定义逻辑
    model.some_custom_transformation()
```

#### 3.2.2 注册系统工作流程

```python
# 在 custom_model_registry.py 中
_CUSTOM_MODEL_REGISTRY: Dict[str, Callable] = {}

def register_custom_model(model_type: str):
    def decorator(fn: Callable[["TransformerModel"], "TransformerModel"]):
        _CUSTOM_MODEL_REGISTRY[model_type] = fn
        return fn
    return decorator

def get_custom_model(model_type: str) -> Optional[Callable]:
    """获取匹配的自定义模型函数"""
    match_key = find_matching_key(_CUSTOM_MODEL_REGISTRY, model_type)
    return _CUSTOM_MODEL_REGISTRY.get(match_key) if match_key else None
```

### 3.3 自定义模型使用示例

#### 3.3.1 基本自定义示例

```python
@register_custom_model("bailing_moe")
def _(model: TransformerModel):
    """BailingMoe模型特殊处理"""
    # 可以选择性调用转换函数
    wrap_model(model)
    maybe_enable_mtp(model)
```

#### 3.3.2 复杂自定义示例

```python
 @register_custom_model("custom_model")
 def _(model: TransformerModel):
     """复杂自定义模型处理"""
     wrap_model(model)
     
     # 自定义注意力处理
     patch_attention(model)
     
     # 自定义旋转位置编码
     patch_rotary_emb(model)
     
     # 跳过其他标准转换步骤
     # 只执行自定义的模块替换
     custom_module_replacement(model)
```

### 3.4 最新更新

#### 3.4.1 通用模型注册支持

自定义模型系统现在通过增强的ModelProfile功能支持所有LLM模型的通用插件注册：

**ModelProfile类结构**：

- **MoE配置**：
  - `moe_module_name`：MoE模块的完整类名
  - `moe_gate_returns_raw_logits`：MoE门控是否返回原始logits
  - `moe_num_experts_key`：获取专家数量的配置键
  - `moe_field_names_override`：MoE的字段名覆盖（使用 `MoEFieldNames` 对象）

- **MTP配置**：
  - `mtp_block_module_name`：MTP块模块的完整类名

- **MLA配置**：
  - `mla_module_name`：MLA模块的完整类名
  - `mla_field_names_override`：MLA的字段名覆盖

- **自定义专家模块**：
  - `custom_expert_module_type`：用于动态创建自定义专家模块的Python类型

- **模型系列**：
  - `model_family`：用于分组相关模型类型的模型系列标识符

- **视觉语言模型**：
  - `vl_patch_method`：视觉语言模型修补方法

- **视觉配置**：增强对具有可配置路径和映射的视觉语言模型的支持

#### 3.4.2 Diffusers模型的转换逻辑复用

转换系统现在通过统一的转换逻辑支持diffusers模型：

**Diffusers模型初始化流程**：
在`DiffusersTransformerModel.__init__()`中，转换函数按顺序调用：

```python
 # 模型创建和设置
 with init_on_device_without_buffers("meta"), no_init_weights():
     self._inner = model_class.from_config(hf_config).to(model_config.dtype)
 self._inner.eval()
 
 # 用于标准模型和diffusers模型的统一转换流水线
 wrap_model(self)           # 处理transformer和diffusers接口
 quantize_model(self)       # 支持两种模型类型的共享量化逻辑
 quantize_linear(self)      # 应用于单独层级的量化
 # 注意: shard_model()在转换函数中被隐式处理
```

   **统一函数支持**：

- `wrap_model()` - 检测模型类型并应用适当的接口标准化
- `quantize_model()` - 处理transformer和diffusers模型的量化
- `quantize_linear()` - 应用带有模型感知优化的线性层量化
