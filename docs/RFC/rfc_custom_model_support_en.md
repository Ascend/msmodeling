# RFC: Support for User Custom Model

## Metadata

| Item | Content                                       |
| :--- |:----------------------------------------------|
| **Status** | Approved                                      |
| **Author(s)** | genius52                                      |
| **Creation Date** | 2026-1-28                                     |
| **Update Date** | 2026-3-11                                     |
| **Related Links** | <https://gitcode.com/Ascend/msmodeling/pull/105> |

---

## 1. Plugin System Execution Workflow

### 1.1 Model Loading Process

During initialization, the system loads model definitions from two directories: `tensor_cast/transformers/builtin_model` and `tensor_cast/custom_model`:

**Built-in Models Registration**:

- **Location**: `tensor_cast/transformers/builtin_model/` directory
- **Registration Execution**: Automatic module import in `__init__.py`

**User Custom Models Registration**:
      - **Location**: `tensor_cast/custom_model/` directory (loaded only if exists)
      - **Registration Execution**: `import_custom_model_modules()` in `custom_model_registry.py:336`
**Registration Mechanism**:

- `@register_custom_model(model_type)`: User custom model registration

### 1.2 Model Initialization

In `TransformerModel.__init__()`:

```python
with self.set_default_dtype():
    # Check if there's a custom model implementation
    custom_fn = get_custom_model(self.hf_config.model_type)
    if custom_fn:
        custom_fn(self)
    else:
        # Apply standard transformation steps
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

### 1.3 Transformation Function Calls

All transformation functions have been extracted to an independent `transformations.py` file and executed in a fixed order:

```python
# Independent transformation functions in transformations.py
def wrap_model(model) -> None:
    # Standardize forward interface wrapper

def maybe_enable_mtp(model) -> None:
    # MTP (Multi-Stage Training) mechanism
    # When model config enables MTP, substitute subsequent training modules with MTP training modules to support multi-stage training architecture

def maybe_reuse_layers(model) -> None:
    # Layer reuse optimization

def patch_rotary_emb(model) -> None:
    # Rotary position encoding adaptation

def patch_attention(model) -> None:
    # Attention module processing

def patch_mla(model) -> None:
    # Multi-head latent attention processing

def patch_moe(model) -> None:
    # Mixture of experts module processing

def quantize_model(model) -> None:
    # Model quantization processing

def shard_model(model) -> None:
    # Model sharding processing
```

### 1.4 Execution Logic

The system uses a simple condition checking mechanism:

- **Condition Check**: If a custom model function is registered, execute custom logic
- **Default Execution**: If no custom model is registered, execute the 9 standard transformation functions in fixed order
- **Linear Execution**: No complex branching logic or multi-level priority mechanism exists

### 1.5 Component Responsibilities

**Registry System** (`custom_model_registry.py`):

- `register_custom_model(model_type)`: Stores user custom model functions
- Pattern matching supports exact matches and wildcards (e.g., "bailing_moe", "deepseek_*")
- Registry is a single global dictionary `_CUSTOM_MODEL_REGISTRY`

**Transformation Function Library** (`transformations.py`):

- 9 independent transformation functions, each handling specific model processing operations
- Executed in fixed order: wrap_model, maybe_enable_mtp, maybe_reuse_layers, patch_rotary_emb, patch_attention, patch_mla, patch_moe, quantize_model, shard_model
  - **maybe_enable_mtp**: MTP multi-stage training mechanism
- Each function receives a `TransformerModel` instance and returns the processed model

**Model Initializer** (`TransformerModel.__init__()`):

- Checks if a custom model function is registered
- Determines whether to execute custom logic or default transformation flow
- Ensures transformations are executed in the correct context

---

## 2. Plugin System Working Principles

Based on the analysis of actual code implementation, the plugin system works as follows:

### 2.1 How the Plugin System Works

**Step 1: Model Type Detection**

- When a model is initialized, the system checks its `model_type` from `self.hf_config.model_type`
- This determines which customization logic should be applied

**Step 2: Registry Loading**

- System automatically loads both built-in models (`builtin_model/`) and user models (`custom_model/`)
- Built-in models load via module import on initialization
- User models load if the directory exists at `tensor_cast/custom_model/`

**Step 3: Execution Logic**

- **Check Custom Model**: Call `get_custom_model()` to check if a custom model function is registered
- **Custom Execution**: If a custom function is found, directly execute custom logic
- **Default Execution**: If no custom function is found, execute the 9 standard transformation functions in fixed order

**Step 4: Step Processing**

- Each step corresponds to a specific model processing operation:
  1. `wrap_model` - Standardize forward interface with standardized wrapper
  2. `maybe_enable_mtp` - Multi-stage training optimization
  3. `maybe_reuse_layers` - Layer reuse optimization
  4. `patch_rotary_emb` - Rotary position encoding adaptation
  5. `patch_attention` - Attention mechanism replacement with custom modules
  6. `patch_mla` - Multi-head latent attention replacement
  7. `patch_moe` - Mixture of experts module replacement
  8. `quantize_model` - Model quantization
  9. `shard_model` - Model sharding

**Key Insight**: The `transformation_mixin` is not a traditional mixin for method inheritance, but a base container that provides basic placeholder implementations and handles the execution flow with actual model property/module replacements.

### 2.2 Key Components

**Registry System** (`custom_model_registry.py`):

- `_CUSTOM_MODEL_REGISTRY`: stores user custom model functions
- Pattern matching supports exact matches and wildcards (e.g., "bailing_moe", "deepseek_*")
- Registration decorator is a single `@register_custom_model`

**Transformation Function Library** (`transformations.py`):

- 9 independent transformation functions, each handling specific model transformation tasks
- Functions are called independently in fixed order, no complex selection logic
- Each function directly receives and operates on a `TransformerModel` instance

**Model Initializer** (`model.py`):

- Implements `get_custom_model()` condition checking logic
- Selects execution path based on check results
- Ensures transformations are executed in the correct context

---

## 3. Implementation Status

### 3.1 Completed Features

- **Step Processing Mechanism**: Successfully replaced single `patch_model` with multi-step processing (9 standard transformation steps)
- **Registration System**: Implemented simple condition checking mechanism to support user custom model registration
- **Transformation Function Extraction**: All transformation functions have been extracted to independent `transformations.py` file

### 3.2 Current Registration Mechanism

#### 3.2.1 Single Registration Decorator

```python
@register_custom_model("model_type")
def _(model: TransformerModel):
    """User custom model processing function"""
    # Implement custom logic
    model.some_custom_transformation()
```

### 3.3 Custom Model Usage Examples

#### 3.3.1 Basic Custom Examples

```python
@register_custom_model("bailing_moe")
def _(model: TransformerModel):
   """BailingMoe special processing"""
   # Can selectively call transformation functions
   wrap_model(model)
   maybe_enable_mtp(model)
```

#### 3.3.2 Complex Custom Examples

```python
@register_custom_model("custom_model")
def _(model: TransformerModel):
   """Complex custom model processing"""
   wrap_model(model)

   # Custom attention processing
   patch_attention(model)

   # Custom rotary position encoding
   patch_rotary_emb(model)

   # Skip other standard transformation steps
   # Only execute custom module replacements
   custom_module_replacement(model)
```

### 3.4 Latest Updates

#### 3.4.1 Universal Model Registration Support

The custom model system now supports universal plugin registration for all LLM models through enhanced ModelProfile functionality:

**ModelProfile Class Structure**:

- **MoE Configuration**:
  - `moe_module_name`: Full-qualified class name of MoE module
  - `moe_gate_returns_raw_logits`: Whether MoE gate returns raw logits
  - `moe_num_experts_key`: Configuration key(s) to get number of experts
  - `moe_field_names_override`: Field name overrides for MoE (using `MoEFieldNames`)

- **MTP Configuration**:
  - `mtp_block_module_name`: Full-qualified class name of MTP block module

- **MLA Configuration**:
  - `mla_module_name`: Full-qualified class name of MLA module
  - `mla_field_names_override`: Field name overrides for MLA

- **Custom Expert Module**:
  - `custom_expert_module_type`: Python type for dynamic custom expert module creation

- **Model Family**:
  - `model_family`: Model family identifier for grouping related model types

- **Visual Language Model**:
  - `vl_patch_method`: Method for visual language model patching

- **Visual Configuration**: Enhanced support for vision-language models with configurable paths and mappings

#### 3.4.2 Transformation Logic Reuse for Diffusers Models

The transformation system now supports diffusers models through unified transformation logic:

**Diffusers Model Initialization Flow**:
In `DiffusersTransformerModel.__init__()`, the transformation functions are called in sequence:

```python
# Model creation and setup
with init_on_device_without_buffers("meta"), no_init_weights():
    self._inner = model_class.from_config(hf_config).to(model_config.dtype)
    self._inner.eval()

    # Unified transformation pipeline for both standard and diffusers models
    wrap_model(self)           # Handles both transformer and diffusers interfaces
    quantize_model(self)       # Supports both model types with shared quantization logic
    quantize_linear(self)      # Applies quantization to individual layers
# Note: shard_model() is implicitly handled within the transformation functions
```

**Unified Function Support**:

- `wrap_model()` - Detects model type and applies appropriate interface standardization
- `quantize_model()` - Handles quantization for both transformer and diffusers models
- `quantize_linear()` - Applies linear layer quantization with model-aware optimization
