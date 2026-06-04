# RFC: Diffusers 模型 Hugging Face Repo ID 自动加载支持

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | Draft (草稿) |
| **作者** | minghang_c |
| **创建日期** | 2026-05-16 |
| **更新日期** | 2026-05-16 |
| **相关 Issue/PR** | TBD |

---

## 1. 概述

### 1.1 简介

本提案建议为 `video_generate` 的 Diffusers 模型加载流程增加 Hugging Face repo id 自动解析能力，使用户可以直接传入类似 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 的模型标识，而无需先手动下载模型目录。

该能力会在保持现有本地目录加载方式不变的前提下，复用 Hugging Face Hub 默认缓存目录下载 Diffusers 模型配置文件，并将解析后的本地快照路径交给现有 Diffusers 配置加载逻辑继续处理。

### 1.2 动机

当前 `tensor_cast.diffusers.load_config_from_file` 要求 `model_id` 必须是本地目录。用户在使用 `TensorCast: Video Generate (wan2.1)` 或命令行 `video_generate` 时，如果希望使用公开 Hugging Face Diffusers 仓库，需要先手动下载模型配置并组织成本地目录。这与 `text_generate` 支持直接传入 Hugging Face repo id 的体验不一致，也增加了视频模型性能建模的上手成本。

Diffusers 建模当前只需要读取 `transformer/config.json`，以及可选的 `vae/config.json`。因此，自动下载阶段只需要拉取配置文件即可满足现有建模路径，不需要下载完整权重文件，能够在改善易用性的同时控制网络与磁盘开销。

### 1.3 目标

目标：

1. `video_generate` 的 `model_id` 同时支持本地 Diffusers 目录和 Hugging Face repo id。
2. 本地目录路径继续走现有流程，不改变已有用户工作流。
3. 非本地目录的 `model_id` 默认视为 Hugging Face repo id，通过 `huggingface_hub.snapshot_download` 下载配置文件到 Hugging Face 默认缓存目录。
4. 下载范围限制为 `**/config.json`，避免拉取模型权重。
5. 下载失败或仓库结构不符合 Diffusers 模型约定时，提供清晰错误信息。
6. 更新 `video_generate` 的 CLI 参数说明，使能力边界对用户可见。

非目标：

1. 不支持 Hugging Face 以外的模型仓库。
2. 不改造 Diffusers 模型核心加载、建模和执行逻辑。
3. 不提供额外认证交互能力；私有仓库、网络失败等场景由 Hugging Face Hub 的环境配置和错误处理承接。
4. 不下载完整模型权重。

## 2. 用例分析

### 2.1 公开 Hugging Face Diffusers 模型快速建模

用户在 `video_generate` 中传入公开 Diffusers 仓库 id，例如：

```bash
python -m cli.inference.video_generate --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers ...
```

系统自动下载仓库中的配置文件到 Hugging Face 默认缓存目录，并继续使用现有 Diffusers 配置加载流程完成模型构建。

### 2.2 本地 Diffusers 目录保持兼容

用户传入已存在的本地模型目录时，系统不访问 Hugging Face Hub，直接按现有逻辑读取本地 `transformer/config.json` 和可选 `vae/config.json`。

### 2.3 DFX 要求

- **兼容性**：本地路径行为不变；新增能力只影响非目录 `model_id`。
- **可维护性**：路径解析逻辑应独立于 Diffusers 模型加载逻辑，避免把下载细节散落到模型构建流程中。
- **可测试性**：下载路径通过 mock `snapshot_download` 做轻量单元测试，不依赖真实网络。
- **可靠性**：下载失败、缺少 `transformer/config.json` 等错误需要明确提示用户原因和替代操作。
- **性能与资源**：只下载配置文件，避免权重文件带来的大额网络和磁盘消耗。

## 3. 方案设计

### 3.1 总体方案

在 Diffusers 模型构建入口增加一个模型路径解析步骤。该步骤接收用户传入的 `model_id`，输出一个可供现有 `load_config_from_file` 使用的本地目录路径。

解析规则如下：

1. 如果 `model_id` 是本地目录，直接返回该路径。
2. 如果 `model_id` 不是本地目录，则将其视为 Hugging Face repo id。
3. 调用 `huggingface_hub.snapshot_download(repo_id=model_id, allow_patterns=["**/config.json"])`。
4. 使用 Hugging Face Hub 默认缓存解析逻辑，不显式指定 cache dir，使其自然遵循 `HF_HOME` / `HF_HUB_CACHE` 等环境变量。
5. 返回 `snapshot_download` 返回的本地快照目录。
6. 后续继续由现有 `load_config_from_file` 查找并读取 `transformer/config.json` 与可选 `vae/config.json`。

建议数据流：

```text
video_generate CLI
  -> UserInputConfig / Diffusers model build path
  -> resolve_diffusers_model_path(model_id)
       -> local directory: return as-is
       -> repo id: snapshot_download(..., allow_patterns=["**/config.json"])
  -> load_config_from_file(resolved_model_path)
  -> build_diffusers_transformer_model
```

该方案将“用户输入解析”和“Diffusers 配置加载”解耦，现有模型加载代码只需要面对本地目录，符合当前架构边界。

### 3.2 技术选型

#### 方案 A：在 Diffusers 构建流程前增加独立 resolver（推荐）

优点：

- 保持 `load_config_from_file` 的职责稳定，仍然只处理本地目录中的配置文件。
- 便于单元测试本地路径、远端 repo id、下载失败等分支。
- 与 `text_generate` 的用户体验对齐，同时不强行复用 Transformer 配置解析实现。

缺点：

- 需要在 Diffusers 构建链路中新增一个解析函数或模块。

#### 方案 B：在 `load_config_from_file` 内直接处理 Hugging Face 下载

优点：

- 调用方改动更少。

缺点：

- `load_config_from_file` 的职责从“读取本地配置文件”扩大为“解析输入、下载远端资源、读取配置”，边界不清晰。
- 后续测试和错误定位更复杂。

#### 方案 C：要求用户继续手动下载

优点：

- 不需要改动代码。

缺点：

- 用户体验与 `text_generate` 不一致。
- 默认使用公开 Diffusers 仓库时仍存在不必要的准备成本。

### 3.3 安全隐私与 DFX 设计

- **网络访问边界**：只有当 `model_id` 不是本地目录时才访问 Hugging Face Hub。
- **下载范围**：使用 `allow_patterns=["**/config.json"]` 限定只下载配置文件，避免默认下载权重或其他大文件。
- **缓存行为**：不自定义缓存目录，遵循 Hugging Face Hub 默认行为，兼容用户已有的 `HF_HOME` / `HF_HUB_CACHE` 配置。
- **认证与私有仓库**：不新增 token 参数或交互式认证流程；用户可继续使用 Hugging Face Hub 支持的环境变量、登录态或本地配置。
- **错误信息**：下载失败时提示自动下载失败，并建议用户手动下载后传入本地路径；缺少 `transformer/config.json` 时提示该 repo 不是当前支持的 Diffusers 模型目录结构。
- **回滚成本**：本地路径逻辑不变，若远端加载存在问题，可以通过传入本地目录绕过新增路径。

### 3.4 编程与调用设计

#### 3.4.1 编程模型基本设计

本能力面向 Python CLI 和现有 TensorCast Diffusers 建模流程，不引入新的外部服务或运行时依赖模式。实现应复用项目已有 Python 环境与 `huggingface_hub` 依赖。

开发者主要关注一个输入输出明确的 resolver：

- 输入：用户传入的 `model_id` 字符串。
- 输出：本地 Diffusers 模型目录路径。
- 副作用：当输入不是本地目录时，可能触发 Hugging Face Hub 配置文件下载。

#### 3.4.2 接口定义与设计

建议新增或等价实现如下内部函数：

```python
def resolve_diffusers_model_path(model_id: str) -> str:
    ...
```

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
| --- | --- | --- | --- | --- |
| model_id | 输入 | str | 本地 Diffusers 目录路径或 Hugging Face repo id | 已存在目录，或合法 Hugging Face repo id |
| return | 输出 | str | 可供 Diffusers 配置加载逻辑读取的本地目录路径 | 本地目录路径 |

异常处理：

- 本地目录路径不存在且 Hugging Face 下载失败：抛出带有明确说明的异常。
- 下载成功但缺少 `transformer/config.json`：抛出当前 Diffusers 模型结构不支持的异常。

变更说明：

- `video_generate` 的 `model_id` CLI help 文案需要更新为“本地 Diffusers 目录或 Hugging Face repo id”。
- 现有本地目录调用方式不需要迁移。

#### 3.4.3 使用说明

本地目录使用方式保持不变：

```bash
python -m cli.inference.video_generate --model_id /path/to/diffusers/model ...
```

新增 Hugging Face repo id 使用方式：

```bash
python -m cli.inference.video_generate --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers ...
```

如自动下载失败，用户可以手动下载模型配置或模型目录，并将 `--model_id` 指向本地目录。

## 4. 测试设计

建议新增轻量单元测试覆盖以下场景：

1. **本地目录路径**：构造包含 `transformer/config.json` 的临时目录，验证 resolver 直接返回该目录且不会调用 `snapshot_download`。
2. **Hugging Face repo id**：传入非目录字符串，mock `huggingface_hub.snapshot_download` 返回临时快照目录，验证调用参数包含 `allow_patterns=["**/config.json"]`。
3. **下载失败**：mock `snapshot_download` 抛出异常，验证最终错误信息提示自动下载失败以及可手动下载后传入本地路径。
4. **缺少 transformer 配置**：构造不包含 `transformer/config.json` 的目录或下载快照，验证错误信息说明 repo 不是当前支持的 Diffusers-style 模型目录。
5. **CLI help 文案**：验证 `video_generate --help` 中 `model_id` 描述包含本地目录和 Hugging Face repo id 两种输入。

不建议在单元测试中访问真实 Hugging Face 网络服务，避免测试不稳定和 CI 环境依赖外部网络。

## 5. 缺点和风险

1. **非目录字符串的解释变化**：过去非目录会直接报错；新增后会尝试作为 Hugging Face repo id 下载。若用户传错本地路径，错误会变成下载失败提示。应在错误信息中同时说明“不是本地目录且自动下载失败”。
2. **网络依赖**：公开 repo 自动加载依赖用户环境能访问 Hugging Face。失败时保留手动下载到本地路径的替代方式。
3. **仓库结构差异**：不是所有 Hugging Face repo 都符合当前 Diffusers 目录约定。缺少 `transformer/config.json` 时应明确提示不支持该仓库结构。
4. **依赖版本差异**：`snapshot_download` 的参数行为依赖 `huggingface_hub` 版本。实现和测试需要基于项目当前依赖版本确认。

## 6. 现有技术

`text_generate` 已支持传入 Hugging Face 模型标识，并依赖 Hugging Face 生态完成配置解析和缓存管理。本提案参考该用户体验，但 Diffusers 当前建模流程只需要本地配置目录，因此采用轻量 resolver 将 repo id 转换为本地快照路径，而不是重写 Diffusers 模型加载逻辑。

Hugging Face Hub 的 `snapshot_download` 已支持 `allow_patterns` 和默认缓存目录管理，能够满足只下载配置文件并复用用户缓存配置的需求。

## 7. 未解决问题

1. **相关 Issue/PR 编号**：当前 RFC 尚未关联具体 Issue/PR，需要在进入评审或提交实现 PR 前补充。
2. **错误类型选择**：实现时需要确认使用项目现有异常类型，还是直接抛出 `ValueError` / `RuntimeError`。
3. **函数放置位置**：建议放在 `tensor_cast.diffusers` 相关模块中，但最终位置应结合当前代码结构和测试可见性确定。

---

## 附录

### 参考资料链接

- `docs/superpowers/specs/2026-03-20-diffusers-hf-autoload-design.md`
- Hugging Face Hub `snapshot_download` API

### 术语表

- **Diffusers-style 模型目录**：包含 `transformer/config.json`，并可选包含 `vae/config.json` 的 Diffusers 模型目录。
- **repo id**：Hugging Face Hub 上的仓库标识，例如 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`。

### 文档更新计划

- RFC 评审后补充关联 Issue/PR。
- 实现完成后同步更新 CLI help 或用户文档中关于 `model_id` 的说明。
