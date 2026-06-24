# RFC: Diffusers 模型远端 Repo ID 自动加载支持

## 元数据

| 项目 | 内容 |
| :--- | :--- |
| **状态** | Draft (草稿，已按实现范围更新) |
| **作者** | minghang_c |
| **创建日期** | 2026-05-16 |
| **更新日期** | 2026-06-13 |
| **相关 Issue/PR** | TBD |

---

## 1. 概述

### 1.1 简介

本提案建议为 `video_generate` 的 Diffusers 模型加载流程增加远端 repo id 自动解析能力，使用户可以直接传入类似 `Wan-AI/Wan2.2-T2V-A14B-Diffusers` 的模型标识，而无需先手动下载模型目录。

该能力会在保持现有本地目录加载方式不变的前提下，复用模型 Hub 默认缓存目录下载 Diffusers 模型配置文件，并将解析后的本地快照路径交给现有 Diffusers 配置加载逻辑继续处理。

本 RFC 最初只覆盖 Hugging Face。实现阶段进一步与 `text_generate` 的远端来源设计对齐，新增了 ModelScope 来源支持，并支持通过 `model_id` 追加子目录的方式选择多 variant aggregate repo 中的具体 Diffusers config 子目录。

### 1.2 动机

当前 `tensor_cast.diffusers.load_config_from_file` 要求 `model_id` 必须是本地目录。用户在使用 `video_generate` 时，如果希望使用公开 Hugging Face 或 ModelScope Diffusers 仓库，需要先手动下载模型配置并组织成本地目录。这与 `text_generate` 支持直接传入远端模型标识的体验不一致，也增加了视频模型性能建模的上手成本。

Diffusers 建模当前主要读取 `transformer/config.json`，以及可选的 `vae/config.json`。因此，自动下载阶段只需要拉取配置文件即可满足现有建模路径，不需要下载完整权重文件，能够在改善易用性的同时控制网络与磁盘开销。

部分公开视频模型仓库是 aggregate repo，例如根目录不直接包含 `transformer/config.json`，而是在 `transformer/<variant>/config.json` 下放置多个 variant。为了支持这类仓库，同时避免新增复杂的自动 variant 推断逻辑，本 RFC 支持用户在 `model_id` 中追加明确子目录。

### 1.3 目标

目标：

1. `video_generate` 的 `model_id` 同时支持本地 Diffusers 目录和远端 repo id。
2. 本地目录路径继续走现有流程，不改变已有用户工作流，也不触发网络访问。
3. 非本地目录的 `model_id` 默认视为 Hugging Face repo id，通过 `huggingface_hub.snapshot_download` 下载配置文件到 Hugging Face 默认缓存目录。
4. 新增 `--remote-source {huggingface,modelscope}`，支持用户显式选择 ModelScope 来源；默认值为 `huggingface`。
5. 下载范围限制为 `config.json` 和 `**/config.json`，避免拉取模型权重。
6. 支持远端 repo id 追加子目录：`<namespace>/<repo>/<subfolder>`，例如 `tencent/HunyuanVideo-1.5/transformer/720p_i2v_distilled_sparse`。
7. 下载失败、指定子目录不存在或仓库结构不符合 Diffusers 模型约定时，提供清晰错误信息。
8. 更新 `video_generate` 的 CLI 参数说明，使能力边界对用户可见。
9. 远端 snapshot 下载阶段隐藏 Hub 客户端的进度输出和噪音日志，避免正常 CLI 输出被大量下载日志干扰。

非目标：

1. 不下载完整模型权重。
2. 不改造 Diffusers 模型核心加载、建模和执行逻辑。
3. 不提供额外认证交互能力；私有仓库、网络失败等场景由 Hugging Face Hub / ModelScope 的环境配置和错误处理承接。
4. 不做自动 variant 推断；aggregate repo 需要用户显式在 `model_id` 中追加具体子目录。
5. 不新增 committed 联网单元测试；真实 Hub 访问只作为交付期端到端验证目标。

## 2. 用例分析

### 2.1 公开 Hugging Face Diffusers 模型快速建模

用户在 `video_generate` 中传入公开 Diffusers 仓库 id，例如：

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device TEST_DEVICE \
  --batch-size 1 \
  --seq-len 128 \
  --frame-num 81 \
  --sample-step 1
```

系统自动下载仓库中的配置文件到 Hugging Face 默认缓存目录，并继续使用现有 Diffusers 配置加载流程完成模型构建。

### 2.2 公开 ModelScope Diffusers 模型快速建模

用户可以通过 `--remote-source modelscope` 指定 ModelScope 来源，例如：

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --remote-source modelscope \
  --device TEST_DEVICE \
  --batch-size 1 \
  --seq-len 128 \
  --frame-num 81 \
  --sample-step 1
```

系统使用 ModelScope 的 `snapshot_download` 下载配置文件，并继续复用同一套本地 Diffusers 配置加载逻辑。

### 2.3 aggregate repo 的显式子目录选择

对于根目录不直接暴露 `transformer/config.json` 的多 variant 仓库，用户可以把子目录拼到 `model_id` 后面，例如：

```bash
python -m cli.inference.video_generate tencent/HunyuanVideo-1.5/transformer/720p_i2v_distilled_sparse \
  --device TEST_DEVICE \
  --batch-size 1 \
  --seq-len 128 \
  --frame-num 121 \
  --sample-step 1
```

解析时会将前两段作为 repo id：

```text
repo_id = tencent/HunyuanVideo-1.5
subfolder = transformer/720p_i2v_distilled_sparse
```

远端下载仍然只下载 config 文件，最终传给 `load_config_from_file` 的路径是本地快照中的指定子目录。

### 2.4 本地 Diffusers 目录保持兼容

用户传入已存在的本地模型目录时，系统不访问 Hugging Face Hub 或 ModelScope，直接按现有逻辑读取本地 `transformer/config.json` 和可选 `vae/config.json`。

### 2.5 DFX 要求

- **兼容性**：本地路径行为不变；新增能力只影响非目录 `model_id`。
- **可维护性**：路径解析逻辑应独立于 Diffusers 模型加载逻辑，避免把下载细节散落到模型构建流程中。
- **可复用性**：ModelScope snapshot 参数兼容逻辑应与 `text_generate` 共享，避免两套实现漂移。
- **可测试性**：下载路径通过 mock `snapshot_download` 做轻量单元测试，不依赖真实网络。
- **可靠性**：下载失败、缺少 `transformer/config.json`、指定子目录不存在等错误需要明确提示用户原因和替代操作。
- **性能与资源**：只下载配置文件，避免权重文件带来的大额网络和磁盘消耗。
- **日志体验**：隐藏 snapshot 下载进度和 Hub 客户端噪音输出；保留真实错误的异常传播。

## 3. 方案设计

### 3.1 总体方案

在 Diffusers 模型构建入口增加一个模型路径解析步骤。该步骤接收用户传入的 `model_id` 和 `remote_source`，输出一个可供现有 `load_config_from_file` 使用的本地目录路径。

解析规则如下：

1. 如果 `model_id` 是本地目录，直接返回该路径，不访问远端 Hub。
2. 如果 `model_id` 不是本地目录，根据 `remote_source` 选择 Hugging Face 或 ModelScope。
3. 对远端 `model_id` 做轻量拆分：
   - 只有两段或更少：整体作为 repo id。
   - 超过两段：前两段作为 repo id，剩余部分作为 snapshot 内部子目录。
4. 调用对应 Hub 的 config-only snapshot 下载函数。
5. 使用 Hub 默认缓存解析逻辑，不显式指定 cache dir，使其自然遵循用户环境变量，例如 `HF_HOME` / `HF_HUB_CACHE`。
6. 如果存在子目录，则在下载后的 snapshot 中定位该子目录，并校验其存在且不逃逸出 snapshot 根目录。
7. 返回最终本地路径。
8. 后续继续由现有 `load_config_from_file` 查找并读取 `transformer/config.json`、兼容的单 Transformer config，以及可选 `vae/config.json`。

建议数据流：

```text
video_generate CLI / Web UI
  -> build_diffusers_transformer_model(model_id, ..., remote_source)
      -> resolve_diffusers_model_path(model_id, remote_source)
           -> local directory: return as-is
           -> Hugging Face repo id: snapshot_download(..., allow_patterns=["config.json", "**/config.json"])
           -> ModelScope repo id: snapshot_download(..., config-only allowlist)
           -> optional subfolder: return snapshot/subfolder
      -> load_config_from_file(resolved_model_path)
      -> DiffusersTransformerModel(...)
```

该方案将“用户输入解析”和“Diffusers 配置加载”解耦，现有模型加载代码只需要面对本地目录，符合当前架构边界。

### 3.2 技术选型

#### 方案 A：在 Diffusers 构建流程前增加独立 resolver（推荐）

优点：

- 保持 `load_config_from_file` 的职责稳定，仍然只处理本地目录中的配置文件。
- 便于单元测试本地路径、远端 repo id、子目录、下载失败等分支。
- 与 `text_generate` 的用户体验对齐，同时不强行复用 Transformer 配置解析实现。

缺点：

- 需要在 Diffusers 构建链路中新增一个解析函数或模块。

#### 方案 B：在 `load_config_from_file` 内直接处理远端下载

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

- **网络访问边界**：只有当 `model_id` 不是本地目录时才访问远端 Hub。
- **下载范围**：使用 `allow_patterns=["config.json", "**/config.json"]` 限定只下载配置文件，避免默认下载权重或其他大文件。
- **ModelScope 兼容性**：ModelScope 不同版本可能使用 `allow_patterns` 或 `allow_file_pattern`，实现应兼容这两类参数；若当前版本无法表达 config-only allowlist，应拒绝调用，避免退化为完整仓库下载。
- **缓存行为**：不自定义缓存目录，遵循 Hub 默认行为，兼容用户已有缓存配置。
- **认证与私有仓库**：不新增 token 参数或交互式认证流程；用户可继续使用 Hub 支持的环境变量、登录态或本地配置。
- **远端子目录安全**：只允许子目录留在 snapshot 根目录内部；拒绝空路径段、`.`、`..` 等不安全输入。
- **日志控制**：snapshot 下载期间隐藏 stdout、stderr、进度条和 warning 级别以下的 Hub 客户端日志；下载完成后的 resolver 细节日志使用 debug 级别，不在默认 info 输出中暴露缓存路径。
- **错误信息**：下载失败时提示自动下载失败，并建议用户手动下载后传入本地路径；缺少 `transformer/config.json` 时提示该 repo 不是当前支持的 Diffusers 模型目录结构；指定子目录不存在时提示该子目录不在 config-only snapshot 中。
- **回滚成本**：本地路径逻辑不变，若远端加载存在问题，可以通过传入本地目录绕过新增路径。

### 3.4 编程与调用设计

#### 3.4.1 编程模型基本设计

本能力面向 Python CLI、Web UI 和现有 TensorCast Diffusers 建模流程，不引入新的外部服务或运行时依赖模式。实现应复用项目已有 Python 环境与 Hub 客户端依赖。

开发者主要关注一个输入输出明确的 resolver：

- 输入：用户传入的 `model_id` 字符串和 `remote_source`。
- 输出：本地 Diffusers 模型目录路径。
- 副作用：当输入不是本地目录时，可能触发 Hugging Face Hub 或 ModelScope 配置文件下载。

#### 3.4.2 接口定义与设计

建议新增或等价实现如下内部函数：

```python
def resolve_diffusers_model_path(
    model_id: str,
    remote_source: str = RemoteSource.huggingface,
) -> str:
    ...
```

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
| --- | --- | --- | --- | --- |
| model_id | 输入 | str | 本地 Diffusers 目录路径、远端 repo id，或远端 repo id 加子目录 | 已存在目录、`<namespace>/<repo>`、`<namespace>/<repo>/<subfolder>` |
| remote_source | 输入 | str | 远端来源 | `huggingface` 或 `modelscope` |
| return | 输出 | str | 可供 Diffusers 配置加载逻辑读取的本地目录路径 | 本地目录路径 |

异常处理：

- `remote_source` 非法：抛出包含可选值的 `ValueError`。
- 本地目录路径不存在且远端下载失败：抛出带有明确说明的 `RuntimeError`。
- 指定远端子目录不存在或不安全：抛出 `ValueError`。
- 下载成功但缺少 `transformer/config.json` 或兼容的单 Transformer config：抛出当前 Diffusers 模型结构不支持的异常。

变更说明：

- `build_diffusers_transformer_model` 增加 `remote_source` 参数，并在内部调用 resolver。
- `video_generate` 的 `model_id` CLI help 文案需要更新为“本地 Diffusers 目录、远端 repo id，或远端 repo id 加子目录”。
- `video_generate` 新增 `--remote-source {huggingface,modelscope}` 参数。
- Web UI 的 video_generate 表单新增 remote-source 下拉框，并把该值纳入任务参数和任务 hash。
- 现有本地目录调用方式不需要迁移。

#### 3.4.3 使用说明

本地目录使用方式保持不变：

```bash
python -m cli.inference.video_generate /path/to/diffusers/model \
  --device TEST_DEVICE \
  --batch-size 1 \
  --seq-len 128
```

Hugging Face repo id 使用方式：

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --device TEST_DEVICE \
  --batch-size 1 \
  --seq-len 128
```

ModelScope repo id 使用方式：

```bash
python -m cli.inference.video_generate Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --remote-source modelscope \
  --device TEST_DEVICE \
  --batch-size 1 \
  --seq-len 128
```

aggregate repo 子目录使用方式：

```bash
python -m cli.inference.video_generate tencent/HunyuanVideo-1.5/transformer/720p_i2v_distilled_sparse \
  --device TEST_DEVICE \
  --batch-size 1 \
  --seq-len 128
```

如自动下载失败，用户可以手动下载模型配置或模型目录，并将 `model_id` 指向本地目录。

## 4. 测试设计

建议新增轻量单元测试覆盖以下场景：

1. **本地目录路径**：构造包含 `transformer/config.json` 的临时目录，验证 resolver 直接返回该目录且不会调用远端下载函数。
2. **Hugging Face repo id**：传入非目录字符串，mock `huggingface_hub.snapshot_download` 返回临时快照目录，验证调用参数包含 `allow_patterns=["config.json", "**/config.json"]`。
3. **ModelScope repo id**：mock `modelscope.snapshot_download`，验证 config-only allowlist 参数兼容 `allow_patterns` 和 `allow_file_pattern`。
4. **远端 repo 子目录**：传入 `<namespace>/<repo>/<subfolder>`，验证下载使用前两段 repo id，返回路径为 snapshot 内部子目录。
5. **子目录错误**：验证不存在子目录、`.`、`..`、空路径段等输入会得到明确错误。
6. **下载失败**：mock snapshot 下载函数抛出异常，验证最终错误信息提示自动下载失败以及可手动下载后传入本地路径。
7. **缺少 transformer 配置**：构造不包含 `transformer/config.json` 或兼容单 Transformer config 的目录或下载快照，验证错误信息说明 repo 不是当前支持的 Diffusers-style 模型目录。
8. **CLI help 文案**：验证 `video_generate --help` 中 `model_id` 描述包含本地目录、远端 repo id、子目录，以及 `--remote-source` 选项。
9. **Web UI 命令构造**：验证默认 Hugging Face 不额外生成 `--remote-source`，ModelScope 会生成 `--remote-source modelscope`，且 `remote_source` 纳入任务参数。
10. **日志控制**：mock snapshot 下载函数输出 stdout/stderr/logging，验证调用方不会看到下载进度噪音。

不建议在单元测试中访问真实 Hugging Face 或 ModelScope 网络服务，避免测试不稳定和 CI 环境依赖外部网络。真实 Wan2.2 / HunyuanVideo1.5 远端加载应作为交付期 E2E 验证目标，而不是 committed 回归测试。

## 5. 缺点和风险

1. **非目录字符串的解释变化**：过去非目录会直接报错；新增后会尝试作为远端 repo id 下载。若用户传错本地路径，错误会变成下载失败提示。应在错误信息中同时说明“不是本地目录且自动下载失败”。
2. **网络依赖**：公开 repo 自动加载依赖用户环境能访问 Hugging Face 或 ModelScope。失败时保留手动下载到本地路径的替代方式。
3. **仓库结构差异**：不是所有远端 repo 都符合当前 Diffusers 目录约定。缺少 `transformer/config.json` 或兼容单 Transformer config 时应明确提示不支持该仓库结构。
4. **子目录解析约定**：远端子目录格式默认取前两段作为 repo id，剩余部分作为 snapshot 内部路径。该设计覆盖当前 Hugging Face / ModelScope 常见 repo id，但不尝试处理更复杂命名空间层级。
5. **依赖版本差异**：`snapshot_download` 的参数行为依赖 Hub 客户端版本。实现和测试需要基于项目当前依赖版本确认，并在 ModelScope 上做参数兼容。
6. **隐藏下载日志的可观测性权衡**：隐藏进度条能改善 CLI 输出体验，但也减少了下载过程中的实时反馈。真实错误仍应通过异常正常暴露。

## 6. 现有技术

`text_generate` 已支持传入远端模型标识，并依赖 Hugging Face / ModelScope 生态完成配置解析和缓存管理。本提案参考该用户体验，但 Diffusers 当前建模流程只需要本地配置目录，因此采用轻量 resolver 将 repo id 转换为本地快照路径，而不是重写 Diffusers 模型加载逻辑。

Hugging Face Hub 的 `snapshot_download` 已支持 `allow_patterns` 和默认缓存目录管理，能够满足只下载配置文件并复用用户缓存配置的需求。

ModelScope 的 `snapshot_download` 在不同版本中存在 allowlist 参数命名差异，实际实现需要兼容 `allow_patterns` 与 `allow_file_pattern`，并在无法表达 config-only 下载时拒绝继续，避免误下载完整模型仓库。

## 7. 未解决问题

1. **相关 Issue/PR 编号**：当前 RFC 尚未关联具体 Issue/PR，需要在进入评审或提交实现 PR 前补充。
2. **官方 HunyuanVideo1.5 aggregate repo 的模型类支持**：子目录解析只能解决 config 定位问题；若选中的 config 使用当前 TensorCast Diffusers mapping 未支持的 transformer class，例如 `HunyuanVideo_1_5_DiffusionTransformer`，仍需要单独补模型类适配。
3. **更显式的子目录参数**：当前方案复用 `model_id` 表达子目录，没有新增 `--remote-subfolder` 或 `--variant` 参数。若后续需要更强校验或 Web UI 结构化输入，可以另行设计。

---

## 附录

### 参考资料链接

- `docs/superpowers/specs/2026-06-12-diffusers-remote-autoload-design.md`
- Hugging Face Hub `snapshot_download` API
- ModelScope `snapshot_download` API

### 术语表

- **Diffusers-style 模型目录**：包含 `transformer/config.json`，并可选包含 `vae/config.json` 的 Diffusers 模型目录。
- **repo id**：模型 Hub 上的仓库标识，例如 `Wan-AI/Wan2.2-T2V-A14B-Diffusers`。
- **remote source**：远端模型来源，目前支持 `huggingface` 和 `modelscope`。
- **snapshot subfolder**：远端 repo 下载到本地 cache 后，在 snapshot 内部进一步选择的子目录，例如 `transformer/720p_i2v_distilled_sparse`。

### 文档更新计划

- RFC 评审后补充关联 Issue/PR。
- 实现完成后同步更新 CLI help 或用户文档中关于 `model_id`、`remote_source`、aggregate repo 子目录格式的说明。
