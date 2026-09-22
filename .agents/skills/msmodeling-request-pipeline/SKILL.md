---
name: msmodeling-request-pipeline
description: >-
  按指定 PR 或当前 diff 跑本地两波流水：先同步最新主仓，再跑 CI + nightly/benchmark/network 相关用例。
  在用户说请求流水、请求流水 PR887、跑流水 887、跑一下流水、本地流水、change pipeline 时使用。
  不是 GitCode compile，也不是整场 nightly。
metadata:
  version: 1.0.0
  source: local
---

# msmodeling 请求流水

用户说「请求流水」时：**先同步最新主仓，再按改动选测并跑两波**。

写了 **PR 号**（例如「请求流水 PR887」「请求流水 887」）时，对的是 **那条 PR 的 head**，不是随便一个 nightly 工作树：

1. `gitcode pr view <N> -R Ascend/msmodeling --json`，记下 head 分支和 base（`master` / `26.2.0`）
2. 检出该 head（已有 worktree 就用；没有就 `git fetch` fork/head 并 `worktree add`，不要搅乱正在跑 nightly 的树）
3. 在该树上 `fetch` + merge 最新主仓 `<base>`（远端必须指向 `Ascend/msmodeling`，优先 `upstream`，否则 `origin`）
4. 跑脚本 `--run --repo <该树> --pr <N>`：选测范围是 **这条 PR 相对最新主仓的 diff**，不是整场 nightly
5. 两波都绿才在 **这个 PR** 上留「本地流水已经通过」

只说「请求流水」、没写 PR 号：用当前仓库当前分支，同样先 sync 再跑。

skill-only 的 PR（例如 887 只加 `.agents/skills`）选测可以为 0，这算通过，不要去跑 8000+ nightly。

## 先同步主仓

跑任何 pytest 之前必须跟上 canonical 主仓 `<base>`（默认 `master`，分支名含 `26.2.0` 则用 `26.2.0`），避免在过旧基线上绿了、合入后 nightly 被别人的 master 打断。同步远端必须指向 `Ascend/msmodeling`：优先 `upstream`，否则 `origin`。`origin` 若是自己的 fork，先加上指向主仓的 `upstream`，不要跟着 fork 的 master。

脚本 `--sync` 默认开：

1. `git fetch <canonical-remote> <base>`
2. 打印 `ahead` / `behind`
3. HEAD 已经被该 base 包含（detached 旧提交、已合入的历史）：停下，不要 fast-forward 到主仓尖端后再报通过
4. `behind=0`：继续
5. 工作区脏且 behind：停，先 commit/stash，不要在落后的树上跑
6. 工作区干净且 behind：`git merge --no-edit <base>`；冲突则 `--abort` 并停下
7. 相对主仓没有 diff：报错，不发「通过」
8. 不要 `reset --hard`，不要 rebase，不要 push

`--dry-run`（没写 `--run` 时的默认）只 fetch，不 merge，不改工作区，避免搅乱正在跑 nightly 的树。真正 `--run` 才 merge。`--no-sync` 仅在用户明确说不要拉主仓时用。

## 保证（以及保证不了什么）

- **能保证**：两波都绿 ⇒ 本次 diff 命中子集（映射用例 + 同文件/同目录的 nightly、benchmark、network 用例）在整场 nightly 中应当通过；若仍失败，优先按 flaky / 跨模块间接影响归因。
- **保证不了**：别人合入的代码、flaky、未映射的产品路径、跨目录间接影响、整场 8000+ 里和本次 diff 无关的失败。那不是本 skill，去跑 `scripts/run_nightly.sh`。

## 不要做的事

- 不要评论 GitCode `compile`（没点名就不触发远端）
- 不要跑整仓 `scripts/run_nightly.sh`
- 不要只用 `build.py test --suite ci_gate`（它丢掉 nightly，过了也会在 nightly 被打断）
- 不要用默认 `addopts`（排除 nightly/network）
- 不要杀掉已在跑的全量 nightly

## 立刻执行

```bash
REPO="$(git rev-parse --show-toplevel)"
PY="${REPO}/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "no ${PY}; run uv sync in the repo, or pass --python" >&2
  exit 1
fi
SCRIPT="${REPO}/.agents/skills/msmodeling-request-pipeline/scripts/run_change_pipeline.py"
$PY "$SCRIPT" --run --repo "$REPO" --pr "<N>"
```

没写 PR 号就去掉 `--pr`。已说「请求流水」则直接 `--run`（会先 fetch/merge 主仓）。选出 >200 个 node 时先报数量再跑。

## 两波流水

1. **ci**：先从选出的 node 里按 `not npu and not nightly and not network` 拆出子集再跑（和门禁/test_map 同一波）。0 条，或 pytest 退出码 5，记为该波通过。
2. **nightly_related**：从同一批里按 `not npu and (nightly or benchmark or network)` 拆出子集再跑（nightly Wave A 里的 `@nightly` + Wave B）。0 条或退出码 5 同样记为通过。

选测：改过的测试文件全量 collect；产品文件走 `test_map`；映射文件再 collect 以补 `@nightly` 兄弟；这些文件所在目录再 collect `nightly or benchmark or network`。

`test_map`：`MSMODELING_TEST_MAP_PATH` → `.msmodeling_cache/test_map/<base>/test_map.json`。产品改动没有 map 必须写明，不能声称 nightly 不会被打断。

## GitCode 留言

**跑完必须在该 PR 评论区回结果**（通过、无映射用例、失败都要发），不要只在聊天里说。

默认 `--comment`：`gitcode pr comment <PR> -R Ascend/msmodeling --body ...`

正文含：本地流水结果、HEAD、base、改动文件数、选测条数、两波 exit。
PR 号：`--pr`、`GITCODE_PR_NUMBER`，或当前分支。没有 `--no-comment-on-pass`；通过和失败都由 `--comment` / `--no-comment` 一起开关。`--no-comment` 仅在用户明确说不要留言时用。

未传 `--pr` 时，用当前分支名在主仓查 open PR。命中项的 head 分支必须相同，且 head 仓库的 owner 必须等于本机 `fork` 远端（没有则用非 Ascend 的 `origin`）的 owner。对不上、无法确定 fork owner、或多条无法区分时，在 stderr 写 `comment=skip`，不取第一条，也不写死某个账号。评论目标固定是 `Ascend/msmodeling`。

## 跑完怎么回

- 主仓 sync：fetch 的 base、ahead/behind、是否 merge
- base、两波 node 数、各自 exit
- 评论区链接；未要求不 commit / push
