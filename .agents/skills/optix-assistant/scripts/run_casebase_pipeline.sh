#!/bin/bash
# 案例底座自动化：一条命令复跑 ①抽取 -> ④审计 -> ③写入 -> ④复检
#
# 归属: agent（Round 0 案例底座装配，见 workflows/environment-preparation.md 步骤 5）
# 契约: references/case-base-automation.md
#
# 用法（本脚本可整体搬移：所有工具路径都相对脚本自身解析，不写死仓库位置）：
#   SRC=<基准/冻结 config.toml 绝对路径> \
#   NEW=<新底座 config.toml 绝对路径> \
#   bash <SKILL_ROOT>/scripts/run_casebase_pipeline.sh              # 干跑：只出清单
#   SRC=... NEW=... APPLY=1 bash <SKILL_ROOT>/scripts/run_casebase_pipeline.sh   # 写入
#
# 可选环境变量：
#   CASE=<案例 md 绝对路径>   显式指定案例；缺省复用 <AUDIT_DIR>/out/case_base_raw.json 里记录的案例
#   AUDIT_DIR=<产物目录>      缺省 <repo>/case_base_out
#   ROOT=<optix 仓根>         缺省按脚本位置上推 4 层
#   PY=<选定环境的 python>    缺省 python3
#   CT=<容器名>               可选；指定后在该容器中执行
#   FORCE_COPY=1              显式允许覆盖已存在的目标配置
#   ENGINE=<vllm|mindie>      缺省 vllm
#   APPLY=1                   真正写入（先 .bak，逐条失败即停，写后自检 + ④ 复检）
#   ACK_REVIEWS=1             第 5 步门禁的显式放行（未闭环项照旧带进本轮 run，慎用）
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
A=${AUDIT_DIR:-$R/case_base_out}
PY=${PY:-python3}
CT=${CT:-}
ENGINE=${ENGINE:-vllm}
CW=$SCRIPT_DIR/config_writer.py
: "${SRC:?需要 SRC=<基准/冻结 config.toml 绝对路径>}"
: "${NEW:?需要 NEW=<新底座 config.toml 绝对路径>}"
APPLY=${APPLY:-0}

say() { printf '\n===== %s =====\n' "$1"; }
run() {
  if [ -n "$CT" ]; then
    docker exec -w "$R" "$CT" "$PY" "$@"
  else
    (cd "$R" && "$PY" "$@")
  fi
}

say "0. 前置检查"
[ -f "$SRC" ] || { echo "ERROR: SRC 不存在: $SRC"; exit 2; }
if [ -n "$CT" ]; then
  command -v docker >/dev/null && docker exec -w "$R" "$CT" "$PY" -c "import sys" || { echo "ERROR: 容器或 Python 不可用: $CT $PY"; exit 2; }
else
  command -v "$PY" >/dev/null || { echo "ERROR: Python 不可用: $PY"; exit 2; }
fi
for f in case_base_extract.py audit_base_coverage.py apply_case_base.py check_base_review.py case_base_rules.json config_writer.py; do
  [ -f "$SCRIPT_DIR/$f" ] || { echo "ERROR: 缺少 $SCRIPT_DIR/$f"; exit 2; }
done
mkdir -p "$A/out" || exit 2
if [ -n "$CT" ]; then
  docker exec "$CT" test -f "$SRC" && docker exec "$CT" test -f "$SCRIPT_DIR/apply_case_base.py" && docker exec "$CT" test -d "$A/out" || { echo "ERROR: 容器中无法访问源配置、脚本或产物目录"; exit 2; }
fi
echo "SCRIPT_DIR = $SCRIPT_DIR"
echo "ROOT       = $R"
echo "SRC        = $SRC"
echo "NEW        = $NEW"
echo "AUDIT_DIR  = $A"
echo "模式       = $([ "$APPLY" = 1 ] && echo APPLY || echo DRY-RUN)"

say "1. ① 抽取案例底座（只读案例库）"
# 案例选择：CASE=<md 路径> 显式指定；否则复用上次 out/case_base_raw.json 里记录的案例（保证可复跑）
if [ -z "${CASE:-}" ] && [ -f "$A/out/case_base_raw.json" ]; then
  CASE=$(run -c "import json;print(json.load(open('$A/out/case_base_raw.json'))['case_file'])")
  echo "[info] 复用上次案例: $CASE"
fi
[ -n "${CASE:-}" ] || { echo "ERROR: 需要 CASE=<案例 md 绝对路径>（当前无历史产物可复用）"; exit 2; }
[ -f "$CASE" ] || { echo "ERROR: 案例不存在: $CASE"; exit 2; }
if [ -n "$CT" ]; then
  docker exec "$CT" test -f "$CASE" || { echo "ERROR: 容器中无法访问案例: $CASE"; exit 2; }
fi
run "$SCRIPT_DIR/case_base_extract.py" --case "$CASE" --out "$A/out/case_base_raw.json" | tail -3 || exit 1

say "2. ④ 审计基准配置（写前）"
run "$SCRIPT_DIR/audit_base_coverage.py" --config "$SRC" --case-json "$A/out/case_base_raw.json" \
  --optix-root "$R" --out "$A/out/base_coverage.json" | tail -3
rc=$?
echo "[基线] 退出码 $rc（1 = 存在未解释缺口，正常）"

say "3. ③ 生成/执行写入（cwd=仓根）"
args=( "$SCRIPT_DIR/apply_case_base.py" --coverage "$A/out/base_coverage.json" --rules "$SCRIPT_DIR/case_base_rules.json" \
       --copy-from "$SRC" --config "$NEW" \
       --optix-root "$R" --engine "$ENGINE" --python "$PY" --config-writer "$CW" \
       --out "$A/out/base_apply_plan.json" )
[ "${FORCE_COPY:-0}" = 1 ] && args+=( --force-copy )
[ "$APPLY" = 1 ] && args+=( --apply )
run "${args[@]}" || { echo "ERROR: ③ 失败（退出码 $?）"; exit 1; }

if [ "$APPLY" != 1 ]; then
  cat <<EOF

（干跑结束。核对上面清单无误后加 APPLY=1 重跑以写入）
EOF
  exit 0
fi

say "4. ④ 复检新底座"
run "$SCRIPT_DIR/audit_base_coverage.py" --config "$NEW" --case-json "$A/out/case_base_raw.json" \
  --optix-root "$R" --out "$A/out/base_coverage.after.json" >/dev/null 2>&1
run -c "
import json
for tag,p in (('before','$A/out/base_coverage.json'),('after','$A/out/base_coverage.after.json')):
    d=json.load(open(p)); c=d.get('counts') or {}
    print('%-7s adopted=%-3s missing=%-3s range_gap=%-2s' % (tag, c.get('adopted'), c.get('missing'), c.get('range_gap')))
    for r in (d.get('missing') or []):
        print('          - [%s] %s (%s)' % (r.get('kind'), r.get('key'), r.get('category')))
"

say "5. ⑥ 底座裁决门禁（needs_review 必须闭环，否则阻断本轮 run）"
gate_rc=0
run "$SCRIPT_DIR/check_base_review.py" --audit "$A/out/base_coverage.after.json" --out-dir "$A/out" || gate_rc=$?
if [ "$gate_rc" = 2 ]; then
  echo "ERROR: 门禁脚本用法/文件问题（退出码 2）"
  exit 2
fi
if [ "$gate_rc" = 3 ]; then
  if [ "${ACK_REVIEWS:-0}" = "1" ]; then
    echo
    echo "WARNING: ACK_REVIEWS=1 —— 已显式放行未闭环项，缺口会被带进本轮 run。"
  else
    cat <<EOF

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
阻断：R0 案例底座仍有未闭环的 missing/needs_review 项，禁止开跑。
  未闭环清单: $A/out/BASE_REVIEW_REQUIRED.md
  裁决模板  : $A/out/base_review_decisions.json（填完重跑本流水线）
  两条合法路径:
    (1) 改造成可自动装配：子键登记进 optix/serve_plan.py 的 JSON_SUBKEY_MAP，
        或 flag 加进 case_base_rules.json 的 role_split_flags，重跑本流水线；
    (2) 逐条裁决：verdict + owner + reason + decided_at 写入裁决文件，重跑本流水线。
  确有理由跳过（例如非案例装配的 run）：ACK_REVIEWS=1 重跑。
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
EOF
    exit 3
  fi
fi
echo "[门禁] 通过（未闭环项 0，或已显式 ACK）"

echo
echo "新底座: $NEW"
echo "计划产物: $A/out/base_apply_plan.json（含 needs_review 的建议命令，未执行）"
echo "下一步:"
echo "  1) 把 R0 证据复制进 run 目录（run_pd_round 的底座门禁要读）:"
echo "     cp $A/out/base_coverage.after.json $A/out/base_review_decisions.json <run-dir>/"
echo "  2) 对 manual 清单逐条裁决 → 复检 config_preflight --check-only"
