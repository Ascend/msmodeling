# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import re
import operator
from typing import Any


# Comparison operators supported in <% if %> expressions
_CMP_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}
_CMP_RE = re.compile(r'^(.+?)\s*(>=|<=|!=|==|>|<)\s*(.+)$')


def resolve_expr_value(expr: str, context: dict) -> Any:
    parts = expr.strip().split(".")

    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in context:
            obj = context[prefix]
            for part in parts[i:]:
                if isinstance(obj, dict):
                    obj = obj.get(part)
                else:
                    obj = getattr(obj, part, None) if hasattr(obj, part) else None
                if obj is None:
                    return None
            return obj

    obj = context.get(parts[0])
    if obj is None:
        return None
    for part in parts[1:]:
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None) if hasattr(obj, part) else None
        if obj is None:
            return None
    return obj


def _eval_condition(expr: str, context: dict) -> Any:
    """Evaluate an <% if %> expression, supporting comparison operators.

    If the expression contains a comparison operator (>, >=, <, <=, ==, !=),
    both sides are resolved and compared. Otherwise falls back to
    resolve_expr_value for simple truthiness checks.
    """
    m = _CMP_RE.match(expr.strip())
    if m:
        left_expr, op, right_expr = m.group(1).strip(), m.group(2), m.group(3).strip()
        left_val = resolve_expr_value(left_expr, context)
        # Right side may be a literal (number/string) or a variable reference
        try:
            right_val = int(right_expr)
        except ValueError:
            try:
                right_val = float(right_expr)
            except ValueError:
                right_val = resolve_expr_value(right_expr, context)
        if left_val is None or right_val is None:
            return False
        return _CMP_OPS[op](left_val, right_val)
    return resolve_expr_value(expr, context)


def format_expr_value(value, default: str = "") -> str:
    if value is None or value is False:
        return default
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value) if value else default
    s = str(value)
    return s if s else default


def _scan_blocks(text: str):
    """Scan text and return matched if/endif block boundaries with correct nesting.

    Returns list of (if_tag_start, if_tag_end, endif_tag_start, endif_tag_end, middle_items)
    for each matched pair.
    """
    full_tag = re.compile(r'<%\s*(if|elif|else|endif)\b[^%]*%>')

    stack = []
    pairs = []

    for m in full_tag.finditer(text):
        tag, tag_start, tag_end = m.group(1), m.start(), m.end()
        if tag == "if":
            stack.append(("if", tag_start, tag_end))
        elif tag in ("elif", "else"):
            if stack:
                stack.append((tag, tag_start, tag_end))
        elif tag == "endif":
            if not stack:
                break
            items = []
            while stack:
                item = stack.pop()
                if item[0] == "if":
                    if_start, if_end = item[1], item[2]
                    pairs.append((if_start, if_end, tag_start, tag_end, items))
                    break
                items.insert(0, item)

    return pairs


def _find_branch_at_depth(
    body: str, full_tag, start: int, depth: int = 0, stop_tags: tuple = ("else", "elif", "endif")
) -> tuple:
    """从 body[start:] 扫描，返回首个在 depth 0 的 stop_tag 位置。

    返回 (tag_type, tag_start, tag_end) 或 (None, -1, -1)。
    用于定位 if 体里的 else/elif/endif（depth 0 = 当前 if 块边界）。
    """
    pos = start
    while pos < len(body):
        m = full_tag.search(body[pos:])
        if not m:
            return (None, -1, -1)
        t = m.group(1)
        abs_start = pos + m.start()
        abs_end = pos + m.end()
        if t == "if":
            depth += 1
        elif t == "endif":
            if depth == 0:
                return (t, abs_start, abs_end)
            depth -= 1
        elif t in ("else", "elif") and depth == 0:
            return (t, abs_start, abs_end)
        pos = abs_end
    return (None, -1, -1)


def _eval_branch(expr: str, context: dict) -> bool:
    """求值 if/elif 表达式，返回 truthy。"""
    value = _eval_condition(expr, context)
    return bool(value) and value not in ("0", 0, "false", "False", "")


def render_template(template: str, context: dict) -> str:
    result = template

    full_tag = re.compile(r'<%\s*(if|elif|else|endif)\b[^%]*%>')

    while True:
        pairs = _scan_blocks(result)
        if not pairs:
            break
        if_start, if_end, endif_start, endif_end, _ = max(pairs, key=lambda x: x[0])
        if_text = result[if_start:if_end]
        expr_match = re.match(r'<%\s*if\s+(.+?)\s*%>', if_text)
        if not expr_match:
            break
        body = result[if_end:endif_start]

        # 遍历 if/elif/else 分支，选首个 truthy
        content = ""
        branch_expr = expr_match.group(1).strip()
        scan_pos = 0
        chosen = False
        while True:
            tag, tag_start, tag_end = _find_branch_at_depth(body, full_tag, scan_pos, 0)
            if tag is None:
                # 无更多分支，取当前到 body 末尾
                if not chosen and branch_expr and _eval_branch(branch_expr, context):
                    content = body[scan_pos:]
                break
            branch_body = body[scan_pos:tag_start]
            if not chosen and branch_expr and _eval_branch(branch_expr, context):
                content = branch_body
                chosen = True
            # 推进到下一分支起点
            scan_pos = tag_end
            if tag == "elif":
                elif_match = re.match(r'<%\s*elif\s+(.+?)\s*%>', body[tag_start:tag_end])
                branch_expr = elif_match.group(1).strip() if elif_match else ""
            elif tag == "else":
                # else 后到 endif 前全是 else 体
                if not chosen:
                    content = body[tag_end:]
                break
            elif tag == "endif":
                break
        result = result[:if_start] + content + result[endif_end:]

    def replace_expr(m):
        raw = m.group(1).strip()
        default = ""
        if ":-" in raw:
            raw, default = raw.split(":-", 1)
            default = default.strip()
        value = resolve_expr_value(raw, context)
        return format_expr_value(value, default)

    result = re.sub(r'<%=\s*(.+?)\s*%>', replace_expr, result)
    return result


def shell_quote(value: str) -> str:
    """Quote a value for safe use in shell `export VAR='...'` or CLI `--arg '...'`.

    Uses single quotes to preserve literal content. Embedded single quotes
    are escaped via the standard '"'\"'"' idiom. If the value is already
    wrapped in single quotes (a common user mistake in config.toml enum
    values), strip them first to avoid double-quoting.
    """
    s = str(value)
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'" and s.count("'") == 2:
        s = s[1:-1]
    return "'" + s.replace("'", "'\"'\"'") + "'"
