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

"""Bind an AISBench Python model config to the generated PD Proxy endpoint."""

import ast
from urllib.parse import urlsplit, urlunsplit


def bind_endpoint(source, *, host, port, model):
    """Edit literal endpoint values without executing the benchmark config."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes = []
    found = set()
    values = {"host_ip": host, "host_port": port, "model": model}

    def replace(name, node):
        if name not in {*values, "url"}:
            return
        if not isinstance(node, ast.Constant):
            raise TypeError(f"AISBench {name} must be a literal for automatic endpoint binding")
        if name == "url":
            if not node.value:
                return
            url = urlsplit(node.value)
            address = f"[{host}]" if ":" in host and not host.startswith("[") else host
            value = urlunsplit(("http", f"{address}:{port}", url.path, url.query, url.fragment))
        else:
            value = values[name]
            if name == "host_port" and isinstance(node.value, str):
                value = str(value)
        found.add(name)
        # AST columns are UTF-8 byte offsets, while edits use Python string offsets.
        start = offsets[node.lineno - 1] + len(lines[node.lineno - 1].encode()[: node.col_offset].decode())
        end = offsets[node.end_lineno - 1] + len(lines[node.end_lineno - 1].encode()[: node.end_col_offset].decode())
        changes.append((start, end, repr(value)))

    # Restrict edits to the models declaration; leave unrelated dataset/tokenizer config intact.
    declarations = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "models" for target in node.targets)
    ]
    if len(declarations) != 1:
        raise ValueError("AISBench config must declare one explicit models list")
    declaration = declarations[0]
    if not isinstance(declaration, (ast.List, ast.Tuple)) or len(declaration.elts) != 1:
        raise ValueError("PD full-service benchmark requires exactly one AISBench model")
    entry = declaration.elts[0]
    if isinstance(entry, ast.Call) and isinstance(entry.func, ast.Name) and entry.func.id == "dict":
        for keyword in entry.keywords:
            replace(keyword.arg, keyword.value)
    elif isinstance(entry, ast.Dict):
        for key, value in zip(entry.keys, entry.values):
            if isinstance(key, ast.Constant):
                replace(key.value, value)
    else:
        raise ValueError("AISBench model must be an explicit dict")
    if "model" not in found or not ({"host_ip", "host_port"} <= found or "url" in found):
        raise ValueError("AISBench model must declare model and host_ip/host_port or url")
    for start, end, value in sorted(changes, reverse=True):
        source = source[:start] + value + source[end:]
    return source
