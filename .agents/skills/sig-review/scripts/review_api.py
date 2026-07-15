#!/usr/bin/env python3
"""GitCode PR 检视 API 工具

自包含的 GitCode API 封装，零外部依赖（仅用 Python 标准库）。
供各类 AI agent（cursor / claude code / opencode / codex 等）通过 bash 调用。

环境变量:
  GITCODE_TOKEN   GitCode API 令牌（可选，推荐用 auth 命令配置）
  GITCODE_OWNER   仓库 owner（默认 Ascend）
  GITCODE_REPO    仓库名称（默认 msmodeling）

配置文件:
  ~/.config/sig-review/config.json（由 auth 命令创建，chmod 600）

令牌配置:
  python3 review_api.py auth --token <你的令牌>   保存令牌到配置文件（一次配置，持久生效）
  python3 review_api.py auth --stdin               从 stdin 读取令牌（不在 shell 历史留痕）
  python3 review_api.py auth                       交互式输入（隐藏输入）

命令:
  auth [--token|--stdin]                            配置 GitCode 令牌（首次使用前执行一次）
  fetch <PR>                                        获取 PR 完整信息（详情 + 文件 + diff + 已有评论）
  assign <PR> [--dry-run]                           分析 PR 变更文件，路由到 SIG，指派 chair，打标签
  list [--user]                                     列出分配给自己（或指定用户）的待检视 PR
  status <PR>                                       查看 PR 状态（assignee / 标签 / 审查状态，轻量）
  comment <PR> --file --line --category --content   提交检视评论（diff_comment 类型）
  verdict <PR> --event --body                       提交检视结论（approved / comment）
  handoff <PR> --to <用户名>                        责任传递：移除自己的 assignee，指派新人
  complete <PR> --to <approver> --event --body      检视完成：提交检视结论 + 指派 approver
  comments <PR>                                     列出 PR 已有评论（用于防重复）
  withdraw <comment_id>                             删除评论

输出: 所有命令输出 JSON 到 stdout，错误输出 JSON 到 stderr。
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.gitcode.com/api/v5"
DEFAULT_OWNER = "Ascend"
DEFAULT_REPO = "msmodeling"
TIMEOUT = 30
PER_PAGE = 100

CATEGORIES = ["逻辑缺陷", "性能优化", "安全风险", "架构设计", "代码规范"]


# ============ 配置 ============

CONFIG_DIR = os.path.expanduser("~/.config/sig-review")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")


def load_config_file():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_config_file(data):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.chmod(CONFIG_PATH, 0o600)


def get_config():
    file_cfg = load_config_file()
    token = file_cfg.get("gitcode_token", "").strip()
    if not token:
        token = os.environ.get("GITCODE_TOKEN", "").strip()
    if not token:
        die(
            "GitCode 令牌未配置。请运行以下命令一次（配置后持久生效）：\n"
            "  python3 review_api.py auth --token <你的令牌>\n"
            "或从 stdin 读取（不在 shell 历史留痕）：\n"
            "  echo '<你的令牌>' | python3 review_api.py auth --stdin"
        )
    return {
        "token": token,
        "owner": file_cfg.get("gitcode_owner") or os.environ.get("GITCODE_OWNER", DEFAULT_OWNER),
        "repo": file_cfg.get("gitcode_repo") or os.environ.get("GITCODE_REPO", DEFAULT_REPO),
    }


# ============ 基础设施 ============


def die(message):
    print(json.dumps({"error": message}, ensure_ascii=False), file=sys.stderr)
    sys.exit(1)


def api_request(method, path, *, query=None, json_body=None, config=None):
    """发起 GitCode API 请求。

    path 格式: /pulls/123 或 /pulls/123/files 或 /pulls/comments/456
    """
    config = config or get_config()
    base = f"{API_BASE}/repos/{config['owner']}/{config['repo']}"
    url = f"{base}{path}"
    if query:
        params = {k: v for k, v in query.items() if v is not None}
        if params:
            url += "?" + urllib.parse.urlencode(params)

    headers = {"PRIVATE-TOKEN": config["token"], "Accept": "application/json"}
    data = None
    if json_body is not None:
        data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        die(f"GitCode API {e.code} {e.reason}: {detail}")
    except urllib.error.URLError as e:
        die(f"网络错误: {e.reason}")


def api_request_raw(method, full_path, *, config=None):
    """发起不需要 owner/repo 前缀的请求（如 DELETE comments）。"""
    config = config or get_config()
    url = f"{API_BASE}/repos/{config['owner']}/{config['repo']}{full_path}"
    headers = {"PRIVATE-TOKEN": config["token"], "Accept": "application/json"}
    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        die(f"网络错误: {e.reason}")


def paginate(method, path, *, query=None, config=None):
    """自动分页获取所有结果。"""
    config = config or get_config()
    results = []
    page = 1
    base_query = query or {}
    while True:
        batch = api_request(
            method,
            path,
            query={**base_query, "page": page, "per_page": PER_PAGE},
            config=config,
        )
        if not batch:
            break
        results.extend(batch)
        if len(batch) < PER_PAGE:
            break
        page += 1
    return results


def extract_patch(file_obj):
    """GitCode API 的 patch 字段有时是字符串，有时是 dict。"""
    patch = file_obj.get("patch")
    if isinstance(patch, dict):
        return patch.get("diff", "")
    return patch or ""


def output_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def get_current_user(config=None):
    """通过 token 获取当前 GitCode 用户名。"""
    config = config or get_config()
    url = f"{API_BASE}/user"
    headers = {"PRIVATE-TOKEN": config["token"], "Accept": "application/json"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("login", "")
    except urllib.error.HTTPError as e:
        die(f"获取用户信息失败: GitCode API {e.code}")
    except urllib.error.URLError as e:
        die(f"网络错误: {e.reason}")


def read_body(args):
    """读取检视正文，支持 --body / --body-stdin / --body-file。"""
    if getattr(args, "body", None) is not None:
        return args.body
    if getattr(args, "body_stdin", False):
        return sys.stdin.read()
    if getattr(args, "body_file", None):
        with open(args.body_file, encoding="utf-8") as f:
            return f.read()
    die("必须提供 --body、--body-file 或 --body-stdin 之一")


def handback_to_author(pr_number, detail, config):
    """移除当前 assignee，指派 PR 作者（diff_comment 后自动移交）。

    成功返回 author 用户名，失败返回 None（调用方应据此输出警告）。
    """
    author = (detail.get("user") or {}).get("login", "")
    if not author:
        return None

    current_assignees = [a.get("login", "") for a in detail.get("assignees", [])]

    if current_assignees:
        try:
            api_request(
                "DELETE",
                f"/pulls/{pr_number}/assignees",
                json_body={"assignees": ",".join(current_assignees)},
                config=config,
            )
        except SystemExit:
            return None

    try:
        api_request(
            "POST",
            f"/pulls/{pr_number}/assignees",
            json_body={"assignees": author},
            config=config,
        )
    except SystemExit:
        return None

    return author


# ============ auth 命令 ============


def cmd_auth(args):
    """配置 GitCode 令牌，保存到 ~/.config/sig-review/config.json。"""
    if args.token:
        token = args.token.strip()
        for i, arg in enumerate(sys.argv):
            if arg == "--token" and i + 1 < len(sys.argv):
                sys.argv[i + 1] = "***"
                break
    elif args.stdin:
        token = sys.stdin.read().strip()
    else:
        import getpass

        try:
            token = getpass.getpass("请输入 GitCode 令牌（输入不可见）: ").strip()
        except Exception:
            die("无法交互式读取输入，请使用 --token 或 --stdin 参数")

    if not token:
        die("令牌不能为空")

    config = load_config_file()
    config["gitcode_token"] = token
    if args.owner:
        config["gitcode_owner"] = args.owner
    if args.repo:
        config["gitcode_repo"] = args.repo

    save_config_file(config)

    output_json(
        {
            "success": True,
            "config_path": CONFIG_PATH,
            "message": "令牌已保存，后续命令将自动读取（无需设置环境变量）",
        }
    )


# ============ 命令实现 ============


def cmd_fetch(args):
    """获取 PR 完整信息：详情 + 文件 + diff + 已有评论。"""
    config = get_config()
    pr_number = args.pr_number

    detail = api_request("GET", f"/pulls/{pr_number}", config=config)
    if not detail:
        die(f"PR #{pr_number} 不存在或无法访问")

    files = paginate("GET", f"/pulls/{pr_number}/files", config=config)
    raw_comments = paginate("GET", f"/pulls/{pr_number}/comments", config=config)

    diff_lines = sum(f.get("additions", 0) + f.get("deletions", 0) for f in files)

    result = {
        "pr_number": pr_number,
        "title": detail.get("title", ""),
        "body": detail.get("body", ""),
        "author": (detail.get("user") or {}).get("login", ""),
        "head_sha": (detail.get("head") or {}).get("sha", ""),
        "state": detail.get("state", ""),
        "labels": [label.get("name", "") for label in detail.get("labels", [])],
        "assignees": [a.get("login", "") for a in detail.get("assignees", [])],
        "approval_reviewers": [r.get("login", "") for r in detail.get("approval_reviewers", [])],
        "html_url": detail.get("html_url", ""),
        "diff_lines": diff_lines,
        "files": [
            {
                "filename": f.get("filename", ""),
                "status": f.get("status", ""),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "patch": extract_patch(f),
            }
            for f in files
        ],
        "existing_comments": [
            {
                "id": c.get("id", ""),
                "note_id": c.get("note_id", ""),
                "author": (c.get("user") or {}).get("login", ""),
                "body": c.get("body", ""),
                "path": c.get("path", ""),
                "position": c.get("position"),
                "comment_type": c.get("comment_type", ""),
                "created_at": c.get("created_at", ""),
            }
            for c in raw_comments
        ],
    }
    output_json(result)


def read_content(args):
    if args.content is not None:
        return args.content
    if args.content_stdin:
        return sys.stdin.read()
    if args.content_file:
        with open(args.content_file, encoding="utf-8") as f:
            return f.read()
    die("必须提供 --content、--content-file 或 --content-stdin 之一")


def cmd_comment(args):
    """提交一条检视评论，并自动移交回 PR 作者。"""
    config = get_config()
    pr_number = args.pr_number
    content = read_content(args)
    category = args.category

    formatted = f"【review】【{category}】{content}"

    detail = api_request("GET", f"/pulls/{pr_number}", config=config)
    head_sha = (detail.get("head") or {}).get("sha", "")
    if not head_sha:
        die(f"无法获取 PR #{pr_number} 的 head_sha")

    result = api_request(
        "POST",
        f"/pulls/{pr_number}/comments",
        json_body={
            "body": formatted,
            "commit_id": head_sha,
            "path": args.file,
            "position": args.line,
        },
        config=config,
    )

    note_id = result.get("note_id", "")
    if not note_id:
        die(f"评论已提交但未返回 note_id，返回数据: {json.dumps(result, ensure_ascii=False)}")

    author = handback_to_author(pr_number, detail, config)

    output_json(
        {
            "success": True,
            "comment_id": str(note_id),
            "comment_hash": result.get("id", ""),
            "file": args.file,
            "line": args.line,
            "category": category,
            "url": result.get("html_url", ""),
            "handed_back_to": author,
            "message": f"检视意见已提交，PR 已转回给作者 {author} 修改"
            if author
            else "检视意见已提交，但移交回作者失败，请手动更新 assignee",
        }
    )


def cmd_verdict(args):
    """提交检视结论。approved 时仅提交，comment 时移交回作者。"""
    config = get_config()
    pr_number = args.pr_number

    api_request(
        "POST",
        f"/pulls/{pr_number}/review",
        json_body={"event": args.event, "body": args.body},
        config=config,
    )

    handed_back_to = None
    if args.event == "comment":
        detail = api_request("GET", f"/pulls/{pr_number}", config=config)
        handed_back_to = handback_to_author(pr_number, detail, config)

    if args.event == "comment" and handed_back_to:
        message = f"检视结论已提交（有修改意见），PR 已转回给作者 {handed_back_to} 修改"
    elif args.event == "comment" and not handed_back_to:
        message = "检视结论已提交，但移交回作者失败，请手动更新 assignee"
    else:
        message = f"检视结论已提交（{args.event}）"

    output_json(
        {
            "success": True,
            "pr_number": pr_number,
            "event": args.event,
            "body": args.body,
            "handed_back_to": handed_back_to,
            "message": message,
        }
    )


def cmd_comments(args):
    """列出 PR 已有评论（用于防重复检查）。"""
    config = get_config()
    pr_number = args.pr_number

    raw_comments = paginate("GET", f"/pulls/{pr_number}/comments", config=config)

    result = [
        {
            "id": c.get("id", ""),
            "note_id": c.get("note_id", ""),
            "author": (c.get("user") or {}).get("login", ""),
            "body": c.get("body", ""),
            "path": c.get("path", ""),
            "position": c.get("position"),
            "comment_type": c.get("comment_type", ""),
            "created_at": c.get("created_at", ""),
        }
        for c in raw_comments
    ]
    output_json(result)


def cmd_withdraw(args):
    """删除一条评论（需要 note_id 数字 ID）。"""
    config = get_config()
    comment_id = str(args.comment_id).strip()

    if not comment_id.isdigit():
        die(
            f"comment_id 必须是数字 ID（note_id），收到 '{comment_id}'。"
            f"GitCode API 返回的 id 字段是 SHA1 哈希，note_id 字段才是数字 ID，"
            f"删除评论时必须使用 note_id。"
        )

    status, body = api_request_raw("DELETE", f"/pulls/comments/{comment_id}", config=config)

    if status in (200, 204):
        output_json({"success": True, "deleted": comment_id})
    else:
        die(f"删除评论失败: HTTP {status}: {body}")


# ============ SIG 路由与分配 ============

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OWNERSHIP_PATH = os.path.join(SCRIPT_DIR, "..", "sig_ownership.json")


def load_ownership(path=None):
    filepath = path or DEFAULT_OWNERSHIP_PATH
    if not os.path.exists(filepath):
        die(f"SIG 归属配置文件不存在: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def path_matches(file_path, sig_path):
    """检查 file_path 是否匹配 sig_path（文件精确匹配或目录前缀匹配）。"""
    if sig_path.endswith("/"):
        return file_path.startswith(sig_path)
    return file_path == sig_path or file_path.startswith(sig_path + "/")


def route_to_sig(file_paths, sigs, fallback_sigs=None):
    """将文件路径列表路由到 SIG（最长前缀匹配 + fallback 兜底）。

    Returns:
        dict: {sig_name -> {"sig": sig_dict, "matched_paths": [paths], "match_type": "explicit"|"fallback"}}
        特殊 key "_unmatched" -> {"sig": None, "matched_paths": [paths]}
    """
    sig_by_name = {s["name"]: s for s in sigs}
    result = {}
    unmatched = []

    for file_path in file_paths:
        best_sig = None
        best_length = 0

        for sig in sigs:
            for sig_path in sig.get("paths", []):
                if path_matches(file_path, sig_path) and len(sig_path) > best_length:
                    best_sig = sig
                    best_length = len(sig_path)

        if best_sig:
            key = best_sig["name"]
            if key not in result:
                result[key] = {
                    "sig": best_sig,
                    "matched_paths": [],
                    "match_type": "explicit",
                }
            result[key]["matched_paths"].append(file_path)
        elif fallback_sigs:
            matched_fallback = False
            for fb_path, fb_sig_name in sorted(fallback_sigs.items(), key=lambda x: len(x[0]), reverse=True):
                if file_path.startswith(fb_path):
                    sig = sig_by_name.get(fb_sig_name)
                    if sig:
                        key = sig["name"]
                        if key not in result:
                            result[key] = {
                                "sig": sig,
                                "matched_paths": [],
                                "match_type": "fallback",
                            }
                        result[key]["matched_paths"].append(file_path)
                        matched_fallback = True
                        break
            if not matched_fallback:
                unmatched.append(file_path)
        else:
            unmatched.append(file_path)

    if unmatched:
        result["_unmatched"] = {
            "sig": None,
            "matched_paths": unmatched,
            "match_type": "none",
        }

    return result


def cmd_assign(args):
    """分析 PR 变更文件，路由到 SIG，指派 chair，打标签。"""
    config = get_config()
    pr_number = args.pr_number
    ownership = load_ownership(args.ownership_file)
    sigs = ownership.get("sigs", [])

    # 1. 获取 PR 详情和文件
    detail = api_request("GET", f"/pulls/{pr_number}", config=config)
    if not detail:
        die(f"PR #{pr_number} 不存在或无法访问")

    author = (detail.get("user") or {}).get("login", "")
    files = paginate("GET", f"/pulls/{pr_number}/files", config=config)
    file_paths = [f.get("filename", "") for f in files]

    if not file_paths:
        die(f"PR #{pr_number} 没有变更文件")

    # 2. 路由到 SIG
    fallback_sigs = ownership.get("fallback_sigs", {})
    routing = route_to_sig(file_paths, sigs, fallback_sigs)
    unmatched = routing.pop("_unmatched", None)

    # 3. 确定每个 SIG 的 assignee
    routing_result = []
    assignees = []
    labels = []
    auto_classified = []

    for sig_name, match in routing.items():
        sig = match["sig"]
        chair = sig.get("chair", "")
        reviewers = sig.get("reviewers", [])
        match_type = match.get("match_type", "explicit")

        # chair == PR 作者 → 改指派 reviewer
        assignee = chair
        reason = "chair"
        if assignee == author and reviewers:
            assignee = reviewers[0]
            reason = "chair==author，改指派 reviewer"

        assignees.append(assignee)
        labels.append(f"sig:{sig_name}")

        entry = {
            "sig": sig_name,
            "chair": chair,
            "assigned": assignee,
            "assign_reason": reason,
            "matched_files": match["matched_paths"],
            "match_type": match_type,
        }
        routing_result.append(entry)

        # fallback 匹配的文件 → 建议补录到 sig_ownership.json
        if match_type == "fallback":
            for fp in match["matched_paths"]:
                parent = "/".join(fp.split("/")[:-1]) + "/"
                auto_classified.append(
                    {
                        "file": fp,
                        "inferred_sig": sig_name,
                        "suggestion": f"建议将 {parent} 目录归属到 {sig_name} SIG",
                    }
                )

    # 跨 SIG 双签
    cross_sig = len(routing) > 1
    if cross_sig:
        labels.append("cross-sig")

    # 根目录未匹配文件 → 需架构 SIG 共审
    needs_arch_review = unmatched is not None

    # 4. 幂等检查：已有任何 assignee 则不重复指派
    existing_assignees = [a.get("login", "") for a in detail.get("assignees", [])]
    new_assignees = [a for a in assignees if a not in existing_assignees]
    already_assigned = bool(existing_assignees)

    # 5. dry-run 模式：只输出分析结果，不调用 API
    if args.dry_run:
        output_json(
            {
                "dry_run": True,
                "pr_number": pr_number,
                "title": detail.get("title", ""),
                "author": author,
                "total_files": len(file_paths),
                "routing": routing_result,
                "cross_sig": cross_sig,
                "needs_arch_review": needs_arch_review,
                "unmatched_files": unmatched["matched_paths"] if unmatched else [],
                "auto_classified": auto_classified,
                "would_assign": assignees,
                "would_label": labels,
                "existing_assignees": existing_assignees,
            }
        )
        return

    # 6. 幂等：已有 assignee 则跳过
    if already_assigned:
        output_json(
            {
                "success": True,
                "pr_number": pr_number,
                "message": "PR 已有 assignee，未重复指派",
                "existing_assignees": existing_assignees,
                "routing": routing_result,
                "cross_sig": cross_sig,
                "needs_arch_review": needs_arch_review,
                "unmatched_files": unmatched["matched_paths"] if unmatched else [],
                "auto_classified": auto_classified,
            }
        )
        return

    # 7. 指派 assignees
    if new_assignees:
        api_request(
            "POST",
            f"/pulls/{pr_number}/assignees",
            json_body={"assignees": ",".join(new_assignees)},
            config=config,
        )

    # 8. 打标签
    labels_applied = True
    if labels:
        try:
            api_request(
                "POST",
                f"/pulls/{pr_number}/labels",
                json_body={"labels": labels},
                config=config,
            )
        except SystemExit:
            labels_applied = False

    # 9. 输出结果
    output_json(
        {
            "success": True,
            "pr_number": pr_number,
            "title": detail.get("title", ""),
            "author": author,
            "total_files": len(file_paths),
            "routing": routing_result,
            "cross_sig": cross_sig,
            "needs_arch_review": needs_arch_review,
            "unmatched_files": unmatched["matched_paths"] if unmatched else [],
            "auto_classified": auto_classified,
            "assignees": assignees,
            "new_assignees": new_assignees,
            "labels": labels,
            "labels_applied": labels_applied,
            "existing_assignees": existing_assignees,
            "message": f"已指派 {', '.join(new_assignees)}，GitCode 将发送站内信通知"
            + ("" if labels_applied else "（标签添加失败，请手动添加）"),
        }
    )


def cmd_list(args):
    """列出分配给自己（或指定用户）的待检视 PR。"""
    config = get_config()
    username = args.user or get_current_user(config)
    if not username:
        die("无法确定当前用户，请用 --user 指定")

    prs = paginate(
        "GET",
        "/pulls",
        query={"state": "open", "assignee": username},
        config=config,
    )

    if not prs:
        output_json({"username": username, "count": 0, "prs": []})
        return

    result = [
        {
            "pr_number": pr.get("number"),
            "title": pr.get("title", ""),
            "author": (pr.get("user") or {}).get("login", ""),
            "labels": [label.get("name", "") for label in pr.get("labels", [])],
            "assignees": [a.get("login", "") for a in pr.get("assignees", [])],
            "approval_reviewers": [r.get("login", "") for r in pr.get("approval_reviewers", [])],
            "html_url": pr.get("html_url", ""),
            "updated_at": pr.get("updated_at", ""),
        }
        for pr in prs
    ]
    output_json({"username": username, "count": len(result), "prs": result})


def cmd_status(args):
    """查看 PR 状态（轻量，不含 diff 和评论）。"""
    config = get_config()
    pr_number = args.pr_number

    detail = api_request("GET", f"/pulls/{pr_number}", config=config)
    if not detail:
        die(f"PR #{pr_number} 不存在或无法访问")

    output_json(
        {
            "pr_number": pr_number,
            "title": detail.get("title", ""),
            "author": (detail.get("user") or {}).get("login", ""),
            "state": detail.get("state", ""),
            "labels": [label.get("name", "") for label in detail.get("labels", [])],
            "assignees": [a.get("login", "") for a in detail.get("assignees", [])],
            "approval_reviewers": [r.get("login", "") for r in detail.get("approval_reviewers", [])],
            "html_url": detail.get("html_url", ""),
            "created_at": detail.get("created_at", ""),
            "updated_at": detail.get("updated_at", ""),
        }
    )


def cmd_handoff(args):
    """责任传递：移除自己的 assignee，指派新人。"""
    config = get_config()
    pr_number = args.pr_number
    to_user = args.to

    me = args.user or get_current_user(config)
    if not me:
        die("无法确定当前用户，请用 --user 指定你的 GitCode 用户名")

    # 1. 移除自己
    api_request(
        "DELETE",
        f"/pulls/{pr_number}/assignees",
        json_body={"assignees": me},
        config=config,
    )

    # 2. 指派新人
    api_request(
        "POST",
        f"/pulls/{pr_number}/assignees",
        json_body={"assignees": to_user},
        config=config,
    )

    output_json(
        {
            "success": True,
            "pr_number": pr_number,
            "removed": me,
            "assigned": to_user,
            "message": f"已将 PR #{pr_number} 从 {me} 移交给 {to_user}，GitCode 将发送站内信通知",
        }
    )


def cmd_complete(args):
    """检视完成：提交检视结论。approved 时指派 approver，comment 时移交回作者。"""
    config = get_config()
    pr_number = args.pr_number
    body = read_body(args)
    approver = args.to
    event = args.event

    # 0. 参数校验：approved 时必须有 --to
    if event == "approved" and not approver:
        die("--event approved 时必须通过 --to 指定 approver 用户名")

    # 1. 提交检视结论
    api_request(
        "POST",
        f"/pulls/{pr_number}/review",
        json_body={"event": event, "body": body},
        config=config,
    )

    if event == "approved":
        # 2a. 通过 → 指派 approver
        api_request(
            "POST",
            f"/pulls/{pr_number}/approval-reviewers",
            json_body={"reviewers": approver},
            config=config,
        )
        output_json(
            {
                "success": True,
                "pr_number": pr_number,
                "event": event,
                "approver": approver,
                "body": body,
                "message": f"检视结论已提交（通过），已指派 approver {approver}，GitCode 将发送站内信通知",
            }
        )
    else:
        # 2b. 有修改意见 → 移交回作者
        detail = api_request("GET", f"/pulls/{pr_number}", config=config)
        author = handback_to_author(pr_number, detail, config)
        output_json(
            {
                "success": True,
                "pr_number": pr_number,
                "event": event,
                "body": body,
                "handed_back_to": author,
                "message": f"检视结论已提交（有修改意见），PR 已转回给作者 {author} 修改"
                if author
                else "检视结论已提交，但移交回作者失败，请手动更新 assignee",
            }
        )


# ============ CLI ============


def main():
    parser = argparse.ArgumentParser(
        description="GitCode PR 检视 API 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "首次使用前配置令牌:\n"
            "  python3 review_api.py auth --token <你的令牌>\n"
            "\n"
            "示例:\n"
            "  python3 review_api.py fetch 123\n"
            "  python3 review_api.py assign 123 --dry-run\n"
            "  python3 review_api.py list\n"
            "  python3 review_api.py status 123\n"
            "  python3 review_api.py comment 123 --file src/main.py --line 42 --category 逻辑缺陷 --content '缺少空指针检查'\n"
            "  python3 review_api.py handoff 123 --to stormchasingg\n"
            "  python3 review_api.py complete 123 --to lutean --event approved --body '检视完成'\n"
            "  python3 review_api.py comments 123\n"
            "  python3 review_api.py withdraw 175776550\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # auth
    p = subparsers.add_parser("auth", help="配置 GitCode 令牌（首次使用前执行一次）")
    token_group = p.add_mutually_exclusive_group()
    token_group.add_argument("--token", help="令牌字符串（会在 shell 历史留痕）")
    token_group.add_argument("--stdin", action="store_true", help="从 stdin 读取令牌")
    p.add_argument("--owner", help="仓库 owner（默认 Ascend）")
    p.add_argument("--repo", help="仓库名称（默认 msmodeling）")
    p.set_defaults(func=cmd_auth)

    # fetch
    p = subparsers.add_parser("fetch", help="获取 PR 完整信息")
    p.add_argument("pr_number", type=int, help="PR 编号")
    p.set_defaults(func=cmd_fetch)

    # assign
    p = subparsers.add_parser("assign", help="分析 PR 变更文件，路由到 SIG，指派 chair，打标签")
    p.add_argument("pr_number", type=int, help="PR 编号")
    p.add_argument("--dry-run", action="store_true", help="只输出分析结果，不实际指派和打标签")
    p.add_argument(
        "--ownership-file",
        help="SIG 归属配置文件路径（默认使用技能目录下的 sig_ownership.json）",
    )
    p.set_defaults(func=cmd_assign)

    # list
    p = subparsers.add_parser("list", help="列出分配给自己的待检视 PR")
    p.add_argument("--user", help="GitCode 用户名（默认从 token 自动获取）")
    p.set_defaults(func=cmd_list)

    # status
    p = subparsers.add_parser("status", help="查看 PR 状态（轻量，不含 diff 和评论）")
    p.add_argument("pr_number", type=int, help="PR 编号")
    p.set_defaults(func=cmd_status)

    # comment
    p = subparsers.add_parser("comment", help="提交检视评论（diff_comment 类型）")
    p.add_argument("pr_number", type=int, help="PR 编号")
    p.add_argument("--file", required=True, help="文件路径（如 tensor_cast/core/config.py）")
    p.add_argument(
        "--line",
        required=True,
        type=int,
        help="行号（PR diff 中新增或修改行的文件行号）",
    )
    p.add_argument("--category", required=True, choices=CATEGORIES, help="评论类别")
    content_group = p.add_mutually_exclusive_group(required=True)
    content_group.add_argument("--content", help="评论内容（短文本，不含代码块时使用）")
    content_group.add_argument("--content-stdin", action="store_true", help="从 stdin 读取内容")
    content_group.add_argument("--content-file", help="从文件读取内容（含代码块时推荐使用）")
    p.set_defaults(func=cmd_comment)

    # verdict
    p = subparsers.add_parser("verdict", help="提交检视结论")
    p.add_argument("pr_number", type=int, help="PR 编号")
    p.add_argument(
        "--event",
        required=True,
        choices=["approved", "comment"],
        help="approved=通过, comment=有意见需修改",
    )
    p.add_argument("--body", required=True, help="检视摘要")
    p.set_defaults(func=cmd_verdict)

    # handoff
    p = subparsers.add_parser("handoff", help="责任传递：移除自己的 assignee，指派新人")
    p.add_argument("pr_number", type=int, help="PR 编号")
    p.add_argument("--to", required=True, help="新 assignee 的 GitCode 用户名")
    p.add_argument("--user", help="你的 GitCode 用户名（默认从 token 自动获取）")
    p.set_defaults(func=cmd_handoff)

    # complete
    p = subparsers.add_parser(
        "complete",
        help="检视完成：提交检视结论（approved 指派 approver，comment 移交回作者）",
    )
    p.add_argument("pr_number", type=int, help="PR 编号")
    p.add_argument("--to", help="approver 的 GitCode 用户名（event=approved 时必须）")
    p.add_argument(
        "--event",
        required=True,
        choices=["approved", "comment"],
        help="approved=通过并指派 approver, comment=有意见并移交回作者",
    )
    body_group = p.add_mutually_exclusive_group(required=True)
    body_group.add_argument("--body", help="检视摘要（须含三项评价：个人理解、功能评价、代码质量评价）")
    body_group.add_argument("--body-stdin", action="store_true", help="从 stdin 读取")
    body_group.add_argument("--body-file", help="从文件读取")
    p.set_defaults(func=cmd_complete)

    # comments
    p = subparsers.add_parser("comments", help="列出 PR 已有评论")
    p.add_argument("pr_number", type=int, help="PR 编号")
    p.set_defaults(func=cmd_comments)

    # withdraw
    p = subparsers.add_parser("withdraw", help="删除评论")
    p.add_argument("comment_id", help="评论数字 ID（note_id，非 SHA1 哈希）")
    p.set_defaults(func=cmd_withdraw)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
