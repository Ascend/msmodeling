#!/usr/bin/env python3
import argparse
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from statistics import mean


STATUS_MAP = {
    "9989": "passed",
    "10060": "failed",
    "128346": "running",
}

TEXT_STATUS_MAP = {
    "SUCCESS": "passed",
    "FAILED": "failed",
    "RUNNING": "running",
    "ABORTED": "aborted",
    "CANCELED": "canceled",
    "INIT": "init",
    "QUEUED": "queued",
    "UNSELECTED": "unselected",
    "SKIPPED": "skipped",
}

SENSITIVE_PATTERNS = [
    (re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(authorization\s*:\s*)[^\r\n]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(token=)[^&\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(access_token=)[^&\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(refresh_token=)[^&\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(signature=)[^&\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(sig=)[^&\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(x-amz-signature=)[^&\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(cookie\s*:\s*)[^\r\n]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(set-cookie\s*:\s*)[^\r\n]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b(ak|sk|secret|password|passwd)\b\s*[:=]\s*[^\s,;]+"), "[REDACTED_SECRET]"),
]


def get_gitcode_binary() -> str:
    candidate = os.environ.get("GITCODE_BIN") or shutil.which("gitcode")
    if not candidate:
        raise SystemExit("gitcode executable not found; set GITCODE_BIN or install GitCode CLI")
    return candidate


def run_gitcode_comments(repo: str, pr: int) -> str:
    result = subprocess.run(
        [get_gitcode_binary(), "pr", "comments", str(pr), "-R", repo],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout or ""


def run_gitcode_pr_list(repo: str, limit: int, state: str = "all"):
    result = subprocess.run(
        [
            get_gitcode_binary(),
            "pr",
            "list",
            "-R",
            repo,
            "--state",
            state,
            "-L",
            str(limit),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    prs = []
    try:
        data = json.loads(result.stdout or "[]")
        for item in data:
            num = item.get("number")
            if num:
                prs.append(int(num))
    except (json.JSONDecodeError, TypeError):
        for line in (result.stdout or "").splitlines():
            match = re.match(r"#(\d+)\s+", line.strip())
            if match:
                prs.append(int(match.group(1)))
    return prs


def split_comment_blocks(comments_text: str):
    if not comments_text:
        return []
    blocks = []
    current = []
    for line in comments_text.splitlines():
        if line.startswith("#") and ") ID:" in line:
            if current:
                blocks.append("\n".join(current))
                current = []
        if line.strip() or current:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def parse_pipeline_comments(comments_text: str):
    blocks = []
    comment_blocks = split_comment_blocks(comments_text)
    for comment in comment_blocks:
        if "流水线任务触发成功" not in comment:
            continue
        link_match = re.search(
            r"https://www\.openlibing\.com/apps/pipelineDetail\?pipelineId=[^'\"\s<]+",
            comment,
        )
        if not link_match:
            continue
        table_match = re.search(r"(<table.*?</table>)", comment, re.S)
        table = table_match.group(1) if table_match else ""
        rows = parse_table_rows(table)
        ts_match = re.search(r"Author: .* at (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", comment)
        blocks.append(
            {
                "name": "pipeline",
                "state": "triggered",
                "link": html.unescape(link_match.group(0)),
                "rows": rows,
                "comment_time": ts_match.group(1) if ts_match else "",
                "raw_comment": comment,
            }
        )
    if blocks:
        return blocks

    # Backward-compatible parser for older comment HTML.
    pattern = re.compile(
        r'流水线 <a href="(?P<link>[^"]+)">(?P<name>[^<]+)</a> (?P<state>[^<]+)</div>(?P<table>.*?</table>)',
        re.S,
    )
    for match in pattern.finditer(comments_text):
        table = match.group("table")
        rows = parse_table_rows(table)
        blocks.append(
            {
                "name": clean_html(match.group("name")),
                "state": clean_html(match.group("state")),
                "link": html.unescape(match.group("link")),
                "rows": rows,
                "comment_time": "",
                "raw_comment": match.group(0),
            }
        )
    return blocks


def redact_text(text: str) -> str:
    value = text or ""
    for pattern, repl in SENSITIVE_PATTERNS:
        value = pattern.sub(repl, value)
    return value


def redact_url(url: str) -> str:
    if not url:
        return url
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = []
    for key, value in query:
        if re.search(r"(?i)(token|signature|sig|secret|password|passwd|cookie)", key):
            redacted_query.append((key, "[REDACTED]"))
        else:
            redacted_query.append((key, value))
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(redacted_query)))


def clean_html(text: str) -> str:
    text = re.sub(r"<.*?>", "", text or "")
    return html.unescape(text).strip()


def parse_table_rows(table_html: str):
    rows = []
    if not table_html:
        return rows

    current_stage = ""
    tr_matches = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)
    for tr in tr_matches[1:]:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)
        if not cells:
            continue
        cleaned_cells = [clean_html(cell) for cell in cells]
        links = re.findall(r'href=["\']?([^"\' >]+)', tr)

        # Current robot format: 任务名称 / 状态 / 日志 / 下载链接
        if len(cleaned_cells) >= 2 and cleaned_cells[0] and cleaned_cells[0] != "任务名称":
            task_name = cleaned_cells[0]
            status_text = cleaned_cells[1].replace("✅", "").replace("❌", "").strip().upper()
            task_link = html.unescape(links[0]) if links else ""
            rows.append(
                {
                    "stage": current_stage,
                    "task": task_name,
                    "status": TEXT_STATUS_MAP.get(status_text, status_text.lower()),
                    "link": task_link,
                }
            )
            continue

    if rows:
        return rows

    last_stage = ""
    tr_matches = re.findall(r"<tr>(.*?)</tr>", table_html, re.S)
    for tr in tr_matches[1:]:
        cells = re.findall(r"<td(?: rowspan=\"\d+\")?>(.*?)</td>", tr, re.S)
        if len(cells) == 4:
            stage_raw, task_raw, status_raw, detail_raw = cells
            last_stage = clean_html(stage_raw)
        elif len(cells) == 3:
            stage_raw = last_stage
            task_raw, status_raw, detail_raw = cells
        else:
            continue
        link_match = re.search(r'href=\"([^\"]+)\"', detail_raw)
        if not link_match:
            continue
        status_code_match = re.search(r'&#(\d+);', status_raw)
        status_code = status_code_match.group(1) if status_code_match else ""
        rows.append(
            {
                "stage": clean_html(stage_raw),
                "task": clean_html(task_raw),
                "status": STATUS_MAP.get(status_code, clean_html(status_raw)),
                "link": html.unescape(link_match.group(1)),
            }
        )
    return rows


def choose_block(blocks, comments_text="", latest_failed_run=False):
    if not blocks:
        return None
    invalidated_after = ""
    for comment in split_comment_blocks(comments_text):
        if re.search(r"source code change are detected, tasks labels is removed", comment, re.I):
            ts_match = re.search(r"Author: .* at (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", comment)
            if ts_match and ts_match.group(1) > invalidated_after:
                invalidated_after = ts_match.group(1)
    effective_blocks = blocks
    if invalidated_after:
        filtered = [block for block in blocks if block.get("comment_time", "") > invalidated_after]
        if filtered:
            effective_blocks = filtered
    blocks_with_rows = [block for block in effective_blocks if block.get("rows")]
    if blocks_with_rows:
        effective_blocks = blocks_with_rows
    if latest_failed_run:
        for block in effective_blocks:
            if find_failed_tasks(block):
                return block
    return effective_blocks[-1]


def find_task(block, task_name=None):
    rows = block["rows"]
    if task_name:
        for row in rows:
            if row["task"] == task_name:
                return row
        raise SystemExit(f"Task not found: {task_name}")
    for row in rows:
        if row["status"] == "failed":
            return row
    for row in rows:
        if row["status"] == "running":
            return row
    return None


def find_failed_tasks(block):
    return [row for row in block["rows"] if row["status"] == "failed" and row["stage"] != "流水线"]


def infer_project_id(block):
    candidates = [block.get("link", "")]
    candidates.extend(row.get("link", "") for row in block.get("rows", []))
    for link in candidates:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
        vals = query.get("projectId")
        if vals and vals[0]:
            return vals[0]
    return None


def extract_pipeline_params(block):
    query = urllib.parse.parse_qs(urllib.parse.urlparse(block["link"]).query)
    pipeline_id = (query.get("pipelineId") or [""])[0]
    pipeline_run_id = (query.get("pipelineRunId") or [""])[0]
    project_id = infer_project_id(block)
    if not pipeline_id or not pipeline_run_id:
        raise SystemExit("Missing pipelineId or pipelineRunId in pipeline link")
    if not project_id:
        raise SystemExit("Missing projectId in pipeline context; pass a PR whose comment links contain projectId")
    return {
        "projectId": project_id,
        "pipelineId": pipeline_id,
        "pipelineRunId": pipeline_run_id,
    }


def extract_params(detail_link: str):
    query = urllib.parse.parse_qs(urllib.parse.urlparse(detail_link).query)
    required = ["projectId", "pipelineId", "pipelineRunId", "jobRunId", "stepRunId"]
    params = {}
    for key in required:
        vals = query.get(key)
        if not vals:
            raise SystemExit(f"Missing {key} in task detail link")
        params[key] = vals[0]
    return params


def fetch_pipeline_detail(pipeline_params):
    url = (
        "https://www.openlibing.com/gateway/openlibing-cicd/project/pipeline/pipeline-run/detail?"
        + urllib.parse.urlencode(pipeline_params)
    )
    body = fetch_json(url)
    if body.get("code") != 200:
        raise SystemExit(f"pipeline detail request failed: {body}")
    return body["data"]


def attach_stage_from_detail(block, detail):
    stage_by_task = {}
    for stage in detail.get("stages", []):
        for job in stage.get("jobs") or []:
            stage_by_task[job.get("name")] = stage.get("name", "")
    for row in block["rows"]:
        if not row.get("stage"):
            row["stage"] = stage_by_task.get(row["task"], "")


def resolve_task_params_from_detail(task, block, detail):
    pipeline_params = extract_pipeline_params(block)
    for stage in detail.get("stages", []):
        for job in stage.get("jobs") or []:
            if job.get("name") != task["task"]:
                continue
            steps = job.get("steps") or []
            if not steps:
                continue
            return {
                "projectId": pipeline_params["projectId"],
                "pipelineId": pipeline_params["pipelineId"],
                "pipelineRunId": pipeline_params["pipelineRunId"],
                "jobRunId": job["id"],
                "stepRunId": steps[0]["id"],
            }
    raise SystemExit(f"Task not found in pipeline detail: {task['task']}")


def fetch_exec_log(params, sort="desc", limit=500, start_offset=0, end_offset=0):
    payload = dict(params)
    payload.update(
        {
            "sort": sort,
            "limit": limit,
            "startOffset": start_offset,
            "endOffset": end_offset,
        }
    )
    req = urllib.request.Request(
        "https://www.openlibing.com/gateway/openlibing-cicd/project/pipeline/exec-log",
        data=json.dumps(payload).encode(),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8", "ignore"))
    if body.get("code") != 200:
        raise SystemExit(f"exec-log request failed: {body}")
    return body["data"]


def fetch_json(url: str, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def format_duration_ms(start_time, end_time, pause_time=0):
    if start_time is None or end_time is None:
        return "-"
    seconds = max(0, int((end_time - start_time - (pause_time or 0)) / 1000))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def get_task_bucket(task_name: str):
    if task_name.startswith("Compile_"):
        return "Compile"
    if task_name.startswith("UT_"):
        return "UT"
    if task_name.startswith("StaticCheck_"):
        return "StaticCheck"
    if (
        task_name in {"codecheck", "SCA", "anti_virus", "Check_Pr", "cocheck_codestyle", "harmony-infer"}
        or task_name.startswith("API_")
        or task_name.startswith("ATK_")
        or task_name.startswith("Smoke_")
    ):
        return "Quality/Test"
    return "Other"


def print_duration_summary(block, detail):
    print("")
    print("Duration summary:")
    print(f"Pipeline total: {format_duration_ms(detail.get('start_time'), detail.get('end_time'))}")
    print(
        f"Pipeline actual: {format_duration_ms(detail.get('start_time'), detail.get('end_time'), detail.get('pause_time', 0))}"
    )
    print("")
    print("Stages:")
    for stage in detail.get("stages", []):
        print(
            f"- {stage.get('name')} [{stage.get('status')}] "
            f"total={format_duration_ms(stage.get('start_time'), stage.get('end_time'))} "
            f"actual={format_duration_ms(stage.get('start_time'), stage.get('end_time'), stage.get('pause_time', 0))} "
            f"jobs={len(stage.get('jobs') or [])}"
        )

    bucket_stats = defaultdict(lambda: {"count": 0, "total_s": 0, "max_s": -1, "max_job": ""})
    top_jobs = []
    for stage in detail.get("stages", []):
        for job in stage.get("jobs") or []:
            if job.get("start_time") is None or job.get("end_time") is None:
                continue
            duration_s = max(0, int((job["end_time"] - job["start_time"]) / 1000))
            bucket = get_task_bucket(job.get("name", ""))
            bucket_stats[bucket]["count"] += 1
            bucket_stats[bucket]["total_s"] += duration_s
            if duration_s > bucket_stats[bucket]["max_s"]:
                bucket_stats[bucket]["max_s"] = duration_s
                bucket_stats[bucket]["max_job"] = job.get("name", "")
            top_jobs.append((duration_s, stage.get("name", ""), job.get("name", ""), job.get("status", "")))

    if bucket_stats:
        print("")
        print("Task buckets:")
        for bucket in sorted(bucket_stats):
            stats = bucket_stats[bucket]
            avg_s = int(round(stats["total_s"] / stats["count"])) if stats["count"] else 0
            print(f"- {bucket}: avg={avg_s}s count={stats['count']} max={stats['max_s']}s ({stats['max_job']})")

    if top_jobs:
        print("")
        print("Top slow jobs:")
        for duration_s, stage_name, job_name, status in sorted(top_jobs, reverse=True)[:10]:
            print(f"- {job_name} [{status}] stage={stage_name} duration={duration_s}s")


def analyze_pr(repo: str, pr: int, latest_failed_run=False):
    comments = run_gitcode_comments(repo, pr)
    blocks = parse_pipeline_comments(comments)
    block = choose_block(blocks, comments, latest_failed_run=latest_failed_run)
    if not block:
        return {"pr": pr, "block": None, "detail": None}

    detail = None
    try:
        detail = fetch_pipeline_detail(extract_pipeline_params(block))
        attach_stage_from_detail(block, detail)
    except SystemExit:
        detail = None
    return {"pr": pr, "block": block, "detail": detail}


def build_duration_metrics(detail):
    if not detail:
        return None

    stages = []
    bucket_stats = defaultdict(lambda: {"count": 0, "total_s": 0, "max_s": -1, "max_job": ""})
    top_jobs = []
    for stage in detail.get("stages", []):
        stages.append(
            {
                "name": stage.get("name"),
                "status": stage.get("status"),
                "duration_s": (
                    max(0, int((stage["end_time"] - stage["start_time"]) / 1000))
                    if stage.get("start_time") is not None and stage.get("end_time") is not None
                    else None
                ),
                "actual_s": (
                    max(
                        0,
                        int((stage["end_time"] - stage["start_time"] - (stage.get("pause_time") or 0)) / 1000),
                    )
                    if stage.get("start_time") is not None and stage.get("end_time") is not None
                    else None
                ),
                "job_count": len(stage.get("jobs") or []),
            }
        )
        for job in stage.get("jobs") or []:
            if job.get("start_time") is None or job.get("end_time") is None:
                continue
            duration_s = max(0, int((job["end_time"] - job["start_time"]) / 1000))
            bucket = get_task_bucket(job.get("name", ""))
            bucket_stats[bucket]["count"] += 1
            bucket_stats[bucket]["total_s"] += duration_s
            if duration_s > bucket_stats[bucket]["max_s"]:
                bucket_stats[bucket]["max_s"] = duration_s
                bucket_stats[bucket]["max_job"] = job.get("name", "")
            top_jobs.append(
                {
                    "name": job.get("name", ""),
                    "status": job.get("status", ""),
                    "stage": stage.get("name", ""),
                    "duration_s": duration_s,
                    "bucket": bucket,
                }
            )

    buckets = {}
    for bucket_name, stats in bucket_stats.items():
        buckets[bucket_name] = {
            "count": stats["count"],
            "avg_s": int(round(stats["total_s"] / stats["count"])) if stats["count"] else 0,
            "max_s": stats["max_s"],
            "max_job": stats["max_job"],
        }

    return {
        "status": detail.get("status"),
        "pipeline_total_s": (
            max(0, int((detail["end_time"] - detail["start_time"]) / 1000))
            if detail.get("start_time") is not None and detail.get("end_time") is not None
            else None
        ),
        "pipeline_actual_s": (
            max(
                0,
                int((detail["end_time"] - detail["start_time"] - (detail.get("pause_time") or 0)) / 1000),
            )
            if detail.get("start_time") is not None and detail.get("end_time") is not None
            else None
        ),
        "stages": stages,
        "buckets": buckets,
        "top_jobs": sorted(top_jobs, key=lambda item: item["duration_s"], reverse=True)[:10],
    }


def collect_failure_details(block, detail):
    failures = []
    for task in find_failed_tasks(block):
        kind = get_task_kind(task)
        entry = {
            "task": task["task"],
            "stage": task["stage"],
            "kind": kind,
            "link": redact_url(task["link"]),
        }
        if kind == "codecheck":
            params = extract_codecheck_params(task["link"])
            report = fetch_codecheck_report(params)
            defects = fetch_codecheck_defects(params, report)
            defect = (defects.get("defects") or [{}])[0]
            entry["summary"] = (
                f"{defect.get('gitUrl') or defect.get('fileName') or ''}:"
                f"{((defect.get('fragment') or [{}])[0]).get('line_num') or '?'} "
                f"{redact_text(defect.get('ruleName', ''))}"
            ).strip()
        elif kind == "sca":
            scan_id = extract_sca_scan_id(task["link"])
            issues = fetch_sca_issues(scan_id)
            issue = (issues.get("list") or [{}])[0]
            entry["summary"] = (
                f"{issue.get('fileName') or ''} {issue.get('type') or ''} "
                f"{issue.get('vendor') or ''}/{issue.get('component') or ''} "
                f"{issue.get('version') or ''}"
            ).strip()
        else:
            try:
                params = extract_params(task["link"])
            except SystemExit:
                if not detail:
                    params = None
                else:
                    params = resolve_task_params_from_detail(task, block, detail)
            if params:
                needle, excerpt = find_failure_excerpt(params)
                entry["trigger"] = needle or ""
                if excerpt:
                    short = redact_text(excerpt[:800]).replace("\n", " ")
                    entry["summary"] = re.sub(r"\s+", " ", short).strip()
                else:
                    entry["summary"] = "No failure excerpt found."
            else:
                entry["summary"] = "Unable to resolve task log parameters."
        failures.append(entry)
    return failures


def print_latest_duration_report(repo: str, limit: int, latest_failed_run=False):
    prs = run_gitcode_pr_list(repo, limit)
    analyses = [analyze_pr(repo, pr, latest_failed_run=latest_failed_run) for pr in prs]
    rows = []
    for item in analyses:
        if not item["block"] or not item["detail"]:
            rows.append({"pr": item["pr"], "status": "no-data"})
            continue
        metrics = build_duration_metrics(item["detail"])
        rows.append({"pr": item["pr"], **metrics})

    print(f"Latest {limit} PR duration report")
    print("")
    for row in rows:
        if row["status"] == "no-data":
            print(f"- PR #{row['pr']}: no pipeline detail")
            continue
        print(
            f"- PR #{row['pr']}: status={row['status']} "
            f"pipeline={row['pipeline_total_s']}s actual={row['pipeline_actual_s']}s"
        )
        stage_summary = ", ".join(
            f"{stage['name']}={stage['actual_s']}s" for stage in row["stages"] if stage["actual_s"] is not None
        )
        print(f"  stages: {stage_summary}")
        bucket_summary = ", ".join(
            f"{name}=avg{stats['avg_s']}s/max{stats['max_s']}s({stats['max_job']})"
            for name, stats in sorted(row["buckets"].items())
        )
        print(f"  buckets: {bucket_summary}")
        slow_summary = ", ".join(f"{job['name']}={job['duration_s']}s" for job in row["top_jobs"][:5])
        print(f"  top: {slow_summary}")

    completed = [row for row in rows if row.get("status") == "COMPLETED"]
    if not completed:
        return

    print("")
    print("Aggregate summary:")
    pipeline_avg = int(round(mean(row["pipeline_total_s"] for row in completed)))
    print(f"- completed PRs: {len(completed)}")
    print(f"- avg pipeline total: {pipeline_avg}s")

    stage_aggregate = defaultdict(list)
    bucket_aggregate = defaultdict(list)
    for row in completed:
        for stage in row["stages"]:
            if stage["actual_s"] is not None:
                stage_aggregate[stage["name"]].append(stage["actual_s"])
        for name, stats in row["buckets"].items():
            bucket_aggregate[name].append(stats["avg_s"])

    for name, values in stage_aggregate.items():
        print(f"- stage {name}: avg {int(round(mean(values)))}s")
    for name, values in sorted(bucket_aggregate.items()):
        print(f"- bucket {name}: avg job {int(round(mean(values)))}s")


def format_markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def emit_output(text, output_path=None):
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
        return
    print(text, end="" if text.endswith("\n") else "\n")


def display_value(value):
    return "-" if value is None else value


def collect_latest_duration_report_data(repo: str, limit: int, latest_failed_run=False):
    prs = run_gitcode_pr_list(repo, limit)
    rows = []
    completed_rows = []
    for pr in prs:
        analysis = analyze_pr(repo, pr, latest_failed_run=latest_failed_run)
        if not analysis["block"] or not analysis["detail"]:
            rows.append(
                {
                    "pr": pr,
                    "status": "no-data",
                    "pipeline_total_s": None,
                    "pipeline_actual_s": None,
                    "stages": [],
                    "buckets": {},
                    "top_jobs": [],
                }
            )
            continue
        metrics = build_duration_metrics(analysis["detail"])
        rows.append({"pr": pr, **metrics})
        if metrics["status"] == "COMPLETED":
            completed_rows.append(metrics)

    aggregate = {
        "completed_prs": len(completed_rows),
        "avg_pipeline_total_s": (
            int(round(mean(row["pipeline_total_s"] for row in completed_rows))) if completed_rows else None
        ),
        "stage_avg_s": {},
        "bucket_avg_job_s": {},
    }
    if completed_rows:
        for stage_name in ["解析CI_BRANCH", "Image", "编译构建", "LLT", "后处理阶段"]:
            values = []
            for metrics in completed_rows:
                for stage in metrics["stages"]:
                    if stage["name"] == stage_name and stage["actual_s"] is not None:
                        values.append(stage["actual_s"])
            if values:
                aggregate["stage_avg_s"][stage_name] = int(round(mean(values)))
        for bucket_name in ["Compile", "UT", "StaticCheck", "Quality/Test", "Other"]:
            values = [
                metrics["buckets"][bucket_name]["avg_s"]
                for metrics in completed_rows
                if bucket_name in metrics["buckets"]
            ]
            if values:
                aggregate["bucket_avg_job_s"][bucket_name] = int(round(mean(values)))

    return {"repo": repo, "limit": limit, "rows": rows, "aggregate": aggregate}


def render_latest_duration_report_table(report):
    buffer = io.StringIO()
    print(f"Latest {report['limit']} PR duration report", file=buffer)
    print("", file=buffer)

    overview_rows = []
    bucket_rows = []
    for row in report["rows"]:
        if row["status"] == "no-data":
            overview_rows.append([f"#{row['pr']}", "no-data", "-", "-", "-", "-", "-", "-"])
            continue
        stage_map = {stage["name"]: stage["actual_s"] for stage in row["stages"]}
        overview_rows.append(
            [
                f"#{row['pr']}",
                row["status"],
                display_value(row["pipeline_total_s"]),
                display_value(stage_map.get("解析CI_BRANCH", "-")),
                display_value(stage_map.get("Image", "-")),
                display_value(stage_map.get("编译构建", "-")),
                display_value(stage_map.get("LLT", "-")),
                display_value(stage_map.get("后处理阶段", "-")),
            ]
        )
        bucket_rows.append(
            [
                f"#{row['pr']}",
                row["buckets"].get("Compile", {}).get("avg_s", "-"),
                row["buckets"].get("UT", {}).get("avg_s", "-"),
                row["buckets"].get("StaticCheck", {}).get("avg_s", "-"),
                row["buckets"].get("Quality/Test", {}).get("avg_s", "-"),
                row["buckets"].get("Other", {}).get("avg_s", "-"),
                (f"{row['top_jobs'][0]['name']}={row['top_jobs'][0]['duration_s']}s" if row["top_jobs"] else "-"),
            ]
        )

    print(
        format_markdown_table(
            ["PR", "status", "pipeline_s", "branch_s", "image_s", "build_s", "llt_s", "post_s"],
            overview_rows,
        ),
        file=buffer,
    )
    if bucket_rows:
        print("", file=buffer)
        print(
            format_markdown_table(
                ["PR", "compile_avg_s", "ut_avg_s", "static_avg_s", "quality_avg_s", "other_avg_s", "slowest_job"],
                bucket_rows,
            ),
            file=buffer,
        )

    aggregate = report["aggregate"]
    if aggregate["completed_prs"]:
        print("", file=buffer)
        print("Aggregate summary:", file=buffer)
        print(f"- completed PRs: {aggregate['completed_prs']}", file=buffer)
        print(f"- avg pipeline total: {aggregate['avg_pipeline_total_s']}s", file=buffer)
        for name, value in aggregate["stage_avg_s"].items():
            print(f"- stage {name}: avg {value}s", file=buffer)
        for name, value in aggregate["bucket_avg_job_s"].items():
            print(f"- bucket {name}: avg job {value}s", file=buffer)

    return buffer.getvalue()


def print_latest_duration_report_table(repo: str, limit: int, latest_failed_run=False):
    prs = run_gitcode_pr_list(repo, limit)
    analyses = [analyze_pr(repo, pr, latest_failed_run=latest_failed_run) for pr in prs]
    rows = []
    completed_rows = []
    for item in analyses:
        if not item["block"] or not item["detail"]:
            rows.append([f"#{item['pr']}", "no-data", "-", "-", "-", "-", "-", "-"])
            continue
        metrics = build_duration_metrics(item["detail"])
        stage_map = {stage["name"]: stage["actual_s"] for stage in metrics["stages"]}
        row = [
            f"#{item['pr']}",
            metrics["status"],
            display_value(metrics["pipeline_total_s"]),
            display_value(stage_map.get("解析CI_BRANCH", "-")),
            display_value(stage_map.get("Image", "-")),
            display_value(stage_map.get("编译构建", "-")),
            display_value(stage_map.get("LLT", "-")),
            display_value(stage_map.get("后处理阶段", "-")),
        ]
        rows.append(row)
        if metrics["status"] == "COMPLETED":
            completed_rows.append(metrics)

    print(f"Latest {limit} PR duration report")
    print("")
    print(
        format_markdown_table(
            ["PR", "status", "pipeline_s", "branch_s", "image_s", "build_s", "llt_s", "post_s"],
            rows,
        )
    )

    bucket_rows = []
    for item in analyses:
        if not item["block"] or not item["detail"]:
            continue
        metrics = build_duration_metrics(item["detail"])
        bucket_rows.append(
            [
                f"#{item['pr']}",
                metrics["buckets"].get("Compile", {}).get("avg_s", "-"),
                metrics["buckets"].get("UT", {}).get("avg_s", "-"),
                metrics["buckets"].get("StaticCheck", {}).get("avg_s", "-"),
                metrics["buckets"].get("Quality/Test", {}).get("avg_s", "-"),
                metrics["buckets"].get("Other", {}).get("avg_s", "-"),
                (
                    f"{metrics['top_jobs'][0]['name']}={metrics['top_jobs'][0]['duration_s']}s"
                    if metrics["top_jobs"]
                    else "-"
                ),
            ]
        )

    if bucket_rows:
        print("")
        print(
            format_markdown_table(
                ["PR", "compile_avg_s", "ut_avg_s", "static_avg_s", "quality_avg_s", "other_avg_s", "slowest_job"],
                bucket_rows,
            )
        )

    if completed_rows:
        print("")
        print("Aggregate summary:")
        print(f"- completed PRs: {len(completed_rows)}")
        print(f"- avg pipeline total: {int(round(mean(row['pipeline_total_s'] for row in completed_rows)))}s")
        stage_names = ["解析CI_BRANCH", "Image", "编译构建", "LLT", "后处理阶段"]
        for stage_name in stage_names:
            values = []
            for metrics in completed_rows:
                for stage in metrics["stages"]:
                    if stage["name"] == stage_name and stage["actual_s"] is not None:
                        values.append(stage["actual_s"])
            if values:
                print(f"- stage {stage_name}: avg {int(round(mean(values)))}s")
        for bucket_name in ["Compile", "UT", "StaticCheck", "Quality/Test", "Other"]:
            values = [
                metrics["buckets"][bucket_name]["avg_s"]
                for metrics in completed_rows
                if bucket_name in metrics["buckets"]
            ]
            if values:
                print(f"- bucket {bucket_name}: avg job {int(round(mean(values)))}s")


def print_latest_failure_report_table(repo: str, limit: int, latest_failed_run=False):
    prs = run_gitcode_pr_list(repo, limit)
    summary_rows = []
    detail_lines = []
    for pr in prs:
        analysis = analyze_pr(repo, pr, latest_failed_run=latest_failed_run)
        block = analysis["block"]
        detail = analysis["detail"]
        if not block:
            summary_rows.append([f"#{pr}", "no-comment", "-", "-"])
            continue
        failures = collect_failure_details(block, detail)
        status = detail.get("status") if detail else block.get("state")
        if not failures:
            summary_rows.append([f"#{pr}", status, "0", "-"])
            continue
        summary_rows.append(
            [
                f"#{pr}",
                status,
                str(len(failures)),
                f"{failures[0]['task']} [{failures[0]['kind']}]",
            ]
        )
        detail_lines.append(f"PR #{pr}:")
        for failure in failures:
            trigger = f" trigger={failure['trigger']}" if failure.get("trigger") else ""
            detail_lines.append(f"- {failure['stage']} / {failure['task']} [{failure['kind']}]{trigger}")
            detail_lines.append(f"  {failure['summary']}")

    print(f"Latest {limit} PR failure report")
    print("")
    print(format_markdown_table(["PR", "status", "failed_count", "first_failure"], summary_rows))
    if detail_lines:
        print("")
        print("Failure details:")
        print("\n".join(detail_lines))


def collect_latest_failure_report_data(repo: str, limit: int, latest_failed_run=False):
    prs = run_gitcode_pr_list(repo, limit)
    rows = []
    for pr in prs:
        analysis = analyze_pr(repo, pr, latest_failed_run=latest_failed_run)
        block = analysis["block"]
        detail = analysis["detail"]
        if not block:
            rows.append(
                {
                    "pr": pr,
                    "status": "no-comment",
                    "failed_count": None,
                    "first_failure": None,
                    "failures": [],
                }
            )
            continue
        failures = collect_failure_details(block, detail)
        status = detail.get("status") if detail else block.get("state")
        rows.append(
            {
                "pr": pr,
                "status": status,
                "failed_count": len(failures),
                "first_failure": (f"{failures[0]['task']} [{failures[0]['kind']}]" if failures else None),
                "failures": failures,
            }
        )
    return {"repo": repo, "limit": limit, "rows": rows}


def render_latest_failure_report_table(report):
    buffer = io.StringIO()
    print(f"Latest {report['limit']} PR failure report", file=buffer)
    print("", file=buffer)
    summary_rows = []
    detail_lines = []
    for row in report["rows"]:
        summary_rows.append(
            [
                f"#{row['pr']}",
                row["status"],
                "-" if row["failed_count"] is None else row["failed_count"],
                row["first_failure"] or "-",
            ]
        )
        if row["failures"]:
            detail_lines.append(f"PR #{row['pr']}:")
            for failure in row["failures"]:
                trigger = f" trigger={failure['trigger']}" if failure.get("trigger") else ""
                detail_lines.append(f"- {failure['stage']} / {failure['task']} [{failure['kind']}]{trigger}")
                detail_lines.append(f"  {failure['summary']}")

    print(
        format_markdown_table(["PR", "status", "failed_count", "first_failure"], summary_rows),
        file=buffer,
    )
    if detail_lines:
        print("", file=buffer)
        print("Failure details:", file=buffer)
        print("\n".join(detail_lines), file=buffer)
    return buffer.getvalue()


def print_latest_failure_report(repo: str, limit: int, latest_failed_run=False):
    prs = run_gitcode_pr_list(repo, limit)
    print(f"Latest {limit} PR failure report")
    print("")
    for pr in prs:
        analysis = analyze_pr(repo, pr, latest_failed_run=latest_failed_run)
        block = analysis["block"]
        detail = analysis["detail"]
        if not block:
            print(f"- PR #{pr}: no pipeline comment")
            continue
        failures = collect_failure_details(block, detail)
        if not failures:
            print(f"- PR #{pr}: no failed tasks")
            continue
        status = detail.get("status") if detail else block.get("state")
        print(f"- PR #{pr}: status={status} failed_tasks={len(failures)}")
        for failure in failures:
            trigger = f" trigger={failure['trigger']}" if failure.get("trigger") else ""
            print(f"  {failure['stage']} / {failure['task']} [{failure['kind']}]{trigger}")
            print(f"  {failure['summary']}")


def fetch_log_pages(params, sort="desc", limit=500, max_pages=12):
    start_offset = 0
    end_offset = 0
    for _ in range(max_pages):
        data = fetch_exec_log(
            params,
            sort=sort,
            limit=limit,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        yield data
        if not data.get("has_more"):
            break
        start_offset = data.get("start_offset", 0)
        end_offset = data.get("end_offset", 0)


def find_failure_excerpt(params):
    needles = [
        "short test summary info",
        "FAILED ",
        "ERROR ",
        "Traceback (most recent call last)",
        "DID NOT RAISE",
    ]
    for data in fetch_log_pages(params, sort="asc", max_pages=40):
        log = data.get("log", "")
        best_idx = -1
        best_needle = None
        for needle in needles:
            idx = log.find(needle)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_needle = needle
        if best_idx != -1:
            excerpt = log[max(0, best_idx - 2500) : best_idx + 8000]
            return best_needle, excerpt
    return None, None


def get_task_kind(task):
    parsed = urllib.parse.urlparse(task["link"])
    path = parsed.path
    if "/apps/entryCheckDashCode/" in path:
        return "codecheck"
    if "/apps/personalScandTaskInfor/" in path:
        return "sca"
    if "pipelineDetail" in path and "stepRunId" in urllib.parse.parse_qs(parsed.query):
        return "exec-log"
    return "unknown"


def extract_codecheck_params(detail_link: str):
    parsed = urllib.parse.urlparse(detail_link)
    query = urllib.parse.parse_qs(parsed.query)
    parts = [part for part in parsed.path.split("/") if part]
    try:
        idx = parts.index("entryCheckDashCode")
        task_id = parts[idx + 1]
        uuid = parts[idx + 2]
    except (ValueError, IndexError):
        raise SystemExit(f"Unsupported CodeCheck detail link: {detail_link}")
    return {
        "taskId": task_id,
        "uuid": uuid,
        "projectId": (query.get("projectId") or [""])[0],
        "projectName": (query.get("projectName") or [""])[0],
    }


def fetch_codecheck_report(params):
    url = (
        "https://www.openlibing.com/gateway/openlibing-codecheck/"
        f"ci-portal/v1/codecheck/event/task/issues/report?uuid={params['uuid']}&taskId={params['taskId']}"
    )
    payload = {
        "pageNum": 1,
        "pageSize": 20,
        "date": None,
        "defectLevel": "",
        "ruleType": "",
        "filePath": "",
        "fileName": "",
        "defectStatus": "",
        "checkType": "",
        "trigger": "",
        "defectCheckerName": "",
        "isDelay": "",
    }
    body = fetch_json(url, payload=payload)
    if body.get("code") != 200:
        raise SystemExit(f"CodeCheck report request failed: {body}")
    return body["result"]


def fetch_codecheck_defects(params, report):
    report_vo = report.get("reportVo") or {}
    url = (
        "https://www.openlibing.com/gateway/openlibing-codecheck/"
        f"ci-portal/v1/event/codecheck/task?uuid={params['uuid']}&taskId={params['taskId']}"
    )
    payload = {
        "pageNum": 1,
        "pageSize": 20,
        "date": None,
        "defectLevel": "",
        "ruleType": "",
        "filePath": "",
        "fileName": "",
        "defectStatus": "",
        "checkType": "",
        "trigger": "",
        "defectCheckerName": "",
        "isDelay": "",
        "flag": "",
        "projectName": report_vo.get("projectName", params["projectName"]),
        "projectId": params["projectId"],
        "repoUrl": report_vo.get("repoUrl", ""),
        "repoName": report_vo.get("repoNameEn", ""),
        "branchName": report_vo.get("git_branch", ""),
        "ruleName": "",
    }
    body = fetch_json(url, payload=payload)
    if body.get("code") != 200:
        raise SystemExit(f"CodeCheck defect request failed: {body}")
    return body["result"]


def print_codecheck_summary(task):
    params = extract_codecheck_params(task["link"])
    report = fetch_codecheck_report(params)
    report_vo = report.get("reportVo") or {}
    defects = fetch_codecheck_defects(params, report)
    defect_list = defects.get("defects") or []
    print("")
    print(f"Task: {task['task']}")
    print(f"Stage: {task['stage']}")
    print(f"Link: {redact_url(task['link'])}")
    print(f"Issue count: {report_vo.get('issue_count') or 0}")
    print(f"New count: {report_vo.get('new_count') or 0}")
    print(f"Risk coefficient: {report_vo.get('risk_coefficient')}")
    if not defect_list:
        print("No CodeCheck defects returned.")
        return
    for defect in defect_list[:5]:
        fragments = defect.get("fragment") or []
        line_nums = [frag.get("line_num") for frag in fragments if frag.get("line_num")]
        line_hint = line_nums[0] if line_nums else "?"
        file_label = defect.get("gitUrl") or defect.get("fileName") or defect.get("filepath") or ""
        print(f"- {file_label}:{line_hint} {redact_text(defect.get('ruleName', ''))}")
        print(f"  Rule: {redact_text(defect.get('defectCheckerName', ''))}")


def extract_sca_scan_id(detail_link: str):
    parsed = urllib.parse.urlparse(detail_link)
    parts = [part for part in parsed.path.split("/") if part]
    try:
        idx = parts.index("personalScandTaskInfor")
        return parts[idx + 2]
    except (ValueError, IndexError):
        raise SystemExit(f"Unsupported SCA detail link: {detail_link}")


def fetch_sca_scan_info(scan_id: str):
    url = f"https://www.openlibing.com/gateway/openlibing-sca/open/person/scan/id?scanId={urllib.parse.quote(scan_id)}"
    body = fetch_json(url)
    if body.get("code") != 200:
        raise SystemExit(f"SCA scan info request failed: {body}")
    return body["data"]


def fetch_sca_issues(scan_id: str):
    url = "https://www.openlibing.com/gateway/openlibing-sca/open/scan/scanIssue/query"
    payload = {"scanId": scan_id, "pageNo": 1, "pageSize": 20}
    body = fetch_json(url, payload=payload)
    if body.get("code") != 200:
        raise SystemExit(f"SCA issues request failed: {body}")
    return body["data"]


def print_sca_summary(task):
    scan_id = extract_sca_scan_id(task["link"])
    info = fetch_sca_scan_info(scan_id)
    data = fetch_sca_issues(scan_id)
    issues = data.get("list") or []
    print("")
    print(f"Task: {task['task']}")
    print(f"Stage: {task['stage']}")
    print(f"Link: {redact_url(task['link'])}")
    print(f"Scan result: {info.get('scanResult')}")
    print(f"Scanned files: {info.get('fileNum')}")
    print(f"Issue total: {data.get('total') or 0}")
    if not issues:
        print("No SCA issues returned.")
        return
    for issue in issues[:5]:
        print(
            f"- {issue.get('fileName')}: {issue.get('type')} {issue.get('matched')} "
            f"{issue.get('vendor')}/{issue.get('component')} {issue.get('version')}"
        )
        print(f"  licenseStatus={issue.get('licenseStatus')} copyrightStatus={issue.get('copyrightStatus')}")


def print_block_summary(block):
    print(f"Pipeline: {block['name']}")
    print(f"State: {block['state']}")
    print(f"Link: {redact_url(block['link'])}")
    print("")
    for row in block["rows"]:
        print(f"[{row['status']}] {row['stage']} / {row['task']}")
        print(f"  {redact_url(row['link'])}")


def print_log_excerpt(task, data):
    print("")
    print(f"Selected task: {task['task']}")
    print(f"Task state: {task['status']}")
    print(f"Task link: {redact_url(task['link'])}")
    print("")
    print("Log page info:")
    print(
        f"  has_more={data.get('has_more')} start_offset={data.get('start_offset')} end_offset={data.get('end_offset')}"
    )
    print("")
    print(redact_text(data.get("log", "")[:12000]))


def print_failure_excerpt(task, params):
    needle, excerpt = find_failure_excerpt(params)
    print("")
    if not excerpt:
        print("No failure excerpt found in fetched log pages.")
        return
    print(f"Failure excerpt trigger: {needle}")
    print("")
    print(redact_text(excerpt))


def print_failed_task_summaries(block):
    failed_tasks = find_failed_tasks(block)
    if not failed_tasks:
        print("")
        print("No failed tasks found in the latest pipeline comment.")
        return False

    print("")
    print("Failed task summaries:")
    for task in failed_tasks:
        kind = get_task_kind(task)
        if kind == "codecheck":
            print_codecheck_summary(task)
            continue
        if kind == "sca":
            print_sca_summary(task)
            continue
        params = extract_params(task["link"])
        print("")
        print(f"Task: {task['task']}")
        print(f"Stage: {task['stage']}")
        print(f"Link: {redact_url(task['link'])}")
        needle, excerpt = find_failure_excerpt(params)
        if not excerpt:
            print("No failure excerpt found in fetched log pages.")
            continue
        print(f"Failure excerpt trigger: {needle}")
        print(redact_text(excerpt[:8000]))
    return True


def main():
    stdout_encoding = getattr(sys.stdout, "encoding", None)
    if stdout_encoding and str(stdout_encoding).lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, io.UnsupportedOperation):
            pass
    stderr_encoding = getattr(sys.stderr, "encoding", None)
    if stderr_encoding and str(stderr_encoding).lower() != "utf-8":
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, io.UnsupportedOperation):
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int)
    parser.add_argument("--latest", type=int)
    parser.add_argument("--task")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--failure-summary", action="store_true")
    parser.add_argument("--latest-failed-run", action="store_true")
    parser.add_argument("--durations", action="store_true")
    parser.add_argument("--failure-details", action="store_true")
    parser.add_argument("--report-format", choices=["plain", "table", "json"], default="plain")
    parser.add_argument("--output")
    args = parser.parse_args()

    if not args.pr and not args.latest:
        raise SystemExit("One of --pr or --latest is required")
    if args.pr and args.latest:
        raise SystemExit("Use either --pr or --latest, not both")

    if args.latest:
        if args.failure_details:
            if args.report_format == "json":
                emit_output(
                    json.dumps(
                        collect_latest_failure_report_data(
                            args.repo,
                            args.latest,
                            latest_failed_run=args.latest_failed_run,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    args.output,
                )
                return
            if args.report_format == "table":
                emit_output(
                    render_latest_failure_report_table(
                        collect_latest_failure_report_data(
                            args.repo,
                            args.latest,
                            latest_failed_run=args.latest_failed_run,
                        )
                    ),
                    args.output,
                )
                return
            if args.output:
                old_stdout = sys.stdout
                buffer = io.StringIO()
                sys.stdout = buffer
                try:
                    print_latest_failure_report(
                        args.repo,
                        args.latest,
                        latest_failed_run=args.latest_failed_run,
                    )
                finally:
                    sys.stdout = old_stdout
                emit_output(buffer.getvalue(), args.output)
                return
            print_latest_failure_report(
                args.repo,
                args.latest,
                latest_failed_run=args.latest_failed_run,
            )
            return
        if args.durations:
            if args.report_format == "json":
                emit_output(
                    json.dumps(
                        collect_latest_duration_report_data(
                            args.repo,
                            args.latest,
                            latest_failed_run=args.latest_failed_run,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    args.output,
                )
                return
            if args.report_format == "table":
                emit_output(
                    render_latest_duration_report_table(
                        collect_latest_duration_report_data(
                            args.repo,
                            args.latest,
                            latest_failed_run=args.latest_failed_run,
                        )
                    ),
                    args.output,
                )
                return
            if args.output:
                old_stdout = sys.stdout
                buffer = io.StringIO()
                sys.stdout = buffer
                try:
                    print_latest_duration_report(
                        args.repo,
                        args.latest,
                        latest_failed_run=args.latest_failed_run,
                    )
                finally:
                    sys.stdout = old_stdout
                emit_output(buffer.getvalue(), args.output)
                return
            print_latest_duration_report(
                args.repo,
                args.latest,
                latest_failed_run=args.latest_failed_run,
            )
            return
        raise SystemExit("--latest currently requires --durations or --failure-details")

    analysis = analyze_pr(args.repo, args.pr, latest_failed_run=args.latest_failed_run)
    block = analysis["block"]
    if not block:
        raise SystemExit("No pipeline table found in PR comments")
    detail = analysis["detail"]

    print_block_summary(block)
    if args.durations and detail:
        print_duration_summary(block, detail)
    if args.summary_only:
        return

    if not args.task:
        found_failed = print_failed_task_summaries(block)
        if found_failed:
            return
        running_task = find_task(block)
        if not running_task:
            print("")
            print("No failed or running tasks found in the latest pipeline comment.")
            return

    task = find_task(block, args.task)
    if not task:
        raise SystemExit("No task rows found in latest pipeline comment")
    kind = get_task_kind(task)
    if args.failure_summary and kind == "codecheck":
        print("")
        print(f"Selected task: {task['task']}")
        print(f"Task state: {task['status']}")
        print(f"Task link: {redact_url(task['link'])}")
        print_codecheck_summary(task)
        return
    if args.failure_summary and kind == "sca":
        print("")
        print(f"Selected task: {task['task']}")
        print(f"Task state: {task['status']}")
        print(f"Task link: {redact_url(task['link'])}")
        print_sca_summary(task)
        return
    if kind == "codecheck":
        print_codecheck_summary(task)
        return
    if kind == "sca":
        print_sca_summary(task)
        return
    try:
        params = extract_params(task["link"])
    except SystemExit:
        if not detail:
            raise
        params = resolve_task_params_from_detail(task, block, detail)
    if args.failure_summary:
        print("")
        print(f"Selected task: {task['task']}")
        print(f"Task state: {task['status']}")
        print(f"Task link: {redact_url(task['link'])}")
        print_failure_excerpt(task, params)
        return
    data = fetch_exec_log(params)
    print_log_excerpt(task, data)


if __name__ == "__main__":
    main()
