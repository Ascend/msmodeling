#!/usr/bin/env python3
"""Validate project skill metadata and directory conventions.

Checks:
- frontmatter fields and name uniqueness (error)
- same-topic skills coexisting (warning) — surfaces capability/workflow
  layering and accidental functional duplicates across skill roots
- cross-root same-topic skills (warning) — e.g. .agents vs .claude overlap
- description token overlap (warning) — backstop for topics not in TOPIC_PATTERNS
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

SKILL_ROOTS = ((".agents", "skills"), (".claude", "skills"))

# Topic -> trigger phrases (case-insensitive). Used to detect functional overlap.
TOPIC_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "pr-review": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"review pr",
            r"pr review",
            r"检视 pr",
            r"检视pr",
            r"评审 pr",
            r"inline review",
            r"行内检视",
            r"代码检视",
            r"检视PR",
        )
    ],
    "pr-feedback": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"review feedback",
            r"检视意见",
            r"评审意见",
            r"apply review",
            r"处理评审",
            r"按评审修改",
            r"apply feedback",
        )
    ],
    "pr-create": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"create.*pr",
            r"pr create",
            r"提.*pr",
            r"创建.*pr",
            r"提交.*mr",
            r"pr description",
            r"pr body",
            r"发合入",
            r"合入请求",
        )
    ],
    "pr-diff": [re.compile(p, re.IGNORECASE) for p in (r"pr diff", r"diff.*pr", r"拉.*diff", r"pr 变更", r"变更内容")],
    "issue-create": [
        re.compile(p, re.IGNORECASE)
        for p in (r"issue.*create", r"create.*issue", r"提.*issue", r"起草.*issue", r"issue 草", r"整理.*issue")
    ],
    "issue-review": [
        re.compile(p, re.IGNORECASE) for p in (r"issue.*review", r"review.*issue", r"评审.*issue", r"issue 评审")
    ],
    "issue-triage": [re.compile(p, re.IGNORECASE) for p in (r"triage", r"backlog", r"队列")],
    "pipeline": [re.compile(p, re.IGNORECASE) for p in (r"pipeline", r"流水线", r"openlibing", r"openLiBing")],
    "precommit": [re.compile(p, re.IGNORECASE) for p in (r"pre.?commit", r"precommit")],
    "security": [re.compile(p, re.IGNORECASE) for p in (r"security", r"安全.*审", r"凭证", r"密钥", r"secret")],
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "and",
    "or",
    "for",
    "of",
    "in",
    "on",
    "use",
    "when",
    "is",
    "are",
    "with",
    "by",
    "from",
    "into",
    "this",
    "that",
    "it",
    "as",
    "be",
    "will",
    "can",
    "has",
    "have",
    "not",
    "but",
    "if",
    "so",
    "do",
    "does",
}

JACCARD_WARN_THRESHOLD = 0.6


@dataclass(frozen=True)
class Finding:
    path: str
    message: str
    severity: str = "error"


def parse_frontmatter(path: Path) -> dict[str, str]:
    """Parse the small frontmatter subset used by project skills."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].lstrip("\ufeff").strip() != "---":
        return {}
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}

    values: dict[str, str] = {}
    in_metadata = False
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0 and ":" in line:
            key, value = line.split(":", 1)
            in_metadata = key.strip() == "metadata"
            values[key.strip()] = value.strip()
        elif in_metadata and indent > 0 and ":" in line:
            key, value = line.split(":", 1)
            values[f"metadata.{key.strip()}"] = value.strip()
    return values


def _topics(description: str) -> set[str]:
    hits: set[str] = set()
    for topic, patterns in TOPIC_PATTERNS.items():
        if any(p.search(description) for p in patterns):
            hits.add(topic)
    return hits


def _tokens(description: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", description.lower()) if w not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class SkillRecord:
    path: str
    root_name: str
    name: str
    description: str
    source: str
    topics: set[str]
    tokens: set[str]


def collect_skills(repo_root: Path) -> tuple[list[SkillRecord], list[Finding]]:
    records: list[SkillRecord] = []
    findings: list[Finding] = []
    names: dict[str, str] = {}

    for parts in SKILL_ROOTS:
        root = repo_root.joinpath(*parts)
        if not root.is_dir():
            continue
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            skill_file = directory / "SKILL.md"
            relative = skill_file.relative_to(repo_root).as_posix()
            if not skill_file.is_file():
                findings.append(Finding(relative, "missing SKILL.md"))
                continue

            data = parse_frontmatter(skill_file)
            for key in ("name", "description", "metadata.version", "metadata.source"):
                if not data.get(key):
                    findings.append(Finding(relative, f"missing frontmatter field: {key}"))

            name = data.get("name", "").strip().strip('"').strip("'")
            if name and not SKILL_NAME_PATTERN.fullmatch(name):
                findings.append(Finding(relative, f"invalid skill name: {name}"))
            if name:
                if name in names:
                    findings.append(Finding(relative, f"duplicate skill name '{name}' also used by {names[name]}"))
                else:
                    names[name] = relative

            description = data.get("description", "")
            records.append(
                SkillRecord(
                    path=relative,
                    root_name=parts[0],
                    name=name,
                    description=description,
                    source=data.get("metadata.source", ""),
                    topics=_topics(description),
                    tokens=_tokens(description),
                )
            )

    return records, findings


def detect_overlap(records: list[SkillRecord]) -> list[Finding]:
    findings: list[Finding] = []

    # Topic-based overlap: same topic shared by >=2 skills -> warn.
    topic_to_skills: dict[str, list[SkillRecord]] = {}
    for rec in records:
        for topic in rec.topics:
            topic_to_skills.setdefault(topic, []).append(rec)

    for topic, skills in sorted(topic_to_skills.items()):
        if len(skills) < 2:
            continue
        names = ", ".join(s.name or s.path for s in skills)
        for s in skills:
            findings.append(
                Finding(
                    s.path,
                    f"topic '{topic}' shared with other skill(s): {names} — confirm this is intended "
                    f"capability/workflow layering, not a duplicate",
                    severity="warning",
                )
            )

    # Cross-root same topic -> warn (e.g. .agents vs .claude overlap).
    for a, b in combinations(records, 2):
        if a.root_name == b.root_name:
            continue
        shared = a.topics & b.topics
        if shared:
            findings.append(
                Finding(
                    a.path,
                    f"cross-root same-topic '{','.join(sorted(shared))}' with {b.path} — "
                    f"redundant local skill outside .agents/skills",
                    severity="warning",
                )
            )

    # Backstop: high description token overlap not captured by topics.
    for a, b in combinations(records, 2):
        if a.topics & b.topics:
            continue
        score = _jaccard(a.tokens, b.tokens)
        if score >= JACCARD_WARN_THRESHOLD:
            findings.append(
                Finding(
                    a.path,
                    f"description overlap {score:.2f} with {b.path} — possible duplicate",
                    severity="warning",
                )
            )

    return findings


def validate_skills(repo_root: Path) -> list[Finding]:
    """Return validation findings for every project skill."""
    records, findings = collect_skills(repo_root)
    findings.extend(detect_overlap(records))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    findings = validate_skills(args.repo_root.resolve())
    has_error = any(f.severity == "error" for f in findings)
    if args.json:
        print(
            json.dumps(
                {"ok": not has_error, "findings": [asdict(item) for item in findings]},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for finding in findings:
            tag = "WARN " if finding.severity == "warning" else "ERROR"
            print(f"{tag} {finding.path}: {finding.message}")
        print(
            f"skills validation: {'passed' if not has_error else 'failed'} "
            f"({sum(1 for f in findings if f.severity == 'warning')} warning(s))"
        )
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
