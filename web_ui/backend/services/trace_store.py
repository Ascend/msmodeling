"""Chrome trace file storage utilities (trace artifacts per job/case).

Trace files follow the same .msmodeling_ui/ convention as logs:
  .msmodeling_ui/chrome_traces/{job_id}/case_{seq}.json

Worker writes {case_hash}.json as a temp name (seq unknown during worker run).
Main process renames {case_hash}.json -> case_{seq}.json after seq assignment.
"""

from __future__ import annotations

from pathlib import Path

from services.capture import msmodeling_ui_dir


def trace_dir(job_id: str) -> Path:
    """Directory for a job's trace files: .msmodeling_ui/chrome_traces/{job_id}/"""
    base = msmodeling_ui_dir()
    traces_dir = base / "chrome_traces" / job_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    return traces_dir


def trace_path(job_id: str, seq: int) -> Path:
    """Path to a case's trace file: .../chrome_traces/{job_id}/case_{seq}.json"""
    return trace_dir(job_id) / f"case_{seq}.json"


def legacy_hash_path(job_id: str, case_hash: str) -> Path:
    """Legacy worker-written path using case_hash: .../chrome_traces/{job_id}/{case_hash}.json

    Worker doesn't know seq, so uses case_hash as temp filename.
    Main process renames these to case_{seq}.json after seq assignment.
    """
    return trace_dir(job_id) / f"{case_hash}.json"


def copy_all_traces(src_job_id: str, dst_job_id: str) -> int:
    """Copy every trace file from ``src_job_id``'s dir into ``dst_job_id``'s dir.

    Used by Phase C cache hits, which reuse an entire prior run (seq is preserved
    1:1 by ``clone_records``, so ``case_{seq}.json`` names line up). Idempotent —
    a destination file that already exists is left as-is. Returns the count copied.
    """
    import shutil

    src_dir = trace_dir(src_job_id)
    if not src_dir.exists():
        return 0
    dst_dir = trace_dir(dst_job_id)
    copied = 0
    for src in src_dir.iterdir():
        if not src.is_file():
            continue
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def materialize_traces(job_id: str, records, skipped_hashes: set[str]) -> None:
    """Place each case's trace file at ``trace_path(job_id, i)`` (i = record index).

    Called after the worker run + case-dedup merge, BEFORE id/job_id/seq are
    reassigned (so ``record.job_id`` / ``record.seq`` still point at the source
    job for cached cases). For a cached case (``case_hash`` in ``skipped_hashes``)
    the trace is copied from its source job; for a fresh case the worker-written
    ``{case_hash}.json`` or ``{case_hash}_*.json`` (``legacy_hash_path``) is renamed into place. Idempotent
    — never overwrites an existing destination.

    For throughput_optimizer, ParallelRunner creates multiple trace files per case
    (one per parallel config: {case_hash}_tp{X}dp{Y}mtp{Z}.json). This function
    finds the first matching file and renames it to case_{seq}.json.
    """
    import shutil

    skipped = set(skipped_hashes)
    for i, record in enumerate(records):
        dst = trace_path(job_id, i)
        if record.case_hash in skipped:
            # Cached case: copy trace from its source job.
            src = trace_path(record.job_id, record.seq)
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        else:
            # Fresh case: rename the worker-written trace file.
            # First try exact match (text_generate style)
            tmp = legacy_hash_path(job_id, record.case_hash)
            if tmp.exists():
                if not dst.exists():
                    tmp.rename(dst)
                else:
                    # Idempotent re-run: destination already materialized, so drop
                    # the orphaned worker temp (mirrors the glob branch's
                    # ``not dst.exists()`` guard below — never overwrite).
                    tmp.unlink(missing_ok=True)
            else:
                # For throughput_optimizer: find files matching {case_hash}_*.json
                tmp_dir = tmp.parent
                case_hash = record.case_hash
                # Find all matching files (e.g., {case_hash}_tp1dp4mtp0.json)
                matches = sorted(tmp_dir.glob(f"{case_hash}_*.json"))
                if matches and not dst.exists():
                    # Copy the first match to case_{seq}.json
                    shutil.copy2(matches[0], dst)
