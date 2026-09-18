#!/bin/bash
# -------------------------------------------------------------------------
# Leftover process cleanup script
#
# Input convention (both channels carry the same value; use whichever you prefer):
#   - the first argument $1
#   - the environment variable MODEL_EVAL_STATE_KILL_PROCESS_NAME
#   Both are comma-separated process name strings, for example:
#       "mindie, mindie-llm, mindieservice_daemon, mindie_llm"
#   A comma may be followed by spaces, so the script trims surrounding whitespace itself.
#
# Note: this script is run by bash (no executable bit or shebang required), and any
#       non-zero exit code is only recorded in the log; it never interrupts the main flow.
#
# Matching scope: `pgrep -i -f` does a case-insensitive substring match on the full
# command line, so any unrelated process whose command line merely contains the target
# name -- e.g. `vim vllm_config.py`, `grep vllm`, `tail -f /var/log/vllm.log` -- is
# also matched and killed. The exclusions below only protect this script's own process
# tree, not third-party processes. This script therefore assumes it runs on a dedicated
# worker node where no interactive/diagnostic processes are expected to run alongside
# the inference workload.
# -------------------------------------------------------------------------
set -u

names="${1:-${MODEL_EVAL_STATE_KILL_PROCESS_NAME:-}}"
if [ -z "${names}" ]; then
    echo "no process name provided, skip"
    exit 0
fi

# Maximum number of retries, to prevent an infinite loop
MAX_RETRY=5
# Wait time (seconds) after each kill, giving the driver time to reclaim NPU memory
WAIT_SECONDS=2

# Collect this script's own PID plus every ancestor, and never kill them.
#
# The match is done on the full command line (-f), and the process names we are asked to
# clean up also appear in our own command line: the script lives under /tmp/vllm and takes
# the name as an argument, so "vllm" matches `bash /tmp/vllm/stop_vllm_process.sh vllm` and
# the `sh -c` wrapper above it. Without this exclusion the first pkill kills the script
# itself before it can verify anything, which leaves the caller's SSH/docker exec hanging
# until its timeout.
#
# PID 1 is excluded as well: in Docker mode this runs inside the container, and killing its
# init tears the whole container down.
protected_pids="1"
_ancestor=$$
while [ -n "${_ancestor}" ] && [ "${_ancestor}" != "0" ] && [ "${_ancestor}" != "1" ]; do
    protected_pids="${protected_pids} ${_ancestor}"
    # /proc/PID/stat field 2 (comm) may contain spaces and parentheses; drop everything up
    # to the final ')' first, after which field 2 of the remainder is ppid.
    _ancestor="$(sed -e 's/^.*) //' "/proc/${_ancestor}/stat" 2>/dev/null | cut -d' ' -f2)"
done

# PID exclusion alone is not enough: `$(pgrep ...)` forks a subshell that inherits this
# script's command line, so it matches -f as well, under a PID that is new every iteration
# and therefore never in protected_pids. Skip anything whose command line names this script,
# which covers that subshell, the `sh -c` wrapper, and the script itself.
self_script="${0##*/}"

# Read a process's command line, or return non-zero when it can no longer be read (the
# process exited, or it is a kernel thread with an empty cmdline). The redirection is wrapped
# in a subshell because the shell reports a missing /proc/PID/cmdline itself, before `tr`
# runs, so a 2>/dev/null on `tr` alone would not suppress it.
read_cmdline() {
    _cmd="$( (tr '\0' ' ' < "/proc/$1/cmdline") 2>/dev/null )"
    [ -n "${_cmd}" ]
}

# Collect the PIDs matching one name that are not part of this script's own process tree.
# -i ignores case (so vllm also matches VLLM::EngineCore); -f matches the full command line.
find_targets() {
    _found=""
    for _pid in $(pgrep -i -f "$1" 2>/dev/null); do
        _skip=""
        for _p in ${protected_pids}; do
            if [ "${_pid}" = "${_p}" ]; then
                _skip="y"
                break
            fi
        done
        [ -n "${_skip}" ] && continue

        # A PID that is already gone by the time we look at it needs no killing. Treating it
        # as a live target instead would keep the retry loop busy for every attempt and end
        # in a spurious "still exists" warning.
        read_cmdline "${_pid}" || continue
        case "${_cmd}" in
            *"${self_script}"*) continue ;;
        esac

        _found="${_found} ${_pid}"
    done
    echo "${_found}"
}

# Split on commas and kill each one in turn
old_ifs="${IFS}"
IFS=','
for raw in ${names}; do
    IFS="${old_ifs}"
    # Trim surrounding whitespace
    name="$(echo "${raw}" | xargs)"
    [ -z "${name}" ] && continue

    retry=0
    targets="$(find_targets "${name}")"
    while [ -n "${targets}" ] && [ ${retry} -lt ${MAX_RETRY} ]; do
        # Kill the PIDs individually rather than via pkill, so this script's own process tree
        # can be filtered out first (see find_targets above).
        echo "killing residual process matching: ${name} (attempt $((retry + 1))/${MAX_RETRY}):${targets}"
        kill -9 ${targets} 2>/dev/null || true
        sleep ${WAIT_SECONDS}
        retry=$((retry + 1))
        # Re-check after the kill, including after the final attempt, so the warning below
        # only fires when something genuinely survived.
        targets="$(find_targets "${name}")"
    done

    if [ -n "${targets}" ]; then
        echo "WARNING: process matching '${name}' still exists after ${MAX_RETRY} attempts:${targets}"
    else
        echo "no residual process matching: ${name}"
    fi
done
IFS="${old_ifs}"

exit 0
