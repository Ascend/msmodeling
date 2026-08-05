# Shared setup for CodeArts entry scripts.
# Caller must set SCRIPT_DIR to scripts/ before sourcing; HELPERS_DIR is scripts/helpers/.
#
# When uv is available and PYTHON is unset, sources common.sh runs:
#   uv sync --frozen --group ci
#
# Environment set or consumed here:
#   PYTHONPATH   Exported to repository root (PROJECT_DIR); overrides any prior value
#   PYTHON       Optional absolute path to interpreter; if unset, uses uv or python3

# Guard against multiple inclusion
[[ -n "${COMMON_SH_LOADED:-}" ]] && return 0
COMMON_SH_LOADED=1

: "${SCRIPT_DIR:?SCRIPT_DIR must be set before sourcing common.sh}"

# Resolve project directory with error handling
PROJECT_DIR=$(readlink -f "${SCRIPT_DIR}/..") || {
  echo "Error: Failed to resolve PROJECT_DIR from ${SCRIPT_DIR}" >&2
  exit 1
}
[[ -d "${PROJECT_DIR}" ]] || {
  echo "Error: PROJECT_DIR '${PROJECT_DIR}' does not exist" >&2
  exit 1
}

HELPERS_DIR="${SCRIPT_DIR}/helpers"
TESTS_SMOKE="${PROJECT_DIR}/tests/smoke"
TESTS_REGRESSION="${PROJECT_DIR}/tests/regression"
TESTS_BENCHMARK="${PROJECT_DIR}/tests/benchmark"

# pytest-xdist: worksteal balances workers when case durations vary widely.
PYTEST_XDIST_ARGS=(-n auto --dist=worksteal)

# Apply scripts/defaults.env with setdefault (never override user-exported values).
# Grammar MUST match scripts/helpers/defaults.py::parse_defaults_env_line:
#   - empty lines ignored
#   - '#' starts a comment through end of line (including mid-line)
#   - KEY=VALUE; trim whitespace; values must not contain '#'
_DEFAULTS_ENV="${SCRIPT_DIR}/defaults.env"
if [[ -f "${_DEFAULTS_ENV}" ]]; then
  while IFS= read -r _line || [[ -n "${_line}" ]]; do
    _line="${_line%%#*}"
    _line="${_line#"${_line%%[![:space:]]*}"}"
    _line="${_line%"${_line##*[![:space:]]}"}"
    [[ -z "${_line}" || "${_line}" != *=* ]] && continue
    _key="${_line%%=*}"
    _val="${_line#*=}"
    # Trim key and value (same as defaults.py strip on both sides).
    _key="${_key#"${_key%%[![:space:]]*}"}"
    _key="${_key%"${_key##*[![:space:]]}"}"
    _val="${_val#"${_val%%[![:space:]]*}"}"
    _val="${_val%"${_val##*[![:space:]]}"}"
    [[ -z "${_key}" ]] && continue
    case "${_key}" in
      UV_INDEX_URL|HF_ENDPOINT|MSMODELING_TEST_BASE_BRANCH|MSMODELING_CACHE)
        if [[ -z "${!_key:-}" ]]; then
          export "${_key}=${_val}"
        fi
        ;;
    esac
  done < "${_DEFAULTS_ENV}"
  unset _line _key _val
fi

export PYTHONPATH="${PROJECT_DIR}"
# Determine if uv should be used (cache the result)
USE_UV=false
if [[ -z "${PYTHON:-}" ]] && command -v uv >/dev/null 2>&1 && [[ -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  USE_UV=true
fi

# Set Python runner
if [[ -n "${PYTHON:-}" ]]; then
  RUN_PY=("${PYTHON}")
elif $USE_UV; then
  RUN_PY=(uv run python)
else
  RUN_PY=(python3)
fi

# Set pytest runner
if $USE_UV; then
  RUN_PYTEST=(uv run pytest)
else
  RUN_PYTEST=(python3 -m pytest)
fi

# Wrapper functions with basic error handling
run_py() {
  if [[ ${#RUN_PY[@]} -eq 0 ]]; then
    echo "Error: RUN_PY is not properly initialized" >&2
    return 1
  fi
  "${RUN_PY[@]}" "$@"
}

run_pytest() {
  if [[ ${#RUN_PYTEST[@]} -eq 0 ]]; then
    echo "Error: RUN_PYTEST is not properly initialized" >&2
    return 1
  fi
  "${RUN_PYTEST[@]}" "$@"
}

if $USE_UV; then
  (
    cd "${PROJECT_DIR}"
    uv sync --frozen --group ci
  )
fi