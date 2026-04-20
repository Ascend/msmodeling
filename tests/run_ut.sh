#!/bin/bash
set -euo pipefail

# ========================== CONFIGURATION ==========================
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_DIR=$(readlink -f "${SCRIPT_DIR}/..")
TENSOR_CAST_DIR="${PROJECT_DIR}/tests/"
SERVING_CAST_DIR="${PROJECT_DIR}/serving_cast/tests/ut"
COVERAGE_THRESHOLD=80

# Color codes
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
NC="\033[0m"


# ========================== UTILITIES ==========================
log() {
    local level=$1
    local msg=$2
    local color=""

    case "$level" in
        INFO)    color=$BLUE   ;;
        SUCCESS) color=$GREEN  ;;
        WARN)    color=$YELLOW ;;
        ERROR)   color=$RED    ;;
        *)       color=$NC     ;;
    esac

    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] ${color}[${level}]${NC} ${msg}"
}


run_module_tests() {
    # Accept space-separated module names and test dirs
    local module_names_str="$1"
    local test_dirs_str="$2"

    # Convert space-separated strings to arrays
    read -ra module_names <<< "$module_names_str"
    read -ra test_dirs <<< "$test_dirs_str"

    local upper_name
    upper_name=$(echo "$module_names_str" | tr '[:lower:]' '[:upper:]' | tr ' ' '_')

    log "INFO" "==================== RUNNING ${upper_name} TESTS ===================="

    # Validate all test directories exist
    for test_dir in "${test_dirs[@]}"; do
        if [ ! -d "${test_dir}" ]; then
            log "ERROR" "Test directory not found: ${test_dir}"
            exit 1
        fi
    done

    log "INFO" "Running tests with coverage (threshold: ${COVERAGE_THRESHOLD}%)..."
    log "INFO" "Test directories: ${test_dirs_str}"
    echo ""

    # Build --cov arguments for each module
    local cov_args=()
    for module_name in "${module_names[@]}"; do
        cov_args+=("--cov=${module_name}")
    done

    # Run tests with pytest-cov
    # --cov-fail-under exits with error if below threshold
    # Test files are omitted via pyproject.toml [tool.coverage.run] config
    local tmp_output
    tmp_output=$(mktemp)

    # Run tests with real-time output, also capture to temp file for parsing
    PYTHONPATH="${PROJECT_DIR}" python -m pytest "${test_dirs[@]}" \
        -n auto \
        -v \
        --tb=short \
        "${cov_args[@]}" \
        --cov-report=term-missing \
        --cov-fail-under=${COVERAGE_THRESHOLD} \
        --cov-branch 2>&1 | tee "$tmp_output"

    local exit_code=${PIPESTATUS[0]}
    echo ""

    # Extract actual coverage percentage from pytest-cov output
    local actual_coverage
    actual_coverage=$(grep 'TOTAL' "$tmp_output" | tail -1 | awk '{print $NF}' | tr -d '%')

    rm -f "$tmp_output"

    if [ $exit_code -eq 0 ]; then
        log "SUCCESS" "${module_names_str} coverage meets threshold ${COVERAGE_THRESHOLD}% (actual: ${actual_coverage}%)"
        return 0
    else
        log "ERROR" "${module_names_str} coverage is below threshold ${COVERAGE_THRESHOLD}% (actual: ${actual_coverage}%) or tests failed"
        return 1
    fi
}


run_tensor_cast_tests() {
    run_module_tests "tensor_cast" "${TENSOR_CAST_DIR}"
}


run_serving_cast_tests() {
    run_module_tests "serving_cast" "${SERVING_CAST_DIR}"
}


run_all_tests() {
    run_module_tests "tensor_cast serving_cast" "${TENSOR_CAST_DIR} ${SERVING_CAST_DIR}"
}


show_help() {
    echo "Usage: $0 <module>"
    echo "Run unit tests with coverage threshold check (${COVERAGE_THRESHOLD}%)"
    echo
    echo "Modules:"
    echo "  tensor_cast   Run tensor_cast tests (${TENSOR_CAST_DIR})"
    echo "  serving_cast  Run serving_cast tests (${SERVING_CAST_DIR})"
    echo "  all           Run both tensor_cast and serving_cast tests"
    echo
    echo "Examples:"
    echo "  $0 tensor_cast"
    echo "  $0 serving_cast"
    echo "  $0 all"
    echo
    echo "Exit codes:"
    echo "  0  All tests passed and coverage meets threshold"
    echo "  1  Tests failed or coverage below threshold"
}


# ========================== MAIN ==========================
main() {
    if [ $# -lt 1 ]; then
        log "ERROR" "Invalid arguments"
        show_help
        exit 1
    fi

    local module="$1"
    local exit_code=0

    case "$module" in
        tensor_cast)
            if ! run_tensor_cast_tests; then
                exit_code=1
            fi
            ;;
        serving_cast)
            if ! run_serving_cast_tests; then
                exit_code=1
            fi
            ;;
        all)
            if ! run_all_tests; then
                exit_code=1
            fi
            ;;
        *)
            log "ERROR" "Unknown module: $module"
            show_help
            exit 1
            ;;
    esac

    # Clean up temporary coverage files
    log "INFO" "Cleaning up coverage files..."
    find "${PROJECT_DIR}" -name ".coverage" -o -name ".coverage.*" | xargs rm -f 2>/dev/null || true
    log "INFO" "Coverage files cleaned up."

    exit $exit_code
}
main "$@"
