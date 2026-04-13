#!/bin/bash
set -euo pipefail

# ========================== CONFIGURATION ==========================
# All paths are resolved relative to the script location
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
COV_DIR=${SCRIPT_DIR}/coverage
PROJECT_DIR=$(readlink -f ${SCRIPT_DIR}/../..)
UT_DIR="${SCRIPT_DIR}/ut"          # Unit-test directory (unittest)
ST_DIR="${SCRIPT_DIR}/st"          # System-test directory
MAIN_PY_PATH="${SCRIPT_DIR}/../main.py"  # Entry point required by ST

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

check_main_py() {
    if [ ! -f "${MAIN_PY_PATH}" ]; then
        log "ERROR" "Main program not found: ${MAIN_PY_PATH}"
        log "ERROR" "Ensure main.py is located in the expected path"
        exit 1
    fi
}

run_ut() {
    log "INFO" "==================== STARTING UNIT TESTS (UT) ===================="
    log "INFO" "UT directory: ${UT_DIR}"

    if [ ! -d "${UT_DIR}" ]; then
        log "ERROR" "UT directory missing: ${UT_DIR}"
        exit 1
    fi

    if [ ! -d ${COV_DIR} ]; then
        mkdir -p ${COV_DIR}
    fi
    # Find all test files recursively, including subdirectories
    ut_files=$(find "${UT_DIR}" -type f -name "test_*.py" ! -path "*/__pycache__/*")
    if [ -z "${ut_files}" ]; then
        log "WARN" "No UT files found (names must start with test_)"
        return 0
    fi

    # Run discovery recursively
    PYTHONPATH=$PROJECT_DIR python -m coverage run \
        --branch \
        --source "${PROJECT_DIR}/serving_cast" \
        --omit="*/tests/*,*/test_*,*/__pycache__/*,*/.pytest_cache/*" \
        -m pytest ${UT_DIR}
    python -m coverage report -m
    python -m coverage xml -o ${COV_DIR}/coverage.xml

    if [ $? -eq 0 ]; then
        log "SUCCESS" "==================== ALL UNIT TESTS PASSED ===================="
    else
        log "ERROR" "==================== UNIT TESTS FAILED ===================="
        exit 1
    fi
}

run_st() {
    log "INFO" "==================== STARTING SYSTEM TESTS (ST) ===================="
    log "INFO" "ST directory: ${ST_DIR}"

    check_main_py
    if ! python -c "import pytest" &>/dev/null; then
        log "ERROR" "pytest is required but not installed"
        log "ERROR" "Install it first: pip install pytest"
        exit 1
    fi

    if [ ! -d "${ST_DIR}" ]; then
        log "ERROR" "ST directory missing: ${ST_DIR}"
        exit 1
    fi

    st_files=$(find "${ST_DIR}" -type f -name "test_*.py")
    if [ -z "${st_files}" ]; then
        log "WARN" "No ST files found (names must start with test_)"
        return 0
    fi

    PYTHONPATH=$PROJECT_DIR python -m pytest "${ST_DIR}" -v -s

    if [ $? -eq 0 ]; then
        log "SUCCESS" "==================== ALL SYSTEM TESTS PASSED ===================="
    else
        log "ERROR" "==================== SYSTEM TESTS FAILED ===================="
        exit 1
    fi
}

show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Run unit tests (UT) or system tests (ST)"
    echo
    echo "Options:"
    echo "  ut     Run unit tests only (unittest)"
    echo "  st     Run system tests only (pytest)"
    echo "  all    Run UT first, then ST (recommended)"
    echo "  help   Show this help"
    echo
    echo "Examples:"
    echo "  $0 ut"
    echo "  $0 st"
    echo "  $0 all"
}

# ========================== MAIN ==========================
main() {
    if [ $# -ne 1 ]; then
        log "ERROR" "Invalid arguments"
        show_help
        exit 1
    fi

    case "$1" in
        ut)
            run_ut
            ;;
        st)
            run_st
            ;;
        all)
            run_ut
            log "INFO" "UT finished, waiting 2 s before ST..."
            sleep 2
            run_st
            ;;
        help)
            show_help
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac

    log "SUCCESS" "Selected tests completed!"
}

main "$@"