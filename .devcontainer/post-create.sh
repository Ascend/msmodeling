#!/usr/bin/env bash
# Initializes the msmodeling development environment after container creation.
# Each step reports its own failure so one unavailable tool does not prevent the
# developer from entering the container and repairing the environment.

set -uo pipefail

log() { printf '[post-create] %s\n' "$*"; }
warn() { printf '[post-create] WARN: %s\n' "$*" >&2; }

run_step() {
    local title="$1"
    local function_name="$2"
    log "START: $title"
    if "$function_name"; then
        log "DONE: $title"
    else
        local exit_code=$?
        warn "FAILED: $title (exit $exit_code)"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
PYTHON_BIN=""

append_line_once() {
    local file="$1"
    local line="$2"
    touch "$file"
    grep -Fqx "$line" "$file" || printf '\n%s\n' "$line" >>"$file"
}

configure_user_paths() {
    mkdir -p "$HOME/.local/bin"
    append_line_once "$HOME/.bashrc" 'export PATH="$HOME/.local/bin:$PATH"'
    append_line_once "$HOME/.bash_profile" 'export PATH="$HOME/.local/bin:$PATH"'

    if command -v npm >/dev/null 2>&1; then
        npm config set prefix "$HOME/.local"
    else
        warn "npm is unavailable; skipping npm prefix configuration"
    fi
}

append_python311_once() {
    local file="$1"
    local marker="# msmodeling devcontainer: Python 3.11"
    touch "$file"
    grep -Fqx "$marker" "$file" && return 0
    cat >>"$file" <<'EOF'

# msmodeling devcontainer: Python 3.11
if [ -r /etc/profile.d/z_python_switch.sh ]; then
    . /etc/profile.d/z_python_switch.sh
fi
if [ -r /usr/local/bin/use-python ]; then
    . /usr/local/bin/use-python 3.11 >/dev/null 2>&1 || true
fi
EOF
}

configure_python311() {
    if [[ -r /etc/profile.d/z_python_switch.sh ]]; then
        # shellcheck disable=SC1091
        . /etc/profile.d/z_python_switch.sh
    fi
    if [[ -r /usr/local/bin/use-python ]]; then
        # The image exposes use-python as a shell helper.
        # shellcheck disable=SC1091
        . /usr/local/bin/use-python 3.11 >/dev/null 2>&1 || true
    fi

    append_python311_once "$HOME/.bashrc"
    append_python311_once "$HOME/.bash_profile"

    if command -v python3.11 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3.11)"
    elif PYTHON_BIN="$(uv python find 3.11 2>/dev/null)"; then
        :
    else
        uv python install 3.11 || return 1
        PYTHON_BIN="$(uv python find 3.11)" || return 1
    fi

    [[ "$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]]
}

sync_git_identity() {
    local host_gitconfig="/tmp/host-gitconfig"
    local git_name=""
    local git_email=""

    if [[ -s "$host_gitconfig" ]]; then
        git_name="$(git config -f "$host_gitconfig" --get user.name 2>/dev/null || true)"
        git_email="$(git config -f "$host_gitconfig" --get user.email 2>/dev/null || true)"
    fi
    git_name="${git_name:-${MSMODELING_GIT_USER_NAME:-}}"
    git_email="${git_email:-${MSMODELING_GIT_USER_EMAIL:-}}"

    local missing=""
    if [[ -n "$git_name" ]]; then
        git config --global user.name "$git_name"
    else
        missing="user.name"
    fi
    if [[ -n "$git_email" ]]; then
        git config --global user.email "$git_email"
    else
        missing="${missing:+$missing, }user.email"
    fi

    if [[ -n "$missing" ]]; then
        warn "host Git identity unavailable: $missing"
        printf '\n%s\n' '==============================================================' >&2
        printf '%s\n' '[post-create] ⚠️  缺少 Git 身份配置' >&2
        printf '%s\n' "[post-create]    缺失项: $missing" >&2
        printf '%s\n' '[post-create]    提交信息将缺失作者身份，且 pre-commit 可能无法正常运行。' >&2
        printf '%s\n' '[post-create]    请在宿主机配置后再重建容器：' >&2
        printf '%s\n' '[post-create]      git config --global user.name  "你的名字"' >&2
        printf '%s\n' '[post-create]      git config --global user.email "你的邮箱"' >&2
        printf '%s\n' '[post-create]    或在容器内直接执行： git config --global user.name/email' >&2
        printf '%s\n' '==============================================================' >&2
    fi
}

prepare_development_dependencies() {
    [[ -n "$PYTHON_BIN" ]] || {
        warn "Python 3.11 is unavailable; skipping dependency preparation"
        return 1
    }
    UV_PYTHON="$PYTHON_BIN" "$PYTHON_BIN" build.py -e only_down_deps=true
}

install_pre_commit_hook() {
    [[ -x .venv/bin/pre-commit ]] || {
        warn ".venv/bin/pre-commit is unavailable"
        return 1
    }
    [[ -f .pre-commit-config.yaml ]] || {
        warn ".pre-commit-config.yaml is unavailable"
        return 1
    }
    .venv/bin/pre-commit install
}

run_step "configure user paths and npm prefix" configure_user_paths
run_step "activate Python 3.11" configure_python311
run_step "synchronize host Git identity" sync_git_identity
run_step "prepare development dependencies through build.py" prepare_development_dependencies
run_step "install pre-commit hook" install_pre_commit_hook

log "initialization finished; review warnings above before development"
exit 0
