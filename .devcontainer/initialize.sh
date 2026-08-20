#!/usr/bin/env bash
# Runs on the host before the dev container is created.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_GITCONFIG="$SCRIPT_DIR/.host-gitconfig"

# Export only the public Git author identity needed inside the container.
: >"$HOST_GITCONFIG"
git_name="$(git config --global --get user.name 2>/dev/null || true)"
git_email="$(git config --global --get user.email 2>/dev/null || true)"
if [[ -n "$git_name" ]]; then
    git config -f "$HOST_GITCONFIG" user.name "$git_name"
fi
if [[ -n "$git_email" ]]; then
    git config -f "$HOST_GITCONFIG" user.email "$git_email"
fi
chmod 600 "$HOST_GITCONFIG"

# Pull the latest image layer before the container is created. The image tag is
# read from devcontainer.json so it stays in sync with that single source of truth.
IMAGE="$(python3 -c "
import re, json
with open('$SCRIPT_DIR/devcontainer.json') as f:
    text = re.sub(r'//.*$|/\*[\s\S]*?\*/', '', f.read(), flags=re.MULTILINE)
    print(json.loads(text)['image'])
")"
echo "==> Pulling image: $IMAGE"
docker pull "$IMAGE"

# Bind-mount sources must exist before Docker creates the container.
mkdir -p "$HOME/.cache/uv" "$HOME/.cache/pre-commit"
chmod 0755 "$HOME/.cache/uv" "$HOME/.cache/pre-commit"
