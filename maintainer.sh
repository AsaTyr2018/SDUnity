#!/bin/bash

# Maintainer script for SDUnity
# Usage: sudo ./maintainer.sh <install|update|uninstall>

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo." >&2
    exit 1
fi

TARGET_DIR="/opt/SDUnity"
VENV_DIR="$TARGET_DIR/venv"

# Repository to clone from when installing. Use HTTPS to avoid SSH key issues
REPO_URL="https://github.com/AsaTyr2018/SDUnity.git"

# Always clone via HTTPS even if the script lives inside a git repository
SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_SRC=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)
if [ -n "$REPO_SRC" ]; then
    LOCAL_REMOTE=$(git -C "$REPO_SRC" config --get remote.origin.url 2>/dev/null || true)
    if [[ "$LOCAL_REMOTE" =~ ^git@github.com:(.*)\.git$ ]]; then
        REPO_URL="https://github.com/${BASH_REMATCH[1]}.git"
    elif [[ "$LOCAL_REMOTE" =~ ^https://github.com/.*\.git$ ]]; then
        REPO_URL="$LOCAL_REMOTE"
    fi
fi

check_deps() {
    for cmd in git python3; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "Error: $cmd is required but not installed." >&2
            exit 1
        fi
    done
}

ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
}

fix_permissions() {
    local user="${SUDO_USER:-$(whoami)}"
    chown -R "$user":"$user" "$TARGET_DIR"
    chmod -R u+rwX,go+rX "$TARGET_DIR"
}

install_sdunity() {
    check_deps
    if [ -d "$TARGET_DIR/.git" ]; then
        echo "SDUnity already installed in $TARGET_DIR"
    else
        echo "Cloning SDUnity into $TARGET_DIR"
        git clone "$REPO_URL" "$TARGET_DIR"
    fi
    ensure_venv
    "$VENV_DIR/bin/pip" install -r "$TARGET_DIR/requirements.txt"
    fix_permissions
    echo "Installation completed."
}

update_sdunity() {
    check_deps
    if [ ! -d "$TARGET_DIR/.git" ]; then
        echo "SDUnity is not installed in $TARGET_DIR" >&2
        exit 1
    fi
    cd "$TARGET_DIR"
    git pull
    ensure_venv
    "$VENV_DIR/bin/pip" install -r requirements.txt
    fix_permissions
    echo "Update completed."
}

uninstall_sdunity() {
    if [ -d "$TARGET_DIR" ]; then
        rm -rf "$TARGET_DIR"
        echo "Removed $TARGET_DIR"
    else
        echo "SDUnity not found at $TARGET_DIR" >&2
    fi
}

case "$1" in
    install)
        install_sdunity
        ;;
    update)
        update_sdunity
        ;;
    uninstall)
        uninstall_sdunity
        ;;
    *)
        echo "Usage: $0 {install|update|uninstall}" >&2
        exit 1
        ;;
 esac

