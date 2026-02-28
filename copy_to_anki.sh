#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="/Users/jschoreels/Library/Application Support/Anki2/addons21/anki-fsrs-preset-merger-splitter"

mkdir -p "$TARGET_DIR"

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$SOURCE_DIR/" "$TARGET_DIR/"

echo "Addon synced to: $TARGET_DIR"
