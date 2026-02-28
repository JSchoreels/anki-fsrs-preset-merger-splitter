#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="/Users/jschoreels/Library/Application Support/Anki2/addons21/anki-fsrs-preset-merger-splitter"

mkdir -p "$TARGET_DIR"

STAGE_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

cp "$SOURCE_DIR/__init__.py" "$STAGE_DIR/"
cp "$SOURCE_DIR/manifest.json" "$STAGE_DIR/"
rsync -a \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$SOURCE_DIR/fsrs_merge_advisor/" "$STAGE_DIR/fsrs_merge_advisor/"

rsync -a --delete "$STAGE_DIR/" "$TARGET_DIR/"

echo "Addon synced to: $TARGET_DIR"
