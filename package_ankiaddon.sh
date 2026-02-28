#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADDON_NAME="anki-fsrs-preset-merger-splitter"
OUT_FILE="${1:-$ROOT_DIR/dist/${ADDON_NAME}.ankiaddon}"

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' is required but not installed." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_FILE")"
rm -f "$OUT_FILE"

STAGE_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

cp "$ROOT_DIR/__init__.py" "$STAGE_DIR/"
cp "$ROOT_DIR/manifest.json" "$STAGE_DIR/"
rsync -a \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$ROOT_DIR/fsrs_merge_advisor/" "$STAGE_DIR/fsrs_merge_advisor/"

(
  cd "$STAGE_DIR"
  zip -qr "$OUT_FILE" __init__.py manifest.json fsrs_merge_advisor
)

echo "Created: $OUT_FILE"
