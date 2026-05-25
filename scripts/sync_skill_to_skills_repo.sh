#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/skills/matters"
TARGET_DIR="${1:-/Users/matthias/code/SKILLS/matters}"
LEGACY_STATE="$TARGET_DIR/scripts/matters.json"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

if [[ ! -f "$SOURCE_DIR/SKILL.md" ]]; then
  echo "Source skill not found: $SOURCE_DIR" >&2
  exit 1
fi

preserve_legacy_state() {
  local destination="${MATTERS_STATE:-$HOME/.local/share/matters/matters.json}"
  local backup_path

  [[ -f "$LEGACY_STATE" ]] || return 0

  mkdir -p "$(dirname "$destination")"

  if [[ ! -f "$destination" ]]; then
    cp "$LEGACY_STATE" "$destination"
    echo "Preserved legacy skill state at: $destination"
    return 0
  fi

  backup_path="${destination}.legacy-from-skill.${TIMESTAMP}"
  cp "$LEGACY_STATE" "$backup_path"
  echo "Preserved legacy skill state backup at: $backup_path"
}

mkdir -p "$TARGET_DIR"
preserve_legacy_state

rsync -a --delete \
  --delete-excluded \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$SOURCE_DIR/" \
  "$TARGET_DIR/"

cat <<EOF
Synced matters skill:
  $SOURCE_DIR
  -> $TARGET_DIR

Next steps:
  cd "$(dirname "$TARGET_DIR")"
  git status --short "$TARGET_DIR"
EOF
