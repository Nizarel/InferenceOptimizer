#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

usage() {
  cat >&2 <<'EOF'
Usage: run_phase_a.sh [MODEL_KEY ...]
Runs the single-DGX serving sweep. With no arguments, runs all four models.
Useful overrides: RESULT_ROOT=... REPEATS=... CONCURRENCY_LEVELS="1 2 4"
EOF
}
[[ "${1:-}" != "--help" ]] || { usage; exit 0; }

require_tools
mkdir -p "$RESULT_ROOT"
printf '%s\n' "phase_a" > "$RESULT_ROOT/phase.txt"
printf 'model\tpath\tserved_name\n' > "$RESULT_ROOT/model_registry.tsv"

if (( $# > 0 )); then
  keys=("$@")
else
  keys=("${MODEL_KEYS[@]}")
fi
for key in "${keys[@]}"; do
  model_path "$key" >/dev/null
  printf '%s\t%s\t%s\n' "$key" "$(model_path "$key")" "$(served_name "$key")" >> "$RESULT_ROOT/model_registry.tsv"
  "$SCRIPT_DIR/bench_one.sh" "$key" "$RESULT_ROOT/$key"
done

log "Phase A complete: $RESULT_ROOT"
