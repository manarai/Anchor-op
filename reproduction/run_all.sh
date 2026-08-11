#!/usr/bin/env bash
# Run every reproduction script in order. Skip scripts whose data deps are missing.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "$ROOT"

log() { printf "\n\033[1;34m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
warn() { printf "\n\033[1;33m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
fail() { printf "\n\033[1;31m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }

FAILED=()
SKIPPED=()

check_h5ad() {
    if [ ! -f "$1" ]; then
        warn "missing data: $1 — skipping"
        return 1
    fi
    return 0
}

for script in reproduction/[0-9]*.py; do
    name=$(basename "$script")
    log "running $name"

    case "$name" in
        02_*) check_h5ad "$ROOT/examples/data/K562_essential_normalized_singlecell_01.h5ad" >/dev/null 2>&1 \
             || { SKIPPED+=("$name (needs K562 aggregate raw 10x — edit DATA_ROOT in script)"); continue; } ;;
        03_*) check_h5ad "$ROOT/examples/data/K562_essential_normalized_singlecell_01.h5ad" \
             || { SKIPPED+=("$name (needs K562 essential h5ad)"); continue; } ;;
        04_*) check_h5ad "$ROOT/examples/data/rpe1_normalized_singlecell_01.h5ad" \
             || { SKIPPED+=("$name (needs RPE1 essential h5ad)"); continue; } ;;
        05_*|10_*)
             if [ ! -f "$ROOT/results/k562_essential_measurement.pkl" ] \
                || [ ! -f "$ROOT/results/rpe1_essential_measurement.pkl" ]; then
                 SKIPPED+=("$name (needs measurement bundles — run 03 and 04 first)"); continue
             fi ;;
        09_*) check_h5ad "$ROOT/examples/data/rpe1_normalized_singlecell_01.h5ad" \
             || { SKIPPED+=("$name (needs RPE1 h5ad + Jost 2020 data)"); continue; }
             check_h5ad "$ROOT/examples/data/jost2020/GSE132080_10X_matrix.mtx.gz" \
             || { SKIPPED+=("$name (needs Jost 2020 GSE132080 downloaded)"); continue; } ;;
    esac

    if ! python3 "$script"; then
        FAILED+=("$name")
        fail "$name FAILED"
    fi
done

log "SUMMARY"
[ ${#SKIPPED[@]} -gt 0 ] && printf "  skipped: %d\n" "${#SKIPPED[@]}" && printf "    %s\n" "${SKIPPED[@]}"
[ ${#FAILED[@]} -gt 0 ] && fail "failed: ${FAILED[*]}" && exit 1
log "all requested figures generated in manuscript_figures/"
