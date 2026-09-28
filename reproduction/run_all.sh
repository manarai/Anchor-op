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
        # Recheck scripts (audit response, 2026-09-27).
        30_*|33_*|34_*|36_*)
             if [ ! -f "$ROOT/results/k562_essential_measurement.pkl" ] \
                || [ ! -f "$ROOT/results/rpe1_essential_measurement.pkl" ] \
                || [ ! -f "$ROOT/results/jost_u_at_d30.pkl" ]; then
                 SKIPPED+=("$name (recheck needs the results/*.pkl bundles)"); continue
             fi ;;
        35_*) # F7 legacy multi-mode selection sensitivity — superseded by 42.
             SKIPPED+=("$name (superseded by reproduction/42_f7_k562_random200.py; manual)"); continue ;;
        42_*) # F7 K562 random-200 refit — heavy (10 GB h5ad load).
             if [ ! -f "$ROOT/examples/data/K562_essential_normalized_singlecell_01.h5ad" ]; then
                 SKIPPED+=("$name (needs Replogle K562 essential h5ad)"); continue
             fi
             SKIPPED+=("$name (F7 K562 random-200 is manual: python reproduction/42_f7_k562_random200.py)"); continue ;;
        37_*) # F3 end-to-end on Jost — refits from raw counts; heavy.
             if [ ! -f "$ROOT/examples/data/jost2020/GSE132080_10X_matrix.mtx.gz" ]; then
                 SKIPPED+=("$name (needs Jost 2020 GSE132080 counts)"); continue
             fi
             SKIPPED+=("$name (F3-Jost refit is manual: python reproduction/37_recheck_F3_jost_end_to_end.py)"); continue ;;
        39_*|41_*) # Jost target-grouped folds + d-sweep — read results/jost_measurement.pkl, minutes.
             if [ ! -f "$ROOT/results/jost_measurement.pkl" ]; then
                 SKIPPED+=("$name (needs results/jost_measurement.pkl — run 37 first)"); continue
             fi ;;
        40_*) # Nested-CV on all three — reads Replogle + Jost pkls, minutes.
             if [ ! -f "$ROOT/results/k562_essential_measurement.pkl" ] \
                || [ ! -f "$ROOT/results/rpe1_essential_measurement.pkl" ] \
                || [ ! -f "$ROOT/results/jost_measurement.pkl" ]; then
                 SKIPPED+=("$name (needs all three measurement pkls; run 37 for Jost first)"); continue
             fi ;;
        43_*) # Interaction-only cosine at matched α — reads Replogle pkls, seconds.
             if [ ! -f "$ROOT/results/k562_essential_measurement.pkl" ] \
                || [ ! -f "$ROOT/results/rpe1_essential_measurement.pkl" ]; then
                 SKIPPED+=("$name (needs Replogle measurement pkls)"); continue
             fi ;;
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
