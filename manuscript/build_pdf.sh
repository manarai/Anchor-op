#!/usr/bin/env bash
# Build the manuscript PDF from MANUSCRIPT.md.
#
# Requirements (install once):
#   mamba install -y -c conda-forge pandoc tectonic
#
# STIX Two Text is bundled with macOS. On Linux install `stix2` (Fedora) or
# `fonts-stix` (Debian/Ubuntu). Swap `--variable mainfont` for any other
# system font with adequate Latin + Greek coverage (Cambria, Charter, etc.).
#
# The header include (`pandoc_header.tex`) remaps a handful of Unicode
# symbols (arrows, math relations, superscripts, PASS/FAIL check marks)
# to their LaTeX equivalents so the built PDF has no missing-character
# warnings.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
in="$repo/MANUSCRIPT.md"
out="$here/anchor-op_preprint_v0.3.1.pdf"

pandoc "$in" \
  --pdf-engine=tectonic \
  --resource-path="$repo" \
  --number-sections \
  -V geometry:margin=1in \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V fontsize=11pt \
  -V mainfont="STIX Two Text" \
  -V monofont="Menlo" \
  -H "$here/pandoc_header.tex" \
  -o "$out"

echo "wrote: $out"
