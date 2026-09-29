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
out="$here/anchor-op_preprint_v0.3.2.pdf"

pandoc "$in" \
  --pdf-engine=tectonic \
  --resource-path="$repo" \
  -V geometry:margin=1in \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V fontsize=11pt \
  -V mainfont="STIX Two Text" \
  -V monofont="Menlo" \
  -H "$here/pandoc_header.tex" \
  -o "$out"

# Notes:
# - --number-sections is deliberately OFF; MANUSCRIPT.md numbers its own
#   sections (§2.1, §3.2, etc.) so pandoc's auto-numbering would double up.
# - Empty ![](path) alt text on figures avoids pandoc inserting duplicate
#   "Figure N: Figure N" labels above the caption line.

echo "wrote: $out"
