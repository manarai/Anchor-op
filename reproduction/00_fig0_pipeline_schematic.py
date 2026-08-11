"""Fig 0: anchor-op pipeline flowchart.

Matplotlib line-art showing the data → basis → S,U → J·P_X pipeline plus the
two gates (rank_tol, linearity). No data required. Runtime <5 s.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)


def box(ax, xy, w, h, text, fc="#e8eef6", ec="#1f4e79", fs=9):
    rect = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                   linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True)


def diamond(ax, xy, w, h, text, fc="#fef4e5", ec="#c65a30", fs=9):
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    diamond_pts = [[cx, cy + h / 2], [cx + w / 2, cy], [cx, cy - h / 2], [cx - w / 2, cy]]
    poly = patches.Polygon(diamond_pts, closed=True, linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, start, end, text=None, color="#333"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=1.4, color=color))
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.15, text,
                ha="center", va="bottom", fontsize=8, color=color, style="italic")


fig, ax = plt.subplots(figsize=(11, 8.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 12); ax.axis("off")

# Row 1: inputs
box(ax, (0.5, 10.5), 3.5, 1.0, "Perturb-seq h5ad\n(cells × genes + guide, target)", fc="#f2ecdc")
box(ax, (5, 10.5), 3.5, 1.0, "Program basis W\n(d-dim, control-fit)", fc="#f2ecdc")

# Row 2: assemble S, U
arrow(ax, (2.25, 10.5), (2.25, 9.6))
arrow(ax, (6.75, 10.5), (6.75, 9.6))
box(ax, (0.5, 8.4), 3.5, 1.2,
    r"S ($d\times m$): per-guide $\Delta z$"
    "\n(rows = programs, cols = guides)")
box(ax, (5, 8.4), 3.5, 1.2,
    r"U ($d\times m$): per-guide input"
    "\n$u_g = -\\kappa_g \\cdot W^T \\delta_g$")

# Efficiency estimation note
box(ax, (9, 8.4), 4.5, 1.2,
    "κ from auto-router:\ncount data → mean_ratio (MLE)\nresidual data → detection_rate shift",
    fc="#e5f3ec", ec="#2b6a3f", fs=8)
arrow(ax, (9, 9.0), (8.5, 9.0))

# Row 3: filter
arrow(ax, (2.25, 8.4), (2.25, 7.5))
arrow(ax, (6.75, 8.4), (6.75, 7.5))
box(ax, (2, 6.4), 5, 1.0,
    "min_control_detection_rate filter (count data only):\ndrop information-limited targets before fit",
    fc="#f0e8f5", ec="#7a4d95", fs=8)

# Row 4: pseudoinverse
arrow(ax, (4.5, 6.4), (4.5, 5.6))
box(ax, (2, 4.4), 5, 1.2,
    r"Regularized inversion: $J \cdot P_X = -U S^+$"
    "\n(TSVD, rank guarded by rank_tol)")

# Gate 1
arrow(ax, (4.5, 4.4), (4.5, 3.6))
diamond(ax, (2.5, 2.2), 4, 1.4,
        "Gate 1 (rank_tol guard):\neffective_response_rank == d ?",
        fc="#fef4e5", ec="#c65a30")

# Gate 1 branches
arrow(ax, (2.5, 2.9), (0.8, 2.9), text="no", color="#c65a30")
box(ax, (0.2, 1.4), 1.5, 1.0,
    "partial\nidentification;\n.J blocked",
    fc="#fde4d4", ec="#c65a30", fs=8)

arrow(ax, (4.5, 2.2), (4.5, 1.6), text="yes", color="#2b6a3f")

# Linearity diagnostics with power-analysis note
box(ax, (2, 0.4), 5, 1.0,
    "Linearity diagnostics run (bin-split + held-out ρ);\n"
    "interpret against matched-scale power analysis (see §3.5–3.6)",
    fc="#e5f3ec", ec="#2b6a3f", fs=8)

# Right column: downstream analyses (gated on both stages)
box(ax, (9, 4.4), 4.5, 1.2,
    "Downstream analyses:\neigenvalues, hyperbolicity,\narchetypes, cross-tool benchmark",
    fc="#eef4fa", ec="#1f4e79", fs=9)
ax.text(11.25, 3.6, "requires: full identifiability\nAND independent linearity evidence\nbeyond current diagnostics' power",
        ha="center", va="center", fontsize=7.5, color="#666", style="italic")

ax.set_title("Fig 0: anchor-op pipeline", fontsize=12, y=0.98)

out = OUT_DIR / "fig0_pipeline_schematic.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
