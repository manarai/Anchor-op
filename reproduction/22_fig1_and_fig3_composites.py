"""Composites for Fig. 1 (K562 + RPE1 identification) and Fig. 3 (linearity power).

Fig. 1 combines existing K562/RPE1 diagnostic panels into a single figure the
manuscript references as `fig1_measurements.png`.

Fig. 3 combines the three linearity-diagnostic panels (real-scale positive
control, rejection-power surface, κ-range sweep) into `fig3_linearity_power.png`.

Both use existing PNGs from `manuscript_figures/`, so runtime is essentially
just PIL/matplotlib rendering.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "manuscript_figures"

def compose(files_and_titles, out_path, suptitle, layout, figsize):
    fig, axes = plt.subplots(*layout, figsize=figsize, constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (fp, label) in zip(axes_flat, files_and_titles):
        if not fp.exists():
            ax.text(0.5, 0.5, f"MISSING: {fp.name}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off"); continue
        img = mpimg.imread(str(fp))
        ax.imshow(img); ax.axis("off"); ax.set_title(label, fontsize=11)
    for ax in axes_flat[len(files_and_titles):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

# --- Fig 1: K562 + RPE1 identification ---
compose(
    files_and_titles=[
        (OUT / "fig3a_k562_essential_diagnostics.png", "(a) K562: identification diagnostic"),
        (OUT / "fig4a_rpe1_essential_diagnostics.png", "(b) RPE1: identification diagnostic"),
        (OUT / "fig3b_k562_essential_drops.png",       "(c) K562: guide-drop breakdown"),
        (OUT / "fig4b_rpe1_essential_drops.png",       "(d) RPE1: guide-drop breakdown"),
    ],
    out_path=OUT / "fig1_measurements.png",
    suptitle="Fig. 1 — Identification on the Replogle essential-gene screens. "
             "Full-rank identification (K562 30/30, RPE1 30/30) at rank_tol=1e-2.",
    layout=(2, 2), figsize=(14, 10.5),
)

# --- Fig 3: linearity-diagnostic power ---
compose(
    files_and_titles=[
        (OUT / "figS10_realscale_positive_control.png", "(a) Matched-scale linear positive control"),
        (OUT / "figS12_rejection_power_surface.png",    "(b) Rejection-power surface (sat=0.5)"),
        (OUT / "figS11_kappa_range_sweep.png",          "(c) κ-range sweep"),
    ],
    out_path=OUT / "fig3_linearity_power.png",
    suptitle="Fig. 3 — Linearity-diagnostic power. Panels drawn from the "
             "same synthetic-linear-under-Replogle-geometry sweeps as Figs. S10–S12.",
    layout=(1, 3), figsize=(18, 5.6),
)
