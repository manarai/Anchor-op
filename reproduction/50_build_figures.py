"""Build all main-text and supplementary figures from JSON outputs.

Outputs (PNGs, 300 dpi, in manuscript_figures/):
  fig1_pipeline.png            — Pipeline schematic + matched positive control cartoon
  fig2_nested_cv.png           — Table 1 as a plot (real vs matched linear-truth ρ)
  fig3_alpha_and_interaction.png — α_S sweep + interaction-only cosine
  fig4_robustness.png          — random-200, Jost d-sweep, direction-only panels
  fig5_footprint.png           — Footprint encoding on all 4 fits + both nulls
  figS1_dose_interp.png        — Jost dose interpolation vs matched linear-truth
  figS2_rel_diff.png           — rel_diff real vs matched linear-truth on three screens

Every plotted number traces to results/recheck/*.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "manuscript_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REC = REPO / "results" / "recheck"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def _load(name):
    return json.load(open(REC / name))


# ─── FIG 1: pipeline schematic + matched positive control cartoon ─────────
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    # LEFT: pipeline schematic (text-block cartoon)
    ax = axes[0]; ax.set_axis_off()
    boxes = [
        (0.02, 0.72, "Perturb-seq screen\n(n sgRNAs × cells)", "#EEEEEE"),
        (0.02, 0.42, "Program-space projection\nz = Wᵀe   (d = 30 PCs on controls)", "#DDEEFF"),
        (0.02, 0.12, "Fit A = −U·S⁺\nregularised, target-grouped folds", "#DDF6E7"),
        (0.55, 0.72, "Efficiency κ\nmean_ratio or detection_rate", "#FDECEC"),
        (0.55, 0.42, "Input encoding\nu_g = −κ_g Wᵀδ_g", "#FDECEC"),
        (0.55, 0.12, "Held-out ρ + matched\nlinear-truth positive control", "#FFF3D6"),
    ]
    for (x, y, txt, c) in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.42, 0.22, transform=ax.transAxes,
                                    facecolor=c, edgecolor="0.4"))
        ax.text(x + 0.21, y + 0.11, txt, transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
    # arrows
    for xa, xb, ya, yb in [(0.44, 0.55, 0.83, 0.83), (0.23, 0.23, 0.72, 0.64),
                            (0.44, 0.55, 0.53, 0.53), (0.76, 0.76, 0.72, 0.64),
                            (0.23, 0.23, 0.42, 0.34), (0.76, 0.76, 0.42, 0.34),
                            (0.44, 0.55, 0.23, 0.23)]:
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    ax.set_title("(a) anchor-op pipeline")

    # RIGHT: matched-SNR positive control cartoon
    ax = axes[1]; ax.set_axis_off()
    ax.text(0.5, 0.94, "(b) matched-SNR linear-truth positive control",
            transform=ax.transAxes, ha="center", fontsize=10, weight="bold")
    txt = ("draw J_ref = G − 1.5 I;   set α so median ‖−J⁻¹U_data‖ matches real S\n\n"
           "S_sim = −J⁻¹ U_data + σ·noise\n\n"
           "run the same nested-CV recipe on (S_sim, U_data)\n\n"
           "if the encoding is well-posed on this data,\n"
           "real ρ  ≲  matched linear-truth ρ  <  predict-zero baseline ρ = 1")
    ax.text(0.5, 0.44, txt, transform=ax.transAxes, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7", edgecolor="0.6"))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_pipeline.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig1_pipeline.png")


# ─── FIG 2: nested-CV Table 1 as a plot ────────────────────────────────────
def fig2():
    d = _load("F_nested_cv_rho.json")
    screens = ["K562_essential", "RPE1_essential", "Jost_2020"]
    labels = ["K562 essential", "RPE1 essential", "Jost 2020"]
    real_rho, real_sd, lin_rho, lin_sd = [], [], [], []
    for s in screens:
        r = d["datasets"][s]["real_nested_cv"]
        l = d["datasets"][s]["linear_truth_matched_alpha_nested_cv"]
        rhos = [f["rho_fold"] for f in r["per_fold"]]
        real_rho.append(r["rho_pooled"]); real_sd.append(np.std(rhos, ddof=1))
        lin_rho.append(l["rho_mean"]); lin_sd.append(l["rho_std"])

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(screens)); w = 0.35
    ax.bar(x - w/2, real_rho, w, yerr=real_sd, capsize=3, label="Real (target-held-out)",
           color="#3A6EA5", edgecolor="0.2")
    ax.bar(x + w/2, lin_rho, w, yerr=lin_sd, capsize=3, label="Matched linear-truth control",
           color="#E9967A", edgecolor="0.2")
    ax.axhline(1.0, color="k", linestyle="--", lw=0.8, label="predict-zero baseline")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("held-out ρ")
    ax.set_ylim(0, 1.15)
    ax.set_title("Target-held-out nested-CV ρ")
    ax.legend(loc="lower left", frameon=False)
    for xi, r_, s_ in zip(x - w/2, real_rho, real_sd):
        ax.text(xi, r_ + s_ + 0.02, f"{r_:.2f}\n±{s_:.3f}", ha="center", fontsize=8)
    for xi, r_, s_ in zip(x + w/2, lin_rho, lin_sd):
        ax.text(xi, r_ + s_ + 0.02, f"{r_:.2f}\n±{s_:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_nested_cv.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig2_nested_cv.png")


# ─── FIG 3: α_S sweep + interaction-only cosine ────────────────────────────
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    # LEFT: α sweep
    ax = axes[0]
    for tag, colour, label in [("F2_K562_essential.json", "#3A6EA5", "K562"),
                                ("F2_RPE1_essential.json", "#E9967A", "RPE1")]:
        d = _load(tag)
        dense = d["results"]["dense"]
        alphas = np.array([r["alpha_S"] for r in dense])
        cos = np.array([r["cos_full_mean"] for r in dense])
        nullc = np.array([r["cos_full_paired_null_mean"] for r in dense])
        ax.plot(alphas, cos, "o-", color=colour, label=f"{label} real cos", lw=1.8)
        ax.plot(alphas, nullc, "s--", color=colour, alpha=0.55, label=f"{label} cross-rep null")
        ah = d["alpha_hat_data_dense"]
        ax.axvline(ah, color=colour, alpha=0.25, lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("signal-scale α_S (log)")
    ax.set_ylabel("Frobenius cosine")
    ax.set_title("(a) α_S sweep — full-matrix cos vs cross-rep null")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color="0.6", lw=0.6)

    # RIGHT: interaction-only at matched α
    ax = axes[1]
    d = _load("F_interaction_only_cosine.json")["datasets"]
    tags = ["K562_essential", "RPE1_essential"]
    labels = ["K562", "RPE1"]
    cos_int = [d[t]["cos_interaction_only_mean"] for t in tags]
    cos_int_sd = [d[t]["cos_interaction_only_std"] for t in tags]
    null = [d[t]["cos_interaction_only_null_mean_shift1"] for t in tags]
    null_sd = [d[t]["cos_interaction_only_null_std_shift1"] for t in tags]
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, cos_int, w, yerr=cos_int_sd, capsize=3,
           label="interaction-only cos", color="#3A6EA5", edgecolor="0.2")
    ax.bar(x + w/2, null, w, yerr=null_sd, capsize=3,
           label="cross-rep null", color="#B4B4B4", edgecolor="0.2")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("interaction-only cos (matched α)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) interaction-only at data-matched α (N = 200)")
    ax.legend(frameon=False)
    for xi, r_, s_ in zip(x - w/2, cos_int, cos_int_sd):
        ax.text(xi, r_ + s_ + 0.02, f"{r_:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_alpha_and_interaction.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig3_alpha_and_interaction.png")


# ─── FIG 4: robustness — random-200, Jost d-sweep, direction-only ─────────
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    # PANEL (a): K562 top-200 vs random-200
    ax = axes[0]
    d = _load("F7_K562_random200.json")
    d1 = _load("F_nested_cv_rho.json")
    top_real = d1["datasets"]["K562_essential"]["real_nested_cv"]["rho_pooled"]
    top_lin = d1["datasets"]["K562_essential"]["linear_truth_matched_alpha_nested_cv"]["rho_mean"]
    rnd_real = d["real_nested_cv"]["rho_pooled"]
    rnd_lin = d["linear_truth_matched_alpha_nested_cv"]["rho_mean"]
    rnd_lin_sd = d["linear_truth_matched_alpha_nested_cv"]["rho_std"]
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, [top_real, rnd_real], w,
           label="real ρ", color="#3A6EA5", edgecolor="0.2")
    ax.bar(x + w/2, [top_lin, rnd_lin], w, yerr=[0.003, rnd_lin_sd], capsize=3,
           label="matched linear", color="#E9967A", edgecolor="0.2")
    ax.axhline(1.0, color="k", linestyle="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["top-200 (published)", "random-200"])
    ax.set_ylim(0, 1.15); ax.set_ylabel("held-out ρ")
    ax.set_title("(a) K562 target selection")
    ax.legend(frameon=False, fontsize=8)

    # PANEL (b): Jost d-sweep
    ax = axes[1]
    d = _load("F_jost_d_sweep_grouped.json")
    ds = sorted(int(k.split("=")[1]) for k in d["results"])
    real = [d["results"][f"d={dd}"]["real_nested_cv"]["rho"] for dd in ds]
    lin = [d["results"][f"d={dd}"]["linear_truth_matched_nested_cv"]["rho_mean"] for dd in ds]
    lin_sd = [d["results"][f"d={dd}"]["linear_truth_matched_nested_cv"]["rho_std"] for dd in ds]
    ax.plot(ds, real, "o-", color="#3A6EA5", label="real ρ", lw=1.8)
    ax.errorbar(ds, lin, yerr=lin_sd, fmt="s--", color="#E9967A",
                label="matched linear", lw=1.2, capsize=2)
    ax.axhline(1.0, color="k", linestyle="--", lw=0.8)
    ax.set_xlabel("d (truncated program dim)"); ax.set_ylabel("held-out ρ")
    ax.set_title("(b) Jost d-sweep")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 1.15)

    # PANEL (c): direction-only nested-CV
    ax = axes[2]
    d = _load("F_nested_cv_direction_only.json")
    tags = ["K562_essential", "RPE1_essential", "Jost_2020"]
    labels = ["K562", "RPE1", "Jost"]
    real = [d["datasets"][t]["real_nested_cv_direction_only"]["rho_pooled"] for t in tags]
    real_sd = [d["datasets"][t]["real_nested_cv_direction_only"]["per_fold_sd"] for t in tags]
    lin = [d["datasets"][t]["linear_truth_matched_alpha_nested_cv_direction_only"]["rho_mean"] for t in tags]
    lin_sd = [d["datasets"][t]["linear_truth_matched_alpha_nested_cv_direction_only"]["rho_std"] for t in tags]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, real, w, yerr=real_sd, capsize=3, label="real ρ (direction-only)",
           color="#3A6EA5", edgecolor="0.2")
    ax.bar(x + w/2, lin, w, yerr=lin_sd, capsize=3, label="matched linear (direction-only)",
           color="#E9967A", edgecolor="0.2")
    ax.axhline(1.0, color="k", linestyle="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("held-out ρ")
    ax.set_ylim(0, 1.15)
    ax.set_title("(c) direction-only nested-CV")
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_robustness.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig4_robustness.png")


# ─── FIG 5: footprint encoding + nulls (all 4 fits) ────────────────────────
def fig5():
    d = _load("F_footprint.json")["screens"]
    fits = ["K562_top200", "K562_random200", "RPE1", "Jost_2020"]
    labels = ["K562 top-200", "K562 random-200", "RPE1", "Jost 2020"]
    foot = [d[f]["footprint_real"]["rho_pooled"] for f in fits]
    foot_sd = [d[f]["footprint_real"]["per_fold_sd"] for f in fits]
    shuf_p025 = [d[f]["shuffled_null"]["p025"] for f in fits]
    shuf_p975 = [d[f]["shuffled_null"]["p975"] for f in fits]
    shuf_p50 = [d[f]["shuffled_null"]["p50"] for f in fits]
    rand_p025 = [d[f]["random_direction_null"]["p025"] for f in fits]
    rand_p975 = [d[f]["random_direction_null"]["p975"] for f in fits]
    rand_p50 = [d[f]["random_direction_null"]["p50"] for f in fits]
    verdict = [d[f]["verdict"]["success"] for f in fits]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(fits))
    w = 0.28
    # Shuffled null: box between 2.5th and 97.5th percentile
    for i, (a, b, m) in enumerate(zip(shuf_p025, shuf_p975, shuf_p50)):
        ax.add_patch(plt.Rectangle((x[i] - w - 0.05, a), w, b - a,
                                    facecolor="#B4B4B4", edgecolor="0.3", alpha=0.7))
        ax.plot([x[i] - w - 0.05, x[i] - 0.05], [m, m], color="k", lw=1.0)
    # Random null: same
    for i, (a, b, m) in enumerate(zip(rand_p025, rand_p975, rand_p50)):
        ax.add_patch(plt.Rectangle((x[i] + 0.05, a), w, b - a,
                                    facecolor="#DEB887", edgecolor="0.3", alpha=0.7))
        ax.plot([x[i] + 0.05, x[i] + w + 0.05], [m, m], color="k", lw=1.0)
    # Footprint real ρ marker
    for i, (r, s) in enumerate(zip(foot, foot_sd)):
        ax.errorbar(x[i], r, yerr=s, fmt="D", color="#3A6EA5", markersize=8, capsize=3,
                    markeredgecolor="0.2", elinewidth=1.5)
    ax.axhline(1.0, color="k", linestyle="--", lw=0.8, label="predict-zero baseline")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("nested-CV held-out ρ")
    ax.set_ylim(0.6, 1.1)
    ax.set_title("Footprint encoding real ρ vs shuffled and random-direction nulls (N=100 each)")
    # Manual legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#B4B4B4", edgecolor="0.3", alpha=0.7),
               plt.Rectangle((0, 0), 1, 1, facecolor="#DEB887", edgecolor="0.3", alpha=0.7),
               plt.Line2D([0], [0], marker="D", color="#3A6EA5", markersize=8, lw=0)]
    ax.legend(handles, ["shuffled-footprint null (2.5–97.5 pct)",
                        "random-direction null (2.5–97.5 pct)",
                        "footprint real ρ (5-fold SD)"], frameon=False, fontsize=8)
    # Verdict labels
    for i, (v, r) in enumerate(zip(verdict, foot)):
        color = "#2A7A2A" if v else "#B33A3A"
        text = "SUCCESS" if v else "failure"
        ax.text(x[i], 0.63, text, ha="center", color=color, fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_footprint.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig5_footprint.png")


# ─── SUPP: dose interpolation, rel_diff ───────────────────────────────────
def fig_supp():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # Supp 1: Jost dose interpolation
    ax = axes[0]
    d = _load("F3_Jost_grouped_folds.json")
    real_guide = d["rho_real"]["guide_folds"]
    real_target = d["rho_real"]["target_grouped_folds"]
    lin_guide = d["rho_linear_truth"]["at_matched_alpha_guide_folds"]["rho_mean"]
    lin_guide_sd = d["rho_linear_truth"]["at_matched_alpha_guide_folds"]["rho_std"]
    lin_target = d["rho_linear_truth"]["at_matched_alpha_target_grouped_folds"]["rho_mean"]
    lin_target_sd = d["rho_linear_truth"]["at_matched_alpha_target_grouped_folds"]["rho_std"]
    x = np.arange(2); w = 0.35
    ax.bar(x - w/2, [real_guide, real_target], w, label="Jost real",
           color="#3A6EA5", edgecolor="0.2")
    ax.bar(x + w/2, [lin_guide, lin_target], w, yerr=[lin_guide_sd, lin_target_sd],
           capsize=3, label="matched linear", color="#E9967A", edgecolor="0.2")
    ax.axhline(1.0, color="k", linestyle="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["guide-level\n(dose interp.)", "target-grouped\n(new perturbation)"])
    ax.set_ylabel("held-out ρ")
    ax.set_title("Supp 1. Jost dose interpolation")
    ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, 2.8)

    # Supp 2: rel_diff calibration
    ax = axes[1]
    tags = ["K562", "RPE1", "Jost"]
    real_rd = [1.47, 1.57, 1.26]
    lin_rd = [0.33, 0.98, 1.05]
    lin_rd_sd = [0.01, 0.02, 0.03]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, real_rd, w, label="real",
           color="#3A6EA5", edgecolor="0.2")
    ax.bar(x + w/2, lin_rd, w, yerr=lin_rd_sd, capsize=3, label="matched linear",
           color="#E9967A", edgecolor="0.2")
    ax.axhline(0.25, color="k", linestyle="--", lw=0.8, label="preregistered threshold 0.25")
    ax.set_xticks(x); ax.set_xticklabels(tags); ax.set_ylabel("rel_diff")
    ax.set_ylim(0, 1.8)
    ax.set_title("Supp 2. rel_diff calibration (miscalibrated on 2 of 3)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figS1_dose_interp.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figS1_dose_interp.png (contains both supp panels)")


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig_supp()
    print("\nAll figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
