"""Fig S1: rank_tol sensitivity sweep on the measurement bundles.

Loads the K562 aggregate, K562 essential, and RPE1 essential measurement
bundles (if present), and reports effective_response_rank and condition
number as rank_tol varies from 1e-3 to 5e-2. Highlights the preregistered
1e-2 as the elbow.

Requires pickled bundles from `../results/`. Runtime <30 s.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import anchorop as ao

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)

BUNDLES = [
    ("K562 aggregate", "k562_measurement.pkl"),
    ("K562 essential", "k562_essential_measurement.pkl"),
    ("RPE1 essential", "rpe1_essential_measurement.pkl"),
]
COLORS = {"K562 aggregate": "#7a4d95", "K562 essential": "#1f4e79", "RPE1 essential": "#c65a30"}

RANK_TOLS = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]

results = {}
for label, filename in BUNDLES:
    path = RESULTS / filename
    if not path.exists():
        print(f"skip {label}: {path} not found (run measurement script first)")
        continue
    with path.open("rb") as f:
        state = pickle.load(f)
    m = state["measurement"]
    d = m.report.d
    ranks = []; conds = []
    for rt in RANK_TOLS:
        m2 = ao.measure_from_sensitivity(
            m.S, m.U, guide_names=m.guide_names,
            guide_efficiencies=m.report.guide_efficiencies,
            reg="tsvd", reg_param="path", rank_tol=rt,
        )
        ranks.append(int(m2.report.effective_response_rank))
        conds.append(float(m2.report.condition_number))
    results[label] = {"d": d, "ranks": ranks, "conds": conds}
    print(f"{label}: ranks {ranks}, conds {conds}")

if not results:
    raise SystemExit("no measurement bundles found; run scripts 02/03/04 first")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
ax = axes[0]
for label, r in results.items():
    ax.plot(RANK_TOLS, r["ranks"], "o-", color=COLORS.get(label, "black"), lw=2, ms=7, label=f"{label} (d={r['d']})")
ax.axvline(1e-2, color="gray", ls=":", alpha=0.6, label="preregistered 1×10⁻²")
ax.set_xscale("log"); ax.set_xlabel("rank_tol"); ax.set_ylabel("effective response rank")
ax.set_title("(a) effective_response_rank vs rank_tol")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
for label, r in results.items():
    ax.semilogy(RANK_TOLS, r["conds"], "o-", color=COLORS.get(label, "black"), lw=2, ms=7, label=label)
ax.axvline(1e-2, color="gray", ls=":", alpha=0.6, label="preregistered 1×10⁻²")
ax.set_xscale("log"); ax.set_xlabel("rank_tol"); ax.set_ylabel("condition number (log)")
ax.set_title("(b) condition number of J·P_X vs rank_tol")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

fig.suptitle("Fig S1: rank_tol sensitivity sweep", fontsize=12, y=1.03)
out = OUT_DIR / "figS1_rank_tol_sweep.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")

# CSV
csv_path = OUT_DIR / "figS1_rank_tol_sweep.csv"
with csv_path.open("w") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "rank_tol", "effective_rank", "condition_number"])
    for label, r in results.items():
        for rt, rk, c in zip(RANK_TOLS, r["ranks"], r["conds"]):
            w.writerow([label, rt, rk, c])
print(f"wrote {csv_path}")
