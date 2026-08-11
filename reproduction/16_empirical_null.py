"""Fig S16: pipeline-matched empirical null for the operator-recovery cosines.

The analytic random-matrix null in §4.7 gives cos SD ≈ 1/d = 0.033 for the full
d×d comparison and 1/√d = 0.183 for cos_1. This script replaces the analytic
argument with a pipeline-preserving empirical null.

Two null constructions:
  (A) Cross-replicate pairing. For each replicate r ∈ {1..N_REPS} draw J_r,
      add σ=0.266 noise to S_true_r, fit A_r via the default TSVD path.
      Compare A_r against a J_{r'} from a different replicate (r' = (r+1) mod N).
      This preserves the full fitted-operator construction; the only thing
      randomized is which ground truth we score against.
  (B) Shuffled-U. For each replicate r, refit using a column-permuted U (so
      U's row-marginal structure is preserved but each guide is paired with a
      random column direction). Compare A_shuffled_r against J_r. This preserves
      J and the geometry of U but destroys the guide→ground-truth correspondence.

Real signal: cos(A_r, J_r) — the actual recovery cosine.

Runs 200 replicates on K562 dense-Gaussian J_true at measured σ=0.266.
Reports mean, std, and empirical (real − null) / std_null z-scores for cos and cos_1.

Runtime ~8 min at N_REPS=200.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from anchorop.identifiability import regularized_pseudoinverse

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript_figures"
OUT_DIR.mkdir(exist_ok=True)

N_REPS = 200
SIGMA = 0.266
RANK_TOL = 1e-2
SEED_BASE = 20260810


def draw_dense_J(d, seed):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(d, d)) / np.sqrt(d)
    return G - 1.5 * np.eye(d)


def fit_operator(S, U):
    S_pinv, *_ = regularized_pseudoinverse(S, method="tsvd", parameter="path", rank_tol=RANK_TOL)
    return -U @ S_pinv


def cos_full(A, J):
    inner = float(np.sum(A * J))
    nA = float(np.linalg.norm(A)); nJ = float(np.linalg.norm(J))
    return inner / max(nA * nJ, 1e-30)


def cos_topk(A, J, S_true, k):
    Us, _, _ = np.linalg.svd(S_true, full_matrices=False)
    Uk = Us[:, :k]
    AUk = A @ Uk; JUk = J @ Uk
    return float(np.sum(AUk * JUk)) / max(np.linalg.norm(AUk) * np.linalg.norm(JUk), 1e-30)


results = {}
for cell_line, filename in [("K562_essential", "k562_essential_measurement.pkl"),
                              ("RPE1_essential", "rpe1_essential_measurement.pkl")]:
    bundle_path = RESULTS / filename
    with bundle_path.open("rb") as f:
        state = pickle.load(f)
    m = state["measurement"]
    U_real = m.U
    d, n_guides = m.S.shape
    print(f"\n{'='*72}\n{cell_line} (d={d}, n_guides={n_guides}, dense J_true, σ={SIGMA})\n"
          f"N_REPS={N_REPS}\n{'='*72}")

    # 1. Generate all replicates: J_r, S_true_r, S_obs_r, A_r
    Js, S_trues, A_fits = [], [], []
    rng_noise = np.random.default_rng(SEED_BASE + 999)
    for r in range(N_REPS):
        J_r = draw_dense_J(d, seed=SEED_BASE + r)
        S_true = -np.linalg.solve(J_r, U_real)
        S_obs = S_true + SIGMA * rng_noise.normal(size=S_true.shape)
        A_r = fit_operator(S_obs, U_real)
        Js.append(J_r); S_trues.append(S_true); A_fits.append(A_r)
        if (r + 1) % 50 == 0:
            print(f"  fit {r+1}/{N_REPS}")

    # 2. Real signal: cos(A_r, J_r) and cos_1
    real_cos = np.array([cos_full(A_fits[r], Js[r]) for r in range(N_REPS)])
    real_cos1 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 1) for r in range(N_REPS)])
    real_cos5 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 5) for r in range(N_REPS)])

    # 3. Cross-replicate null: cos(A_r, J_{r'}), r' = (r + shift) mod N_REPS
    #    Use shift ∈ {1, 2, ..., 10} to get 10*N_REPS null samples
    null_cross_cos, null_cross_cos1, null_cross_cos5 = [], [], []
    for shift in range(1, 11):
        for r in range(N_REPS):
            rp = (r + shift) % N_REPS
            null_cross_cos.append(cos_full(A_fits[r], Js[rp]))
            null_cross_cos1.append(cos_topk(A_fits[r], Js[rp], S_trues[rp], 1))
            null_cross_cos5.append(cos_topk(A_fits[r], Js[rp], S_trues[rp], 5))
    null_cross_cos = np.array(null_cross_cos)
    null_cross_cos1 = np.array(null_cross_cos1)
    null_cross_cos5 = np.array(null_cross_cos5)

    # 4. Shuffled-U null: refit using a column-permuted U, compare to J_r
    #    (fresh noise per shuffle to fully randomize the fitted-operator construction)
    null_shuf_cos, null_shuf_cos1, null_shuf_cos5 = [], [], []
    rng_shuf = np.random.default_rng(SEED_BASE + 888)
    for r in range(N_REPS):
        perm = rng_shuf.permutation(n_guides)
        U_shuf = U_real[:, perm]
        S_true_shuf = -np.linalg.solve(Js[r], U_shuf)
        S_obs_shuf = S_true_shuf + SIGMA * rng_shuf.normal(size=S_true_shuf.shape)
        A_shuf = fit_operator(S_obs_shuf, U_real)  # fit uses ORIGINAL U (guide labels shuffled)
        null_shuf_cos.append(cos_full(A_shuf, Js[r]))
        null_shuf_cos1.append(cos_topk(A_shuf, Js[r], S_trues[r], 1))
        null_shuf_cos5.append(cos_topk(A_shuf, Js[r], S_trues[r], 5))
    null_shuf_cos = np.array(null_shuf_cos)
    null_shuf_cos1 = np.array(null_shuf_cos1)
    null_shuf_cos5 = np.array(null_shuf_cos5)

    def summarize(real, null_cross, null_shuf, label):
        mr, sr = float(real.mean()), float(real.std())
        mn_c, sn_c = float(null_cross.mean()), float(null_cross.std())
        mn_s, sn_s = float(null_shuf.mean()), float(null_shuf.std())
        # z-scores of real mean against each null
        se_r = sr / np.sqrt(len(real))  # SE of real mean
        z_c = (mr - mn_c) / max(sn_c, 1e-9)   # in null SD units
        z_s = (mr - mn_s) / max(sn_s, 1e-9)
        # empirical one-sided p from null distribution
        p_c = float(np.mean(null_cross >= mr))
        p_s = float(np.mean(null_shuf  >= mr))
        return {
            "real_mean": mr, "real_std": sr, "real_SE": float(se_r),
            "null_cross_mean": mn_c, "null_cross_std": sn_c,
            "null_shuf_mean":  mn_s, "null_shuf_std":  sn_s,
            "z_cross": float(z_c), "z_shuf": float(z_s),
            "p_cross_ge_real": p_c, "p_shuf_ge_real": p_s,
        }

    line = {
        "n_reps": N_REPS,
        "sigma": SIGMA,
        "cos_full": summarize(real_cos, null_cross_cos, null_shuf_cos, "cos"),
        "cos_1":    summarize(real_cos1, null_cross_cos1, null_shuf_cos1, "cos_1"),
        "cos_5":    summarize(real_cos5, null_cross_cos5, null_shuf_cos5, "cos_5"),
    }
    results[cell_line] = line

    for metric_name in ["cos_full", "cos_1", "cos_5"]:
        s = line[metric_name]
        print(f"\n  {metric_name}: real mean = {s['real_mean']:+.4f} (SE={s['real_SE']:.4f}, SD={s['real_std']:.3f})")
        print(f"    cross-replicate null:   mean {s['null_cross_mean']:+.4f} ± {s['null_cross_std']:.3f}  "
              f"→ z = {s['z_cross']:+.2f}, p(null≥real) = {s['p_cross_ge_real']:.3f}")
        print(f"    shuffled-U null:        mean {s['null_shuf_mean']:+.4f} ± {s['null_shuf_std']:.3f}  "
              f"→ z = {s['z_shuf']:+.2f}, p(null≥real) = {s['p_shuf_ge_real']:.3f}")

(OUT_DIR / "empirical_null.json").write_text(json.dumps(results, indent=2))

# Figure: 2 rows (K562, RPE1) × 3 cols (cos_full, cos_1, cos_5), showing real vs cross-replicate vs shuffled-U distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for row, cell_line in enumerate(["K562_essential", "RPE1_essential"]):
    # We need the raw arrays too — regenerate them for plotting from the summary is not possible.
    # Save them alongside; use a second-pass here. For brevity we re-derive from what we have
    # by re-running the loop's key arrays. Simpler: rerun in-memory below.
    pass

# We'll rerun a plotting pass by keeping arrays in the outer loop.
# Actually the simpler fix: rewrite the loop to also stash arrays for plotting.
# --> Restructured plotting: reuse variables from the last outer iteration is wrong;
#     to keep code simple, rerun the fits once more only for plotting on K562.

def plot_pass(cell_line, filename, ax_row):
    bundle_path = RESULTS / filename
    with bundle_path.open("rb") as f:
        state = pickle.load(f)
    m = state["measurement"]
    U_real = m.U
    d, n_guides = m.S.shape
    Js, S_trues, A_fits = [], [], []
    rng_noise = np.random.default_rng(SEED_BASE + 999)
    for r in range(N_REPS):
        J_r = draw_dense_J(d, seed=SEED_BASE + r)
        S_true = -np.linalg.solve(J_r, U_real)
        S_obs = S_true + SIGMA * rng_noise.normal(size=S_true.shape)
        Js.append(J_r); S_trues.append(S_true); A_fits.append(fit_operator(S_obs, U_real))
    real_cos = np.array([cos_full(A_fits[r], Js[r]) for r in range(N_REPS)])
    real_cos1 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 1) for r in range(N_REPS)])
    real_cos5 = np.array([cos_topk(A_fits[r], Js[r], S_trues[r], 5) for r in range(N_REPS)])
    null_c, null_c1, null_c5 = [], [], []
    for shift in range(1, 11):
        for r in range(N_REPS):
            rp = (r + shift) % N_REPS
            null_c.append(cos_full(A_fits[r], Js[rp]))
            null_c1.append(cos_topk(A_fits[r], Js[rp], S_trues[rp], 1))
            null_c5.append(cos_topk(A_fits[r], Js[rp], S_trues[rp], 5))
    null_s, null_s1, null_s5 = [], [], []
    rng_shuf = np.random.default_rng(SEED_BASE + 888)
    for r in range(N_REPS):
        perm = rng_shuf.permutation(n_guides)
        U_shuf = U_real[:, perm]
        S_true_shuf = -np.linalg.solve(Js[r], U_shuf)
        S_obs_shuf = S_true_shuf + SIGMA * rng_shuf.normal(size=S_true_shuf.shape)
        A_shuf = fit_operator(S_obs_shuf, U_real)
        null_s.append(cos_full(A_shuf, Js[r]))
        null_s1.append(cos_topk(A_shuf, Js[r], S_trues[r], 1))
        null_s5.append(cos_topk(A_shuf, Js[r], S_trues[r], 5))
    triples = [(real_cos, null_c, null_s, "cos(A, J)", 0),
                (real_cos1, null_c1, null_s1, "cos_1(A, J | U_S top-1)", 1),
                (real_cos5, null_c5, null_s5, "cos_5(A, J | U_S top-5)", 2)]
    for real, nullc, nulls, title, col in triples:
        ax = axes[ax_row, col]
        bins = np.linspace(min(np.min(nullc), np.min(nulls), np.min(real)) - 0.02,
                           max(np.max(nullc), np.max(nulls), np.max(real)) + 0.02, 40)
        ax.hist(nullc, bins=bins, color="#7a7a7a", alpha=0.55, label="cross-rep null", density=True)
        ax.hist(nulls, bins=bins, color="#8899cc", alpha=0.45, label="shuffled-U null", density=True)
        ax.hist(real, bins=bins, color="#c65a30" if "RPE1" in cell_line else "#1f4e79",
                 alpha=0.7, label="real: cos(A_r, J_r)", density=True)
        ax.axvline(float(np.mean(real)), color="black", ls="--", lw=1.2)
        ax.axvline(0.0, color="0.5", lw=0.5)
        ax.set_title(f"{cell_line.replace('_essential','')}: {title}", fontsize=10)
        ax.set_xlabel("cosine"); ax.set_ylabel("density")
        if col == 0 and ax_row == 0:
            ax.legend(fontsize=8, loc="upper left")

plot_pass("K562_essential", "k562_essential_measurement.pkl", 0)
plot_pass("RPE1_essential", "rpe1_essential_measurement.pkl", 1)

fig.suptitle(f"Fig S16: Pipeline-matched empirical null (N={N_REPS} replicates, dense J_true, σ={SIGMA})",
             fontsize=11, y=1.02)
out = OUT_DIR / "figS16_empirical_null.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out}")
