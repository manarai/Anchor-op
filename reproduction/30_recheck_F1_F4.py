"""Recheck F1-F4 against the reviewer's audit.

F1: amplitude of the positive-control simulation vs the observed responses.
F2: signal-scale (alpha) sweep — recovery as a function of J_true amplitude.
F3: linearity diagnostic on stored measurements vs matched-SNR linear simulations.
F4: shared response mode, target-input orthogonality, footprint encoding.

Runs on the stored K562 and RPE1 essential measurements (results/*.pkl).
All outputs land in results/recheck/.

Seed: SEED_BASE = 20260927.
"""
from __future__ import annotations
import json, pickle, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import anchorop as ao
from anchorop.identifiability import regularized_pseudoinverse

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

DATASET_SIGMA = {"K562_essential": 0.240, "RPE1_essential": 0.352}
BUNDLE = {"K562_essential": "k562_essential_measurement.pkl",
          "RPE1_essential": "rpe1_essential_measurement.pkl"}
RANK_TOL = 1e-2
SEED_BASE = 20260927
N_REPS = 200
STRUCTURES = ["dense", "sparse_10pct", "sparse_2pct", "low_rank_5"]


def draw_J(d, seed, structure, c=1.5):
    rng = np.random.default_rng(seed)
    if structure == "dense":
        G = rng.normal(size=(d, d)) / np.sqrt(d)
    elif structure == "sparse_10pct":
        mask = rng.random((d, d)) < 0.10
        G = np.zeros((d, d))
        G[mask] = rng.normal(size=int(mask.sum())) / np.sqrt(max(mask.sum(), 1) / d)
    elif structure == "sparse_2pct":
        mask = rng.random((d, d)) < 0.02
        G = np.zeros((d, d))
        G[mask] = rng.normal(size=int(mask.sum())) / np.sqrt(max(mask.sum(), 1) / d)
    elif structure == "low_rank_5":
        Ul = rng.normal(size=(d, 5)) / np.sqrt(d)
        Vl = rng.normal(size=(5, d)) / np.sqrt(d)
        G = Ul @ Vl
    else:
        raise ValueError(structure)
    return G - c * np.eye(d)


def fit_A(S_obs, U):
    S_pinv, *_ = regularized_pseudoinverse(S_obs, method="tsvd", parameter="path", rank_tol=RANK_TOL)
    return -U @ S_pinv


def cos_full(A, J):
    return float(np.sum(A * J)) / max(np.linalg.norm(A) * np.linalg.norm(J), 1e-30)


def _derive_targets(guide_names):
    """essential-screen convention: 'guide_<TARGET_ID>'. Return {guide: target}."""
    out = {}
    for gn in guide_names:
        gn = str(gn)
        if gn.startswith("guide_"):
            out[gn] = gn[len("guide_"):]
        else:
            out[gn] = gn  # target == guide
    return out


def cos_topk_of(A, B, S_true, k):
    Us, _, _ = np.linalg.svd(S_true, full_matrices=False)
    Uk = Us[:, :k]
    AUk = A @ Uk
    BUk = B @ Uk
    return float(np.sum(AUk * BUk)) / max(np.linalg.norm(AUk) * np.linalg.norm(BUk), 1e-30)


# -----------------------------------------------------------------------------
# F1: amplitude comparison for each cell line, dense J_true, alpha=1 (published).
# -----------------------------------------------------------------------------
def f1_amplitude(state, sigma, tag):
    U = state["measurement"].U
    S_real = state["measurement"].S
    d, m = U.shape
    rng = np.random.default_rng(SEED_BASE)
    sim_norms = []
    obs_noise_norms = []
    for r in range(N_REPS):
        J = draw_J(d, SEED_BASE + r, "dense")
        S_true = -np.linalg.solve(J, U)
        sim_norms.append(np.linalg.norm(S_true, axis=0))
        S_obs = S_true + sigma * rng.normal(size=S_true.shape)
        obs_noise_norms.append(np.linalg.norm(S_obs, axis=0))
    sim_norms = np.concatenate(sim_norms)
    obs_noise_norms = np.concatenate(obs_noise_norms)
    real_col_norms = np.linalg.norm(S_real, axis=0)
    U_col_norms = np.linalg.norm(U, axis=0)
    return {
        "cell_line": tag,
        "d": d,
        "n_guides": m,
        "sigma": sigma,
        "median_norm_S_sim_alpha1_dense": float(np.median(sim_norms)),
        "median_norm_S_sim_plus_noise": float(np.median(obs_noise_norms)),
        "median_norm_S_observed": float(np.median(real_col_norms)),
        "mean_norm_S_observed": float(np.mean(real_col_norms)),
        "median_norm_U": float(np.median(U_col_norms)),
        "sim_over_obs_ratio_median": float(np.median(real_col_norms) / max(np.median(sim_norms), 1e-30)),
        # per-entry SNR: signal std vs noise std, per matrix entry
        "sim_per_entry_snr": float(np.std(sim_norms) / max(sigma, 1e-30)),  # crude
        "obs_per_entry_snr": float(np.std(real_col_norms) / max(sigma, 1e-30)),
    }


# -----------------------------------------------------------------------------
# F2: signal-scale sweep.
#   alpha_S scales the noise-free response directly:
#     J_true(alpha_S) = J_ref / alpha_S,
#     S_true          = -J_true^{-1} U = alpha_S * (-J_ref^{-1} U).
#   Cosine is scale-invariant in J, so cos(A_hat, J_true) tracks SNR alone.
#   We also compute alpha_hat_data: the multiplier that puts median sim
#   ||S|| at the observed median.
# -----------------------------------------------------------------------------
def f2_alpha_sweep(state, sigma, tag, alphas, structures=STRUCTURES, n_reps=N_REPS):
    U = state["measurement"].U
    d, m = U.shape
    S_real_median = float(np.median(np.linalg.norm(state["measurement"].S, axis=0)))
    out = {"cell_line": tag, "sigma": sigma, "S_real_median": S_real_median,
           "n_reps": n_reps, "structures": structures, "alphas_S": list(alphas), "results": {}}
    for struct in structures:
        by_alpha = []
        for alpha in alphas:
            cos_full_list, cos1_list, cos5_list, mag_ratio = [], [], [], []
            null_cos_full, null_cos1 = [], []
            sim_S_medians = []
            rng = np.random.default_rng(SEED_BASE + 7)
            Js, S_trues, A_fits = [], [], []
            for r in range(n_reps):
                J_ref = draw_J(d, SEED_BASE + r, struct)
                J = J_ref / alpha  # J_true = J_ref / alpha_S
                S_true = -np.linalg.solve(J, U)  # = alpha_S * (-J_ref^{-1} U)
                sim_S_medians.append(np.median(np.linalg.norm(S_true, axis=0)))
                S_obs = S_true + sigma * rng.normal(size=S_true.shape)
                A_hat = fit_A(S_obs, U)
                Js.append(J); S_trues.append(S_true); A_fits.append(A_hat)
                cos_full_list.append(cos_full(A_hat, J))
                cos1_list.append(cos_topk_of(A_hat, J, S_true, 1))
                cos5_list.append(cos_topk_of(A_hat, J, S_true, min(5, d)))
                mag_ratio.append(np.linalg.norm(A_hat) / max(np.linalg.norm(J), 1e-30))
            for r in range(n_reps):
                rp = (r + 1) % n_reps
                null_cos_full.append(cos_full(A_fits[r], Js[rp]))
                null_cos1.append(cos_topk_of(A_fits[r], Js[rp], S_trues[rp], 1))
            by_alpha.append({
                "alpha_S": float(alpha),
                "median_sim_norm_S": float(np.median(sim_S_medians)),
                "cos_full_mean": float(np.mean(cos_full_list)),
                "cos_full_median": float(np.median(cos_full_list)),
                "cos_full_p05": float(np.percentile(cos_full_list, 5)),
                "cos_full_p95": float(np.percentile(cos_full_list, 95)),
                "cos_full_paired_null_mean": float(np.mean(null_cos_full)),
                "cos_full_paired_diff_mean": float(np.mean(cos_full_list) - np.mean(null_cos_full)),
                "cos_1_mean": float(np.mean(cos1_list)),
                "cos_1_paired_null_mean": float(np.mean(null_cos1)),
                "cos_1_paired_diff_mean": float(np.mean(cos1_list) - np.mean(null_cos1)),
                "cos_5_mean": float(np.mean(cos5_list)),
                "magnitude_ratio_mean": float(np.mean(mag_ratio)),
            })
        out["results"][struct] = by_alpha
    # alpha_hat_data: since ||S|| ∝ alpha_S linearly here (S_true = alpha_S * X),
    # alpha_hat = observed_median / (median_sim_norm at alpha_S = 1).
    dense = out["results"]["dense"]
    # find the alpha_S = 1 row
    ref = next(row for row in dense if abs(row["alpha_S"] - 1.0) < 1e-9)
    alpha_hat = float(S_real_median / max(ref["median_sim_norm_S"], 1e-30))
    out["alpha_hat_data_dense"] = alpha_hat
    return out


# -----------------------------------------------------------------------------
# F3: linearity diagnostic (held_out_prediction_check + linearity_check)
#     on stored measurements AND on linear simulations at published and matched
#     SNR. Confirms rho > 1 is not overfitting (900 params, ~150 guides).
# -----------------------------------------------------------------------------
def f3_linearity(state, sigma, tag, alpha_hat, n_reps=15):
    U = state["measurement"].U
    d, m = U.shape

    # 1) Observed on stored measurement.
    meas_real = state["measurement"]
    hop_real = ao.held_out_prediction_check(meas_real, n_folds=5, seed=SEED_BASE)
    linc_real = ao.linearity_check(meas_real)

    # Helper: build a synthetic MeasuredOperator with same guide_names / kappa etc.
    def _sim_measure(J, sigma_val, seed):
        S_true = -np.linalg.solve(J, U)
        S_obs = S_true + sigma_val * np.random.default_rng(seed).normal(size=S_true.shape)
        report = meas_real.report
        # guide_targets: derive from guide names ("guide_<ENSG>") if not on report.
        gt = getattr(report, "guide_targets", None) or _derive_targets(list(report.retained_guides))
        return ao.measure_from_sensitivity(
            S=S_obs, U=U,
            guide_names=list(report.retained_guides),
            guide_efficiencies=dict(report.guide_efficiencies),
            guide_targets=gt,
            reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
        )

    def _rho_stats(alpha_S, sigma_val, tag_):
        rhos_lin, rel_diffs = [], []
        for r in range(n_reps):
            J_ref = draw_J(d, SEED_BASE + r, "dense")
            J = J_ref / max(alpha_S, 1e-30)  # so ||S_true|| = alpha_S * ||-J_ref^{-1} U||
            sim = _sim_measure(J, sigma_val, SEED_BASE + 1000 + r)
            hop = ao.held_out_prediction_check(sim, n_folds=5, seed=SEED_BASE + r)
            linc = ao.linearity_check(sim)
            rhos_lin.append(float(hop.rho_pooled))
            rel_diffs.append(float(linc.relative_difference))
        return {"tag": tag_, "alpha_S": float(alpha_S),
                "rho_mean": float(np.mean(rhos_lin)), "rho_median": float(np.median(rhos_lin)),
                "rho_std": float(np.std(rhos_lin)),
                "rel_diff_mean": float(np.mean(rel_diffs)), "rel_diff_median": float(np.median(rel_diffs)),
                "n_reps": n_reps}

    sim_pub = _rho_stats(1.0, sigma, "linear-sim @ published alpha_S=1")
    sim_matched = _rho_stats(alpha_hat, sigma, f"linear-sim @ matched alpha_S={alpha_hat:.2f}")

    return {
        "cell_line": tag, "sigma": sigma, "alpha_hat_S": alpha_hat,
        "d": d, "n_guides": m,
        "params_in_A": d * d,  # 30x30 = 900
        "rho_real": float(hop_real.rho_pooled),
        "rel_diff_real": float(linc_real.relative_difference),
        "sim_linear_published_snr": sim_pub,
        "sim_linear_matched_snr": sim_matched,
    }


# -----------------------------------------------------------------------------
# F4: shared response mode, target-input orthogonality, footprint encoding.
# -----------------------------------------------------------------------------
def f4_shared_mode(state, tag):
    meas = state["measurement"]
    S = meas.S  # d x m
    U = meas.U
    basis = state["basis"]
    W = np.asarray(basis.loadings)  # g x d, orthonormal columns
    d, m = S.shape

    # (a) Top singular direction energy of S.
    _, sv, _ = np.linalg.svd(S, full_matrices=False)
    energy = (sv ** 2)
    top_energy_frac = float(energy[0] / energy.sum())

    # (b) between-target mean cosine (columns of S are per-guide; group by target)
    guide_names = list(meas.report.retained_guides)
    tgt = getattr(meas.report, "guide_targets", None) or _derive_targets(guide_names)
    # Per-target mean response (columns for guides that share a target).
    from collections import defaultdict
    tgt_cols = defaultdict(list)
    for j, gn in enumerate(guide_names):
        t = tgt.get(gn)
        if t is not None:
            tgt_cols[t].append(j)
    # target-level response = mean over its guides (in program coords)
    tgt_names = sorted(tgt_cols)
    S_tgt = np.stack([S[:, tgt_cols[t]].mean(axis=1) for t in tgt_names], axis=1)  # d x T
    # normalize
    Sn = S_tgt / np.maximum(np.linalg.norm(S_tgt, axis=0, keepdims=True), 1e-30)
    C = Sn.T @ Sn
    iu = np.triu_indices_from(C, k=1)
    between_target_mean_cos = float(np.mean(C[iu]))

    # (c) |cos(s_g, u_g)| median per retained sgRNA
    cos_su = []
    for j in range(m):
        s = S[:, j]; u = U[:, j]
        c = abs(float(s @ u) / max(np.linalg.norm(s) * np.linalg.norm(u), 1e-30))
        cos_su.append(c)
    median_abs_cos_su = float(np.median(cos_su))
    # chance level under random unit d-vectors: mean |cos| ~ 2/(pi * sqrt(d-1))
    from math import pi, sqrt
    chance_abs_cos = float(2.0 / (pi * sqrt(max(d - 1, 1))))

    # (d) held-out rho after projecting out mean-response direction.
    v = S.mean(axis=1)
    v /= max(np.linalg.norm(v), 1e-30)
    P = np.eye(d) - np.outer(v, v)
    S_dm = P @ S
    U_dm = P @ U  # keep the same projection so that we're testing the same coord change
    report = meas.report
    meas_dm = ao.measure_from_sensitivity(
        S=S_dm, U=U_dm,
        guide_names=list(report.retained_guides),
        guide_efficiencies=dict(report.guide_efficiencies),
        guide_targets=(getattr(report, "guide_targets", None)
                       or _derive_targets(list(report.retained_guides))),
        reg="tsvd", reg_param="path", rank_tol=RANK_TOL,
    )
    hop_dm = ao.held_out_prediction_check(meas_dm, n_folds=5, seed=SEED_BASE)

    # (e) target lies mostly outside 30-PC span: ||W^T delta_g|| per target.
    gene_index = {g: i for i, g in enumerate(basis.gene_names)}
    in_span_norms = []
    out_of_span = 0
    for t in tgt_names:
        i = gene_index.get(t)
        if i is None:
            out_of_span += 1
            continue
        in_span_norms.append(float(np.linalg.norm(W[i])))  # rows of W give W^T delta_g
    # response norm per target
    resp_norms = np.linalg.norm(S_tgt, axis=0)

    # (f) footprint encoding u_g_foot ∝ W^T Sigma_ctrl delta_g. Exploratory only:
    # we don't have Sigma_ctrl on disk. Skip and note.

    return {
        "cell_line": tag,
        "d": d, "n_targets": len(tgt_names), "n_guides_retained": m,
        "top_sv_energy_fraction": top_energy_frac,
        "singular_values_top6": [float(x) for x in sv[:6]],
        "between_target_mean_cos": between_target_mean_cos,
        "median_abs_cos_s_u": median_abs_cos_su,
        "chance_abs_cos": chance_abs_cos,
        "rho_real_original_measurement": float(ao.held_out_prediction_check(meas, n_folds=5, seed=SEED_BASE).rho_pooled),
        "rho_after_meanproj_removed": float(hop_dm.rho_pooled),
        "targets_in_hvg_span": len(in_span_norms),
        "targets_outside_hvg_span": out_of_span,
        "median_delta_g_projected_norm": float(np.median(in_span_norms)) if in_span_norms else None,
        "median_response_norm_per_target": float(np.median(resp_norms)),
    }


def main():
    all_results = {"seed": SEED_BASE, "rank_tol": RANK_TOL, "n_reps": N_REPS}
    alpha_grid = [1.0, 3.0, 10.0, 30.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 3000.0]
    for tag, path in BUNDLE.items():
        with (RESULTS / path).open("rb") as f:
            state = pickle.load(f)
        sigma = DATASET_SIGMA[tag]
        print(f"\n### {tag} @ sigma={sigma}\n")

        f1 = f1_amplitude(state, sigma, tag)
        print("F1:", json.dumps(f1, indent=2))
        (OUT / f"F1_{tag}.json").write_text(json.dumps(f1, indent=2))

        f2 = f2_alpha_sweep(state, sigma, tag, alpha_grid, n_reps=50)
        print("F2 alpha_hat_data (dense):", f2["alpha_hat_data_dense"])
        (OUT / f"F2_{tag}.json").write_text(json.dumps(f2, indent=2))

        f3 = f3_linearity(state, sigma, tag, f2["alpha_hat_data_dense"], n_reps=15)
        print("F3:", json.dumps(f3, indent=2))
        (OUT / f"F3_{tag}.json").write_text(json.dumps(f3, indent=2))

        f4 = f4_shared_mode(state, tag)
        print("F4:", json.dumps(f4, indent=2))
        (OUT / f"F4_{tag}.json").write_text(json.dumps(f4, indent=2))

        all_results[tag] = {"F1": f1, "F2_alpha_hat": f2["alpha_hat_data_dense"], "F3": f3, "F4": f4}
    (OUT / "F1_F4_summary.json").write_text(json.dumps(all_results, indent=2))
    print("\nsaved:", OUT / "F1_F4_summary.json")


if __name__ == "__main__":
    main()
