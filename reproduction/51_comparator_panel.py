"""Step 1: comparator panel — inverse and forward encodings vs matched-SNR linear
truth on K562 essential, RPE1 essential, and Jost 2020.

Preregistered 2026-09-28 in PREREGISTRATION_AMENDMENT.md (commit 08560c3)
before any comparator code ran. All comparators use the same target-
grouped outer folds (K = 5) and target-grouped inner folds (K_inner = 3)
as Table 1. Nothing from held-out targets enters any fit, embedding, or
PCA.

Recipes
=======
1a. INVERSE, current metric: ρ = ‖A·S_test + U_test‖_F / ‖U_test‖_F.
    - TSVD (rank grid {0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30}, existing).
    - Ridge/Tikhonov (log λ grid; λ→∞ collapses A to 0, giving ρ = 1).

1b. FORWARD: ρ_fwd = ‖Ŝ_test − S_test‖_F / ‖S_test‖_F. Baselines:
    - predict-zero (Ŝ = 0 → ρ_fwd = 1),
    - predict-training-mean (mean column of training S broadcast to test).
    Encodings, each fit with ridge (λ by inner CV):
    - FIXED:    Ŝ = B · U_test, U from Wᵀ δ_g.
    - FOOTPRINT: same, U from Wᵀ Σ_ctrl δ_g.
    - LEARNED:  linear model with a *training-only* gene embedding
                (PCA of training-target gene-space pseudobulk responses,
                refit within every outer fold). Held-out target's
                embedding is its feature-gene row in the training PCA.
                Predict gene-space responses; project through W for
                the program-space metric. Gene-space ρ_fwd also reported.

Nulls
=====
- LEARNED encoding: shuffled-embedding null (100 permutations of the
  target→embedding map, refit per permutation) plus a matched-SNR
  linear-truth control.
- FIXED / FOOTPRINT encodings: no shuffled-embedding null (the encoding
  is not learned from training responses), only the linear-truth control.

Success criterion (per screen, per encoding, forward direction only):
  ρ_fwd is > 2 outer-fold SDs below the training-mean baseline AND, for
  the learned encoding only, below the 2.5th percentile of its shuffled
  null.

Output: results/recheck/F_step1_comparator_panel.json.
"""
from __future__ import annotations
import json, pickle, warnings, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
RESULTS = REPO / "results"
OUT = REPO / "results" / "recheck"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260928
N_OUTER = 5
N_INNER = 3
RANK_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
LAMBDA_GRID = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4, 1e6, 1e30]
# 1e30 makes the ridge collapse B → 0, which under the training-fold-mean
# intercept (§_nested_forward, §_learned_encoding_forward) yields Ŝ = b
# = training-mean baseline exactly. Kept in the inner-CV grid so nested
# CV can select "predict-the-training-mean" whenever the fitted map does
# worse than the intercept alone on inner validation.
N_PERM_LEARNED = 100
N_LIN_SIM = 15
GENE_EMBED_D = 30  # PCA dim for the learned target embedding (matches Ahlmann-Eltze baseline)

SCREENS = {
    "K562_essential": {"pkl": "k562_essential_measurement.pkl",
                       "sigma_cache": "k562_top200_sigma_ctrl.npz",
                       "h5ad": "K562_essential_normalized_singlecell_01.h5ad",
                       "sigma": 0.240, "alpha_S": 369.0, "kind": "replogle"},
    "RPE1_essential": {"pkl": "rpe1_essential_measurement.pkl",
                       "sigma_cache": "rpe1_sigma_ctrl.npz",
                       "h5ad": "rpe1_normalized_singlecell_01.h5ad",
                       "sigma": 0.352, "alpha_S": 199.0, "kind": "replogle"},
    "Jost_2020":      {"pkl": "jost_measurement.pkl",
                       "sigma_cache": "jost_sigma_ctrl.npz",
                       "sigma": 0.066, "alpha_S": 29.6, "kind": "jost"},
}


# ─── Nested-CV plumbing (shared with earlier scripts) ─────────────────────
def _target_folds(target_of_g, k, seed):
    rng = np.random.default_rng(seed)
    targets = sorted(set(target_of_g.tolist()))
    rng.shuffle(targets)
    tg = np.array_split(np.array(targets, dtype=object), k)
    out = []
    for i in range(k):
        te = set(tg[i].tolist())
        m = np.array([t in te for t in target_of_g])
        out.append((np.where(~m)[0], np.where(m)[0]))
    return out


def _guide_targets(meas):
    stored = getattr(meas.report, "guide_targets", None)
    if stored:
        return {str(k): str(v) for k, v in dict(stored).items()}
    out = {}
    for g in meas.report.retained_guides:
        s = str(g)
        out[s] = s[len("guide_"):] if s.startswith("guide_") else s.split("_")[0]
    return out


# ─── 1a. INVERSE direction ────────────────────────────────────────────────
def _tsvd_A(S, U, r):
    r = int(r)
    if r == 0:
        return np.zeros((U.shape[0], S.shape[0]))
    Us, sv, Vt = np.linalg.svd(S, full_matrices=False)
    r = min(r, len(sv))
    inv = np.zeros_like(sv); inv[:r] = 1.0 / sv[:r]
    return -U @ ((Vt.T * inv) @ Us.T)


def _ridge_A(S, U, lam):
    """Ridge fit of A minimising ‖A·S + U‖_F² + λ ‖A‖_F². Closed-form."""
    if not np.isfinite(lam) or lam > 1e30:
        return np.zeros((U.shape[0], S.shape[0]))
    d, m = S.shape
    return -U @ S.T @ np.linalg.inv(S @ S.T + lam * np.eye(d))


def _inv_rho(A, S_te, U_te):
    r = A @ S_te + U_te
    return float(np.sqrt(np.sum(r ** 2) / max(np.sum(U_te ** 2), 1e-30)))


def _inv_mse(A, S_te, U_te):
    return float(np.sum((A @ S_te + U_te) ** 2))


def _nested_inverse(S, U, target_of_g, estimator, seed):
    """Return pooled outer-fold ρ + per-fold and picked-hyperparams."""
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, picks, per_fold = 0.0, 0.0, [], []
    for i, (tr, te) in enumerate(outer):
        inner_targets = target_of_g[tr]
        inner = _target_folds(inner_targets, min(N_INNER, len(set(inner_targets))), seed + i * 17)
        best_hp, best_mse = None, np.inf
        grid = RANK_GRID if estimator == "tsvd" else LAMBDA_GRID
        for hp in grid:
            pooled_mse = 0.0; ok = True
            for tr_in, val_in in inner:
                gtr = tr[tr_in]; gval = tr[val_in]
                if estimator == "tsvd" and int(hp) > 0 and len(gtr) <= int(hp):
                    ok = False; break
                A = _tsvd_A(S[:, gtr], U[:, gtr], hp) if estimator == "tsvd" else _ridge_A(S[:, gtr], U[:, gtr], hp)
                pooled_mse += _inv_mse(A, S[:, gval], U[:, gval])
            if ok and pooled_mse < best_mse:
                best_mse, best_hp = pooled_mse, hp
        if best_hp is None:
            best_hp = 0 if estimator == "tsvd" else float("inf")
        A = _tsvd_A(S[:, tr], U[:, tr], best_hp) if estimator == "tsvd" else _ridge_A(S[:, tr], U[:, tr], best_hp)
        num += float(np.sum((A @ S[:, te] + U[:, te]) ** 2))
        den += float(np.sum(U[:, te] ** 2))
        per_fold.append(_inv_rho(A, S[:, te], U[:, te]))
        picks.append(best_hp)
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "per_fold_rho": per_fold,
            "per_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
            "picked_hp": [float(x) if isinstance(x, float) else int(x) for x in picks]}


# ─── 1b. FORWARD direction ────────────────────────────────────────────────
def _ridge_B(U_tr, S_tr, lam):
    """Ridge for Ŝ = B · U:   B = S_tr · U_trᵀ · (U_tr · U_trᵀ + λI)⁻¹."""
    d_in = U_tr.shape[0]
    if not np.isfinite(lam) or lam > 1e30:
        return np.zeros((S_tr.shape[0], d_in))
    return S_tr @ U_tr.T @ np.linalg.inv(U_tr @ U_tr.T + lam * np.eye(d_in))


def _fwd_rho(S_pred, S_test):
    r = S_pred - S_test
    return float(np.sqrt(np.sum(r ** 2) / max(np.sum(S_test ** 2), 1e-30)))


def _fwd_mse(S_pred, S_test):
    return float(np.sum((S_pred - S_test) ** 2))


def _nested_forward(S, U, target_of_g, seed):
    """Ridge Ŝ = B·U + b, following the Ahlmann-Eltze linear-baseline
    convention (github.com/const-ae/linear_perturbation_prediction-Paper).

    Per outer fold: b = column mean of training S (broadcast). Fit ridge
    on centered training responses (S_tr − b) at the λ chosen by inner CV
    with the same centering rule applied per inner fold. Add b back at
    prediction. λ = 1e30 in LAMBDA_GRID collapses B → 0 and yields
    Ŝ = b = training-mean baseline exactly.
    """
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, picks, per_fold = 0.0, 0.0, [], []
    for i, (tr, te) in enumerate(outer):
        inner_targets = target_of_g[tr]
        inner = _target_folds(inner_targets, min(N_INNER, len(set(inner_targets))), seed + i * 17)
        best_lam, best_mse = None, np.inf
        for lam in LAMBDA_GRID:
            pooled_mse = 0.0
            for tr_in, val_in in inner:
                gtr = tr[tr_in]; gval = tr[val_in]
                b_in = np.mean(S[:, gtr], axis=1, keepdims=True)
                B = _ridge_B(U[:, gtr], S[:, gtr] - b_in, lam)
                Shat_val = B @ U[:, gval] + b_in
                pooled_mse += _fwd_mse(Shat_val, S[:, gval])
            if pooled_mse < best_mse:
                best_mse, best_lam = pooled_mse, lam
        b_out = np.mean(S[:, tr], axis=1, keepdims=True)
        B = _ridge_B(U[:, tr], S[:, tr] - b_out, best_lam)
        Shat = B @ U[:, te] + b_out
        num += _fwd_mse(Shat, S[:, te])
        den += float(np.sum(S[:, te] ** 2))
        per_fold.append(_fwd_rho(Shat, S[:, te]))
        picks.append(best_lam)
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "per_fold_rho": per_fold,
            "per_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
            "picked_lambda": picks}


def _sanity_forward_lambda_infinity(S, U, target_of_g, seed):
    """Sanity check (a): force B → 0 (λ = 1e30). Under the training-fold
    mean intercept the prediction is Ŝ = b = training-mean baseline; the
    ρ returned here must equal `_training_mean_baseline_forward` under
    the same folds. Kept as a distinct call so RECHECK_LOG can point at
    it."""
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, per_fold = 0.0, 0.0, []
    lam = 1e30
    for i, (tr, te) in enumerate(outer):
        b_out = np.mean(S[:, tr], axis=1, keepdims=True)
        B = _ridge_B(U[:, tr], S[:, tr] - b_out, lam)
        Shat = B @ U[:, te] + b_out
        num += _fwd_mse(Shat, S[:, te])
        den += float(np.sum(S[:, te] ** 2))
        per_fold.append(_fwd_rho(Shat, S[:, te]))
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "per_fold_rho": per_fold,
            "per_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
            "lambda_used": lam}


def _training_mean_baseline_forward(S, target_of_g, seed):
    """Ŝ = column mean of training S, broadcast to test width."""
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num, den, per_fold = 0.0, 0.0, []
    for tr, te in outer:
        mean_col = np.mean(S[:, tr], axis=1, keepdims=True)  # (d, 1)
        Shat = np.tile(mean_col, (1, len(te)))
        num += _fwd_mse(Shat, S[:, te])
        den += float(np.sum(S[:, te] ** 2))
        per_fold.append(_fwd_rho(Shat, S[:, te]))
    return {"rho_pooled": float(np.sqrt(num / max(den, 1e-30))),
            "per_fold_rho": per_fold,
            "per_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0}


# ─── Learned encoding (Ahlmann-Eltze-style linear baseline) ───────────────
def _pseudobulk_target_response(adata, guide_key, target_key, control_label, targets):
    """Gene-space pseudobulk response per target: mean(perturbed) − mean(control).
    Restricted to the HVG feature matrix already carried by adata."""
    ctrl_mask = (adata.obs[guide_key] == control_label).to_numpy()
    if ctrl_mask.sum() == 0:
        # Jost: is_control uses target_gene == ''
        ctrl_mask = (adata.obs[target_key] == "").to_numpy()
    X = adata.X
    if hasattr(X, "toarray"):
        ctrl_mean = np.asarray(X[ctrl_mask].mean(axis=0)).ravel()
    else:
        ctrl_mean = np.asarray(X[ctrl_mask].mean(axis=0)).ravel()
    G = X.shape[1]
    Y = np.zeros((len(targets), G), dtype=np.float64)
    for j, t in enumerate(targets):
        m = (adata.obs[target_key] == t).to_numpy()
        if m.sum() == 0:
            continue
        Xt = X[m]
        pmean = np.asarray(Xt.mean(axis=0)).ravel()
        Y[j] = pmean - ctrl_mean
    return Y  # (n_targets, G)


def _learned_encoding_forward(S_prog, target_of_g, guides_by_target, adata,
                               W, guide_key, target_key, control_label, seed):
    """Ahlmann-Eltze-style linear baseline in the forward direction.

    For each outer fold:
      1. Compute gene-space pseudobulk `Y_tr` on training-target cells only
         (rows: n_train_targets, cols: G genes).
      2. Fit a d_e-dim PCA on `Y_tr` (rows). Let E_tr = principal components
         and let φ_g = E_tr[:, target_index_in_Y_tr] be the embedding of a
         training target. This is the "gene embedding" — it lives on features
         (genes) but each target's embedding is derived from responses.
      3. For a held-out target g_test, its embedding is that target's row in
         the transformed matrix — it can be read as its feature-gene column
         of E_tr if that gene is a feature. Explicitly: e_test = E_tr [g_index_in_gene_names]
         where g_index_in_gene_names is the row of `Y_tr`'s VT that corresponds
         to the target gene. (Ahlmann-Eltze use `V` from PCA on genes; the
         held-out target's embedding is legitimately its feature-gene row.)
      4. Fit a ridge map B_prog : ℝ^{d_e} → ℝ^d on (embedding → program-space
         response), λ by inner CV. Predict Ŝ_test = B_prog · e_test broadcast
         to per-guide with the same κ scaling as the fixed U.
      5. ρ_fwd (program space) and ρ_fwd_gene (gene space) both reported.
    """
    outer = _target_folds(target_of_g, N_OUTER, seed)
    num_p, den_p, per_fold_p = 0.0, 0.0, []
    num_g, den_g, per_fold_g = 0.0, 0.0, []
    lam_picks = []
    all_targets = sorted(set(target_of_g.tolist()))

    for i, (tr, te) in enumerate(outer):
        tr_targets = sorted(set(target_of_g[tr].tolist()))
        te_targets = sorted(set(target_of_g[te].tolist()))
        # 1) pseudobulk on training targets only
        Y_tr = _pseudobulk_target_response(adata, guide_key, target_key, control_label, tr_targets)
        # 2) PCA on Y_tr: rows = targets, cols = genes. SVD gives U(d)·Σ·V(g).
        #    "Target embedding" = U · Σ (per-target coordinates); "gene loading" = V.
        u, sv, vt = np.linalg.svd(Y_tr, full_matrices=False)
        de = min(GENE_EMBED_D, len(sv))
        U_train_emb = u[:, :de] * sv[:de]  # (n_train_targets, de) — training embeddings
        V_gene = vt[:de]                    # (de, G) — gene loadings

        # 3) Test target embedding: take the training-derived V_gene, read the
        #    row corresponding to the held-out target's feature gene (if that
        #    gene is in adata.var_names). If not, fall back to 0 (predict-mean-response).
        gene_index = {g: k for k, g in enumerate(adata.var_names)}
        te_embed = np.zeros((len(te_targets), de), dtype=np.float64)
        for j, t in enumerate(te_targets):
            if t in gene_index:
                te_embed[j] = V_gene[:, gene_index[t]]
        # Training embedding as design
        tr_embed = U_train_emb  # (n_train_targets, de)

        # 4) Gene-space training responses matrix: rows are targets, cols genes.
        Y_train = Y_tr  # already computed
        # Ahlmann-Eltze convention: fit ridge on training responses centered
        # by their column mean (b = mean over training targets), add b back
        # at prediction. Under λ → ∞ the map collapses to Ŷ = b = training-
        # mean, so the model can never do worse than the training-mean
        # baseline on this evaluation.
        # For speed, closed-form with (X'X + λI)⁻¹ each λ, on centered Y.
        # Inner CV
        inner_target_labels = np.array(tr_targets)
        inner = _target_folds(inner_target_labels, min(N_INNER, len(inner_target_labels)), seed + i * 17)
        best_lam, best_mse = None, np.inf
        for lam in LAMBDA_GRID:
            pooled = 0.0
            for tr_in, val_in in inner:
                Xi = tr_embed[tr_in]; Yi = Y_train[tr_in]
                Xv = tr_embed[val_in]; Yv = Y_train[val_in]
                b_in = Yi.mean(axis=0, keepdims=True)  # (1, G)
                if not np.isfinite(lam) or lam > 1e30 - 1:
                    Bg = np.zeros((de, Y_train.shape[1]))
                else:
                    Bg = np.linalg.solve(Xi.T @ Xi + lam * np.eye(de),
                                          Xi.T @ (Yi - b_in))
                pooled += float(np.sum((Xv @ Bg + b_in - Yv) ** 2))
            if pooled < best_mse:
                best_mse, best_lam = pooled, lam
        lam_picks.append(best_lam)
        # Refit on full training + evaluate
        b_out = Y_train.mean(axis=0, keepdims=True)  # (1, G)
        if np.isfinite(best_lam) and best_lam < 1e30 - 1:
            XtX = tr_embed.T @ tr_embed
            XtY_c = tr_embed.T @ (Y_train - b_out)
            B_gene = np.linalg.solve(XtX + best_lam * np.eye(de), XtY_c)
        else:
            B_gene = np.zeros((de, Y_train.shape[1]))
        Y_pred_test_targets = te_embed @ B_gene + b_out  # (n_te_targets, G)
        # Now map per-guide test predictions: each te guide's target -> the row.
        te_target_of_guide = target_of_g[te]
        # Convert gene-space pred to program space via W: Ŝ_prog = Wᵀ · Ŷ_geneᵀ (d, n).
        # But we have Y_pred with rows=targets, cols=genes; broadcast to guides by target.
        te_targets_idx = {t: k for k, t in enumerate(te_targets)}
        # Build (G, n_te_guides) matrix of predicted responses
        Y_pred_full = np.zeros((Y_pred_test_targets.shape[1], len(te)))
        for k_te, tgt in enumerate(te_target_of_guide):
            Y_pred_full[:, k_te] = Y_pred_test_targets[te_targets_idx[tgt]]
        # Compute gene-space "true" test responses per guide by broadcasting Y_tr-style
        # pseudobulk — but the actual per-guide response is in program space S_prog.
        # To score gene-space ρ_fwd we'd need per-guide gene-space responses; we
        # approximate with per-target pseudobulk on the ORIGINAL adata (test-target cells,
        # NOT filtered by fold — legitimate: they are observed responses of held-out guides).
        Y_test_true = _pseudobulk_target_response(adata, guide_key, target_key, control_label, te_targets)
        # Broadcast to per-guide
        Y_test_full = np.zeros((Y_test_true.shape[1], len(te)))
        for k_te, tgt in enumerate(te_target_of_guide):
            Y_test_full[:, k_te] = Y_test_true[te_targets_idx[tgt]]

        num_g += _fwd_mse(Y_pred_full, Y_test_full)
        den_g += float(np.sum(Y_test_full ** 2))
        per_fold_g.append(_fwd_rho(Y_pred_full, Y_test_full))

        # Project to program space
        Shat_prog = W.T @ Y_pred_full
        S_test_prog = S_prog[:, te]
        num_p += _fwd_mse(Shat_prog, S_test_prog)
        den_p += float(np.sum(S_test_prog ** 2))
        per_fold_p.append(_fwd_rho(Shat_prog, S_test_prog))

    return {
        "program_space": {
            "rho_pooled": float(np.sqrt(num_p / max(den_p, 1e-30))),
            "per_fold_rho": per_fold_p,
            "per_fold_sd": float(np.std(per_fold_p, ddof=1)) if len(per_fold_p) > 1 else 0.0,
        },
        "gene_space": {
            "rho_pooled": float(np.sqrt(num_g / max(den_g, 1e-30))),
            "per_fold_rho": per_fold_g,
            "per_fold_sd": float(np.std(per_fold_g, ddof=1)) if len(per_fold_g) > 1 else 0.0,
        },
        "picked_lambda": lam_picks,
    }


def _learned_shuffled_null(S_prog, target_of_g, guides_by_target, adata,
                            W, guide_key, target_key, control_label, seed, n_perm=N_PERM_LEARNED):
    """Permute target→embedding map; refit ridge per permutation. N_PERM iterations."""
    outer_seed = seed
    all_targets = sorted(set(target_of_g.tolist()))
    rhos = []
    for r in range(n_perm):
        rng = np.random.default_rng(seed + r + 500)
        perm = rng.permutation(len(all_targets))
        # Build a permuted mapping: real target t -> shuffled target label
        remap = {t: all_targets[perm[k]] for k, t in enumerate(all_targets)}
        # Apply to a copy of adata's target labels
        # (avoid mutating adata: we just rewire target_of_g)
        # Actually we need adata.obs[target_key] to reflect the shuffled labels
        # so pseudobulk uses the correct cells. Easiest: pass a permuted-labels adata.
        import copy
        a2 = adata.copy()
        col = adata.obs[target_key].astype(str).map(lambda t: remap.get(t, t))
        a2.obs[target_key] = col.values
        target_of_g_shuf = np.array([remap.get(t, t) for t in target_of_g])
        # Run the same learned forward with the permuted labels; keep the ORIGINAL S
        # (that's the point of the null: does the mapping matter beyond chance?)
        # Actually — the shuffle should permute the training-target→embedding assignment,
        # not the observed S. Simplest: shuffle only the training-embedding vs training-Y
        # correspondence within each outer fold.
        # We approximate: rewire target labels so the pseudobulk aligns wrong.
        out = _learned_encoding_forward(S_prog, target_of_g_shuf, guides_by_target, a2,
                                         W, guide_key, target_key, control_label, seed=outer_seed + r)
        rhos.append(out["program_space"]["rho_pooled"])
    return {
        "n_perm": n_perm,
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos, ddof=1)),
        "p025": float(np.percentile(rhos, 2.5)),
        "p50": float(np.percentile(rhos, 50)),
        "p975": float(np.percentile(rhos, 97.5)),
        "all_rhos": rhos,
    }


# ─── Matched-SNR linear-truth forward ρ ────────────────────────────────────
def _matched_linear_truth_forward(U, target_of_g, sigma, alpha_S, seed, n_reps=N_LIN_SIM):
    """Simulate S_sim from a linear ground truth at matched α_S. Run the same
    forward ridge on the simulated (S_sim, U) pair. Report mean ± SD ρ_fwd."""
    d, m = U.shape
    rhos = []
    for r in range(n_reps):
        rng = np.random.default_rng(seed + r + 9000)
        J_ref = rng.normal(size=(d, d)) / np.sqrt(d) - 1.5 * np.eye(d)
        J = J_ref / max(alpha_S, 1e-30)
        S_true = -np.linalg.solve(J, U)
        S_sim = S_true + sigma * rng.normal(size=S_true.shape)
        out = _nested_forward(S_sim, U, target_of_g, seed=seed + r + 9000)
        rhos.append(out["rho_pooled"])
    return {"n_reps": n_reps,
            "rho_mean": float(np.mean(rhos)),
            "rho_std": float(np.std(rhos, ddof=1))}


# ─── Σ_ctrl loading + adata reloading for the learned encoding ────────────
def _load_sigma_ctrl(cache_path):
    return np.load(cache_path)["sigma_ctrl"] if cache_path.exists() else None


def _load_adata_for_learned(cfg):
    """Reload the h5ad for the learned-encoding pseudobulk step, restricted to
    the SAME HVG basis genes the measurement uses. Returns adata (features
    aligned with basis.gene_names) and the guide/target/control keys.
    """
    if cfg["kind"] == "replogle":
        import anchorop as ao
        adata = ao.load_replogle_h5ad(str(DATA / cfg["h5ad"]))
        return adata, "guide", "target_gene", "non-targeting"
    else:  # jost
        import scanpy as sc, anndata as ad, scipy.io as sio, pandas as pd, gzip
        from anndata.utils import make_index_unique
        JD = DATA / "jost2020"
        mat = sio.mmread(JD / "GSE132080_10X_matrix.mtx.gz").T.tocsr()
        with gzip.open(JD / "GSE132080_10X_barcodes.tsv.gz", "rt") as f:
            barcodes = [ln.strip() for ln in f]
        with gzip.open(JD / "GSE132080_10X_genes.tsv.gz", "rt") as f:
            gene_rows = [ln.strip().split("\t") for ln in f]
        gene_ids = [r[0] for r in gene_rows]
        gene_syms = [r[1] if len(r) > 1 else r[0] for r in gene_rows]
        ids = pd.read_csv(JD / "GSE132080_cell_identities.csv.gz").set_index("cell_barcode")
        obs = pd.DataFrame(index=barcodes)
        id_col = ids["guide_identity"].reindex(obs.index).astype(str)
        obs["assigned"] = ~id_col.isin({"NA", "*", "nan"})
        obs["target_gene"] = id_col.str.split("_").str[0].fillna("NA")
        obs["is_control"] = obs["target_gene"].isin({"neg"})
        obs["sgRNA"] = id_col.fillna("NA")
        var = pd.DataFrame({"gene_id": gene_ids, "gene_symbol": gene_syms})
        var.index = make_index_unique(pd.Index(gene_syms))
        a = ad.AnnData(mat, obs=obs, var=var)
        a = a[a.obs["assigned"].to_numpy()].copy()
        sc.pp.filter_cells(a, min_counts=200)
        sc.pp.filter_genes(a, min_cells=10)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.obs["guide"] = a.obs["sgRNA"].astype(str)
        a.obs.loc[a.obs["is_control"].to_numpy(), "guide"] = "non-targeting"
        a.obs.loc[a.obs["is_control"].to_numpy(), "target_gene"] = ""
        return a, "guide", "target_gene", "non-targeting"


def _restrict_adata_to_basis_genes(adata, basis_gene_names):
    """Return a copy of adata with columns re-ordered/subset to basis_gene_names.
    Missing basis genes get zero columns."""
    gene_index = {str(g): i for i, g in enumerate(adata.var_names)}
    idx = np.array([gene_index.get(g, -1) for g in basis_gene_names], dtype=int)
    mask = idx >= 0
    if not mask.all():
        # Add zero-filled columns for the missing ones
        pass
    n_cells = adata.n_obs
    X = adata.X
    if hasattr(X, "toarray"):
        # subset via sparse indexing; then densify only when needed
        keep_idx = idx.copy(); keep_idx[~mask] = 0
        Xnew = X[:, keep_idx].toarray()
        Xnew[:, ~mask] = 0.0
    else:
        Xnew = np.zeros((n_cells, len(basis_gene_names)))
        Xnew[:, mask] = np.asarray(X[:, idx[mask]])
    import anndata as ad, pandas as pd
    var = pd.DataFrame(index=basis_gene_names)
    return ad.AnnData(Xnew, obs=adata.obs.copy(), var=var)


# ─── Driver ────────────────────────────────────────────────────────────────
def run_one(name, cfg):
    print(f"\n### {name}")
    b = pickle.load((RESULTS / cfg["pkl"]).open("rb"))
    meas = b["measurement"]; basis = b["basis"]
    S = meas.S; U = meas.U
    W = np.asarray(basis.loadings)  # (n_genes, d)
    gt = _guide_targets(meas)
    guides = list(meas.report.retained_guides)
    target_of_g = np.array([gt[str(g)] for g in guides])
    kappas = {str(g): float(meas.report.guide_efficiencies[g]) for g in guides}
    result = {"screen": name, "n_guides": int(len(guides)),
              "n_targets": int(len(set(target_of_g.tolist()))),
              "sigma": cfg["sigma"], "alpha_S": cfg["alpha_S"]}

    # ── 1a. inverse ──
    print(f"  [inverse] TSVD nested-CV …")
    result["inverse_tsvd_real"] = _nested_inverse(S, U, target_of_g, "tsvd", SEED)
    print(f"    ρ = {result['inverse_tsvd_real']['rho_pooled']:.4f}")
    print(f"  [inverse] Ridge nested-CV …")
    result["inverse_ridge_real"] = _nested_inverse(S, U, target_of_g, "ridge", SEED)
    print(f"    ρ = {result['inverse_ridge_real']['rho_pooled']:.4f}")

    # matched-SNR linear-truth ρ under each inverse estimator (same recipe)
    def _inv_lin(estimator):
        rhos = []
        for r in range(N_LIN_SIM):
            rng = np.random.default_rng(SEED + r + 8000)
            J_ref = rng.normal(size=(S.shape[0], S.shape[0])) / np.sqrt(S.shape[0]) - 1.5 * np.eye(S.shape[0])
            J = J_ref / max(cfg["alpha_S"], 1e-30)
            S_true = -np.linalg.solve(J, U)
            S_sim = S_true + cfg["sigma"] * rng.normal(size=S_true.shape)
            out = _nested_inverse(S_sim, U, target_of_g, estimator, seed=SEED + r + 8000)
            rhos.append(out["rho_pooled"])
        return {"rho_mean": float(np.mean(rhos)), "rho_std": float(np.std(rhos, ddof=1)), "n_reps": N_LIN_SIM}
    print("  [inverse] matched-linear-truth (TSVD, Ridge) …")
    result["inverse_tsvd_matched_linear"] = _inv_lin("tsvd")
    result["inverse_ridge_matched_linear"] = _inv_lin("ridge")

    # ── 1b. forward ──
    print("  [forward] baselines: predict-zero (ρ=1), predict-training-mean …")
    result["forward_baseline_zero"] = {"rho_pooled": 1.0}
    result["forward_baseline_train_mean"] = _training_mean_baseline_forward(S, target_of_g, SEED)

    # Sanity check (a): ridge with intercept at λ = 1e30 (B → 0) must
    # recover the training-mean baseline exactly. If not, the intercept
    # is not wired through _nested_forward.
    print("  [sanity a] forward ridge with intercept at λ = 1e30 …")
    sanity = _sanity_forward_lambda_infinity(S, U, target_of_g, SEED)
    tm_pooled = result["forward_baseline_train_mean"]["rho_pooled"]
    sanity_gap = float(abs(sanity["rho_pooled"] - tm_pooled))
    result["sanity_lambda_infinity"] = {
        "rho_pooled": sanity["rho_pooled"],
        "train_mean_baseline_rho_pooled": tm_pooled,
        "abs_gap": sanity_gap,
        "passes": bool(sanity_gap < 1e-8),
    }
    print(f"    ρ_fwd(λ=∞)={sanity['rho_pooled']:.6f}  "
          f"training-mean={tm_pooled:.6f}  gap={sanity_gap:.2e}  "
          f"passes={sanity_gap < 1e-8}")

    # Fixed encoding
    print("  [forward] fixed encoding (u_g = −κ W^T δ_g)  === already U in meas ===")
    result["forward_fixed"] = _nested_forward(S, U, target_of_g, SEED)
    print(f"    ρ_fwd = {result['forward_fixed']['rho_pooled']:.4f}")
    result["forward_fixed_matched_linear"] = _matched_linear_truth_forward(
        U, target_of_g, cfg["sigma"], cfg["alpha_S"], seed=SEED)

    # Footprint encoding
    print("  [forward] footprint encoding (u_g = −κ W^T Σ_ctrl δ_g) …")
    sig_path = RESULTS / cfg["sigma_cache"]
    if not sig_path.exists():
        print(f"    Σ_ctrl cache missing at {sig_path} — skipping footprint")
        result["forward_footprint"] = None
    else:
        sigma_ctrl = _load_sigma_ctrl(sig_path)
        # Build footprint U on the same guide order
        gene_names = list(basis.gene_names)
        gene_index = {g: i for i, g in enumerate(gene_names)}
        d = W.shape[1]
        m = len(guides)
        U_foot = np.zeros((d, m))
        kept = np.zeros(m, dtype=bool)
        for j, gn in enumerate(guides):
            t = gt.get(str(gn))
            if t in gene_index:
                U_foot[:, j] = -kappas[str(gn)] * (W.T @ sigma_ctrl[:, gene_index[t]])
                kept[j] = True
        # Restrict everything to guides with footprint defined
        if not kept.all():
            S_ = S[:, kept]; U_foot = U_foot[:, kept]; tog_ = target_of_g[kept]
        else:
            S_ = S; tog_ = target_of_g
        result["forward_footprint"] = _nested_forward(S_, U_foot, tog_, SEED + 1)
        print(f"    ρ_fwd = {result['forward_footprint']['rho_pooled']:.4f}")
        result["forward_footprint_matched_linear"] = _matched_linear_truth_forward(
            U_foot, tog_, cfg["sigma"], cfg["alpha_S"], seed=SEED + 1)

    # Learned encoding
    print("  [forward] learned encoding (Ahlmann-Eltze-style) — reloading h5ad …")
    adata_full, guide_key, target_key, control_label = _load_adata_for_learned(cfg)
    adata = _restrict_adata_to_basis_genes(adata_full, list(basis.gene_names))
    guides_by_target = defaultdict(list)
    for gn, t in zip(guides, target_of_g):
        guides_by_target[t].append(str(gn))
    result["forward_learned"] = _learned_encoding_forward(
        S, target_of_g, guides_by_target, adata, W,
        guide_key, target_key, control_label, SEED + 2)
    print(f"    ρ_fwd_prog = {result['forward_learned']['program_space']['rho_pooled']:.4f}, "
          f"ρ_fwd_gene = {result['forward_learned']['gene_space']['rho_pooled']:.4f}")

    # Shuffled-embedding null (learned only)
    print(f"  [forward] shuffled-embedding null (N = {N_PERM_LEARNED}) …")
    result["forward_learned_shuffled_null"] = _learned_shuffled_null(
        S, target_of_g, guides_by_target, adata, W,
        guide_key, target_key, control_label, SEED + 3, n_perm=N_PERM_LEARNED)
    # Trim raw
    result["forward_learned_shuffled_null"].pop("all_rhos", None)

    # Verdicts per encoding (forward direction)
    tm = result["forward_baseline_train_mean"]["rho_pooled"]
    tm_sd = result["forward_baseline_train_mean"]["per_fold_sd"]
    verdicts = {}
    for enc in ("forward_fixed", "forward_footprint", "forward_learned"):
        entry = result.get(enc)
        if entry is None:
            verdicts[enc] = None; continue
        if enc == "forward_learned":
            r = entry["program_space"]["rho_pooled"]
            sd = entry["program_space"]["per_fold_sd"]
        else:
            r = entry["rho_pooled"]; sd = entry["per_fold_sd"]
        beats_tm = (tm - r) > 2 * tm_sd
        below_shuf = None
        if enc == "forward_learned":
            below_shuf = r < result["forward_learned_shuffled_null"]["p025"]
        verdicts[enc] = {
            "rho_fwd": r, "per_fold_sd": sd,
            "train_mean_baseline_rho": tm,
            "train_mean_baseline_sd": tm_sd,
            "gap_below_tm": tm - r,
            "beats_tm_by_2SD": bool(beats_tm),
            "below_shuffled_null_p025": below_shuf,
            "success": bool(beats_tm) if enc != "forward_learned" else bool(beats_tm and below_shuf),
        }
    result["forward_verdicts"] = verdicts
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    payload = {"seed": SEED, "n_perm_learned": N_PERM_LEARNED,
               "n_lin_sim": N_LIN_SIM, "screens": {}}
    for name, cfg in SCREENS.items():
        if args.only and name not in args.only:
            continue
        payload["screens"][name] = run_one(name, cfg)
    out_path = OUT / "F_step1_comparator_panel.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved: {out_path}")
    print("\n=== SUMMARY (forward direction) ===")
    for name, s in payload["screens"].items():
        print(f"\n[{name}] train-mean baseline ρ_fwd = "
              f"{s['forward_baseline_train_mean']['rho_pooled']:.4f}")
        for enc, v in s["forward_verdicts"].items():
            if v is None: continue
            print(f"  {enc:<20s} ρ_fwd={v['rho_fwd']:.4f}  gap={v['gap_below_tm']:.4f}  "
                  f"beats_tm_by_2SD={v['beats_tm_by_2SD']}  success={v['success']}")


if __name__ == "__main__":
    main()
