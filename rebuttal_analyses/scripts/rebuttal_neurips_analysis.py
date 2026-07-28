#!/usr/bin/env python3
"""
NeurIPS 2026 rebuttal: targeted additional analyses on the released predictions.

Addresses (from metareview + reviews):
  A. Act-or-defer / gating experiment (AC, odoV, Wkv8)
     - risk-coverage curves, AURC / E-AURC
     - selective accuracy at fixed coverage, coverage at fixed threshold
     - high-confidence error rate, worst-category selective accuracy
     - deferral-cost utility
     - correlation of ECE / Brier / group-ECE / PVC-VUS / PM-C-PVC-VUS
       with gating outcomes
  B. Group-wise ECE baselines (k5SN, AC)
     - macro / weighted / worst-category ECE, category Brier, equal-mass ECE
  C. Matched random-partition null test (AC, odoV)
     - 1000 random partitions matched to semantic category sizes
     - worst-group SEA / calib error, between-group variance, PM-C-PVC-VUS
     - percentile of semantic value in random null
  D. Held-out category-specific gating thresholds (generic-binning question)
     - calibration/evaluation split; global vs semantic vs random-group thresholds
  E. Bootstrap CIs for headline quantities

No new model inference; pure re-analysis of problem_evaluations_*.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 20260724
BASE = "./data/problem_evaluations"
OUT = "./results"
os.makedirs(OUT, exist_ok=True)

SMALL_FILES = {
    "Math360": "problem_evaluations_mathbenchmark.csv",
    "TruthfulQA": "problem_evaluations_truthfulQA.csv",
    "CSQA": "problem_evaluations_CSQA.csv",
    "Math500": "problem_evaluations_math500.csv",
}
HAIKU_FILES = {
    "Math360": "problem_evaluations_math_benchmark_haiku.csv",
    "TruthfulQA": "problem_evaluations_truthfulQA_haiku.csv",
    "CSQA": "problem_evaluations_CSQA_haiku.csv",
    "Math500": "problem_evaluations_math_500_haiku.csv",
}
MAIN_DATASETS = ["Math360", "TruthfulQA", "CSQA"]  # paper Table 1 aggregate
KEEP_LARGE = {"all_c35_haiku"}  # exclude Nova, as in prior rebuttal analyses

MODEL_LABEL = {"all_c35_haiku": "Claude 3.5 Haiku"}

# Six models highlighted in paper Table 2 (ranking-reversal set)
TABLE2_MODELS = [
    "JiuZhang3.0-7B", "Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct",
    "s1.1-7B", "Qwen2.5-32B-Instruct", "Claude 3.5 Haiku",
]

# dataviz palette (light mode)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.edgecolor": "#b5b4ad", "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
})


def load_all():
    frames = []
    for ds, fname in SMALL_FILES.items():
        df = pd.read_csv(os.path.join(BASE, fname))
        df["dataset"] = ds
        frames.append(df)
    for ds, fname in HAIKU_FILES.items():
        p = os.path.join(BASE, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = df[df["model_id"].isin(KEEP_LARGE)]
            df["dataset"] = ds
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["self_eval_correct"] = pd.to_numeric(df["self_eval_correct"], errors="coerce")
    df["self_eval_confidence"] = pd.to_numeric(df["self_eval_confidence"], errors="coerce")
    df = df.dropna(subset=["self_eval_correct", "self_eval_confidence"])
    df["model"] = df["model_id"].map(lambda m: MODEL_LABEL.get(m, m))
    return df


# ---------------------------------------------------------------- metrics ---

def ece_equal_width(conf, correct, n_bins=10):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return ece


def ece_equal_mass(conf, correct, n_bins=10):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    order = np.argsort(conf, kind="stable")
    splits = np.array_split(order, n_bins)
    n = len(conf); ece = 0.0
    for s in splits:
        if len(s):
            ece += (len(s) / n) * abs(conf[s].mean() - correct[s].mean())
    return ece


def brier(conf, correct):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2))


def groupwise_ece(df, group_col="category", n_bins=10):
    """macro / weighted / worst per-group equal-width ECE, group Brier."""
    rows = []
    for g, gdf in df.groupby(group_col):
        rows.append({
            "group": g, "n": len(gdf),
            "ece": ece_equal_width(gdf["self_eval_confidence"], gdf["self_eval_correct"], n_bins),
            "brier": brier(gdf["self_eval_confidence"], gdf["self_eval_correct"]),
            "sea": gdf["self_eval_correct"].mean(),
            "calib_gap": abs(gdf["self_eval_confidence"].mean() - gdf["self_eval_correct"].mean()),
        })
    g = pd.DataFrame(rows)
    w = g["n"] / g["n"].sum()
    return {
        "macro_ECE": g["ece"].mean(),
        "weighted_ECE": float((g["ece"] * w).sum()),
        "worst_cat_ECE": g["ece"].max(),
        "macro_Brier": g["brier"].mean(),
        "worst_cat_Brier": g["brier"].max(),
        "worst_cat_calib_gap": g["calib_gap"].max(),
        "worst_cat_SEA": g["sea"].min(),
        "between_group_SEA_var": g["sea"].var(ddof=0),
    }, g


def pm_cpvc_vus(df, group_col="category"):
    """PVC-VUS / C-PVC-VUS / PM-C-PVC-VUS on the paper's exact grid
    (gamma, tau in [0,1] with 0.01 steps; weight 2/(gamma-0.5) on the
    positive-margin region gamma - tau - 0.5 > 0, gamma > 0.5), matching
    pvc_analysis.calculate_vus_metrics."""
    cat = df.groupby(group_col).agg(
        sea=("self_eval_correct", "mean"),
        conf=("self_eval_confidence", "mean"),
    )
    cat["calib"] = (cat["conf"] - cat["sea"]).abs()
    gammas = np.round(np.arange(0.0, 1.0001, 0.01), 4)
    taus = np.round(np.arange(0.0, 1.0001, 0.01), 4)
    sea = cat["sea"].values[None, None, :]
    calib = cat["calib"].values[None, None, :]
    G = gammas[None, :, None]; T = taus[:, None, None]
    pvc_per_gamma = (cat["sea"].values[None, :] >= gammas[:, None]).sum(axis=1).astype(float)
    cpvc_grid = ((sea >= G) & (calib <= T)).sum(axis=2).astype(float)  # [tau, gamma]
    G2, T2 = G[:, :, 0], T[:, :, 0]
    pm_mask = (G2 - T2 - 0.5 > 0) & (G2 > 0.5)
    denom = np.where(G2 > 0.5, G2 - 0.5, 1.0)
    weight = np.where(pm_mask, 2.0 / denom, 0.0)
    area = (gammas[-1] - gammas[0]) * (taus[-1] - taus[0])
    pvc_vus = np.trapezoid(np.tile(pvc_per_gamma[None, :], (len(taus), 1)), taus, axis=0)
    pvc_vus = np.trapezoid(pvc_vus, gammas) / area
    cpvc_vus = np.trapezoid(np.trapezoid(cpvc_grid, taus, axis=0), gammas) / area
    pm = np.trapezoid(np.trapezoid(cpvc_grid * weight, taus, axis=0), gammas)
    return {"PVC_VUS": float(pvc_vus), "CPVC_VUS": float(cpvc_vus), "PM_CPVC_VUS": float(pm)}


# ------------------------------------------------------------- A. gating ---

def risk_coverage(conf, correct):
    """Return coverage grid, selective risk, AURC, E-AURC."""
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    n = len(conf)
    order = np.argsort(-conf, kind="stable")
    c = correct[order]
    cum_acc = np.cumsum(c) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    risk = 1 - cum_acc
    aurc = float(np.trapezoid(risk, cov))
    # optimal AURC: sort by correctness (oracle ordering)
    c_opt = np.sort(correct)[::-1]
    risk_opt = 1 - np.cumsum(c_opt) / np.arange(1, n + 1)
    aurc_opt = float(np.trapezoid(risk_opt, cov))
    return cov, risk, aurc, aurc - aurc_opt


def selective_at_coverage(conf, correct, target_cov):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    n = len(conf)
    k = max(1, int(round(target_cov * n)))
    order = np.argsort(-conf, kind="stable")
    return float(correct[order[:k]].mean())


def gate_at_threshold(conf, correct, thr):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    act = conf >= thr
    cov = float(act.mean())
    acc = float(correct[act].mean()) if act.sum() else np.nan
    return cov, acc


def deferral_utility(conf, correct, thr, lam=1.0):
    """U = P(act & correct) - lam * P(act & wrong); defer contributes 0.
    Ties (0.5) contribute half of each."""
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    act = conf >= thr
    return float(np.mean(np.where(act, correct - lam * (1 - correct), 0.0)))


def analysis_A_gating(df, metric_table):
    print("\n[A] Act-or-defer gating analysis")
    recs, curves = [], {}
    for scope_name, scope_df in [("cross3", df[df.dataset.isin(MAIN_DATASETS)])] + \
            [(ds, df[df.dataset == ds]) for ds in SMALL_FILES]:
        for model, mdf in scope_df.groupby("model"):
            conf, corr = mdf["self_eval_confidence"].values, mdf["self_eval_correct"].values
            cov, risk, aurc, eaurc = risk_coverage(conf, corr)
            if scope_name == "cross3":
                curves[model] = (cov, risk)
            row = {"scope": scope_name, "model": model, "N": len(mdf),
                   "SEA": corr.mean(), "AURC": aurc, "E_AURC": eaurc}
            for tc in [0.2, 0.5, 0.8]:
                row[f"sel_acc@cov{tc}"] = selective_at_coverage(conf, corr, tc)
            for thr in [0.8, 0.9, 0.95]:
                c, a = gate_at_threshold(conf, corr, thr)
                row[f"cov@thr{thr}"] = c
                row[f"acc@thr{thr}"] = a
                row[f"hc_err@thr{thr}"] = (1 - a) if not np.isnan(a) else np.nan
            row["util@thr0.9_lam1"] = deferral_utility(conf, corr, 0.9, 1.0)
            row["util@thr0.9_lam2"] = deferral_utility(conf, corr, 0.9, 2.0)
            # worst-category selective accuracy at 50% coverage
            wc = []
            for _, cdf in mdf.groupby("category"):
                if len(cdf) >= 10:
                    wc.append(selective_at_coverage(
                        cdf["self_eval_confidence"].values, cdf["self_eval_correct"].values, 0.5))
            row["worst_cat_sel_acc@cov0.5"] = min(wc) if wc else np.nan
            recs.append(row)
    gat = pd.DataFrame(recs)
    gat.to_csv(os.path.join(OUT, "A_gating_metrics.csv"), index=False)

    # --- correlation of diagnostics with gating outcomes (cross3 scope) ---
    g3 = gat[gat.scope == "cross3"].set_index("model")
    mt = metric_table.set_index("model")
    joined = g3.join(mt, how="inner")
    targets = ["AURC", "E_AURC", "sel_acc@cov0.5", "sel_acc@cov0.8",
               "hc_err@thr0.9", "worst_cat_sel_acc@cov0.5", "util@thr0.9_lam1"]
    predictors = ["ECE", "Brier", "macro_ECE", "worst_cat_ECE", "equal_mass_ECE",
                  "PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS"]
    corr_rows = []
    for t in targets:
        for p in predictors:
            sub = joined[[t, p]].dropna()
            if len(sub) >= 5:
                rho, pv = stats.spearmanr(sub[p], sub[t])
                corr_rows.append({"gating_outcome": t, "diagnostic": p,
                                  "spearman_rho": rho, "p_value": pv, "n_models": len(sub)})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(OUT, "A_diagnostic_vs_gating_correlation.csv"), index=False)
    print(corr.pivot(index="diagnostic", columns="gating_outcome", values="spearman_rho").round(3))

    # --- figure: risk-coverage for Table-2 models ---
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, m in enumerate([m for m in TABLE2_MODELS if m in curves]):
        cov, risk = curves[m]
        ax.plot(cov, risk, color=PAL[i % len(PAL)], lw=2, label=m)
    ax.set_xlabel("Coverage (fraction acted on)")
    ax.set_ylabel("Selective risk (1 − accuracy)")
    ax.set_title("Risk–coverage under confidence gating (Math360+TruthfulQA+CSQA)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "A_fig_risk_coverage.png"))
    plt.close(fig)

    # --- figure: PM-C-PVC-VUS vs gating outcome, ECE vs gating outcome ---
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    tgt = "sel_acc@cov0.5"
    for axi, pred, lab in [(0, "ECE", "ECE (lower = better)"),
                           (1, "PM_CPVC_VUS", "PM-C-PVC-VUS (higher = better)")]:
        sub = joined[[pred, tgt]].dropna()
        rho, _ = stats.spearmanr(sub[pred], sub[tgt])
        axes[axi].scatter(sub[pred], sub[tgt], s=34, color=PAL[0], zorder=3)
        for name, r in sub.iterrows():
            axes[axi].annotate(name.replace("-Instruct", "-I").replace("-Distill-Qwen", ""),
                               (r[pred], r[tgt]), fontsize=6, alpha=0.8,
                               xytext=(3, 3), textcoords="offset points")
        axes[axi].set_xlabel(lab)
        axes[axi].set_ylabel("Selective accuracy @ 50% coverage")
        axes[axi].set_title(f"Spearman ρ = {rho:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "A_fig_diagnostics_vs_gating.png"))
    plt.close(fig)
    return gat, corr, joined


# -------------------------------------------------------- B. group ECE ---

def analysis_B_group_ece(df):
    print("\n[B] Group-wise ECE baselines")
    recs = []
    for scope_name, scope_df in [("cross3", df[df.dataset.isin(MAIN_DATASETS)])] + \
            [(ds, df[df.dataset == ds]) for ds in SMALL_FILES]:
        for model, mdf in scope_df.groupby("model"):
            conf, corr = mdf["self_eval_confidence"].values, mdf["self_eval_correct"].values
            row = {"scope": scope_name, "model": model, "N": len(mdf),
                   "ECE": ece_equal_width(conf, corr),
                   "equal_mass_ECE": ece_equal_mass(conf, corr),
                   "Brier": brier(conf, corr)}
            gsum, _ = groupwise_ece(mdf)
            row.update(gsum)
            if scope_name == "cross3":
                # paper Table 1 convention: per-dataset VUS, then averaged
                # (category sets differ across datasets, so pooling is invalid)
                per_ds = [pm_cpvc_vus(dsub) for _, dsub in mdf.groupby("dataset")]
                row.update({k: float(np.mean([v[k] for v in per_ds]))
                            for k in ("PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS")})
            else:
                row.update(pm_cpvc_vus(mdf))
            recs.append(row)
    tab = pd.DataFrame(recs)
    tab.to_csv(os.path.join(OUT, "B_groupwise_ece_table.csv"), index=False)
    c3 = tab[tab.scope == "cross3"].sort_values("PM_CPVC_VUS", ascending=False)
    cols = ["model", "ECE", "equal_mass_ECE", "macro_ECE", "weighted_ECE",
            "worst_cat_ECE", "worst_cat_calib_gap", "PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS"]
    print(c3[cols].round(3).to_string(index=False))
    return tab


# ------------------------------------------- C. random-partition null ---

def matched_random_partition(mdf, sizes, rng):
    """Assign rows of mdf to random groups matching semantic category sizes."""
    idx = rng.permutation(len(mdf))
    labels = np.empty(len(mdf), dtype=int)
    start = 0
    for gi, s in enumerate(sizes):
        labels[idx[start:start + s]] = gi
        start += s
    return labels


def analysis_C_random_partitions(df, n_seeds=1000):
    print(f"\n[C] Matched random-partition null test ({n_seeds} seeds)")
    rng = np.random.default_rng(RNG_SEED)
    recs = []
    scope_df = df[df.dataset.isin(MAIN_DATASETS)]
    for (model, ds), mdf in scope_df.groupby(["model", "dataset"]):
        mdf = mdf.reset_index(drop=True)
        sizes = mdf.groupby("category").size().values
        sem_sum, sem_groups = groupwise_ece(mdf)
        sem_pm = pm_cpvc_vus(mdf)["PM_CPVC_VUS"]
        sem_worst_gap = sem_sum["worst_cat_calib_gap"]
        sem_worst_sea = sem_sum["worst_cat_SEA"]
        sem_var = sem_sum["between_group_SEA_var"]

        null_gap, null_sea, null_var, null_pm = [], [], [], []
        corr_arr = mdf["self_eval_correct"].values
        conf_arr = mdf["self_eval_confidence"].values
        for _ in range(n_seeds):
            lab = matched_random_partition(mdf, sizes, rng)
            seas = np.array([corr_arr[lab == g].mean() for g in range(len(sizes))])
            confs = np.array([conf_arr[lab == g].mean() for g in range(len(sizes))])
            gaps = np.abs(confs - seas)
            null_gap.append(gaps.max())
            null_sea.append(seas.min())
            null_var.append(seas.var())
            # PM-C-PVC-VUS on random groups
            tmp = mdf.copy()
            tmp["rand_group"] = lab
            null_pm.append(pm_cpvc_vus(tmp, group_col="rand_group")["PM_CPVC_VUS"])
        null_gap = np.array(null_gap); null_sea = np.array(null_sea)
        null_var = np.array(null_var); null_pm = np.array(null_pm)
        recs.append({
            "model": model, "dataset": ds, "n_categories": len(sizes),
            "sem_worst_calib_gap": sem_worst_gap,
            "null_worst_calib_gap_mean": null_gap.mean(),
            "null_worst_calib_gap_p95": np.quantile(null_gap, 0.95),
            "pctile_worst_calib_gap": (null_gap < sem_worst_gap).mean() * 100,
            "sem_worst_SEA": sem_worst_sea,
            "null_worst_SEA_mean": null_sea.mean(),
            "null_worst_SEA_p05": np.quantile(null_sea, 0.05),
            "pctile_worst_SEA_low": (null_sea > sem_worst_sea).mean() * 100,
            "sem_between_SEA_var": sem_var,
            "null_between_SEA_var_mean": null_var.mean(),
            "pctile_between_var": (null_var < sem_var).mean() * 100,
            "sem_PM_CPVC_VUS": sem_pm,
            "null_PM_CPVC_VUS_mean": null_pm.mean(),
            "null_PM_CPVC_VUS_p05": np.quantile(null_pm, 0.05),
            "null_PM_CPVC_VUS_p95": np.quantile(null_pm, 0.95),
        })
        print(f"  {model:32s} {ds:10s} worst-gap pct={recs[-1]['pctile_worst_calib_gap']:5.1f} "
              f"between-var pct={recs[-1]['pctile_between_var']:5.1f}")
    tab = pd.DataFrame(recs)
    tab.to_csv(os.path.join(OUT, "C_random_partition_null.csv"), index=False)

    # summary: how often semantic partition is in the extreme tail of the null
    summ = {
        "n_model_dataset_pairs": len(tab),
        "worst_calib_gap_above_p95": int((tab["pctile_worst_calib_gap"] >= 95).sum()),
        "between_var_above_p95": int((tab["pctile_between_var"] >= 95).sum()),
        "worst_SEA_below_p05": int((tab["pctile_worst_SEA_low"] >= 95).sum()),
        "median_pctile_worst_calib_gap": float(tab["pctile_worst_calib_gap"].median()),
        "median_pctile_between_var": float(tab["pctile_between_var"].median()),
    }
    pd.DataFrame([summ]).to_csv(os.path.join(OUT, "C_random_partition_summary.csv"), index=False)
    print("  summary:", summ)
    return tab


# ------------------------------------- D. held-out threshold transfer ---

def pick_threshold(conf, corr, lam=1.0, grid=None):
    if grid is None:
        grid = np.round(np.arange(0.5, 1.0001, 0.025), 4)
    best_u, best_t = -np.inf, grid[0]
    for t in grid:
        u = deferral_utility(conf, corr, t, lam)
        if u > best_u:
            best_u, best_t = u, t
    return best_t


def analysis_D_threshold_transfer(df, n_splits=50, lam=1.0):
    print(f"\n[D] Held-out threshold transfer ({n_splits} splits, lambda={lam})")
    rng = np.random.default_rng(RNG_SEED + 1)
    scope_df = df[df.dataset.isin(MAIN_DATASETS)]
    recs = []
    for (model, ds), mdf in scope_df.groupby(["model", "dataset"]):
        mdf = mdf.reset_index(drop=True)
        sizes = mdf.groupby("category").size().values
        utils = {"global": [], "semantic": [], "random_group": []}
        for _ in range(n_splits):
            # stratified 50/50 split within category
            cal_idx = np.zeros(len(mdf), dtype=bool)
            for _, cdf in mdf.groupby("category"):
                ids = cdf.index.values
                pick = rng.permutation(len(ids))[: len(ids) // 2]
                cal_idx[ids[pick]] = True
            cal, ev = mdf[cal_idx], mdf[~cal_idx]

            # global threshold
            tg = pick_threshold(cal["self_eval_confidence"].values,
                                cal["self_eval_correct"].values, lam)
            utils["global"].append(deferral_utility(
                ev["self_eval_confidence"].values, ev["self_eval_correct"].values, tg, lam))

            # semantic per-category thresholds
            u_ev, n_ev = 0.0, 0
            for c, cdf in cal.groupby("category"):
                edf = ev[ev["category"] == c]
                if len(edf) == 0:
                    continue
                tc = pick_threshold(cdf["self_eval_confidence"].values,
                                    cdf["self_eval_correct"].values, lam)
                u_ev += deferral_utility(edf["self_eval_confidence"].values,
                                         edf["self_eval_correct"].values, tc, lam) * len(edf)
                n_ev += len(edf)
            utils["semantic"].append(u_ev / n_ev)

            # random-group per-group thresholds (matched sizes, shared split)
            lab = matched_random_partition(mdf, sizes, rng)
            mtmp = mdf.copy(); mtmp["rg"] = lab
            calr, evr = mtmp[cal_idx], mtmp[~cal_idx]
            u_ev, n_ev = 0.0, 0
            for g, gdf in calr.groupby("rg"):
                edf = evr[evr["rg"] == g]
                if len(edf) == 0:
                    continue
                tc = pick_threshold(gdf["self_eval_confidence"].values,
                                    gdf["self_eval_correct"].values, lam)
                u_ev += deferral_utility(edf["self_eval_confidence"].values,
                                         edf["self_eval_correct"].values, tc, lam) * len(edf)
                n_ev += len(edf)
            utils["random_group"].append(u_ev / n_ev)

        rec = {"model": model, "dataset": ds}
        for k, v in utils.items():
            v = np.array(v)
            rec[f"util_{k}_mean"] = v.mean()
            rec[f"util_{k}_se"] = v.std(ddof=1) / np.sqrt(len(v))
        rec["semantic_minus_global"] = rec["util_semantic_mean"] - rec["util_global_mean"]
        rec["semantic_minus_random"] = rec["util_semantic_mean"] - rec["util_random_group_mean"]
        recs.append(rec)
        print(f"  {model:32s} {ds:10s} sem-glob={rec['semantic_minus_global']:+.4f} "
              f"sem-rand={rec['semantic_minus_random']:+.4f}")
    tab = pd.DataFrame(recs)
    tab.to_csv(os.path.join(OUT, "D_threshold_transfer.csv"), index=False)
    # paired summary
    summ = {
        "pairs": len(tab),
        "semantic_beats_global": int((tab["semantic_minus_global"] > 0).sum()),
        "semantic_beats_random": int((tab["semantic_minus_random"] > 0).sum()),
        "mean_semantic_minus_global": float(tab["semantic_minus_global"].mean()),
        "mean_semantic_minus_random": float(tab["semantic_minus_random"].mean()),
        "wilcoxon_sem_vs_rand_p": float(stats.wilcoxon(
            tab["semantic_minus_random"]).pvalue) if len(tab) > 5 else np.nan,
        "wilcoxon_sem_vs_glob_p": float(stats.wilcoxon(
            tab["semantic_minus_global"]).pvalue) if len(tab) > 5 else np.nan,
    }
    pd.DataFrame([summ]).to_csv(os.path.join(OUT, "D_threshold_transfer_summary.csv"), index=False)
    print("  summary:", summ)
    return tab


# --------------------------------------------------- E. bootstrap CIs ---

def analysis_E_bootstrap(df, n_boot=1000):
    print(f"\n[E] Bootstrap CIs ({n_boot} resamples)")
    rng = np.random.default_rng(RNG_SEED + 2)
    scope_df = df[df.dataset.isin(MAIN_DATASETS)]
    recs = []
    for model, mdf in scope_df.groupby("model"):
        conf = mdf["self_eval_confidence"].values
        corr = mdf["self_eval_correct"].values
        per_ds = [d.reset_index(drop=True) for _, d in mdf.groupby("dataset")]
        n = len(mdf)
        pm_vals, aurc_vals, gap_vals = [], [], []
        for _ in range(n_boot):
            # PM-C-PVC-VUS: resample within each dataset, average per-dataset
            # values (paper Table 1 convention)
            pms = []
            for d in per_ds:
                bidx = rng.integers(0, len(d), len(d))
                pms.append(pm_cpvc_vus(d.iloc[bidx])["PM_CPVC_VUS"])
            pm_vals.append(float(np.mean(pms)))
            # AURC and confidence gap: pooled resample
            bidx = rng.integers(0, n, n)
            _, _, a, _ = risk_coverage(conf[bidx], corr[bidx])
            aurc_vals.append(a)
            c1 = conf[bidx][corr[bidx] == 1.0]
            c0 = conf[bidx][corr[bidx] == 0.0]
            gap_vals.append(c1.mean() - c0.mean() if len(c1) and len(c0) else np.nan)
        for name, vals in [("PM_CPVC_VUS", pm_vals), ("AURC", aurc_vals),
                           ("conf_gap", gap_vals)]:
            vals = np.array(vals, float)
            recs.append({"model": model, "quantity": name,
                         "mean": np.nanmean(vals),
                         "ci_lo": np.nanquantile(vals, 0.025),
                         "ci_hi": np.nanquantile(vals, 0.975)})
        print(f"  {model:32s} PM-C-PVC-VUS "
              f"{np.mean(pm_vals):.3f} [{np.quantile(pm_vals,0.025):.3f}, {np.quantile(pm_vals,0.975):.3f}]")
    tab = pd.DataFrame(recs)
    tab.to_csv(os.path.join(OUT, "E_bootstrap_cis.csv"), index=False)
    return tab


# ------------------------------------------------------------------ main ---

def main():
    df = load_all()
    print(f"Loaded {len(df)} predictions, {df['model'].nunique()} models, "
          f"{df['dataset'].nunique()} datasets")
    print(df.groupby("dataset").size())

    # metric table (cross-3 aggregate) reused by analysis A
    b_tab = analysis_B_group_ece(df)
    metric_table = b_tab[b_tab.scope == "cross3"][
        ["model", "ECE", "Brier", "macro_ECE", "worst_cat_ECE", "equal_mass_ECE",
         "PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS"]]

    analysis_A_gating(df, metric_table)

    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    analysis_C_random_partitions(df, n_seeds=n_seeds)
    analysis_D_threshold_transfer(df)
    analysis_E_bootstrap(df)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
