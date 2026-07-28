#!/usr/bin/env python3
"""
Addendum analyses for the NeurIPS 2026 rebuttal, per internal review:

  A2. AUROC/AUPRC baselines for the diagnostic-vs-gating correlation table
  A3. Deferral-utility lambda sensitivity sweep (0.5, 1, 2, 5)
  A4. Permutation p-values (10k) for key Spearman correlations, n stated
  A5. Per-dataset correlations (incl. MATH-500) as clustered robustness
  C2. Random-partition effect sizes: median inflation, semantic percentile
      within the null (same RNG seed -> identical nulls as main run)

Outputs to analysis_outputs/ alongside the main run.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuttal_neurips_analysis import (
    load_all, pm_cpvc_vus, ece_equal_width, ece_equal_mass, brier,
    groupwise_ece, risk_coverage, selective_at_coverage, gate_at_threshold,
    deferral_utility, matched_random_partition,
    OUT, MAIN_DATASETS, SMALL_FILES, RNG_SEED,
)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def auroc(conf, correct):
    """Mann-Whitney AUROC of confidence for correct-vs-incorrect; ties in the
    label (0.5 rows) are excluded."""
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    m = (correct == 1.0) | (correct == 0.0)
    conf, correct = conf[m], correct[m]
    pos, neg = conf[correct == 1.0], conf[correct == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def spearman_perm(x, y, n_perm=10000, rng=None):
    rho, _ = stats.spearmanr(x, y)
    if rng is None:
        rng = np.random.default_rng(RNG_SEED + 99)
    y = np.asarray(y)
    count = 0
    for _ in range(n_perm):
        r, _ = stats.spearmanr(x, rng.permutation(y))
        if abs(r) >= abs(rho):
            count += 1
    return rho, (count + 1) / (n_perm + 1)


def main():
    df = load_all()
    print(f"Loaded {len(df)} rows")

    # ------------------------------------------------------------------
    # A2/A3: per-model AUROC + lambda sweep, cross-3 scope
    # ------------------------------------------------------------------
    scope = df[df.dataset.isin(MAIN_DATASETS)]
    LAMBDAS = [0.5, 1.0, 2.0, 5.0]
    rows = []
    for model, mdf in scope.groupby("model"):
        conf, corr = mdf["self_eval_confidence"].values, mdf["self_eval_correct"].values
        row = {"model": model, "AUROC": auroc(conf, corr)}
        for lam in LAMBDAS:
            row[f"util@0.9_lam{lam}"] = deferral_utility(conf, corr, 0.9, lam)
        rows.append(row)
    a2 = pd.DataFrame(rows)
    a2.to_csv(os.path.join(OUT, "A2_auroc_lambda_sweep.csv"), index=False)
    print("\n[A2/A3] AUROC + lambda sweep")
    print(a2.round(3).to_string(index=False))

    # lambda rank stability
    print("\nUtility ranking stability across lambda:")
    base_rank = a2.set_index("model")["util@0.9_lam1.0"].rank(ascending=False)
    for lam in LAMBDAS:
        r = a2.set_index("model")[f"util@0.9_lam{lam}"].rank(ascending=False)
        rho = stats.spearmanr(base_rank, r).statistic
        jz = a2.set_index("model")
        jz_vs_qw = (jz.loc["JiuZhang3.0-7B", f"util@0.9_lam{lam}"] <
                    jz.loc["Qwen2.5-32B-Instruct", f"util@0.9_lam{lam}"])
        print(f"  lambda={lam}: Spearman vs lam=1 rank {rho:.3f}; "
              f"JiuZhang < Qwen32B: {jz_vs_qw}")

    # ------------------------------------------------------------------
    # A4: correlation table with AUROC + permutation p-values (cross-3)
    # ------------------------------------------------------------------
    b = pd.read_csv(os.path.join(OUT, "B_groupwise_ece_table.csv"))
    g = pd.read_csv(os.path.join(OUT, "A_gating_metrics.csv"))
    b3 = b[b.scope == "cross3"].set_index("model")
    g3 = g[g.scope == "cross3"].set_index("model")
    j = g3.join(b3, how="inner", lsuffix="", rsuffix="_b").join(
        a2.set_index("model"), how="inner")

    predictors = ["PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS", "ECE", "Brier",
                  "macro_ECE", "worst_cat_ECE", "equal_mass_ECE", "AUROC"]
    targets = ["sel_acc@cov0.5", "sel_acc@cov0.8", "hc_err@thr0.9",
               "AURC", "worst_cat_sel_acc@cov0.5"]
    rng = np.random.default_rng(RNG_SEED + 99)
    recs = []
    for t in targets:
        for p in predictors:
            sub = j[[p, t]].dropna()
            rho, pperm = spearman_perm(sub[p].values, sub[t].values, rng=rng)
            recs.append({"gating_outcome": t, "diagnostic": p,
                         "spearman_rho": rho, "perm_p": pperm, "n": len(sub)})
    a4 = pd.DataFrame(recs)
    a4.to_csv(os.path.join(OUT, "A4_correlations_perm_p.csv"), index=False)
    print("\n[A4] Correlations with permutation p (n=13):")
    piv = a4.pivot(index="diagnostic", columns="gating_outcome",
                   values="spearman_rho").round(2)
    print(piv.to_string())
    pivp = a4.pivot(index="diagnostic", columns="gating_outcome",
                    values="perm_p").round(4)
    print("\npermutation p-values:")
    print(pivp.to_string())

    # ------------------------------------------------------------------
    # A5: per-dataset correlations (incl. MATH-500)
    # ------------------------------------------------------------------
    recs = []
    for ds in SMALL_FILES:
        dsub = df[df.dataset == ds]
        rows = []
        for model, mdf in dsub.groupby("model"):
            conf, corr = mdf["self_eval_confidence"].values, mdf["self_eval_correct"].values
            r = {"model": model,
                 "sel50": selective_at_coverage(conf, corr, 0.5),
                 "hce9": 1 - gate_at_threshold(conf, corr, 0.9)[1],
                 "ECE": ece_equal_width(conf, corr),
                 "AUROC": auroc(conf, corr)}
            r.update(pm_cpvc_vus(mdf))
            rows.append(r)
        t = pd.DataFrame(rows)
        for p in ["PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS", "ECE", "AUROC"]:
            for tgt in ["sel50", "hce9"]:
                sub = t[[p, tgt]].dropna()
                rho, _ = stats.spearmanr(sub[p], sub[tgt])
                recs.append({"dataset": ds, "diagnostic": p, "outcome": tgt,
                             "spearman_rho": rho, "n": len(sub)})
    a5 = pd.DataFrame(recs)
    a5.to_csv(os.path.join(OUT, "A5_per_dataset_correlations.csv"), index=False)
    print("\n[A5] Per-dataset rho (sel50):")
    print(a5[a5.outcome == "sel50"].pivot(index="diagnostic", columns="dataset",
          values="spearman_rho").round(2).to_string())

    # ------------------------------------------------------------------
    # C2: random-partition effect sizes (same seed -> same nulls)
    # ------------------------------------------------------------------
    print("\n[C2] Random-partition effect sizes (1000 seeds, same RNG)")
    rng = np.random.default_rng(RNG_SEED)
    recs = []
    for (model, ds), mdf in scope.groupby(["model", "dataset"]):
        mdf = mdf.reset_index(drop=True)
        sizes = mdf.groupby("category").size().values
        sem_pm = pm_cpvc_vus(mdf)["PM_CPVC_VUS"]
        # regenerate null in the identical draw order as the main run
        corr_arr = mdf["self_eval_correct"].values
        conf_arr = mdf["self_eval_confidence"].values
        null_pm = []
        for _ in range(1000):
            lab = matched_random_partition(mdf, sizes, rng)
            # consume the same per-seed group stats draws as main run did
            # (main run computed seas/confs/gaps from lab; identical here)
            tmp = mdf.copy()
            tmp["rand_group"] = lab
            null_pm.append(pm_cpvc_vus(tmp, group_col="rand_group")["PM_CPVC_VUS"])
        null_pm = np.array(null_pm)
        recs.append({
            "model": model, "dataset": ds,
            "sem_PM": sem_pm,
            "null_median": float(np.median(null_pm)),
            "null_mean": float(null_pm.mean()),
            "inflation_delta_mean": float(null_pm.mean() - sem_pm),
            "sem_percentile_in_null": float((null_pm < sem_pm).mean() * 100),
            "sem_below_null_median": bool(sem_pm < np.median(null_pm)),
        })
    c2 = pd.DataFrame(recs)
    c2.to_csv(os.path.join(OUT, "C2_random_partition_effect_sizes.csv"), index=False)
    print(c2.round(4).to_string(index=False))
    print("\nSummary:")
    print("  pairs where semantic below null median:",
          int(c2.sem_below_null_median.sum()), "/", len(c2))
    print("  median semantic percentile in null: "
          f"{c2.sem_percentile_in_null.median():.1f}")
    print(f"  median inflation delta (null mean - semantic): "
          f"{c2.inflation_delta_mean.median():.4f}")
    nz = c2[c2.null_mean > 0]
    ratio = (nz.null_mean / nz.sem_PM.clip(lower=1e-9))
    print(f"  median multiplicative inflation on pairs with nonzero null: "
          f"{ratio.median():.1f}x (n={len(nz)})")


if __name__ == "__main__":
    main()
