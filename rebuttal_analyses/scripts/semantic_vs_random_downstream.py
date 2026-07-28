#!/usr/bin/env python3
"""
Semantic-vs-random downstream comparison (AC's "outperform random" question).

For each of 1,000 matched random partitions (same seeds as the main
C-analysis, via identical RNG initialization):
  1. Compute per-model PM-C-PVC-VUS / C-PVC-VUS / PVC-VUS on the random
     partition (cross-3 aggregate: per-dataset VUS averaged, matching the
     paper's Table-1 convention).
  2. Correlate (Spearman) with realized gating outcomes across the 13
     models: SelAcc@50%, HC-Err@0.9, AURC.
  3. Locate the semantic partition's correlation within the random-null
     distribution (percentile).

Also: model-selection regret. Each scheme's diagnostic-selected model
(top PM-C-PVC-VUS) is evaluated on realized SelAcc@50%; regret = best
realized minus selected realized. Compare semantic regret to the random
regret distribution.

No new inference; pure re-analysis of the released predictions.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuttal_neurips_analysis import (
    load_all, pm_cpvc_vus, matched_random_partition,
    OUT, MAIN_DATASETS, RNG_SEED,
)

N_SEEDS = 1000


def main():
    df = load_all()
    scope = df[df.dataset.isin(MAIN_DATASETS)].reset_index(drop=True)
    models = sorted(scope.model.unique())
    print(f"{len(models)} models, {len(scope)} rows")

    # realized gating outcomes (fixed, from the completed gating analysis)
    g = pd.read_csv(os.path.join(OUT, "A_gating_metrics.csv"))
    g3 = g[g.scope == "cross3"].set_index("model")
    outcomes = {
        "sel50": g3["sel_acc@cov0.5"],
        "hce9": g3["hc_err@thr0.9"],
        "aurc": g3["AURC"],
    }

    # semantic diagnostic scores (per-dataset VUS averaged)
    sem_scores = {}
    for m in models:
        vals = []
        for ds in MAIN_DATASETS:
            sub = scope[(scope.model == m) & (scope.dataset == ds)]
            vals.append(pm_cpvc_vus(sub))
        sem_scores[m] = {k: float(np.mean([v[k] for v in vals]))
                         for k in ("PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS")}
    sem = pd.DataFrame(sem_scores).T

    def corrs(scores):
        out = {}
        for name, y in outcomes.items():
            aligned = y.reindex(scores.index)
            rho, _ = stats.spearmanr(scores.values, aligned.values)
            out[name] = rho
        return out

    sem_corr = {k: corrs(sem[k]) for k in ("PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS")}
    print("Semantic correlations:")
    for k, v in sem_corr.items():
        print(f"  {k}: " + ", ".join(f"{n}={r:+.3f}" for n, r in v.items()))

    # semantic regret (PM-C-PVC-selected model)
    def regret(scores):
        pick = scores.idxmax()
        best = outcomes["sel50"].max()
        return float(best - outcomes["sel50"][pick]), pick

    sem_regret, sem_pick = regret(sem["PM_CPVC_VUS"])
    print(f"Semantic PM-selected model: {sem_pick}, regret={sem_regret:.4f}")

    # ---- random partitions: same per-(model,dataset) draw structure as
    # analysis C (one shared RNG; group stats recomputed per seed) ----
    # Pre-split data per (model, dataset) with sizes
    groups = {}
    for (m, ds), sub in scope.groupby(["model", "dataset"]):
        sub = sub.reset_index(drop=True)
        sizes = sub.groupby("category").size().values
        groups[(m, ds)] = (sub, sizes)

    rng = np.random.default_rng(RNG_SEED + 400)
    rand_corr = {k: {n: [] for n in outcomes} for k in ("PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS")}
    rand_regret = []
    for seed_i in range(N_SEEDS):
        scores = {k: {} for k in ("PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS")}
        for m in models:
            vals = []
            for ds in MAIN_DATASETS:
                sub, sizes = groups[(m, ds)]
                lab = matched_random_partition(sub, sizes, rng)
                tmp = sub.copy()
                tmp["rg"] = lab
                vals.append(pm_cpvc_vus(tmp, group_col="rg"))
            for k in scores:
                scores[k][m] = float(np.mean([v[k] for v in vals]))
        for k in scores:
            sc = pd.Series(scores[k])
            c = corrs(sc)
            for n in outcomes:
                rand_corr[k][n].append(c[n])
        r, _ = regret(pd.Series(scores["PM_CPVC_VUS"]))
        rand_regret.append(r)
        if (seed_i + 1) % 100 == 0:
            print(f"  seed {seed_i+1}/{N_SEEDS}", flush=True)

    # ---- summarize: percentile of semantic within random null ----
    rows = []
    for k in ("PVC_VUS", "CPVC_VUS", "PM_CPVC_VUS"):
        for n in outcomes:
            null = np.array(rand_corr[k][n])
            semv = sem_corr[k][n]
            # for sel50 higher rho is better; for hce9/aurc more-negative is better
            if n == "sel50":
                pct = (null < semv).mean() * 100
            else:
                pct = (null > semv).mean() * 100  # semantic more negative than pct% of null
            rows.append({"diagnostic": k, "outcome": n,
                         "semantic_rho": semv,
                         "null_mean": null.mean(), "null_p5": np.quantile(null, .05),
                         "null_p95": np.quantile(null, .95),
                         "pct_of_null_beaten": pct})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "H_semantic_vs_random_downstream.csv"), index=False)
    print("\n=== Semantic vs random downstream association ===")
    print(tab.round(3).to_string(index=False))

    rr = np.array(rand_regret)
    reg_pct = (rr > sem_regret).mean() * 100
    print(f"\n=== Model-selection regret (PM-C-PVC pick, SelAcc@50%) ===")
    print(f"semantic regret: {sem_regret:.4f} (pick={sem_pick})")
    print(f"random-null regret: mean={rr.mean():.4f}, p5={np.quantile(rr,.05):.4f}, "
          f"p95={np.quantile(rr,.95):.4f}")
    print(f"semantic regret lower than {reg_pct:.1f}% of random partitions")
    pd.DataFrame([{"semantic_regret": sem_regret, "semantic_pick": sem_pick,
                   "null_mean": rr.mean(), "null_p5": np.quantile(rr,.05),
                   "null_p95": np.quantile(rr,.95),
                   "pct_null_worse": reg_pct}]).to_csv(
        os.path.join(OUT, "H_regret_summary.csv"), index=False)


if __name__ == "__main__":
    main()
