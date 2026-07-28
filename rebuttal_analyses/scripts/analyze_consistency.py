#!/usr/bin/env python3
"""
Analyze consistency-based confidence results vs verbalized confidence.

Input: consistency_results/*.jsonl produced by consistency_bedrock.py
       (or consistency_experiment.py on GPU).

For each model-dataset file, compute under BOTH confidence sources
(against the same fixed judge-reference labels):
  - ECE, Brier
  - PVC-VUS / C-PVC-VUS / PM-C-PVC-VUS (paper grid)
  - selective accuracy @50% coverage, HC-Err@0.9, AURC
  - confidence gap (correct vs incorrect)
Report the aggregate-vs-category-level comparison per source.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuttal_neurips_analysis import (
    pm_cpvc_vus, ece_equal_width, brier, risk_coverage,
    selective_at_coverage, gate_at_threshold, OUT,
)

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "consistency_results")


def metrics_for(df, conf_col, corr_col):
    conf = df[conf_col].astype(float).values
    corr = df[corr_col].astype(float).values
    out = {
        "N": len(df),
        "SEA": corr.mean(),
        "mean_conf": conf.mean(),
        "ECE": ece_equal_width(conf, corr),
        "Brier": brier(conf, corr),
        "sel_acc@50": selective_at_coverage(conf, corr, 0.5),
    }
    _, acc9 = gate_at_threshold(conf, corr, 0.9)
    out["HC_err@0.9"] = 1 - acc9 if not np.isnan(acc9) else np.nan
    out["cov@0.9"] = gate_at_threshold(conf, corr, 0.9)[0]
    _, _, aurc, _ = risk_coverage(conf, corr)
    out["AURC"] = aurc
    tmp = pd.DataFrame({"category": df["category"],
                        "self_eval_correct": corr,
                        "self_eval_confidence": conf})
    out.update(pm_cpvc_vus(tmp))
    c1 = conf[corr == 1.0]; c0 = conf[corr == 0.0]
    out["conf_gap"] = c1.mean() - c0.mean() if len(c1) and len(c0) else np.nan
    return out


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(RES_DIR, "*.jsonl"))):
        name = os.path.basename(path).replace("_consistency.jsonl", "")
        recs = [json.loads(l) for l in open(path)]
        df = pd.DataFrame(recs)
        n_total = len(df)
        df = df[df.cons_selected.notna() & df.judge_selected.notna() &
                df.orig_selected.notna() & df.orig_confidence.notna()]
        # correctness against the same fixed judge reference
        df["corr_orig"] = (df.orig_selected == df.judge_selected).astype(float)
        df["corr_cons"] = (df.cons_selected == df.judge_selected).astype(float)
        df["orig_confidence"] = pd.to_numeric(df.orig_confidence, errors="coerce")
        df = df.dropna(subset=["orig_confidence", "cons_confidence"])
        if "fixed_confidence" not in df.columns:
            # derive from stored samples: agreement with original decision
            def fixed_from_samples(row):
                valid = [s for s in row["samples"] if s in ("A", "B")]
                if not valid or row["orig_selected"] not in ("A", "B"):
                    return np.nan
                return valid.count(row["orig_selected"]) / len(valid)
            df["fixed_confidence"] = df.apply(fixed_from_samples, axis=1)
        has_fixed = df.fixed_confidence.notna().any()
        print(f"\n=== {name} (usable {len(df)}/{n_total}) ===")
        variants = [("verbalized", "orig_confidence", "corr_orig"),
                    ("modal-cons", "cons_confidence", "corr_cons")]
        if has_fixed:
            # fixed-decision variant: original single-shot decision kept,
            # confidence = repeated-selection agreement with that decision
            variants.insert(1, ("fixed-cons", "fixed_confidence", "corr_orig"))
        for src, conf_col, corr_col in variants:
            sub = df.dropna(subset=[conf_col])
            m = metrics_for(sub, conf_col, corr_col)
            m.update({"model_dataset": name, "source": src})
            rows.append(m)
            print(f"  [{src:11s}] SEA={m['SEA']:.3f} conf={m['mean_conf']:.3f} "
                  f"ECE={m['ECE']:.3f} Brier={m['Brier']:.3f} "
                  f"sel50={m['sel_acc@50']:.3f} HCerr={m['HC_err@0.9']:.3f} "
                  f"AURC={m['AURC']:.3f} PM={m['PM_CPVC_VUS']:.3f} "
                  f"gap={m['conf_gap']:.4f}")
        # agreement between the two decision rules
        agree = (df.orig_selected == df.cons_selected).mean()
        print(f"  decision agreement verbalized vs consistency: {agree:.3f}")

    tab = pd.DataFrame(rows)
    outp = os.path.join(OUT, "G_consistency_comparison.csv")
    tab.to_csv(outp, index=False)
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
