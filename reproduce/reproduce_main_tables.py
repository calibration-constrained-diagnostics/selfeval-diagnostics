#!/usr/bin/env python3
"""
Reproduce the tables and plots that appear in the NeurIPS 2026 E&D submission
"Evaluating LLM Self-Evaluation with Calibration-Constrained Diagnostics"
from the released ``model_outputs/*_self_eval.csv`` files.

Outputs (all written under ``reproduce/outputs/``):

  Main paper
    table1_cross_dataset_diagnostic_decomposition.csv   (Table 1 in the paper)
    table2_scale_and_ranking_reversal.csv               (Table 2 in the paper)
    cross_dataset_average_pvc_plot.png                  (Figure 2(a))
    cross_domain_pvc_cpvc_comparison.png                (Figure 2(b))
    cross_dataset_average_cpvc_3d_grid.png              (Figure 3)

  Appendix
    appendix_per_dataset_breakdown.csv                  (per-dataset VUS table)
    appendix_math500_table.csv                          (MATH-500 robustness table)
    appendix_judge_correlation.csv                      (judge correlation study)

Run inside the ``selfeval`` conda environment:

    conda activate selfeval
    python reproduce/reproduce_main_tables.py

This script does not write anywhere outside ``reproduce/outputs/`` and does
not invoke the standalone ``compute_pvc_cpvc.py`` CLI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless Matplotlib backend (must be set before importing the analysis module).
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "reproduce" / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# Import the analysis library (import-safe, no side effects).
from selfeval.scripts.compute_pvc_cpvc import (  # noqa: E402
    calculate_vus_metrics,
    generate_parameter_sweep_table_futures,
    generate_parameter_sweep_table_sequential,
    plot_all_models_cpvc_3d_grid,
)
from selfeval.scripts.compute_cross_dataset import (  # noqa: E402
    plot_averaged_pvc_line_plot,
    plot_cross_domain_pvc_cpvc_comparison,
)

# Three cross-domain datasets used in the paper's main Table 1 / Figures 2-3.
# MATH-500 is reported separately in the appendix.
DATASETS = {
    "Math360":    ROOT / "model_outputs" / "math360_self_eval.csv",
    "TruthfulQA": ROOT / "model_outputs" / "truthfulqa_self_eval.csv",
    "CSQA":       ROOT / "model_outputs" / "commonsenseqa_self_eval.csv",
}
MATH500_CSV = ROOT / "model_outputs" / "math500_self_eval.csv"

# 11 open 7--8B models used for Table 1 (paper ordering).
LEGEND_ORDER = [
    "Qwen2.5-7B",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-Math-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "OpenThinker2-7B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Bespoke-Stratos-7B",
    "JiuZhang3.0-7B",
    "Ministral-8B-Instruct-2410",
    "Open-Reasoner-Zero-7B",
    "s1.1-7B",
]

# Six models used for Table 2 (scale + ranking reversal).
TABLE2_MODELS = [
    "JiuZhang3.0-7B",
    "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "s1.1-7B",
    "Qwen2.5-32B-Instruct",
    "Claude 3.5 Haiku",
]


# ---------- small helpers ----------

def _calibration_df(data: pd.DataFrame) -> pd.DataFrame:
    """Per-(model,category) accuracy / confidence / calibration_error frame."""
    accuracy = (data.groupby(["model_id", "category"])["self_eval_correct"]
                    .mean().reset_index()
                    .rename(columns={"self_eval_correct": "actual_accuracy"}))
    confidence = (data.groupby(["model_id", "category"])["self_eval_confidence"]
                      .mean().reset_index())
    cal = pd.merge(accuracy, confidence, on=["model_id", "category"])
    cal["calibration_error"] = cal["self_eval_confidence"] - cal["actual_accuracy"]
    return cal


def _per_model_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Per-model aggregate ECE / Brier / SEA / CalibError from raw eval rows."""
    rows = []
    sea_all = data.groupby("model_id")["self_eval_correct"].mean()
    for model, mdf in data.groupby("model_id"):
        conf = mdf["self_eval_confidence"].values
        correct = mdf["self_eval_correct"].values
        # Brier: mean squared error between confidence and correctness indicator
        brier = float(np.mean((conf - correct) ** 2))
        # ECE: binning on confidence
        num_bins = 10
        edges = np.linspace(0.0, 1.0, num_bins + 1)
        idx = np.clip(np.digitize(conf, edges) - 1, 0, num_bins - 1)
        bin_counts = np.bincount(idx, minlength=num_bins).astype(float)
        bin_conf = np.bincount(idx, weights=conf, minlength=num_bins) / np.maximum(bin_counts, 1)
        bin_acc = np.bincount(idx, weights=correct, minlength=num_bins) / np.maximum(bin_counts, 1)
        total = max(bin_counts.sum(), 1)
        ece = float(np.sum(bin_counts * np.abs(bin_conf - bin_acc)) / total)
        # CalibError: overall mean(|conf - correct|) at the example level
        calib_err = float(np.mean(np.abs(conf - correct)))
        rows.append({
            "Model": model,
            "ECE": round(ece, 4),
            "Brier": round(brier, 4),
            "CalibError": round(calib_err, 4),
            "SEA": round(float(sea_all[model]), 4),
        })
    return pd.DataFrame(rows)


def _cached_sweep(tag: str, cal_df: pd.DataFrame) -> pd.DataFrame:
    """Load a previously computed (gamma, tau) sweep table, or compute it.

    The sweep is cached under ``reproduce/outputs/_sweep_{tag}.csv`` so that
    subsequent reruns are instantaneous. The sequential implementation is
    used because it is the most reliable across Python / OS / multiprocess
    setups; the first call per dataset takes ~30-60 s.
    """
    path = OUTDIR / f"_sweep_{tag}.csv"
    if path.exists():
        sweep = pd.read_csv(path)
        if len(sweep) > 0:
            return sweep
    sweep = generate_parameter_sweep_table_sequential(cal_df, tag)
    sweep.to_csv(path, index=False)
    return sweep


def _ordered(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    present = [m for m in order if m in df["Model"].tolist()]
    rest = [m for m in df["Model"].tolist() if m not in order]
    return (df.set_index("Model")
              .reindex(present + rest)
              .reset_index())


# ---------- per-dataset pipeline ----------

def run_single_dataset(tag: str, csv_path: Path) -> dict:
    """Compute the per-model VUS + calibration summary for one dataset."""
    print(f"[{tag}] loading {csv_path.name}")
    data = pd.read_csv(csv_path)
    cal_df = _calibration_df(data)
    sweep = _cached_sweep(tag, cal_df)
    vus = calculate_vus_metrics(sweep)
    summary = _per_model_summary(data)
    merged = pd.merge(vus, summary, on="Model", how="outer")
    return {"data": data, "cal_df": cal_df, "sweep": sweep, "summary": merged}


# ---------- main paper tables ----------

def build_table1(per_ds: dict) -> pd.DataFrame:
    """Table 1: cross-dataset aggregate over Math360/TruthfulQA/CSQA (11 models)."""
    frames = []
    for tag in ("Math360", "TruthfulQA", "CSQA"):
        df = per_ds[tag]["summary"].copy()
        df["dataset"] = tag
        frames.append(df)
    stacked = pd.concat(frames, ignore_index=True)
    numeric_cols = [c for c in stacked.columns if c not in ("Model", "dataset")]
    agg = stacked.groupby("Model")[numeric_cols].mean().round(4).reset_index()
    agg = _ordered(agg, LEGEND_ORDER)
    column_order = ["Model", "PVC-VUS", "C-PVC-VUS", "Gap",
                    "PM-C-PVC-VUS", "CalibError", "SEA"]
    return agg[[c for c in column_order if c in agg.columns]]


def build_appendix_per_dataset(per_ds: dict) -> pd.DataFrame:
    """Appendix per-dataset breakdown with the full 9-column schema."""
    rows = []
    for tag in ("Math360", "TruthfulQA", "CSQA"):
        df = per_ds[tag]["summary"].copy()
        df.insert(1, "Dataset", tag)
        rows.append(df)
    stacked = pd.concat(rows, ignore_index=True)
    # Reorder model rows to LEGEND_ORDER, datasets in [Math360, TruthfulQA, CSQA]
    order_key = {m: i for i, m in enumerate(LEGEND_ORDER)}
    ds_key = {"Math360": 0, "TruthfulQA": 1, "CSQA": 2}
    stacked["_mk"] = stacked["Model"].map(order_key).fillna(99)
    stacked["_dk"] = stacked["Dataset"].map(ds_key)
    stacked = stacked.sort_values(["_mk", "_dk"]).drop(columns=["_mk", "_dk"])
    column_order = ["Model", "Dataset",
                    "PVC-VUS", "C-PVC-VUS", "Gap",
                    "PM-PVC-VUS", "PM-C-PVC-VUS", "PM-SC-VUS",
                    "CalibError", "SEA"]
    return stacked[[c for c in column_order if c in stacked.columns]]


def build_table2(per_ds: dict) -> pd.DataFrame:
    """Table 2: scale extension + ranking-reversal. Six-model aggregate.

    Uses the same cross-dataset averaging as Table 1, but restricted to the
    six models shown in the paper (includes Qwen2.5-32B-Instruct and
    Claude 3.5 Haiku when present in the released outputs).
    """
    agg = build_table1(per_ds)
    mask = agg["Model"].isin(TABLE2_MODELS)
    subset = agg[mask].copy()
    # Scale tag for readability
    scale_tag = {
        "JiuZhang3.0-7B": "7B",
        "Qwen2.5-7B-Instruct": "7B",
        "Llama-3.1-8B-Instruct": "8B",
        "s1.1-7B": "7B",
        "Qwen2.5-32B-Instruct": "32B",
        "Claude 3.5 Haiku": "Proprietary",
    }
    subset.insert(1, "Scale", subset["Model"].map(scale_tag).fillna("—"))
    cols = ["Model", "Scale", "ECE", "Brier",
            "PVC-VUS", "C-PVC-VUS", "Gap", "PM-C-PVC-VUS"]
    subset = subset[[c for c in cols if c in subset.columns]]
    order = {m: i for i, m in enumerate(TABLE2_MODELS)}
    subset["_k"] = subset["Model"].map(order)
    subset = subset.sort_values("_k").drop(columns=["_k"]).reset_index(drop=True)
    return subset


def build_math500_table(math500: dict) -> pd.DataFrame:
    df = math500["summary"].copy()
    df = _ordered(df, LEGEND_ORDER)
    cols = ["Model", "PVC-VUS", "C-PVC-VUS", "Gap",
            "PM-PVC-VUS", "PM-C-PVC-VUS", "PM-SC-VUS",
            "CalibError", "SEA"]
    return df[[c for c in cols if c in df.columns]]


def build_judge_correlation(data_by_tag: dict) -> pd.DataFrame:
    """Overall judge-correlation summary (appendix Table)."""
    # Use Math-360 raw data; the paper reports overall Pearson + agreement.
    data = data_by_tag["Math360"]["data"]
    rows = []
    for j, label in [("a", "Claude 3.7 Sonnet"),
                     ("b", "Amazon Nova Premier"),
                     ("c", "DeepSeek-R1")]:
        correct = (data["self_eval_answer"] == data["correct_answer"]).astype(int)
        judge_correct = (data[f"judge_{j}_answer"] == data["correct_answer"]).astype(int)
        # Pearson correlation on matched-correctness series (paper convention).
        if correct.std() > 0 and judge_correct.std() > 0:
            corr = float(np.corrcoef(correct, judge_correct)[0, 1])
        else:
            corr = float("nan")
        agreement = float((data["self_eval_answer"] == data[f"judge_{j}_answer"]).mean())
        rows.append({
            "LLM Judge": label,
            "Correlation": round(corr, 4),
            "Agreement": round(agreement, 4),
        })
    return pd.DataFrame(rows)


# ---------- main paper figures ----------

def build_main_figures(per_ds: dict) -> None:
    """Figures 2 and 3 of the paper, all cross-dataset averaged over
    Math360 / TruthfulQA / CSQA.

    File names match the ``\\includegraphics{./figures/<name>}`` references
    in the paper source:

      - Figure 2(a): ``cross_dataset_average_pvc_plot.png``
          -- number of gamma-shattered categories vs gamma, one line per model
      - Figure 2(b): ``cross_domain_pvc_cpvc_comparison.png``
          -- scatter of PVC-VUS vs C-PVC-VUS, one point per model
      - Figure 3:    ``cross_dataset_average_cpvc_3d_grid.png``
          -- 4x3 grid of 3-D C-PVC surfaces, one per model.
    """
    import matplotlib.pyplot as plt

    # Cross-dataset averaged (gamma, tau) sweep (input for all three figures)
    combined = []
    for tag in ("Math360", "TruthfulQA", "CSQA"):
        sweep = per_ds[tag]["sweep"].copy()
        sweep["dataset"] = tag
        combined.append(sweep)
    all_sweep = pd.concat(combined, ignore_index=True)
    avg_sweep = (all_sweep.groupby(["model", "gamma", "tau"])[["pvc", "c_pvc", "sample_complexity"]]
                          .mean().reset_index())

    # Figure 2, panel (a)
    fig2a = OUTDIR / "cross_dataset_average_pvc_plot.png"
    plot_averaged_pvc_line_plot(avg_sweep, str(fig2a))
    plt.close("all")
    print(f"[fig2a] saved to {fig2a}")

    # Figure 2, panel (b)
    fig2b = OUTDIR / "cross_domain_pvc_cpvc_comparison.png"
    plot_cross_domain_pvc_cpvc_comparison(avg_sweep, str(fig2b))
    plt.close("all")
    print(f"[fig2b] saved to {fig2b}")

    # Figure 3: 4x3 grid of 3-D C-PVC surfaces
    fig3 = OUTDIR / "cross_dataset_average_cpvc_3d_grid.png"
    plot_all_models_cpvc_3d_grid(avg_sweep, "Cross-Dataset", str(fig3))
    plt.close("all")
    print(f"[fig3] saved to {fig3}")


# ---------- main ----------

def main() -> None:
    per_ds = {}
    for tag, path in DATASETS.items():
        if not path.exists():
            print(f"[warn] missing {path}; skipping {tag}")
            continue
        per_ds[tag] = run_single_dataset(tag, path)

    # Main-paper tables
    t1 = build_table1(per_ds)
    t1.to_csv(OUTDIR / "table1_cross_dataset_diagnostic_decomposition.csv", index=False)
    print("[table1] written")

    t2 = build_table2(per_ds)
    t2.to_csv(OUTDIR / "table2_scale_and_ranking_reversal.csv", index=False)
    print("[table2] written")

    # Appendix tables
    per_ds_table = build_appendix_per_dataset(per_ds)
    per_ds_table.to_csv(OUTDIR / "appendix_per_dataset_breakdown.csv", index=False)
    print("[appendix_per_dataset] written")

    if MATH500_CSV.exists():
        math500 = run_single_dataset("Math500", MATH500_CSV)
        t_m500 = build_math500_table(math500)
        t_m500.to_csv(OUTDIR / "appendix_math500_table.csv", index=False)
        print("[appendix_math500] written")
    else:
        print("[warn] no MATH-500 outputs; skipping appendix_math500_table.csv")

    jc = build_judge_correlation(per_ds)
    jc.to_csv(OUTDIR / "appendix_judge_correlation.csv", index=False)
    print("[appendix_judge_correlation] written")

    # Main-paper figures (cross-dataset averaged)
    build_main_figures(per_ds)

    print(f"\nAll outputs written under {OUTDIR}")


if __name__ == "__main__":
    main()
