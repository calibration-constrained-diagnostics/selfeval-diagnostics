#!/usr/bin/env python3
"""
Exactly-one-correct / both-correct / both-wrong stratification for the
NeurIPS 2026 rebuttal (Reviewer odoV W1/Q1).

Uses the archived detailed-results JSONs (candidate solution texts +
gold reference answers) to grade each candidate against gold, stratify
pairs, and recompute the core quantities per stratum:

  - stratum proportions per model/dataset
  - SEA (judge-reference agreement) per stratum
  - gold-selection accuracy on exactly-one-correct pairs
    (did the model pick the candidate whose final answer matches gold?)
  - confidence gap (correct vs incorrect decisions) per stratum
  - selective accuracy @50% coverage and HC-Err@0.9 per stratum
  - ECE and PM-C-PVC-VUS per stratum (where sample size allows)

Answer extraction: last \\boxed{...}, else "answer is[:] ...", else last
number. Normalization: strip $, spaces, trailing periods; numeric
equivalence when both parse as numbers. Coverage reported honestly.
"""

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuttal_neurips_analysis import (
    pm_cpvc_vus, ece_equal_width, selective_at_coverage, gate_at_threshold,
    OUT, RNG_SEED,
)

DL = "./data/archived_candidates"
DIRS = {
    # dataset -> {model_id: dir}
    "Math360": {
        "JiuZhang3.0-7B": f"{DL}/JiuZhang3.0-7B_math_benchmark",
        "Qwen2.5-32B-Instruct": f"{DL}/Qwen2.5-32B-Instruct_math_benchmark",
        "Qwen2.5-7B": f"{DL}/Qwen2.5-7B_math_benchmark",
        "s1.1-7B": f"{DL}/s1.1-7B_math_benchmark",
    },
    "Math500": {
        "JiuZhang3.0-7B": f"{DL}/JiuZhang3.0-7B_math-500",
        "Llama-3.1-8B-Instruct": f"{DL}/Llama-3.1-8B-Instruct_math-500",
        "Ministral-8B-Instruct-2410": f"{DL}/Ministral-8B-Instruct-2410_math-500",
        "Open-Reasoner-Zero-7B": f"{DL}/Open-Reasoner-Zero-7B_math-500",
        "Qwen2.5-7B": f"{DL}/Qwen2.5-7B_math-500",
        "Qwen2.5-7B-Instruct": f"{DL}/Qwen2.5-7B-Instruct_math-500",
        "Qwen2.5-Math-7B-Instruct": f"{DL}/Qwen2.5-Math-7B-Instruct_math-500",
    },
}


# ------------------------------------------------------------------
# answer extraction & grading
# ------------------------------------------------------------------

def extract_boxed(text):
    """Return content of the last \\boxed{...} handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


ANSWER_RE = re.compile(
    r"(?:the\s+)?answer\s+is[:\s]+\$?([^\n\.$]+)", re.IGNORECASE)


def extract_answer(text):
    if not isinstance(text, str) or not text.strip():
        return None
    b = extract_boxed(text)
    if b is not None:
        return b
    m = None
    for m in ANSWER_RE.finditer(text):
        pass
    if m:
        return m.group(1)
    return None


UNIT_WORDS = (r"cm²|cm2|cm\^2|m²|m2|m\^2|km|cm|mm|meters?|inches?|feet|ft|"
              r"seconds?|minutes?|hours?|days?|degrees?|dollars?|units?|"
              r"inchespersecond|persecond")


def normalize(ans):
    if ans is None:
        return None
    s = str(ans).strip()
    s = s.replace("\\$", "").replace("$", "").replace("\\!", "").replace("\\,", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"^\\?\(|\\?\)$", "", s.strip())
    # strip leading variable assignment, e.g. "x = 22", "f'(x) = 2x sin(x^4)"
    s = re.sub(r"^[a-zA-Z]'?\([a-zA-Z]\)\s*=\s*", "", s)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    # strip leading commentary, e.g. "correctly 7", "correct, and ... is 7"
    s = re.sub(r"^(correctly|correct,?)\s+", "", s, flags=re.IGNORECASE)
    s = s.replace(" ", "").rstrip(".").lower()
    # strip trailing unit words after a number
    s = re.sub(rf"(?<=[\d\)\}}])({UNIT_WORDS})$", "", s)
    s = re.sub(r"^\{(.*)\}$", r"\1", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.rstrip("}") if s.count("}") > s.count("{") else s
    return s if s else None


def to_number(s):
    if s is None:
        return None
    t = s.replace(",", "")
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", t)
    if m:
        return float(t)
    m = re.fullmatch(r"\\?frac\{(-?\d+)\}\{(-?\d+)\}", t)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", t)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    # mixed number, e.g. "262/3" won't parse; handle "26 2/3" pre-normalize
    m = re.fullmatch(r"(-?\d+)\s+(\d+)/(\d+)", s.strip())
    if m and float(m.group(3)) != 0:
        whole = float(m.group(1))
        frac = float(m.group(2)) / float(m.group(3))
        return whole + frac if whole >= 0 else whole - frac
    return None


def answers_match(a, b):
    na, nb = normalize(a), normalize(b)
    if na is None or nb is None:
        return None
    if na == nb:
        return True
    xa, xb = to_number(na), to_number(nb)
    if xa is not None and xb is not None:
        return abs(xa - xb) < 1e-6
    return False


# ------------------------------------------------------------------
# load + grade
# ------------------------------------------------------------------

def load_detailed(dataset, model, d):
    files = glob.glob(os.path.join(d, "all_detailed_results_*.json"))
    if not files:
        return None
    recs = json.load(open(files[0]))
    rows = []
    for r in recs:
        gold = r.get("reference_answer")
        sa, sb = r.get("solution_a", ""), r.get("solution_b", "")
        ea, eb = extract_answer(sa), extract_answer(sb)
        ma, mb = answers_match(ea, gold), answers_match(eb, gold)
        se = r.get("self_evaluation", {})
        je = r.get("judge_evaluation", {})
        rows.append({
            "dataset": dataset, "model": model,
            "problem_id": r.get("problem_id"),
            "category": r.get("category"),
            "gold": gold,
            "ans_a": ea, "ans_b": eb,
            "a_correct": ma, "b_correct": mb,
            "self_sel": se.get("selected_solution"),
            "self_eval_confidence": se.get("confidence"),
            "judge_sel": je.get("selected_solution"),
            "self_eval_correct": float(se.get("selected_solution") ==
                                       je.get("selected_solution"))
                                 if se.get("selected_solution") and je.get("selected_solution")
                                 else np.nan,
        })
    return pd.DataFrame(rows)


def stratum(row):
    a, b = row["a_correct"], row["b_correct"]
    if a is None or b is None:
        return "ungraded"
    if a and b:
        return "both_correct"
    if not a and not b:
        return "both_wrong"
    return "exactly_one"


def main():
    frames = []
    for ds, models in DIRS.items():
        for model, d in models.items():
            t = load_detailed(ds, model, d)
            if t is None:
                print(f"  MISSING: {model} / {ds}")
                continue
            frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    df["stratum"] = df.apply(stratum, axis=1)
    df = df.dropna(subset=["self_eval_confidence"])

    # coverage report
    cov = (df.groupby(["dataset", "model"])["stratum"]
             .value_counts(normalize=True).unstack(fill_value=0).round(3))
    graded = df[df.stratum != "ungraded"].copy()
    print("Answer-extraction coverage (fraction ungraded):")
    print(cov.get("ungraded", pd.Series(dtype=float)).to_string())
    print(f"\nTotal rows {len(df)}, graded {len(graded)} "
          f"({len(graded)/len(df)*100:.1f}%)")

    # gold-selection correctness on exactly-one pairs
    def gold_pick(r):
        if r["stratum"] != "exactly_one" or r["self_sel"] not in ("A", "B"):
            return np.nan
        winner = "A" if r["a_correct"] else "B"
        return float(r["self_sel"] == winner)

    graded["gold_sel_correct"] = graded.apply(gold_pick, axis=1)

    def judge_gold_pick(r):
        if r["stratum"] != "exactly_one" or r["judge_sel"] not in ("A", "B"):
            return np.nan
        winner = "A" if r["a_correct"] else "B"
        return float(r["judge_sel"] == winner)

    graded["judge_gold_correct"] = graded.apply(judge_gold_pick, axis=1)

    # ---------- per model/dataset/stratum table ----------
    recs = []
    for (ds, model, st), g in graded.groupby(["dataset", "model", "stratum"]):
        conf = g["self_eval_confidence"].astype(float).values
        sea = g["self_eval_correct"].astype(float)
        row = {"dataset": ds, "model": model, "stratum": st, "N": len(g),
               "frac": len(g) / len(graded[(graded.dataset == ds) &
                                           (graded.model == model)]),
               "SEA_judge_ref": sea.mean(),
               "mean_conf": conf.mean()}
        if st == "exactly_one":
            gs = g["gold_sel_correct"].dropna()
            row["gold_sel_acc"] = gs.mean() if len(gs) else np.nan
            jg = g["judge_gold_correct"].dropna()
            row["judge_gold_acc"] = jg.mean() if len(jg) else np.nan
            # gating quantities against GOLD on the unambiguous subset
            sub = g.dropna(subset=["gold_sel_correct"])
            if len(sub) >= 20:
                c = sub["self_eval_confidence"].astype(float).values
                y = sub["gold_sel_correct"].values
                row["sel_acc@cov0.5_gold"] = selective_at_coverage(c, y, 0.5)
                _, acc9 = gate_at_threshold(c, y, 0.9)
                row["hc_err@0.9_gold"] = 1 - acc9 if not np.isnan(acc9) else np.nan
                row["ECE_gold"] = ece_equal_width(c, y)
                c1 = c[y == 1.0]; c0 = c[y == 0.0]
                row["conf_gap_gold"] = (c1.mean() - c0.mean()
                                        if len(c1) and len(c0) else np.nan)
        recs.append(row)
    tab = pd.DataFrame(recs).sort_values(["dataset", "model", "stratum"])
    tab.to_csv(os.path.join(OUT, "F_stratified_analysis.csv"), index=False)
    print("\n[F] Stratified table:")
    print(tab.round(3).to_string(index=False))

    # ---------- headline: reversal on exactly-one-correct subset ----------
    print("\n[F2] Exactly-one-correct subset — Math360 key pair:")
    key = tab[(tab.stratum == "exactly_one") & (tab.dataset == "Math360")]
    cols = ["model", "N", "gold_sel_acc", "sel_acc@cov0.5_gold",
            "hc_err@0.9_gold", "ECE_gold", "conf_gap_gold", "judge_gold_acc"]
    print(key[[c for c in cols if c in key.columns]].round(3).to_string(index=False))

    key5 = tab[(tab.stratum == "exactly_one") & (tab.dataset == "Math500")]
    print("\n[F2] Exactly-one-correct subset — Math500:")
    print(key5[[c for c in cols if c in key5.columns]].round(3).to_string(index=False))

    # ---------- PM-C-PVC on exactly-one subset (against gold) ----------
    print("\n[F3] PM-C-PVC-VUS on exactly-one subset (gold-graded), where N allows:")
    recs = []
    for (ds, model), g in graded[graded.stratum == "exactly_one"].groupby(["dataset", "model"]):
        sub = g.dropna(subset=["gold_sel_correct"]).copy()
        if len(sub) < 40:
            continue
        sub = sub.rename(columns={"gold_sel_correct": "self_eval_correct_gold"})
        tmp = pd.DataFrame({
            "category": sub["category"],
            "self_eval_correct": sub["self_eval_correct_gold"],
            "self_eval_confidence": sub["self_eval_confidence"].astype(float),
        })
        v = pm_cpvc_vus(tmp)
        full = graded[(graded.dataset == ds) & (graded.model == model)]
        tmp_full = pd.DataFrame({
            "category": full["category"],
            "self_eval_correct": full["self_eval_correct"].astype(float),
            "self_eval_confidence": full["self_eval_confidence"].astype(float),
        }).dropna()
        v_full = pm_cpvc_vus(tmp_full)
        recs.append({"dataset": ds, "model": model, "N_exactly_one": len(sub),
                     "PM_CPVC_VUS_exactly_one_gold": v["PM_CPVC_VUS"],
                     "PM_CPVC_VUS_full_judge": v_full["PM_CPVC_VUS"],
                     "ECE_exactly_one_gold": ece_equal_width(
                         tmp["self_eval_confidence"], tmp["self_eval_correct"])})
    f3 = pd.DataFrame(recs)
    f3.to_csv(os.path.join(OUT, "F3_pm_cpvc_exactly_one.csv"), index=False)
    print(f3.round(3).to_string(index=False))

    # ---------- judge-reference validation against gold ----------
    print("\n[F4] Judge-majority vs gold on exactly-one pairs (validates reference):")
    f4 = (graded[graded.stratum == "exactly_one"]
          .groupby("dataset")["judge_gold_correct"].agg(["mean", "count"]).round(3))
    print(f4.to_string())
    f4.to_csv(os.path.join(OUT, "F4_judge_vs_gold.csv"))


if __name__ == "__main__":
    main()
