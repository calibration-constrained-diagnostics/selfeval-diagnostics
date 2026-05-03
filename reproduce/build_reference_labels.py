#!/usr/bin/env python3
"""
Convert the released model_outputs CSV files into reference-label JSONL files
using the schema documented in reference_labels/schema.md.

Usage:
    python reproduce/build_reference_labels.py

Input:
    model_outputs/{math360, truthfulqa, commonsenseqa, math500}_self_eval.csv

Output:
    reference_labels/{math360, truthfulqa, commonsenseqa, math500}.jsonl
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reference_labels"
IN_DIR = ROOT / "model_outputs"

DATASETS = [
    ("math360", "math360_self_eval.csv"),
    ("truthfulqa", "truthfulqa_self_eval.csv"),
    ("commonsenseqa", "commonsenseqa_self_eval.csv"),
    ("math500", "math500_self_eval.csv"),
]

JUDGES = [
    ("judge_a", "claude-3.7-sonnet"),
    ("judge_b", "amazon-nova-premier"),
    ("judge_c", "deepseek-r1"),
]


def _cand_hash(model_id: str, problem_id: str, letter: str) -> str:
    key = f"{model_id}|{problem_id}|{letter}".encode("utf-8")
    return f"cand_{letter.lower()}_{hashlib.sha1(key).hexdigest()[:12]}"


def _vote(row: dict, prefix: str) -> str | None:
    v = row.get(f"{prefix}_answer")
    if v in ("A", "B"):
        return v
    return None


def _majority(votes: list[str | None]) -> tuple[str | None, str]:
    valid = [v for v in votes if v in ("A", "B")]
    if not valid:
        return None, "invalid"
    a = valid.count("A")
    b = valid.count("B")
    if a == b:
        return None, "tie"
    if a == 3 or b == 3:
        return ("A" if a > b else "B"), "3-0"
    return ("A" if a > b else "B"), "2-1"


def _confidence(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def build_one(name: str, csv_name: str) -> tuple[int, int]:
    in_path = IN_DIR / csv_name
    out_path = OUT_DIR / f"{name}.jsonl"
    kept = 0
    skipped = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            votes = [_vote(row, p) for p, _ in JUDGES]
            majority, agreement = _majority(votes)
            invalid = any(v is None for v in votes)

            record = {
                "example_id": row["problem_id"],
                "dataset": name,
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "model_id": row["model_id"],
                "candidate_a_id": _cand_hash(row["model_id"], row["problem_id"], "A"),
                "candidate_b_id": _cand_hash(row["model_id"], row["problem_id"], "B"),
                "target_selected": row.get("self_eval_answer"),
                "target_confidence": _confidence(row, "self_eval_confidence"),
                "judge_1_model": JUDGES[0][1],
                "judge_1_vote": votes[0],
                "judge_1_confidence": _confidence(row, "judge_a_confidence"),
                "judge_2_model": JUDGES[1][1],
                "judge_2_vote": votes[1],
                "judge_2_confidence": _confidence(row, "judge_b_confidence"),
                "judge_3_model": JUDGES[2][1],
                "judge_3_vote": votes[2],
                "judge_3_confidence": _confidence(row, "judge_c_confidence"),
                "majority_reference": majority,
                "agreement_type": agreement,
                "invalid_judge_response": invalid,
                "reference_answer_available": row.get("correct_answer") is not None,
            }
            if majority is None:
                skipped += 1
                continue
            fout.write(json.dumps(record) + "\n")
            kept += 1
    return kept, skipped


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, csv_name in DATASETS:
        kept, skipped = build_one(name, csv_name)
        print(f"[{name}] kept={kept} skipped={skipped} -> {OUT_DIR / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
