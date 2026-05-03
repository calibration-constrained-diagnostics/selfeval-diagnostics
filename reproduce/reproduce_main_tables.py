#!/usr/bin/env python3
"""
Level-1 reproduction: compute all main-paper metrics (PVC, C-PVC, VUS, PM-VUS,
SEA, CalibError) from the released model_outputs CSVs. Writes a single summary
CSV per dataset under ``reproduce/outputs/``.

This corresponds to Reproducibility Level 1 in README.md: no model or judge
re-execution is required; only the released outputs and the selfeval metric
code are used.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "reproduce" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    script = ROOT / "selfeval" / "scripts" / "compute_pvc_cpvc.py"
    if not script.exists():
        print(f"ERROR: missing {script}", file=sys.stderr)
        sys.exit(1)

    # The metric script reads a CSV named via ``output_csv`` at the top of the
    # file; we invoke it once per released dataset via a small wrapper env var.
    datasets = [
        ("math360", ROOT / "model_outputs" / "math360_self_eval.csv"),
        ("truthfulqa", ROOT / "model_outputs" / "truthfulqa_self_eval.csv"),
        ("commonsenseqa", ROOT / "model_outputs" / "commonsenseqa_self_eval.csv"),
        ("math500", ROOT / "model_outputs" / "math500_self_eval.csv"),
    ]
    for name, csv_path in datasets:
        out_path = OUTPUTS / f"{name}_metrics.csv"
        print(f"[{name}] computing metrics -> {out_path}")
        subprocess.check_call([
            sys.executable, str(script),
            "--input", str(csv_path),
            "--output", str(out_path),
        ])
    print("Level-1 reproduction complete. See reproduce/outputs/*.csv.")


if __name__ == "__main__":
    main()
