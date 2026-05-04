# Calibration-Constrained Diagnostics for Evaluating LLM Self-Evaluation

**Anonymous repository for double-blind review:**
`https://github.com/calibration-constrained-diagnostics/selfeval-diagnostics/`

This repository releases the **complete diagnostic evaluation package** accompanying the NeurIPS 2026 Evaluations & Datasets Track submission of the same title. The released artifact is **not a single dataset**: it is a reusable evaluation protocol, diagnostic metrics, a controlled diagnostic testbed (Math-360), cross-domain metadata, judge-reference labels, and reproduction scripts, together with a dataset/evaluation card and Croissant metadata. `metadata/croissant.json` has been validated against the [MLCommons Croissant 1.0 schema](https://mlcommons.org/working-groups/data/croissant/) (`mlcroissant validate`) and loads cleanly via `mlcroissant.Dataset`; file MD5 checksums for every redistributed input are listed in `metadata/checksums.txt`.

## What this artifact is / is not

**Enables**

- Reproducing the PVC / C-PVC / PM-VUS diagnostic scores in the paper.
- Comparing whether a model's reported confidence tracks judge-referenced self-selection correctness.
- Auditing category-level calibration failures hidden by aggregate uncertainty summaries (ECE, Brier).
- Extending the protocol to new models and datasets.

**Does not enable**

- Absolute human-ground-truth correctness claims.
- Literal VC-dimension estimation.
- Deployment safety certification for any particular model.
- A general-purpose math leaderboard.

Math-360 is a **controlled diagnostic testbed**, not a standalone general-purpose math benchmark. It has a supporting role; the central contribution is the evaluation methodology and diagnostic framework.

## Directory layout

```
selfeval_diagnostics/
├── README.md                    ← you are here
├── LICENSE                      ← license for the package
├── requirements.txt             ← Python dependencies
├── pyproject.toml               ← installable selfeval package
│
├── data/
│   ├── math360/
│   │   └── questions.jsonl      ← 360 original problems, 8 domains, 5 subcategories × 3 difficulties
│   └── cross_domain/
│       ├── truthfulqa_categories.jsonl   ← category mapping + IDs only
│       └── commonsenseqa_categories.jsonl← category mapping + IDs only
│
├── prompts/
│   ├── candidate_generation_prompt_A.txt
│   ├── candidate_generation_prompt_B.txt
│   ├── self_selection_prompt.txt
│   └── judge_prompt.txt
│
├── reference_labels/
│   ├── schema.md                ← JSONL schema
│   ├── math360.jsonl            ← majority judge votes + target self-selection + confidence
│   ├── truthfulqa.jsonl
│   ├── commonsenseqa.jsonl
│   └── math500.jsonl
│
├── model_outputs/
│   ├── math360_self_eval.csv    ← raw per-pair target/judge outputs (wide CSV)
│   ├── truthfulqa_self_eval.csv
│   ├── commonsenseqa_self_eval.csv
│   └── math500_self_eval.csv
│
├── selfeval/                    ← installable package: protocol + metric code
│   ├── __init__.py
│   ├── models.py
│   ├── judges.py
│   ├── evaluation.py
│   ├── experiment.py
│   ├── utils.py
│   ├── configs/
│   │   ├── models.yaml
│   │   ├── datasets.yaml
│   │   ├── decoding.yaml
│   │   └── prompts.yaml
│   └── scripts/
│       ├── run_eval_pipeline.py        ← Level 3: full generation + judge labeling
│       ├── compute_pvc_cpvc.py         ← PVC / C-PVC / VUS / PM-VUS
│       └── compute_cross_dataset.py    ← cross-dataset aggregation
│
├── reproduce/
│   ├── reproduce_main_tables.py        ← Level 1: all paper tables + figures
│   ├── build_reference_labels.py       ← model_outputs CSV → reference_labels JSONL
│   ├── outputs/                        ← generated tables and figures
│   └── notebooks/
│       └── reproduce_tables_and_figures.ipynb  ← inline notebook rendering
│
├── metadata/
│   ├── croissant.json            ← validated Croissant dataset metadata
│   ├── dataset_card.md           ← per-dataset card (intended use, limits, license)
│   └── evaluation_card.md        ← evaluation-protocol card (what is measured, how)
│
└── docs/
    └── release_strategy.md       ← hosting/anonymity/release plan
```

## Reproducibility levels

**Level 1 — Metric reproduction from released outputs (≤10 min, CPU).**
Reproduce every main-paper table from the released `model_outputs/*.csv` without rerunning any model or judge.

```bash
pip install -r requirements.txt
python reproduce/reproduce_main_tables.py
# → reproduce/outputs/{math360, truthfulqa, commonsenseqa, math500}_metrics.csv
```

**Level 2 — Reference-label recomputation (requires judge API access).**
Rerun the three-judge ensemble on the released candidates and rebuild `reference_labels/*.jsonl`.

```bash
python selfeval/scripts/run_eval_pipeline.py \
    --mode judge_relabel \
    --candidates model_outputs/ \
    --judges c37_sonnet nova_premier deepseek_r1
python reproduce/build_reference_labels.py
```

**Level 3 — Full target-model generation (requires GPU or model API).**
Rerun the full three-stage protocol end-to-end on selected target models.

```bash
python selfeval/scripts/run_eval_pipeline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --judges c37_sonnet nova_premier deepseek_r1 \
    --problems data/math360/questions.jsonl \
    --output model_outputs/
```

## Datasets

| Dataset | Role | Release form |
|---|---|---|
| Math-360 | controlled diagnostic testbed | **full release** (questions + solutions + categories) |
| TruthfulQA | cross-domain validation | IDs + category mappings + derived evaluation outputs only |
| CommonsenseQA | cross-domain validation | IDs + category mappings + derived evaluation outputs only |
| MATH-500 | cross-domain validation | derived evaluation outputs only |

For TruthfulQA / CommonsenseQA / MATH-500 we do **not** redistribute original question text. Please obtain the original datasets from their official sources and use the scripts in `selfeval/` to reconstruct the evaluation splits.

## Citation

The accompanying paper is currently under double-blind review. Citation information will be added upon acceptance.

## License

Math-360 original questions and solutions are released under **CC-BY-4.0**.
Code in `selfeval/` and `reproduce/` is released under **Apache-2.0**.
Derived evaluation outputs (`reference_labels/`, `model_outputs/`) are released under **CC-BY-4.0** where permitted by upstream model / API provider terms.

See `LICENSE` for full text and `metadata/dataset_card.md` for dataset-specific terms.
