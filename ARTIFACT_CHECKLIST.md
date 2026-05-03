# Artifact Checklist

Reviewer-facing summary of what is included in this release and which claims
the artifact is designed to support.

**Anonymous repository:**
`https://github.com/calibration-constrained-diagnostics/selfeval-diagnostics/`

## Release contents

- [x] Math-360 diagnostic testbed released in full (`data/math360/questions.jsonl`, 360 problems)
- [x] Cross-domain category splits released as IDs + category mappings (TruthfulQA, CommonsenseQA); original question text is **not** redistributed
- [x] Judge-reference labels released (`reference_labels/*.jsonl` for Math-360, TruthfulQA, CSQA, MATH-500)
- [x] Model outputs released (`model_outputs/*_self_eval.csv` for all evaluated models across four datasets)
- [x] Prompts released verbatim (`prompts/`: candidate generation A/B, self-selection, judge)
- [x] Diagnostic metric code released (`selfeval/scripts/compute_pvc_cpvc.py`, `compute_cross_dataset.py`)
- [x] Reproduction scripts released (`reproduce/reproduce_main_tables.py`, `reproduce/build_reference_labels.py`)
- [x] Configs released (`selfeval/configs/` for datasets, decoding, models, prompts)
- [x] File checksums (`metadata/checksums.txt`, MD5 for every distributed file)
- [x] Croissant metadata (`metadata/croissant.json`, including core + RAI fields; populated with real `url` and MD5 hashes for every `FileObject`; should be run through the MLCommons Croissant online validator before camera-ready)
- [x] Dataset card (`metadata/dataset_card.md`)
- [x] Evaluation card (`metadata/evaluation_card.md`)
- [x] License (`LICENSE`: CC-BY-4.0 for data / derived outputs; code under Apache-2.0 via `pyproject.toml`)

## Reproducibility levels

- [x] **L1** — main-paper tables from released `model_outputs/*.csv` (no API keys, CPU, $<$10 min)
  - Command: `python reproduce/reproduce_main_tables.py`
- [x] **L2** — judge-label recomputation from released candidate pairs (requires judge-API access)
  - Entry point: `selfeval/scripts/run_eval_pipeline.py` in judge mode + `reproduce/build_reference_labels.py`
- [x] **L3** — end-to-end self-evaluation rerun from prompts/configs (requires target-model API or checkpoints)
  - Entry point: `selfeval/scripts/run_eval_pipeline.py`

## Claim boundary

This artifact is designed to support:

- [x] Comparing whether model-reported confidence tracks judge-referenced self-selection correctness.
- [x] Detecting category-level calibration / discrimination failures hidden by aggregate UQ summaries.
- [x] Analyzing whether calibration-constrained self-evaluation behavior survives in the positive-margin region.
- [x] Reproducing and extending PVC / C-PVC / PM-VUS surfaces across new models and datasets under the declared protocol.
- [x] Diagnosing deployment-relevant failure modes for confidence-gated downstream decisions.

This artifact does **not** support:

- [ ] Certifying absolute correctness of model reasoning.
- [ ] Providing human-preference alignment without additional human validation.
- [ ] Literal VC-dimension estimation or distribution-free sample-complexity bounds.
- [ ] Universal claims that LLMs cannot self-evaluate outside the declared protocol.
- [ ] Certifying deployment safety of any specific model.

## Anonymity

- [x] Commit history uses `Anonymous Authors <anonymous@example.com>`.
- [x] No author names, emails, affiliations, institutional paths, CI badges, wandb URLs, or internal hostnames in any file.
- [x] Upstream dataset URLs (`sylinrl/TruthfulQA`, `HuggingFaceH4/MATH-500`) are unrelated to author identity.
- [x] Croissant `url` points at the anonymous review repository; will be replaced by a persistent DOI at camera-ready.
- [x] No citation file is shipped during double-blind review.

## After acceptance

- [ ] Transfer (or rename) the anonymous GitHub repository to the author's public org.
- [ ] Mint a Zenodo DOI (or Harvard Dataverse record) for an archival snapshot.
- [ ] Freeze a `v1.0` tag.
- [ ] Update `README.md`, `metadata/croissant.json`, and the paper with the DOI and final author information.
