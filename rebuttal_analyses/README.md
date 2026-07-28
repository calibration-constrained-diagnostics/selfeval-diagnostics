# Rebuttal Analyses (NeurIPS 2026 Author Response)

Targeted additional analyses referenced in the author response. All analyses
in `results/` re-use the released 17,420 predictions (`model_outputs/`) and
archived candidate outputs; the consistency experiment re-runs only the
self-selection step with a selection-only prompt (protocol in
`scripts/consistency_vllm.py` / `scripts/consistency_bedrock.py`).

## Reviewer concern → files

**Act-or-defer gating experiment (13 models)**
- `results/A_gating_metrics.csv` — risk–coverage/AURC, selective accuracy at fixed coverage, HC-Err@t = P(wrong | conf ≥ t), act-or-defer utility U_t = P(act ∧ correct) − λ·P(act ∧ wrong) with act ≡ conf ≥ t (t = 0.9; λ ∈ {0.5, 1, 2, 5} in `A2_auroc_lambda_sweep.csv`)
- `results/A4_correlations_perm_p.csv` — Spearman associations with permutation p-values (10,000 permutations), incl. AUROC baseline
- `results/A5_per_dataset_correlations.csv` — per-dataset breakdown (incl. MATH-500)
- `results/E_paired_bootstrap_reversal.csv`, `results/E2_paired_bootstrap_hce_util.csv` — paired bootstrap (questions resampled within datasets, model pairing preserved; 1,000 resamples). Note: the observed pooled selective-accuracy difference for JiuZhang3.0-7B vs Qwen2.5-32B-Instruct is −0.084; the bootstrap resample mean is −0.082 (CI [−0.148, −0.019]).
- `scripts/rebuttal_neurips_analysis.py`, `scripts/rebuttal_addendum.py`

**Group-wise calibration baselines**
- `results/B_groupwise_ece_table.csv` — macro-, weighted-, worst-category, equal-mass (adaptive) ECE; category Brier; PVC/C-PVC/PM-C-PVC-VUS per model

**Matched random-partition null (semantic vs. generic binning)**
- `results/C_random_partition_null.csv` — 39 model–dataset pairs × 1,000 partitions matched to semantic category sizes
- `results/C_random_partition_summary.csv`, `results/C2_random_partition_effect_sizes.csv`, `results/C_pm_inflation_summary.csv`

**Gold correctness stratification (MATH-500 / Math-360)**
- `results/F_stratified_analysis.csv` — exactly-one / both-correct / both-wrong strata
- `results/F9_exactly_one_adjudication.csv` — the "incorrect"-labeled candidate of all 371 apparent MATH-500 exactly-one pairs, rechecked against gold (19 both-correct pairs identified and removed)
- `results/F10_exactly_one_adjudicated.csv` — per-model results on the cleaned subset
- `results/F4_judge_vs_gold.csv` — judge-majority vs. gold agreement
- `results/F5_extraction_coverage.csv`, `results/F6_within_model_bias_check.csv` — extraction-coverage validity checks
- `results/F7_manual_audit_sample.csv`, `results/F8_heldout_audit_sample.csv` — two-stage matcher audit samples (first audit surfaced 10 systematic misses, corrected; held-out audit: zero false positives)
- `scripts/rebuttal_stratification.py`

**Consistency-based confidence (targeted four-model ablation)**
- `results/G_consistency_comparison.csv` — verbalized vs. fixed-decision vs. modal variants, MATH-500 and Math-360 (K = 10 balanced-order stochastic selections per archived pair; temperature 0.7, top-p 0.9)
- `scripts/consistency_vllm.py` (open models, vLLM), `scripts/consistency_bedrock.py` (API), `scripts/analyze_consistency.py`

**Category-threshold transfer (negative result, reported for completeness)**
- `results/D_threshold_transfer.csv`, `results/D_threshold_transfer_summary.csv`

**Additional robustness**
- `results/E_bootstrap_cis.csv` — bootstrap CIs for per-model PM-C-PVC-VUS, AURC, confidence gap
- `scripts/semantic_vs_random_downstream.py` — semantic-vs-random downstream association (supplementary)

## Notes

- The gating analyses use the three cross-domain datasets available for all
  13 models (Math-360, TruthfulQA, CSQA), matching the paper's Table 1 pool;
  the paper's tables use different aggregations, so some ECE and PM-C-PVC
  values differ slightly across tables.
- Archived single-shot decisions come from the original HF pipeline; the
  consistency re-selections use vLLM on the same checkpoints and precision.
- Script paths assume the released `model_outputs/` CSVs and archived
  candidate JSONs; adjust the `BASE`/`DL` constants at the top of each script.
