# Dataset Card — LLM Self-Evaluation Diagnostics Package (v1.0)

## Name
LLM Self-Evaluation Diagnostics — Calibration-Constrained Benchmark and Diagnostic Protocol.

## Summary
A reusable diagnostic evaluation package for LLM self-evaluation reliability. It combines a controlled diagnostic testbed (Math-360), processed cross-domain category splits (TruthfulQA, CommonsenseQA, MATH-500), prompts, reference labels from a three-judge LLM ensemble, released model outputs, and diagnostic metric code.

## Intended use
- Diagnosing whether LLM-reported confidence is reliable enough for confidence-gated downstream decisions.
- Comparing models under a fixed reference signal.
- Auditing category-level calibration/discrimination failures hidden by aggregate UQ summaries.
- Extending the protocol and metrics to new models and datasets.

## Out-of-scope use
- General-purpose math leaderboard ranking.
- Absolute correctness certification.
- Human-preference alignment claims without human validation.
- Deployment safety certification.

## Data provenance

| Component | Source | Release form |
|---|---|---|
| Math-360 questions | Original, authored by the paper's authors | Full release (CC-BY-4.0) |
| TruthfulQA categories | [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA) | IDs + category mapping only |
| CommonsenseQA categories | [CommonsenseQA](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa) | IDs + category mapping only |
| MATH-500 subset | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | IDs + derived outputs only |
| Reference labels | This work (three-judge LLM ensemble) | JSONL release (CC-BY-4.0) |
| Model outputs | This work (self-selection + confidence) | CSV release (CC-BY-4.0) |
| Prompts and scripts | This work | Apache-2.0 |

## Reference-labeling procedure
Three independent LLM judges — Claude 3.7 Sonnet, Amazon Nova Premier, DeepSeek-R1 — are each prompted with the pairwise comparison template in `prompts/judge_prompt.txt`. Each judge is given access to the reference answer; the target model is not. The reference preference is the majority vote. Invalid / missing responses (<0.4%) are excluded. Reference labels are a **fixed comparative reference signal**, not an oracle.

## Known limitations / risks
- **LLM-as-judge bias.** Judges may have self-preference, position bias, or length bias. Mitigated by a three-judge ensemble, randomized A/B order, and label-perturbation robustness checks (see paper §5.4).
- **Controlled set size.** Math-360 is 360 items; it is a diagnostic testbed, not a leaderboard. Cross-domain validation is provided by TruthfulQA / CommonsenseQA / MATH-500.
- **Proprietary judge drift.** Claude / Nova / DeepSeek-R1 API versions are pinned at the time of the experiments; outputs may drift under vendor updates.
- **License variability.** Cross-domain datasets are governed by their upstream licenses; we redistribute only derived metadata.

## Versioning
`v1.0` — frozen at initial release. Future revisions will be versioned.

## License
See repository-root `LICENSE`. Summary: CC-BY-4.0 for data and derived outputs, Apache-2.0 for code, upstream licenses preserved for re-used datasets.
