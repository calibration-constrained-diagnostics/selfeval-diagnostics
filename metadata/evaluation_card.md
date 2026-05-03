# Evaluation Card — LLM Self-Evaluation Diagnostics (v1.0)

## Protocol
Three-stage black-box pairwise self-evaluation:

1. **Candidate generation.** For each question, the target model generates two candidate solutions with distinct prompts (`prompts/candidate_generation_prompt_A.txt`, `..._B.txt`) at temperature 0.7, top-*p* 0.9, max 4096 tokens.
2. **Self-selection.** The target model is shown both candidates in randomized order (no reference-label access) and returns (i) a binary choice A/B and (ii) a confidence on [0, 100] (`prompts/self_selection_prompt.txt`).
3. **Reference labeling.** A three-judge LLM ensemble (Claude 3.7 Sonnet, Amazon Nova Premier, DeepSeek-R1) independently returns a preference for the same pair with reference-answer access (`prompts/judge_prompt.txt`). The majority vote is the reference signal.

## Metrics

| Name | Definition | Interpretation |
|---|---|---|
| SEA | Category-level agreement between target selection and majority reference | Reference-agreement self-evaluation accuracy (not absolute correctness) |
| CalibError | \|mean reported confidence − SEA\| per category | Category-level miscalibration |
| PVC | Number of categories with SEA ≥ γ | Category-level discriminative self-evaluation |
| C-PVC | Number of categories with SEA ≥ γ and CalibError ≤ τ | PVC under per-category calibration |
| VUS | Volume-under-surface of the corresponding metric over (γ, τ) | Threshold-invariant summary |
| PM-VUS | VUS restricted to γ > τ + 1/2 (positive-margin region) | Stringent, theory-motivated summary |

All metrics are **operational diagnostic proxies**, not literal VC-dimension estimates. PM (positive-margin) is the region in which the uniform-convergence argument in the paper's Theorem 1 applies, under an assumed capacity envelope on the induced classes. PM-VUS is not an empirical generalization guarantee.

## What this protocol measures
- Whether a model's reported confidence tracks judge-referenced self-selection correctness.
- Whether discrimination and calibration survive **simultaneously** across semantic categories.
- Whether calibration-constrained self-evaluation behavior survives in the positive-margin region.

## What this protocol does not measure
- Absolute correctness of model reasoning (requires human or ground-truth validation).
- Human-preference alignment (requires human annotation).
- Open-ended generation quality or multi-turn self-critique.
- Deployment safety.

## Reference-label validity claim
Reference labels are treated as a **fixed comparative reference**, not as oracle ground truth. The diagnostic signal is the **difference between models under a fixed reference**, not the absolute correctness of any single label.

## Reproducibility levels
- **Level 1.** Recompute all metrics from released `model_outputs/` (<10 min on CPU).
- **Level 2.** Rebuild reference labels from released candidates with judge API access.
- **Level 3.** Rerun full target-model generation with GPU or model API.

## Versioning
v1.0 — frozen at initial release; corresponds to the NeurIPS 2026 E&D submission under review.
