# Reviewer Entry Point — Evaluating LLM Self-Evaluation with Calibration-Constrained Diagnostics

**Anonymous repository URL:**
`https://github.com/calibration-constrained-diagnostics/selfeval-diagnostics`

This repository accompanies the anonymous submission
*Evaluating LLM Self-Evaluation with Calibration-Constrained Diagnostics*
to the NeurIPS 2026 Evaluations & Datasets Track.

The repository is anonymized for double-blind review. Author identities,
affiliations, contact information, and persistent DOIs will be added after the
review period.

## What you can do in ~10 minutes (Level 1)

```bash
pip install -r requirements.txt
python reproduce/reproduce_main_tables.py
# Metrics are written to reproduce/outputs/{math360, truthfulqa, commonsenseqa, math500}_metrics.csv
```

This reproduces every main-paper diagnostic table (PVC / C-PVC / VUS / PM-VUS /
SEA / CalibError) directly from the released `model_outputs/*.csv`. No model
or judge reruns are required.

## What else is here

See `README.md` for the full directory layout, and `metadata/evaluation_card.md`
for the evaluation protocol specification. Release policy is documented in
`docs/release_strategy.md`.

## Anonymity notes

- `LICENSE` uses a placeholder author identifier; citation information is omitted during review.
- `metadata/croissant.json` uses the anonymous repository URL as `url`; a persistent DOI will replace it at camera-ready.
- Git commit history uses `Anonymous Authors <anonymous@example.com>`.
- No author names, emails, affiliations, institutional paths, personal GitHub
  usernames, CI badges, or internal hostnames appear anywhere in this tree.
- Upstream dataset URLs (`sylinrl/TruthfulQA`, `HuggingFaceH4/MATH-500`) are
  the only external references and are unrelated to author identity.

If you discover an identity leak during review, please flag it via OpenReview.
It will be treated as a bug against the anonymization, not as deliberate.
