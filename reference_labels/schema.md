# Reference-label schema

Each reference-label JSONL record in `reference_labels/*.jsonl` (one file per dataset) follows this schema:

```json
{
  "example_id":            "math360_algebra_easy_001",
  "model_id":              "qwen2.5-7b-instruct",
  "candidate_a_id":        "cand_a_<hash>",
  "candidate_b_id":        "cand_b_<hash>",
  "target_selected":       "A",
  "target_confidence":     0.82,
  "judge_1_model":         "claude-3.7-sonnet",
  "judge_1_vote":          "A",
  "judge_2_model":         "amazon-nova-premier",
  "judge_2_vote":          "B",
  "judge_3_model":         "deepseek-r1",
  "judge_3_vote":          "A",
  "majority_reference":    "A",
  "agreement_type":        "2-1",
  "presentation_order_seed": 1234,
  "invalid_judge_response": false
}
```

## Field reference

| Field | Type | Description |
|---|---|---|
| `example_id` | str | Unique example id across the package (`{dataset}_{category}_{difficulty}_{index}`). |
| `model_id` | str | Target model under evaluation. |
| `candidate_a_id`, `candidate_b_id` | str | Deterministic hash IDs for the two candidate solutions shown to the target model. |
| `target_selected` | `"A"` \| `"B"` | Target model's selection. |
| `target_confidence` | float in [0, 1] | Confidence that the selected candidate matches the reference preference. |
| `judge_{1..3}_model` | str | Judge identity. |
| `judge_{1..3}_vote` | `"A"` \| `"B"` \| `null` | Judge vote; `null` when the judge response was invalid/missing. |
| `majority_reference` | `"A"` \| `"B"` | Majority vote among valid judges. Always defined when at least 2 judges produced a valid vote. |
| `agreement_type` | `"3-0"` \| `"2-1"` | Unanimous vs.\ split decision. |
| `presentation_order_seed` | int | Seed that determined the A/B presentation order. |
| `invalid_judge_response` | bool | True when any judge returned an invalid response; the pair is excluded from the main aggregate when this prevents a valid majority. |

## Notes

- The `majority_reference` column is the evaluation signal used throughout the paper; it is treated as a **fixed comparative reference**, not as absolute ground truth.
- Raw judge rationale strings are not redistributed in this release; they can be regenerated via `selfeval/scripts/run_eval_pipeline.py` with judge access.
- Less than 0.4% of pairs have `invalid_judge_response=true`; these are excluded from the aggregates.
