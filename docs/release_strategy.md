# Release strategy

This document records the hosting, anonymity, and release-level plan for the
diagnostic evaluation package. It applies during the NeurIPS 2026 E&D Track
double-blind review period and after acceptance.

## 1. Hosting plan

| Target | Host | During review | After acceptance |
|---|---|---|---|
| Code + data (everything in `selfeval_diagnostics/`: `data/`, `reference_labels/`, `model_outputs/`, `metadata/`, `prompts/`, `selfeval/`, `reproduce/`, configs) | **Single anonymous GitHub repository** | private (or public) reviewer-accessible repo under an anonymous GitHub org, linked from OpenReview | public author repo |
| Upstream MATH-500 text | Hugging Face Hub (upstream, `HuggingFaceH4/MATH-500`) | loaded at runtime by `selfeval/utils.py` | unchanged |
| Archival snapshot | — | — | Zenodo or Harvard Dataverse DOI on the data |

## 2. Submission procedure

1. Prepare a clean local copy of the `selfeval_diagnostics/` tree (no upstream `.git`, no runtime logs).
2. Create an anonymous GitHub account and org (e.g., `selfeval-diagnostics-anon`), then create a single repository (e.g., `selfeval-diagnostics`) containing the full `selfeval_diagnostics/` tree.
3. Commit with an anonymous git author (`Anonymous Authors <anonymous@example.com>`); do not enable GitHub Actions, issue templates, or features that reveal maintainers.
4. Link the repository URL in the OpenReview submission (supplementary field or abstract footnote). Attach the same `selfeval_diagnostics/` as a zipped supplementary file as a fallback in case the repository is unreachable during review.
5. Cite as "an anonymous reviewer-accessible GitHub repository containing code, data splits, judge-reference labels, model outputs, and reproduction scripts; a mirror of the same tree is attached as supplementary material on OpenReview."

## 3. Three-level release

| Level | Required materials | Target user |
|---|---|---|
| L1 | `model_outputs/`, `selfeval/scripts/compute_pvc_cpvc.py`, `reproduce/reproduce_main_tables.py` | Reviewer who wants to reproduce every main-paper table without model or judge access |
| L2 | L1 + `selfeval/scripts/run_eval_pipeline.py` judge mode + `reproduce/build_reference_labels.py` | User with judge API access who wants to rebuild reference labels |
| L3 | L2 + full target-model generation | User with GPU or model API access who wants end-to-end reruns |

## 4. Redistribution policy

- **Math-360 original questions and solutions**: redistributed under CC-BY-4.0.
- **TruthfulQA / CommonsenseQA / MATH-500**: only IDs, category mappings, and
  derived evaluation outputs are redistributed; original question text must be
  obtained from upstream sources.
- **Target and judge model text outputs**: we redistribute selections, votes,
  and confidences in aggregated form (CSV / JSONL). Raw rationale strings are
  not redistributed; they can be regenerated via the provided pipeline scripts
  with the relevant API / model access.

## 5. Anonymization workflow (used to produce this tree)

The steps below document how `selfeval_diagnostics/` was prepared to avoid identity
leaks. They are reproducible for future revisions.

```bash
# 1. Start from a clean directory (no upstream .git)
mkdir selfeval-diagnostics-anon
cd selfeval-diagnostics-anon

# 2. Initialize a fresh git history with an anonymous author
git init
git config --local user.name  "Anonymous Authors"
git config --local user.email "anonymous@example.com"

# 3. Copy only the reviewer-facing tree (not the original repo's .git or logs)
cp -r <source>/selfeval_diagnostics/* .

# 4. Scan for identity leaks: replace the placeholders below with the real
#    author first/last name, usernames, employer/affiliation tokens, and
#    institutional email domains before running. Keep this list in a private
#    checklist; do NOT commit it populated.
grep -RIn -E "<FIRSTNAME>|<LASTNAME>|<USERNAME>|<EMPLOYER>|<GROUP>|@<INSTITUTION_DOMAIN>|/Users/|/home/" .
grep -RIn "github.com/" . | grep -v "sylinrl/TruthfulQA"  # only upstream OK
grep -RIn "huggingface.co/" . | grep -v "HuggingFaceH4/MATH-500"

# 5. Clear notebook outputs and metadata
jupyter nbconvert --clear-output --inplace selfeval/notebooks/*.ipynb

# 6. Remove runtime artifacts (see .gitignore)
rm -rf wandb outputs/raw_api_logs .credentials* .env

# 7. Commit
git add .
git commit -m "Initial anonymous release"
```

## 6. Review-period leak checklist

The following items must all pass before submission:

- [ ] No author names, emails, affiliations in any file (README,
      pyproject.toml, LICENSE, configs, notebooks, scripts, docs).
- [ ] No absolute paths like `/Users/<name>/` or `/home/<name>/`.
- [ ] No personal GitHub usernames (upstream dataset URLs are
      fine: `sylinrl/TruthfulQA`, `HuggingFaceH4/MATH-500`, etc.).
- [ ] No CI/CD badges that link to an author account.
- [ ] No Docker image references that include an author namespace.
- [ ] No wandb project URLs, S3 bucket names, internal hostnames.
- [ ] Git commit history uses `Anonymous Authors <anonymous@example.com>`.
- [ ] Jupyter notebooks have no execution metadata referencing a username.
- [ ] `LICENSE` uses a placeholder author identifier during review; no separate citation file is shipped.

## 7. Post-acceptance plan

- Transfer (or rename) the anonymous GitHub repository to the author's public org.
- Mint a Zenodo DOI (or Harvard Dataverse record) for an archival snapshot.
- Freeze a `v1.0` tag on the repository.
- Update `README.md`, `metadata/croissant.json`, and the paper with the DOIs
  and final author information.

## 8. Paper wording during review

> Code and data are available in an anonymous reviewer-accessible GitHub
> repository. The release includes Math-360, processed category mappings for
> TruthfulQA / CommonsenseQA / MATH-500, prompts, judge-reference labels,
> model confidence outputs, and scripts for reproducing all main tables and
> figures. The repository will be de-anonymized upon acceptance.
