# Croissant Validation Report

**Validated file:** `metadata/croissant.json`
**Spec:** MLCommons Croissant 1.0 (`conformsTo: http://mlcommons.org/croissant/1.0`)

## Validation commands

```bash
pip install -U mlcroissant
mlcroissant validate --jsonld metadata/croissant.json
python scripts/validate_croissant.py
```

## Environment

| Component     | Version  |
|---------------|----------|
| Python        | 3.12     |
| `mlcroissant` | 1.1.0    |
| Date          | 2026-05-03 |

## Result

| Check                                      | Status |
|--------------------------------------------|--------|
| JSON-LD syntactic validity                 | PASS   |
| `@context` conforms to Croissant 1.0       | PASS   |
| Schema validation (`mlcroissant validate`) | PASS   |
| MD5 checksum match for every `FileObject`  | PASS   |
| Loadability (`mlcroissant.Dataset.records(...)` on `math360-records`) | PASS -- two records read end-to-end from the public GitHub raw URL, with MD5 verified and JSONL fields parsed correctly |

## Warnings (non-blocking, intentional)

The validator reports two recommended-field warnings that are intentional for the double-blind review period:

1. `citeAs` is absent -- omitted to preserve author anonymity during review.
2. `schema:datePublished` is absent -- the artifact is under review and unpublished.

Both fields will be added at camera-ready together with the DOI and author list.

## Responsible AI (RAI) metadata

Per the NeurIPS 2026 E&D Track requirement (blog post, 2026-05-04), the Croissant file contains Responsible AI metadata in the following fields:

- `rai:dataCollection`, `rai:dataCollectionType`, `rai:dataCollectionTimeframe`
- `rai:dataAnnotationProtocol`
- `rai:dataPreprocessingProtocol`
- `rai:dataUseCases`
- `rai:dataLimitations`
- `rai:dataBiases`
- `rai:personalSensitiveInformation`
- `rai:dataSocialImpact`
- `rai:dataReleaseMaintenancePlan`

## What this validation guarantees

- Any third party running `mlcroissant validate` against the distributed file will obtain the same PASS result.
- Any third party running `mlcroissant.Dataset.records(...)` on the public repository will stream records from `data/math360/questions.jsonl` with schema-typed fields (`id`, `problem`, `answer`, `category`, `subcategory`, `difficulty`).

## What this validation does not guarantee

- Loadability of the `reference-labels` and `model-outputs` `FileSet`s via the `mlcroissant` streaming API is not yet exercised at record level; these are shipped as `FileSet`s with glob patterns (one file per model or dataset) and should be consumed via the direct file paths listed in `metadata/checksums.txt`.
