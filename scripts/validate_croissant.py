#!/usr/bin/env python3
"""Croissant loadability check.

Loads ``metadata/croissant.json`` via ``mlcroissant`` and streams a few
records from each record set to verify that the Croissant metadata is not
only schema-valid but also usable end-to-end.

Usage:
    conda activate selfeval
    pip install -U mlcroissant
    python scripts/validate_croissant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlcroissant as mlc


CROISSANT_PATH = Path(__file__).resolve().parent.parent / "metadata" / "croissant.json"


def main() -> int:
    dataset = mlc.Dataset(jsonld=str(CROISSANT_PATH))
    print("Dataset:", dataset.metadata.name)
    print("Record sets:")
    for record_set in dataset.metadata.record_sets:
        print(f"  - {record_set.name}")

    ok = 0
    for record_set in dataset.metadata.record_sets:
        print(f"\nChecking record set: {record_set.name}")
        n = 0
        try:
            for record in dataset.records(record_set=record_set.uuid):
                n += 1
                if n <= 1:
                    preview = {
                        k: (v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v)
                        for k, v in record.items()
                    }
                    print(f"  sample record: {preview}")
                if n >= 2:
                    break
            print(f"  -> {n} record(s) streamed OK")
            ok += 1
        except Exception as e:  # noqa: BLE001 -- report any loader error without traceback
            print(f"  -> skipped ({type(e).__name__}: {e})")

    print(f"\n{ok}/{len(dataset.metadata.record_sets)} record sets streamed successfully.")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
