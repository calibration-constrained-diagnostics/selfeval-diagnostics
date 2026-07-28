#!/usr/bin/env python3
"""
Consistency-based confidence via the Bedrock API (NeurIPS 2026 rebuttal,
Reviewer odoV W2/Q2).

Design (per rebuttal plan):
  - Reuse each ARCHIVED candidate pair verbatim (candidates fixed).
  - Query the same target model K=10 times with stochastic decoding
    (temperature 0.7, top-p 0.9) through the black-box API.
  - Selection-only prompt: no numerical confidence is requested.
  - A/B presentation order randomized per call, mapped back to the
    archived orientation.
  - Consistency confidence = modal-selection frequency max(nA,nB)/n_valid.

Usage:
    python consistency_bedrock.py <bedrock_model_id> <pairs_json> <out_jsonl>

Incremental writes; resumes by skipping problem_ids already in output.
"""

import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config

PROFILE = "default"
REGION = "us-east-1"
K = 10
MAX_RETRIES_PER_SAMPLE = 3
PARALLEL_PAIRS = 8

SYSTEM_PROMPT = """You are an expert mathematics evaluator.
Your task is to carefully evaluate two different solutions to the same mathematical problem and determine which solution is superior in terms of correctness, clarity, and mathematical rigor."""

USER_TEMPLATE = """Please analyze the following mathematical problem and two proposed solutions:

PROBLEM:
{problem}

SOLUTION A:
{solution_a}

SOLUTION B:
{solution_b}

INSTRUCTIONS:
1. Carefully evaluate both solutions for mathematical correctness.
2. Determine which solution is of higher quality overall.

Respond with ONLY the single letter A or B. Do not include any explanation or other text.

ANSWER:"""


def parse_choice(text):
    m = re.search(r"\b([AB])\b", text.strip()[:40])
    return m.group(1) if m else None


def make_client():
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    cfg = Config(retries={"max_attempts": 8, "mode": "adaptive"},
                 read_timeout=120)
    return session.client("bedrock-runtime", config=cfg)


def sample_once(client, model_id, r, rng):
    """One stochastic selection with randomized A/B order.
    Returns 'A'/'B' in the ARCHIVED orientation, or None."""
    flip = rng.random() < 0.5
    sa, sb = (r["solution_b"], r["solution_a"]) if flip else (r["solution_a"], r["solution_b"])
    user = USER_TEMPLATE.format(problem=r.get("problem_text", ""),
                                solution_a=sa, solution_b=sb)
    for _ in range(MAX_RETRIES_PER_SAMPLE):
        try:
            resp = client.converse(
                modelId=model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": 8, "temperature": 0.7, "topP": 0.9},
            )
            text = resp["output"]["message"]["content"][0]["text"]
        except Exception:
            continue
        c = parse_choice(text)
        if c in ("A", "B"):
            if flip:
                c = "B" if c == "A" else "A"
            return c
    return None


def run_pair(client, model_id, r, seed):
    rng = random.Random(seed)
    sels = [sample_once(client, model_id, r, rng) for _ in range(K)]
    valid = [s for s in sels if s]
    if valid:
        n_a = valid.count("A")
        n_b = len(valid) - n_a
        if n_a == n_b:
            maj, conf = r.get("self_evaluation", {}).get("selected_solution"), 0.5
        else:
            maj = "A" if n_a > n_b else "B"
            conf = max(n_a, n_b) / len(valid)
    else:
        maj, conf = None, None
    return {
        "problem_id": r.get("problem_id"),
        "category": r.get("category"),
        "correct_answer": r.get("correct_answer"),
        "judge_selected": (r.get("judge_evaluation") or {}).get("selected_solution"),
        "orig_selected": (r.get("self_evaluation") or {}).get("selected_solution"),
        "orig_confidence": (r.get("self_evaluation") or {}).get("confidence"),
        "samples": sels,
        "n_valid": len(valid),
        "cons_selected": maj,
        "cons_confidence": conf,
    }


def main():
    model_id, pairs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    records = json.load(open(pairs_path))
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["problem_id"])
                except Exception:
                    pass
    todo = [r for r in records
            if r.get("problem_id") not in done
            and r.get("solution_a") and r.get("solution_b")]
    print(f"{model_id}: {len(records)} pairs, {len(done)} done, {len(todo)} to run "
          f"x K={K}", flush=True)

    client = make_client()
    lock = threading.Lock()
    fout = open(out_path, "a")
    n_done = 0

    def work(idx_r):
        idx, r = idx_r
        return run_pair(client, model_id, r, seed=20260724 + idx)

    with ThreadPoolExecutor(max_workers=PARALLEL_PAIRS) as ex:
        futures = [ex.submit(work, (i, r)) for i, r in enumerate(todo)]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  {n_done}/{len(todo)}", flush=True)
    fout.close()
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
