#!/usr/bin/env python3
"""
Consistency-based confidence on GPU via vLLM (NeurIPS 2026 rebuttal, odoV W2/Q2).

Design:
  - Archived candidate pairs fixed verbatim.
  - K=10 stochastic selections per pair (temperature 0.7, top-p 0.9),
    selection-only prompt (no numerical confidence requested).
  - A/B presentation order balanced: 5 samples in archived order, 5 flipped,
    all mapped back to the archived orientation (controls position bias).
  - Per-sample selections stored so analysis can compute BOTH
      (a) fixed-decision variant: original single-shot decision kept,
          confidence = fraction of K samples agreeing with it;
      (b) modal variant: majority decision + modal frequency.

Usage:
    python consistency_vllm.py <model_key> <pairs_json> <out_jsonl>
"""

import json
import re
import sys

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_IDS = {
    "JiuZhang3.0-7B": "ToheartZhang/JiuZhang3.0-7B",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "s1.1-7B": "simplescaling/s1.1-7B",
}

K_PER_ORDER = 5  # x2 orders = K=10 total

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


def build_prompt(tok, r, flipped):
    sa, sb = ((r["solution_b"], r["solution_a"]) if flipped
              else (r["solution_a"], r["solution_b"]))
    user = USER_TEMPLATE.format(problem=r.get("problem_text", ""),
                                solution_a=sa, solution_b=sb)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    except Exception:
        prompt = SYSTEM_PROMPT + "\n\n" + user
    ids = tok(prompt, truncation=True, max_length=7000)["input_ids"]
    return tok.decode(ids, skip_special_tokens=False)


def main():
    model_key, pairs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    model_id = MODEL_IDS[model_key]

    records = [r for r in json.load(open(pairs_path))
               if r.get("solution_a") and r.get("solution_b")]
    print(f"{model_key}: {len(records)} pairs x K={2*K_PER_ORDER}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=8192)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8,
                        n=K_PER_ORDER, seed=20260724)

    # two prompt variants per pair: archived order + flipped order
    prompts, meta = [], []
    for r in records:
        for flipped in (False, True):
            prompts.append(build_prompt(tok, r, flipped))
            meta.append((r, flipped))

    outputs = llm.generate(prompts, sp)

    per_pair = {}
    for (r, flipped), out in zip(meta, outputs):
        pid = r.get("problem_id")
        sels = []
        for o in out.outputs:
            c = parse_choice(o.text)
            if c in ("A", "B") and flipped:
                c = "B" if c == "A" else "A"
            sels.append(c)
        per_pair.setdefault(pid, {"r": r, "samples": []})["samples"].extend(sels)

    with open(out_path, "w") as f:
        for pid, d in per_pair.items():
            r = d["r"]
            valid = [s for s in d["samples"] if s in ("A", "B")]
            orig_sel = (r.get("self_evaluation") or {}).get("selected_solution")
            if valid:
                n_a = valid.count("A")
                n_b = len(valid) - n_a
                if n_a == n_b:
                    maj, conf = orig_sel, 0.5
                else:
                    maj = "A" if n_a > n_b else "B"
                    conf = max(n_a, n_b) / len(valid)
                fixed_conf = (valid.count(orig_sel) / len(valid)
                              if orig_sel in ("A", "B") else None)
            else:
                maj, conf, fixed_conf = None, None, None
            f.write(json.dumps({
                "problem_id": pid,
                "category": r.get("category"),
                "correct_answer": r.get("correct_answer"),
                "judge_selected": (r.get("judge_evaluation") or {}).get("selected_solution"),
                "orig_selected": orig_sel,
                "orig_confidence": (r.get("self_evaluation") or {}).get("confidence"),
                "samples": d["samples"],
                "n_valid": len(valid),
                "cons_selected": maj,
                "cons_confidence": conf,
                "fixed_confidence": fixed_conf,
            }) + "\n")
    print(f"wrote {out_path} ({len(per_pair)} pairs)", flush=True)


if __name__ == "__main__":
    main()
