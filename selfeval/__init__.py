"""selfeval: diagnostic evaluation package for LLM self-evaluation.

This package accompanies the paper *Calibration-Constrained Diagnostics
for Evaluating LLM Self-Evaluation*. It exposes the model, judge,
evaluation, and experiment utilities used to produce the released outputs,
and is re-usable on new models through the black-box three-stage protocol.

The top-level imports are kept minimal so that L1 reproduction
(``reproduce/reproduce_main_tables.py``) does not require the full
model/judge stack. Submodules such as ``selfeval.models`` (which depends
on torch / transformers) are imported lazily.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRS = {
    "HFModel": ("selfeval.models", "HFModel"),
    "BedrockModel": ("selfeval.models", "BedrockModel"),
    "BEDROCK_MODEL_IDS": ("selfeval.models", "BEDROCK_MODEL_IDS"),
    "JudgeEnsemble": ("selfeval.judges", "JudgeEnsemble"),
    "generate_quality_solution_1st": ("selfeval.evaluation", "generate_quality_solution_1st"),
    "generate_quality_solution_2nd": ("selfeval.evaluation", "generate_quality_solution_2nd"),
    "evaluate_solutions": ("selfeval.evaluation", "evaluate_solutions"),
    "test_problem_pvc": ("selfeval.evaluation", "test_problem_pvc"),
    "setup_logging": ("selfeval.utils", "setup_logging"),
    "load_math_problems": ("selfeval.utils", "load_math_problems"),
    "convert_to_json_serializable": ("selfeval.utils", "convert_to_json_serializable"),
    "run_pvc_experiment": ("selfeval.experiment", "run_pvc_experiment"),
}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_path, attr = _LAZY_ATTRS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'selfeval' has no attribute {name!r}")
