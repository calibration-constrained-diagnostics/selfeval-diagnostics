"""selfeval: diagnostic evaluation package for LLM self-evaluation reliability.

This package accompanies the paper *Evaluating LLM Self-Evaluation: A
Calibration-Constrained Benchmark and Diagnostic Protocol*. It exposes the
model, judge, evaluation, and experiment utilities used to produce the
released outputs, and is re-usable on new models through the black-box
three-stage protocol.
"""

from .models import HFModel, BedrockModel, BEDROCK_MODEL_IDS
from .judges import JudgeEnsemble
from .evaluation import (
    generate_quality_solution_1st,
    generate_quality_solution_2nd,
    evaluate_solutions,
    test_problem_pvc,
)
from .utils import setup_logging, load_math_problems, convert_to_json_serializable
from .experiment import run_pvc_experiment

__all__ = [
    "HFModel",
    "BedrockModel",
    "BEDROCK_MODEL_IDS",
    "JudgeEnsemble",
    "generate_quality_solution_1st",
    "generate_quality_solution_2nd",
    "evaluate_solutions",
    "test_problem_pvc",
    "setup_logging",
    "load_math_problems",
    "convert_to_json_serializable",
    "run_pvc_experiment",
]