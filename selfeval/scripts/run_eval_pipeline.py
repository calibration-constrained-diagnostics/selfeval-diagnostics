"""
Example Command (Experiment with Bedrock models as judges)
python main.py \
  --model Qwen/Qwen2-7B-Instruct \
  --judges c37_sonnet nova_premier deepseek_r1 \
  --problems math_benchmark.jsonl \
  --output results
"""

import os
import argparse
import random
import logging
import torch
import numpy as np

from models import HFModel, BedrockModel, BEDROCK_MODEL_IDS
from judges import JudgeEnsemble
from utils import setup_logging, load_math_problems
from experiment import run_pvc_experiment

def main():
    parser = argparse.ArgumentParser(description="Probabilistic VC Dimension Measurement with Self-Evaluation")
    parser.add_argument("--problems", type=str, default="math_benchmark.jsonl", 
                      help="Path to problems JSONL file or 'math-500' for Hugging Face dataset")
    parser.add_argument("--output", type=str, default="results", help="Base output directory for results")
    parser.add_argument("--model", type=str, default="open-thoughts/OpenThinker2-7B", help="Model ID/name for solution generation")
    parser.add_argument("--judges", type=str, nargs="+", default=["c37_sonnet", "nova_premier", "deepseek_r1"], help="List of judge model IDs")
    parser.add_argument("--voting-method", type=str, choices=["majority", "weighted"], default="majority", 
                        help="Method for combining judge decisions")
    parser.add_argument("--categories", type=str, nargs="+", default=[], help="Specific categories to test")
    parser.add_argument("--min-problems", type=int, default=1, help="Minimum problems per category")
    parser.add_argument("--max-problems", type=int, default=500, help="Maximum problems per category")
    parser.add_argument("--seed", type=int, default=13579, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine dataset name
    if args.problems.lower() == "math-500":
        dataset_name = "math-500"
    else:
        dataset_name = os.path.basename(args.problems).split('.')[0]
    
    # Create model_dataset specific output directory
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    output_dir = os.path.join(args.output, f"{model_name}_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging with the dataset-model specific directory
    logger = setup_logging(output_dir)
    logger.info(f"Starting experiment: model={args.model}, judges={args.judges}, dataset={dataset_name}")
    
    # Load solution generation model
    if args.model in BEDROCK_MODEL_IDS:
        model = BedrockModel(args.model)
    else:
        model = HFModel(args.model)
    
    # Load judge models
    judge_models = []
    for judge_model_id in args.judges:
        if judge_model_id in BEDROCK_MODEL_IDS:
            judge_models.append(BedrockModel(judge_model_id))
        else:
            judge_models.append(HFModel(judge_model_id))
        logger.info(f"Loaded judge model: {judge_model_id}")
    
    # Create judge ensemble
    judge_ensemble = JudgeEnsemble(judge_models, args.voting_method)
    
    # Load math problems
    problems_by_category = load_math_problems(args.problems)
    logger.info(f"Loaded categories: {list(problems_by_category.keys())}")
    
    # Log reference answer availability
    for category, problems in problems_by_category.items():
        has_answers = sum(1 for p in problems if 'answer' in p)
        logger.info(f"Category '{category}': {has_answers}/{len(problems)} problems have reference answers")
    
    # Update args to use the dataset-model specific output directory
    args.output = output_dir
    
    # Run experiment and save detailed JSONL files
    run_pvc_experiment(model, judge_ensemble, problems_by_category, args)
    
    logger.info("Experiment completed - detailed JSONL files saved for post-processing")

if __name__ == "__main__":
    main()
