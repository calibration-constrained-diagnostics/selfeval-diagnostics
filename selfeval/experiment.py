import os
import random
import logging
import json
from tqdm import tqdm
from utils import convert_to_json_serializable
from evaluation import test_problem_pvc

def run_pvc_experiment(model, judge_ensemble, problems_by_category, args):
    """
    Run PVC dimension experiment across categories
    
    Args:
        model: Solution generation model
        judge_ensemble: Ensemble of judge models
        problems_by_category: Dictionary of problems by category
        args: Command-line arguments
        
    Returns:
        Experiment results
    """
    logger = logging.getLogger(__name__)
    
    # Setup categories
    categories = args.categories or list(problems_by_category.keys())
    
    # Store experiment results
    raw_results_by_category = {}
    
    for category in categories:
        if category not in problems_by_category:
            logger.warning(f"Category '{category}' not found.")
            continue
        
        problems = problems_by_category[category]
        
        # Apply problem count limits
        if args.max_problems and len(problems) > args.max_problems:
            problems = random.sample(problems, args.max_problems)
            logger.info(f"Sampled {args.max_problems} problems from category '{category}'")
        
        if len(problems) < args.min_problems:
            logger.warning(f"Category '{category}' has too few problems: {len(problems)} < {args.min_problems}")
            continue
            
        logger.info(f"Testing category '{category}' with {len(problems)} problems")
        logger.info(f"Reference answers available: {sum(1 for p in problems if 'answer' in p)}/{len(problems)}")
        
        # Run the actual tests
        results = []
        
        for i, problem in enumerate(tqdm(problems, desc=f"Testing {category} problems")):
            result = test_problem_pvc(model, judge_ensemble, problem, logger)
            if result:
                results.append(result)
        
        if not results:
            logger.warning(f"No valid results for category {category}")
            continue
            
        # Store the raw results including detailed solution texts
        raw_results_by_category[category] = results
    
    # Save detailed raw results
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    judge_name = "_".join([j.split('/')[-1] if '/' in j else j for j in args.judges])
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save each category's detailed results individually
        for category, results in raw_results_by_category.items():
            # Create a sanitized filename-friendly category name
            safe_category = category.replace(' ', '_').replace('/', '_').lower()
            
            # Save complete detailed results with solutions for this category
            category_path = os.path.join(output_dir, f'detailed_{safe_category}_{model_name}_with_{judge_name}.jsonl')
            with open(category_path, 'w') as f:
                for result in results:
                    # Ensure result is JSON serializable
                    serializable_result = convert_to_json_serializable(result)
                    f.write(json.dumps(serializable_result) + '\n')
            
            logger.info(f"Saved detailed results for category '{category}': {category_path}")
        
        # Save all detailed results in a single file
        all_results = []
        for category, results in raw_results_by_category.items():
            for result in results:
                # Add category to each result for identification
                result_copy = dict(result)  # Make a copy to avoid modifying original
                if "category" not in result_copy:
                    result_copy["category"] = category
                all_results.append(result_copy)
        
        # Save all results to JSONL
        all_path = os.path.join(output_dir, f'all_detailed_results_{model_name}_with_{judge_name}.jsonl')
        with open(all_path, 'w') as f:
            for result in all_results:
                serializable_result = convert_to_json_serializable(result)
                f.write(json.dumps(serializable_result) + '\n')
        
        logger.info(f"Saved all detailed results: {all_path}")
        
    except Exception as e:
        logger.error(f"Error saving detailed results: {str(e)}")
    
    return raw_results_by_category