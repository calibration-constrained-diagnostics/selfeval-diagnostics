import os
import json
import logging
import time
import numpy as np

def setup_logging(output_dir):
    """Configure comprehensive logging for experiment tracking"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(output_dir, f"pvc_experiment_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    return logger

def convert_to_json_serializable(obj):
    """
    Convert all NumPy types to JSON serializable Python native types
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj

def load_math_problems(file_path):
    """
    Load and categorize math problems from JSONL file or Hugging Face dataset
    
    Args:
        file_path: Path to JSONL file with math problems or "math-500" for HF dataset
        
    Returns:
        Dictionary of problems grouped by category
    """
    logger = logging.getLogger(__name__)
    
    if file_path.lower() == "math-500":
        # Load from Hugging Face math-500 dataset
        try:
            from datasets import load_dataset
            logger.info("Loading math-500 dataset from Hugging Face")
            dataset = load_dataset("HuggingFaceH4/MATH-500")
            
            # Convert to our format
            problems_by_category = {}
            
            for split in ["train", "test"]:
                if split in dataset:
                    for item in dataset[split]:
                        # Map HF dataset fields to our format
                        problem = {
                            "id": item.get("unique_id", ""),
                            "problem": item.get("problem", ""),
                            "answer": item.get("answer", ""),
                            "category": item.get("subject", "Uncategorized"),
                            "difficulty": item.get("level", 1)
                        }
                        
                        category = problem["category"]
                        if category not in problems_by_category:
                            problems_by_category[category] = []
                        problems_by_category[category].append(problem)
            
            logger.info(f"Loaded {sum(len(probs) for probs in problems_by_category.values())} problems from math-500 dataset")
            logger.info(f"Categories: {list(problems_by_category.keys())}")
            
            return problems_by_category
            
        except ImportError:
            logger.error("Failed to import 'datasets' library. Please install with: pip install datasets")
            raise
        
        except Exception as e:
            logger.error(f"Error loading math-500 dataset: {str(e)}")
            raise
            
    else:
        # Load from JSONL file
        try:
            with open(file_path, 'r') as f:
                problems = [json.loads(line) for line in f]

            problems_by_category = {}
            for problem in problems:
                assert "problem" in problem
                assert "answer" in problem
                assert "category" in problem
                assert "difficulty" in problem
                category = problem.get('category', 'uncategorized')
                if category not in problems_by_category:
                    problems_by_category[category] = []
                problems_by_category[category].append(problem)
            
            logger.info(f"Loaded {sum(len(probs) for probs in problems_by_category.values())} problems from {file_path}")
            logger.info(f"Categories: {list(problems_by_category.keys())}")

            return problems_by_category
            
        except Exception as e:
            logger.error(f"Error loading problems from {file_path}: {str(e)}")
            raise