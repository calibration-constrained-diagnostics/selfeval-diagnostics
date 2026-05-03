import re
import random
import logging

def generate_quality_solution_1st(model, problem):
    """
    Generate a solution with expert-level reasoning
    
    Args:
        model: LLM model
        problem: Problem dictionary with problem and answer
        
    Returns:
        Solution to a mathematical problem with reasoning
    """
    
    system_prompt = """You are an expert mathematics educator with deep knowledge across all mathematical domains. 
Your task is to solve a math problem with exceptional clarity and accuracy."""

    user_prompt = f"""Please solve the following mathematical problem step by step, ensuring complete accuracy:

PROBLEM:
{problem['problem']}

Provide a detailed solution that demonstrates expert-level understanding. Include:
1. Clear explanation of your approach
2. Each step fully worked out
3. Proper mathematical notation
4. Verification of your answer

Make sure your final answer is clearly indicated and correct.
"""

    prompt = model.format_chat_prompt(system_prompt, user_prompt)
    return model.generate(prompt)

def generate_quality_solution_2nd(model, problem):
    """
    Generate a solution with thorough explanation and verification
    
    Args:
        model: LLM model
        problem: Problem dictionary with problem and answer
        
    Returns:
        Solution to a mathematical problem with detailed verifications
    """
    system_prompt = """You are a brilliant mathematics educator known for your creative problem-solving approaches.
Your task is to solve a math problem using your new methods."""

    user_prompt = f"""Please solve the following mathematical problem:

PROBLEM:
{problem['problem']}

Provide an extremely detailed solution that demonstrates understanding. Include:
1. A clear explanation of your approach using an alternative and creative solution path
2. Each step fully worked out with mathematical reasoning
3. Proper mathematical notation and formalism
4. If applicable, connect this problem to broader mathematical concepts or real-world applications

Make your solution as thorough and educational as possible, showing multiple ways to think about and verify the solution.
Focus on approaches that demonstrate mathematical insight and elegance.
"""

    prompt = model.format_chat_prompt(system_prompt, user_prompt)
    return model.generate(prompt)

def evaluate_solutions(model, problem, solution_a, solution_b, reference_answer=None):
    """
    Have the model self-evaluate and choose the better solution
    
    Args:
        model: LLM model
        problem: Problem dictionary
        solution_a: First solution
        solution_b: Second solution
        
    Returns:
        Model's self-evaluation result with confidence
    """
    system_prompt = """You are an expert mathematics evaluator.
Your task is to carefully evaluate two different solutions to the same mathematical problem and determine which solution is superior in terms of correctness, clarity, and mathematical rigor."""

    # Include reference answer in the prompt if available
    reference_part = f"\nREFERENCE ANSWER: \n{reference_answer}" if reference_answer else ""
    
    user_prompt = f"""Please analyze the following mathematical problem and two proposed solutions:

PROBLEM:
{problem['problem']}
{reference_part}

SOLUTION A:
{solution_a}

SOLUTION B:
{solution_b}

INSTRUCTIONS:
1. Carefully evaluate both solutions for mathematical correctness.
2. Identify any errors or flaws in either solution.
3. Determine which solution is of higher quality overall.
4. Provide your reasoning process.
5. Assign a confidence score (0-100) reflecting your certainty in this assessment.

Please format your response using these exact tags:
<ANALYSIS>
Your detailed analysis of both solutions
</ANALYSIS>

<WINNER>A or B only - Do not include any explanations</WINNER>
<SCORE>Your confidence score (0-100)</SCORE>
"""

    max_attempts = 5  # Maximum retry attempts
    
    for attempt in range(max_attempts):
        prompt = model.format_chat_prompt(system_prompt, user_prompt)
        response = model.generate(prompt)
        
        # Parse structured response using regex
        better_solution_match = re.search(r'<WINNER>\s*([AB])\s*</WINNER>', response, re.IGNORECASE)
        confidence_match = re.search(r'<SCORE>\s*(\d+)\s*</SCORE>', response, re.IGNORECASE)
        
        # Try alternative formats if standard format not found
        if not better_solution_match:
            better_solution_match = re.search(r'Winner:\s+([AB])\s+is', response, re.IGNORECASE)
            
        if not confidence_match:
            confidence_match = re.search(r'(\d+)%\s*confiden', response, re.IGNORECASE)
        
        better_solution = better_solution_match.group(1) if better_solution_match else None
        confidence = int(confidence_match.group(1))/100 if confidence_match and confidence_match.group(1).isdigit() else None
        
        # If we successfully parsed both values, return the result
        if better_solution is not None and confidence is not None:
            return {
                "better_solution": better_solution,
                "confidence": confidence,
                "full_response": response
            }
    
    # If all attempts failed, return a default response
    return {
        "better_solution": "A" if random.random() < 0.5 else "B",  # Random choice as fallback
        "confidence": 0.5,  # Neutral confidence
        "full_response": response,
        "parsing_failed": True  # Flag to indicate parsing failure
    }

def test_problem_pvc(model, judge_ensemble, problem, logger):
    """
    Test a single problem for PVC capability
    
    Args:
        model: Solution generation model
        judge_ensemble: Ensemble of judge models
        problem: Problem dictionary
        logger: Logger instance
        
    Returns:
        Problem test result
    """
    try:
        # Extract reference answer if available
        reference_answer = problem.get('answer', None)
        
        # Generate high and low quality solutions
        logger.info(f"problem: {problem}")
        high_quality = generate_quality_solution_1st(model, problem)
        logger.info(f"quality 1st: {high_quality}")
        low_quality = generate_quality_solution_2nd(model, problem)
        logger.info(f"quality 2nd: {low_quality}")
        
        logger.debug(f"Generated solutions for problem {problem.get('id', 'unknown')}")
        
        # Randomize solution order to prevent positional bias
        if random.random() > 0.5:
            solution_a, solution_b = low_quality, high_quality
            correct_answer = "B"
            logger.info("quality solutions have been flipped.")
        else:
            solution_a, solution_b = high_quality, low_quality
            correct_answer = "A"
        
        # Model self-evaluates which solution is better
        self_evaluation = evaluate_solutions(model, problem, solution_a, solution_b)
        logger.info(f"self_evaluation: {self_evaluation}")
        
        # Have the judge ensemble evaluate solutions (ground truth)
        judge_evaluation = judge_ensemble.run_ensemble_evaluations(problem, solution_a, solution_b, reference_answer)
        logger.info(f"judge_evaluation: {judge_evaluation}")
        
        # Make sure we have valid self_evaluation and judge_evaluation data
        if not isinstance(self_evaluation, dict) or "better_solution" not in self_evaluation:
            logger.warning(f"Invalid self_evaluation format: {self_evaluation}")
            self_evaluation = {"better_solution": None, "confidence": 0.5}
            
        if not isinstance(judge_evaluation, dict) or "better_solution" not in judge_evaluation:
            logger.warning(f"Invalid judge_evaluation format: {judge_evaluation}")
            judge_evaluation = {"better_solution": None, "confidence": 0.5}
        
        # Check if model's self-evaluation matches judge's evaluation
        self_eval_correct = self_evaluation["better_solution"] == judge_evaluation["better_solution"] 
        self_eval_correct &= self_evaluation["better_solution"] != None
        
        result = {
            "problem_id": problem.get('id', 'unknown'),
            "problem_text": problem.get('problem', ''),
            "category": problem.get('category', 'uncategorized'),
            "subcategory": problem.get('subcategory', 'general'),
            "solution_a": solution_a,
            "solution_b": solution_b,
            "self_evaluation": {
                "selected_solution": self_evaluation["better_solution"],
                "confidence": self_evaluation["confidence"],
                "full_response": self_evaluation.get("full_response", "")
            },
            "judge_evaluation": {
                "selected_solution": judge_evaluation["better_solution"],
                "confidence": judge_evaluation["confidence"],
                "judgments": judge_evaluation.get("judgments", [])
            },
            "correct_answer": correct_answer,
            "self_eval_correct": self_eval_correct,
            "had_reference_answer": reference_answer is not None,
            "reference_answer": reference_answer
        }
        
        logger.info(f"Problem {problem.get('id', 'unknown')}: Self-eval correct: {self_eval_correct}, "
                   f"Confidence: {self_evaluation['confidence']:.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing problem {problem.get('id', 'unknown')}: {str(e)}")
        # Return a default structure on error
        return {
            "problem_id": problem.get('id', 'unknown'),
            "error": str(e),
            "self_evaluation": {"selected_solution": None, "confidence": 0.0},
            "judge_evaluation": {"selected_solution": None, "confidence": 0.0},
            "correct_answer": None,
            "self_eval_correct": False,
            "had_reference_answer": False
        }