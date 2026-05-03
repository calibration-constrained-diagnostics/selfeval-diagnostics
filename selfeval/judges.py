import logging
import random
import numpy as np
from typing import List
from models import LLMModel
from evaluation import evaluate_solutions

class JudgeEnsemble:
    """Ensemble of multiple judges for more robust evaluation"""
    def __init__(self, judges: List[LLMModel], voting_method: str = "majority"):
        """
        Initialize judge ensemble
        
        Args:
            judges: List of LLM models to use as judges
            voting_method: Method for combining judge decisions (majority, weighted)
        """
        self.judges = judges
        self.voting_method = voting_method
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized judge ensemble with {len(judges)} judges using {voting_method} voting")
        
        for i, judge in enumerate(judges):
            judge_name = getattr(judge, "model_name", f"Judge-{i+1}")
            self.logger.info(f"Judge {i+1}: {judge_name}")
    
    def run_ensemble_evaluations(self, problem, solution_a, solution_b, reference_answer=None):
        """
        Evaluate solutions using multiple judges
        
        Args:
            problem: Problem statement
            solution_a: First solution
            solution_b: Second solution
            reference_answer: Reference answer if available
            
        Returns:
            Aggregated judgment result
        """
        judgments = []
        
        # Get judgments from all judges
        for i, judge in enumerate(self.judges):
            judge_name = getattr(judge, "model_name", f"Judge-{i+1}")
            self.logger.info(f"Getting judgment from {judge_name}")
            
            try:
                judgment = evaluate_solutions(judge, problem, solution_a, solution_b, reference_answer)
                judgments.append(judgment)
                self.logger.info(f"Judge {i+1} ({judge_name}) selected {judgment['better_solution']} "
                               f"with confidence {judgment['confidence']:.2f}")
            except Exception as e:
                self.logger.error(f"Error with judge {i+1} ({judge_name}): {str(e)}")
        
        # Combine judgments based on voting method
        if not judgments:
            self.logger.error("No valid judgments received")
            return {"better_solution": None, "confidence": 0.0, "judgments": judgments}
        
        return self._combine_judgments(judgments)
        
    def _combine_judgments(self, judgments):
        """
        Combine multiple judgments into a final decision
        
        Args:
            judgments: List of individual judge judgments
            
        Returns:
            Combined judgment
        """
        if self.voting_method == "majority":
            return self._majority_voting(judgments)
        elif self.voting_method == "weighted":
            return self._weighted_voting(judgments)
        else:
            self.logger.warning(f"Unknown voting method: {self.voting_method}, falling back to majority voting")
            return self._majority_voting(judgments)
    
    def _majority_voting(self, judgments):
        """Implement majority voting"""
        # Count votes for A and B
        votes_a = sum(1 for j in judgments if j["better_solution"] == "A")
        votes_b = sum(1 for j in judgments if j["better_solution"] == "B")
        
        # Determine winner
        if votes_a > votes_b:
            better_solution = "A"
            vote_ratio = votes_a / len(judgments)
        elif votes_b > votes_a:
            better_solution = "B"
            vote_ratio = votes_b / len(judgments)
        else:
            # In case of tie, use the average confidence to decide
            avg_conf_a = np.mean([j["confidence"] for j in judgments if j["better_solution"] == "A"])
            avg_conf_b = np.mean([j["confidence"] for j in judgments if j["better_solution"] == "B"])
            
            if avg_conf_a >= avg_conf_b:
                better_solution = "A"
                vote_ratio = 0.5
            else:
                better_solution = "B"
                vote_ratio = 0.5
        
        # Calculate the average confidence of judges who voted for the winning solution
        winner_confidences = [j["confidence"] if j["better_solution"] == better_solution else 1-j["confidence"] for j in judgments]
        avg_confidence = np.mean(winner_confidences) if winner_confidences else 0.5
        
        return {
            "better_solution": better_solution,
            "confidence": avg_confidence,
            "vote_ratio": vote_ratio,
            "judgments": judgments
        }
    
    def _weighted_voting(self, judgments):
        """Implement confidence-weighted voting"""
        # Calculate weighted votes
        weight_a = sum(j["confidence"] for j in judgments if j["better_solution"] == "A")
        weight_b = sum(j["confidence"] for j in judgments if j["better_solution"] == "B")
        
        # Determine winner
        if weight_a > weight_b:
            better_solution = "A"
            weight_ratio = weight_a / (weight_a + weight_b)
        elif weight_b > weight_a:
            better_solution = "B"
            weight_ratio = weight_b / (weight_a + weight_b)
        else:
            # In case of tie, choose randomly
            better_solution = random.choice(["A", "B"])
            weight_ratio = 0.5
        
        return {
            "better_solution": better_solution,
            "confidence": weight_ratio,
            "vote_ratio": weight_ratio,
            "judgments": judgments
        }