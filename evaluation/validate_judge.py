"""Validation framework for LLM-as-a-Judge evaluation.

This module provides validation functionality for the LLM-as-a-Judge component
used in MEQ-Bench. It allows comparison of LLM scores with human ratings to
assess the reliability and validity of automated evaluation.

The validation process includes correlation analysis, statistical significance
testing, and detailed performance metrics to ensure the LLM judge produces
reliable and consistent scores.

Example:
    ```python
    from validate_judge import validate_llm_judge
    from src.evaluator import MEQBenchEvaluator
    
    # Prepare validation data
    predictions = [
        {'generated_explanation': 'Patient explanation...', 'human_rating': 4.2},
        {'generated_explanation': 'Another explanation...', 'human_rating': 3.8},
    ]
    
    # Initialize LLM judge
    llm_judge = MEQBenchEvaluator()
    
    # Run validation
    correlation, p_value = validate_llm_judge(predictions, llm_judge)
    print(f"Correlation: {correlation:.3f}, p-value: {p_value:.3f}")
    ```
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.stats import spearmanr, pearsonr
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger('meq_bench.validation')


@dataclass
class ValidationResult:
    """Container for validation results.
    
    Attributes:
        correlation_coefficient: Spearman's rank correlation coefficient.
        p_value: Statistical significance p-value.
        n_samples: Number of samples used in validation.
        llm_scores: List of LLM-generated scores.
        human_ratings: List of human ratings.
        score_differences: List of differences between LLM and human scores.
        mean_absolute_error: Mean absolute error between LLM and human scores.
        correlation_type: Type of correlation used ('spearman' or 'pearson').
    """
    correlation_coefficient: float
    p_value: float
    n_samples: int
    llm_scores: List[float]
    human_ratings: List[float]
    score_differences: List[float]
    mean_absolute_error: float
    correlation_type: str


def validate_llm_judge(
    predictions: List[Dict[str, Any]], 
    llm_judge: Any,
    medical_content: Optional[str] = None,
    audience: str = 'patient',
    correlation_type: str = 'spearman'
) -> Tuple[float, float]:
    """Validate LLM-as-a-Judge by comparing with human ratings.
    
    This function evaluates the reliability of an LLM judge by comparing its
    scores with human ratings using correlation analysis. It provides a 
    quantitative measure of how well the LLM judge aligns with human judgment.
    
    Args:
        predictions: List of dictionaries containing validation data. Each dictionary
            should have:
            - 'generated_explanation': The explanation text to be scored
            - 'human_rating': The human-provided rating (float, typically 1-5 scale)
            Optional keys:
            - 'medical_content': Original medical content (if not provided globally)
            - 'audience': Target audience (if different from global setting)
            
        llm_judge: Instance of the LLM judge class (e.g., MEQBenchEvaluator).
            Must have a method to score explanations.
            
        medical_content: Original medical content for context. If None, each
            prediction should include its own medical_content.
            
        audience: Target audience for scoring ('physician', 'nurse', 'patient', 'caregiver').
            Defaults to 'patient'.
            
        correlation_type: Type of correlation to compute ('spearman' or 'pearson').
            Spearman is recommended for ordinal ratings.
            
    Returns:
        Tuple containing:
        - correlation_coefficient: Correlation between LLM and human scores
        - p_value: Statistical significance of the correlation
        
    Raises:
        ValueError: If predictions list is empty, contains invalid data, or if
            required fields are missing.
        AttributeError: If llm_judge doesn't have required scoring methods.
        
    Example:
        ```python
        # Example validation data
        predictions = [
            {
                'generated_explanation': 'High blood pressure means your heart works harder.',
                'human_rating': 4.2
            },
            {
                'generated_explanation': 'Hypertension is cardiovascular condition...',
                'human_rating': 3.1
            }
        ]
        
        # Run validation
        correlation, p_value = validate_llm_judge(
            predictions, 
            llm_judge,
            medical_content='Hypertension explanation',
            audience='patient'
        )
        ```
    """
    # Input validation
    if not predictions:
        raise ValueError("Predictions list cannot be empty")
    
    if not hasattr(llm_judge, 'evaluate_explanation') and not hasattr(llm_judge, 'evaluate_all_audiences'):
        raise AttributeError("LLM judge must have evaluation methods")
    
    if correlation_type not in ['spearman', 'pearson']:
        raise ValueError("Correlation type must be 'spearman' or 'pearson'")
    
    logger.info(f"Starting LLM judge validation with {len(predictions)} samples")
    
    # Extract LLM scores and human ratings
    llm_scores = []
    human_ratings = []
    
    for i, prediction in enumerate(predictions):
        try:
            # Validate prediction structure
            if not isinstance(prediction, dict):
                logger.warning(f"Skipping prediction {i}: not a dictionary")
                continue
                
            if 'generated_explanation' not in prediction:
                logger.warning(f"Skipping prediction {i}: missing 'generated_explanation'")
                continue
                
            if 'human_rating' not in prediction:
                logger.warning(f"Skipping prediction {i}: missing 'human_rating'")
                continue
            
            explanation = prediction['generated_explanation']
            human_rating = prediction['human_rating']
            
            # Validate explanation text
            if not isinstance(explanation, str) or not explanation.strip():
                logger.warning(f"Skipping prediction {i}: invalid explanation text")
                continue
            
            # Validate human rating
            try:
                human_rating = float(human_rating)
            except (ValueError, TypeError):
                logger.warning(f"Skipping prediction {i}: invalid human rating")
                continue
            
            # Get medical content for this prediction
            current_medical_content = prediction.get('medical_content', medical_content)
            if not current_medical_content:
                logger.warning(f"Skipping prediction {i}: no medical content available")
                continue
            
            # Get audience for this prediction  
            current_audience = prediction.get('audience', audience)
            
            # Score the explanation using the LLM judge
            llm_score = _score_explanation(
                llm_judge, 
                current_medical_content, 
                explanation, 
                current_audience
            )
            
            if llm_score is None:
                logger.warning(f"Skipping prediction {i}: LLM scoring failed")
                continue
            
            # Store valid scores
            llm_scores.append(llm_score)
            human_ratings.append(human_rating)
            
            logger.debug(f"Prediction {i}: LLM={llm_score:.3f}, Human={human_rating:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing prediction {i}: {e}")
            continue
    
    # Check if we have enough valid data
    if len(llm_scores) < 2:
        raise ValueError(f"Need at least 2 valid predictions for correlation analysis, got {len(llm_scores)}")
    
    logger.info(f"Successfully processed {len(llm_scores)} predictions for validation")
    
    # Calculate correlation coefficient
    try:
        if correlation_type == 'spearman':
            correlation_coef, p_value = spearmanr(llm_scores, human_ratings)
        else:  # pearson
            correlation_coef, p_value = pearsonr(llm_scores, human_ratings)
            
        # Handle NaN results (can occur with constant values)
        if np.isnan(correlation_coef):
            logger.warning("Correlation coefficient is NaN - may indicate constant values")
            correlation_coef = 0.0
            p_value = 1.0
        
        logger.info(f"Validation complete: {correlation_type} correlation = {correlation_coef:.3f}, p-value = {p_value:.3f}")
        
        return correlation_coef, p_value
        
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        raise


def validate_llm_judge_detailed(
    predictions: List[Dict[str, Any]], 
    llm_judge: Any,
    medical_content: Optional[str] = None,
    audience: str = 'patient',
    correlation_type: str = 'spearman'
) -> ValidationResult:
    """Perform detailed validation of LLM-as-a-Judge with comprehensive metrics.
    
    This function provides a more comprehensive validation analysis including
    additional metrics like mean absolute error, score distributions, and
    detailed diagnostic information.
    
    Args:
        predictions: List of prediction dictionaries (same format as validate_llm_judge).
        llm_judge: Instance of the LLM judge class.
        medical_content: Original medical content for context.
        audience: Target audience for scoring.
        correlation_type: Type of correlation to compute.
        
    Returns:
        ValidationResult object containing comprehensive validation metrics.
        
    Example:
        ```python
        result = validate_llm_judge_detailed(predictions, llm_judge)
        print(f"Correlation: {result.correlation_coefficient:.3f}")
        print(f"MAE: {result.mean_absolute_error:.3f}")
        print(f"Sample size: {result.n_samples}")
        ```
    """
    # Get basic correlation results
    correlation_coef, p_value = validate_llm_judge(
        predictions, llm_judge, medical_content, audience, correlation_type
    )
    
    # Extract scores for detailed analysis
    llm_scores = []
    human_ratings = []
    
    for prediction in predictions:
        try:
            if not isinstance(prediction, dict):
                continue
                
            explanation = prediction.get('generated_explanation')
            human_rating = prediction.get('human_rating')
            
            if not explanation or human_rating is None:
                continue
            
            current_medical_content = prediction.get('medical_content', medical_content)
            current_audience = prediction.get('audience', audience)
            
            if not current_medical_content:
                continue
            
            llm_score = _score_explanation(
                llm_judge, current_medical_content, explanation, current_audience
            )
            
            if llm_score is not None:
                llm_scores.append(llm_score)
                human_ratings.append(float(human_rating))
                
        except Exception:
            continue
    
    # Calculate additional metrics
    score_differences = [abs(llm - human) for llm, human in zip(llm_scores, human_ratings)]
    mean_absolute_error = np.mean(score_differences) if score_differences else 0.0
    
    return ValidationResult(
        correlation_coefficient=correlation_coef,
        p_value=p_value,
        n_samples=len(llm_scores),
        llm_scores=llm_scores,
        human_ratings=human_ratings,
        score_differences=score_differences,
        mean_absolute_error=mean_absolute_error,
        correlation_type=correlation_type
    )


def _score_explanation(
    llm_judge: Any, 
    medical_content: str, 
    explanation: str, 
    audience: str
) -> Optional[float]:
    """Score an explanation using the LLM judge.
    
    This helper function abstracts the scoring process to handle different
    LLM judge interfaces and provides consistent error handling.
    
    Args:
        llm_judge: Instance of the LLM judge.
        medical_content: Original medical content.
        explanation: Generated explanation to score.
        audience: Target audience.
        
    Returns:
        Overall score as float, or None if scoring failed.
    """
    try:
        # Try different scoring methods based on available interface
        if hasattr(llm_judge, 'evaluate_explanation'):
            # Single explanation evaluation
            result = llm_judge.evaluate_explanation(medical_content, explanation, audience)
            
            # Extract overall score from result
            if hasattr(result, 'overall'):
                return float(result.overall)
            elif isinstance(result, dict) and 'overall' in result:
                return float(result['overall'])
            elif isinstance(result, (int, float)):
                return float(result)
            else:
                logger.warning("Cannot extract score from evaluation result")
                return None
                
        elif hasattr(llm_judge, 'evaluate_all_audiences'):
            # Multi-audience evaluation
            explanations = {audience: explanation}
            results = llm_judge.evaluate_all_audiences(medical_content, explanations)
            
            if audience in results:
                result = results[audience]
                if hasattr(result, 'overall'):
                    return float(result.overall)
                elif isinstance(result, dict) and 'overall' in result:
                    return float(result['overall'])
            
            logger.warning(f"No score found for audience '{audience}'")
            return None
            
        else:
            logger.error("LLM judge has no recognized scoring method")
            return None
            
    except Exception as e:
        logger.error(f"Error scoring explanation: {e}")
        return None


def save_validation_results(
    result: ValidationResult, 
    output_path: Union[str, Path]
) -> None:
    """Save detailed validation results to a JSON file.
    
    Args:
        result: ValidationResult object to save.
        output_path: Path where results should be saved.
        
    Example:
        ```python
        result = validate_llm_judge_detailed(predictions, llm_judge)
        save_validation_results(result, 'validation_results.json')
        ```
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert result to dictionary
    result_dict = {
        'correlation_coefficient': result.correlation_coefficient,
        'p_value': result.p_value,
        'n_samples': result.n_samples,
        'mean_absolute_error': result.mean_absolute_error,
        'correlation_type': result.correlation_type,
        'llm_scores': result.llm_scores,
        'human_ratings': result.human_ratings,
        'score_differences': result.score_differences,
        'summary_statistics': {
            'llm_scores': {
                'mean': float(np.mean(result.llm_scores)),
                'std': float(np.std(result.llm_scores)),
                'min': float(np.min(result.llm_scores)),
                'max': float(np.max(result.llm_scores))
            },
            'human_ratings': {
                'mean': float(np.mean(result.human_ratings)),
                'std': float(np.std(result.human_ratings)),
                'min': float(np.min(result.human_ratings)),
                'max': float(np.max(result.human_ratings))
            }
        }
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save validation results: {e}")
        raise


def load_validation_data(data_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load validation data from a JSON file.
    
    Args:
        data_path: Path to the validation data file.
        
    Returns:
        List of prediction dictionaries.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        
    Example:
        ```python
        predictions = load_validation_data('validation_data.json')
        result = validate_llm_judge(predictions, llm_judge)
        ```
    """
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Validation data file not found: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("Validation data must be a list of predictions")
            
        logger.info(f"Loaded {len(data)} validation predictions from {data_file}")
        return data
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in validation data file: {e}",
            e.doc, e.pos
        )