"""
Main MEQ-Bench benchmark implementation
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from .config import config
from .prompt_templates import AudienceAdaptivePrompt
from .evaluator import MEQBenchEvaluator, EvaluationScore

logger = logging.getLogger('meq_bench.benchmark')


@dataclass
class MEQBenchItem:
    """Single benchmark item"""
    id: str
    medical_content: str
    complexity_level: str  # "basic", "intermediate", "advanced"
    source_dataset: str
    reference_explanations: Optional[Dict[str, str]] = None


class MEQBench:
    """Main benchmark class for MEQ-Bench"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize MEQ-Bench
        
        Args:
            data_path: Path to benchmark data directory
        """
        self.data_path = self._resolve_data_path(data_path)
        # Initialize evaluator with graceful fallback for missing dependencies
        try:
            self.evaluator = MEQBenchEvaluator()
        except Exception as e:
            logger.warning(f"Could not initialize full evaluator: {e}")
            logger.info("Some evaluation features may be limited due to missing dependencies or configuration")
        self.prompt_template = AudienceAdaptivePrompt()
        self.benchmark_items: List[MEQBenchItem] = []
        
        # Load benchmark data if available
        self._load_benchmark_data()
    
    def _resolve_data_path(self, data_path: Optional[str] = None) -> Path:
        """Resolve data directory path with fallback options.
        
        Args:
            data_path: Optional custom data path
            
        Returns:
            Resolved Path object for data directory
        """
        if data_path:
            # Use provided path (can be relative or absolute)
            resolved_path = Path(data_path).resolve()
        else:
            # Try multiple fallback locations
            possible_paths = [
                # Relative to package directory
                Path(__file__).parent.parent / "data",
                # Current working directory
                Path.cwd() / "data",
                # Environment variable if set
                Path(os.environ.get('MEQ_BENCH_DATA_PATH', '')) if os.environ.get('MEQ_BENCH_DATA_PATH') else None,
                # Config-based path
                Path(config.get_data_path()) if hasattr(config, 'get_data_path') else None
            ]
            
            # Find first existing path or use first option as default
            resolved_path = None
            for path in possible_paths:
                if path and path.exists():
                    resolved_path = path.resolve()
                    break
            
            if not resolved_path:
                # Default to package relative path
                resolved_path = (Path(__file__).parent.parent / "data").resolve()
        
        # Ensure directory exists
        resolved_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using data directory: {resolved_path}")
        return resolved_path
    
    def _load_benchmark_data(self):
        """Load benchmark data from JSON files with error handling."""
        try:
            benchmark_file = self.data_path / "benchmark_items.json"
            
            if not benchmark_file.exists():
                logger.warning(f"Benchmark data file not found: {benchmark_file}")
                logger.info("Use create_sample_dataset() to generate sample data")
                return
            
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError("Benchmark data must be a list of items")
                
            for i, item_data in enumerate(data):
                try:
                    # Validate required fields
                    required_fields = ['id', 'medical_content', 'complexity_level', 'source_dataset']
                    for field in required_fields:
                        if field not in item_data:
                            raise KeyError(f"Missing required field '{field}' in item {i}")
                    
                    item = MEQBenchItem(
                        id=item_data['id'],
                        medical_content=item_data['medical_content'],
                        complexity_level=item_data['complexity_level'],
                        source_dataset=item_data['source_dataset'],
                        reference_explanations=item_data.get('reference_explanations')
                    )
                    self.benchmark_items.append(item)
                except (KeyError, ValueError) as e:
                    logger.error(f"Error loading benchmark item {i}: {e}")
                    continue
            
            logger.info(f"Loaded {len(self.benchmark_items)} benchmark items from {benchmark_file}")
            
        except FileNotFoundError:
            logger.warning(f"Benchmark data directory not found: {self.data_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in benchmark data file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading benchmark data: {e}")
    
    def add_benchmark_item(self, item: MEQBenchItem):
        """Add a new benchmark item with validation."""
        if not isinstance(item, MEQBenchItem):
            raise TypeError("item must be an instance of MEQBenchItem")
        
        # Validate item data
        if not item.id or not isinstance(item.id, str):
            raise ValueError("item.id must be a non-empty string")
        
        if not item.medical_content or not isinstance(item.medical_content, str):
            raise ValueError("item.medical_content must be a non-empty string")
        
        if item.complexity_level not in ['basic', 'intermediate', 'advanced']:
            raise ValueError("item.complexity_level must be 'basic', 'intermediate', or 'advanced'")
        
        # Check for duplicate IDs
        if any(existing_item.id == item.id for existing_item in self.benchmark_items):
            raise ValueError(f"Item with ID '{item.id}' already exists")
        
        self.benchmark_items.append(item)
        logger.debug(f"Added benchmark item: {item.id}")
    
    def generate_explanations(self, medical_content: str, model_func: callable) -> Dict[str, str]:
        """
        Generate audience-adaptive explanations using a model
        
        Args:
            medical_content: Medical information to adapt
            model_func: Function that takes prompt and returns model response
            
        Returns:
            Dictionary mapping audience to explanation
            
        Raises:
            ValueError: If medical_content is empty or invalid
            TypeError: If model_func is not callable
        """
        # Input validation
        if not medical_content or not isinstance(medical_content, str):
            raise ValueError("medical_content must be a non-empty string")
        
        if medical_content.strip() == "":
            raise ValueError("medical_content cannot be empty or contain only whitespace")
        
        if len(medical_content.strip()) < 10:
            raise ValueError("medical_content must be at least 10 characters long")
        
        if not callable(model_func):
            raise TypeError("model_func must be a callable function")
        
        # Additional content validation
        if len(medical_content) > 10000:  # Reasonable upper limit
            logger.warning(f"Medical content is very long ({len(medical_content)} chars). Consider splitting.")
        
        # Sanitize content - remove excessive whitespace
        sanitized_content = ' '.join(medical_content.split())
        
        try:
            prompt = self.prompt_template.format_prompt(sanitized_content)
            logger.debug(f"Generated prompt with {len(prompt)} characters")
            
            response = model_func(prompt)
            
            # Validate model response
            if not response or not isinstance(response, str):
                raise ValueError("Model function returned empty or invalid response")
            
            if response.strip() == "":
                raise ValueError("Model function returned empty response")
            
            explanations = self.prompt_template.parse_response(response)
            
            # Validate parsed explanations
            if not explanations:
                raise ValueError("Failed to parse any explanations from model response")
            
            # Log successful generation
            logger.info(f"Generated explanations for {len(explanations)} audiences")
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            if isinstance(e, (ValueError, TypeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error during explanation generation: {e}") from e
    
    def evaluate_model(self, model_func: callable, max_items: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a model on the full benchmark
        
        Args:
            model_func: Function that takes prompt and returns model response
            max_items: Maximum number of items to evaluate (for testing)
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {
            'model_name': getattr(model_func, '__name__', 'unknown'),
            'total_items': len(self.benchmark_items[:max_items]) if max_items else len(self.benchmark_items),
            'audience_scores': {
                'physician': [],
                'nurse': [],
                'patient': [],
                'caregiver': []
            },
            'complexity_scores': {
                'basic': [],
                'intermediate': [],
                'advanced': []
            },
            'detailed_results': []
        }
        
        items_to_evaluate = self.benchmark_items[:max_items] if max_items else self.benchmark_items
        
        for item in items_to_evaluate:
            # Generate explanations
            explanations = self.generate_explanations(item.medical_content, model_func)
            
            # Evaluate explanations
            evaluation_results = self.evaluator.evaluate_all_audiences(
                item.medical_content, explanations
            )
            
            # Store detailed results
            item_result = {
                'item_id': item.id,
                'complexity_level': item.complexity_level,
                'source_dataset': item.source_dataset,
                'explanations': explanations,
                'scores': {
                    audience: {
                        'readability': score.readability,
                        'terminology': score.terminology,
                        'safety': score.safety,
                        'coverage': score.coverage,
                        'quality': score.quality,
                        'overall': score.overall
                    }
                    for audience, score in evaluation_results.items()
                }
            }
            results['detailed_results'].append(item_result)
            
            # Aggregate scores by audience
            for audience, score in evaluation_results.items():
                if audience in results['audience_scores']:
                    results['audience_scores'][audience].append(score.overall)
            
            # Aggregate scores by complexity
            avg_overall = sum(score.overall for score in evaluation_results.values()) / len(evaluation_results)
            results['complexity_scores'][item.complexity_level].append(avg_overall)
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_stats(results)
        
        return results
    
    def _calculate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results"""
        summary = {}
        
        # Audience-level statistics
        for audience, scores in results['audience_scores'].items():
            if scores:
                summary[f'{audience}_mean'] = sum(scores) / len(scores)
                summary[f'{audience}_std'] = (sum((x - summary[f'{audience}_mean']) ** 2 for x in scores) / len(scores)) ** 0.5
                summary[f'{audience}_min'] = min(scores)
                summary[f'{audience}_max'] = max(scores)
        
        # Complexity-level statistics
        for complexity, scores in results['complexity_scores'].items():
            if scores:
                summary[f'{complexity}_mean'] = sum(scores) / len(scores)
                summary[f'{complexity}_std'] = (sum((x - summary[f'{complexity}_mean']) ** 2 for x in scores) / len(scores)) ** 0.5
        
        # Overall statistics
        all_scores = []
        for audience_scores in results['audience_scores'].values():
            all_scores.extend(audience_scores)
        
        if all_scores:
            summary['overall_mean'] = sum(all_scores) / len(all_scores)
            summary['overall_std'] = (sum((x - summary['overall_mean']) ** 2 for x in all_scores) / len(all_scores)) ** 0.5
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file with proper path handling."""
        output_file = Path(output_path).resolve()
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
            raise
    
    def create_sample_dataset(self, output_path: Optional[str] = None) -> List[MEQBenchItem]:
        """
        Create a sample dataset for testing
        
        Args:
            output_path: Path to save sample dataset
            
        Returns:
            List of sample MEQBenchItem objects
        """
        sample_items = [
            MEQBenchItem(
                id="sample_001",
                medical_content="Hypertension, also known as high blood pressure, is a condition where the force of blood against artery walls is consistently too high. It can lead to heart disease, stroke, and kidney problems if left untreated. Treatment typically involves lifestyle changes and medication.",
                complexity_level="basic",
                source_dataset="sample"
            ),
            MEQBenchItem(
                id="sample_002",
                medical_content="Myocardial infarction occurs when blood flow to a part of the heart muscle is blocked, usually by a blood clot in a coronary artery. This results in damage or death of heart muscle cells. Immediate treatment with medications to dissolve clots or procedures to restore blood flow is critical.",
                complexity_level="intermediate",
                source_dataset="sample"
            ),
            MEQBenchItem(
                id="sample_003",
                medical_content="Diabetic ketoacidosis (DKA) is a serious complication of diabetes mellitus characterized by hyperglycemia, ketosis, and metabolic acidosis. It typically occurs in type 1 diabetes due to absolute insulin deficiency. Treatment involves IV fluids, insulin therapy, and electrolyte replacement while addressing underlying precipitating factors.",
                complexity_level="advanced",
                source_dataset="sample"
            )
        ]
        
        if output_path:
            # Save sample dataset with proper path handling
            output_file = Path(output_path).resolve()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            data_to_save = []
            for item in sample_items:
                data_to_save.append({
                    'id': item.id,
                    'medical_content': item.medical_content,
                    'complexity_level': item.complexity_level,
                    'source_dataset': item.source_dataset,
                    'reference_explanations': item.reference_explanations
                })
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                logger.info(f"Sample dataset saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save sample dataset to {output_file}: {e}")
                raise
        
        return sample_items
    
    def get_benchmark_stats(self) -> Dict[str, Any]:
        """Get statistics about the benchmark dataset"""
        if not self.benchmark_items:
            return {'total_items': 0, 'message': 'No benchmark items loaded'}
        
        stats = {
            'total_items': len(self.benchmark_items),
            'complexity_distribution': {},
            'source_distribution': {}
        }
        
        for item in self.benchmark_items:
            # Count by complexity
            if item.complexity_level not in stats['complexity_distribution']:
                stats['complexity_distribution'][item.complexity_level] = 0
            stats['complexity_distribution'][item.complexity_level] += 1
            
            # Count by source
            if item.source_dataset not in stats['source_distribution']:
                stats['source_distribution'][item.source_dataset] = 0
            stats['source_distribution'][item.source_dataset] += 1
        
        return stats
    
    def validate_benchmark(self) -> Dict[str, Any]:
        """Validate the benchmark dataset and return validation report.
        
        Returns:
            Dictionary containing validation results and any issues found
        """
        validation_report = {
            'valid': True,
            'total_items': len(self.benchmark_items),
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        if not self.benchmark_items:
            validation_report['valid'] = False
            validation_report['issues'].append("No benchmark items loaded")
            return validation_report
        
        # Check for duplicate IDs
        ids = [item.id for item in self.benchmark_items]
        duplicate_ids = set([id for id in ids if ids.count(id) > 1])
        if duplicate_ids:
            validation_report['valid'] = False
            validation_report['issues'].append(f"Duplicate item IDs found: {duplicate_ids}")
        
        # Validate complexity level distribution
        complexity_counts = {}
        for item in self.benchmark_items:
            complexity = item.complexity_level
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        validation_report['statistics']['complexity_distribution'] = complexity_counts
        
        # Check for balanced distribution
        if len(complexity_counts) < 3:
            validation_report['warnings'].append("Not all complexity levels represented")
        
        # Validate content length
        content_lengths = [len(item.medical_content) for item in self.benchmark_items]
        avg_length = sum(content_lengths) / len(content_lengths)
        validation_report['statistics']['average_content_length'] = avg_length
        
        if avg_length < 50:
            validation_report['warnings'].append("Average content length is quite short")
        elif avg_length > 2000:
            validation_report['warnings'].append("Average content length is quite long")
        
        # Check for empty or very short content
        short_content_items = [
            item.id for item in self.benchmark_items 
            if len(item.medical_content.strip()) < 20
        ]
        if short_content_items:
            validation_report['valid'] = False
            validation_report['issues'].append(f"Items with very short content: {short_content_items}")
        
        logger.info(f"Benchmark validation completed. Valid: {validation_report['valid']}")
        return validation_report