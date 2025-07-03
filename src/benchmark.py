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
        self.data_path = Path(data_path) if data_path else Path(__file__).parent.parent / "data"
        self.evaluator = MEQBenchEvaluator()
        self.prompt_template = AudienceAdaptivePrompt()
        self.benchmark_items: List[MEQBenchItem] = []
        
        # Load benchmark data if available
        if self.data_path.exists():
            self._load_benchmark_data()
    
    def _load_benchmark_data(self):
        """Load benchmark data from JSON files"""
        benchmark_file = self.data_path / "benchmark_items.json"
        
        if benchmark_file.exists():
            with open(benchmark_file, 'r') as f:
                data = json.load(f)
                
            for item_data in data:
                item = MEQBenchItem(
                    id=item_data['id'],
                    medical_content=item_data['medical_content'],
                    complexity_level=item_data['complexity_level'],
                    source_dataset=item_data['source_dataset'],
                    reference_explanations=item_data.get('reference_explanations')
                )
                self.benchmark_items.append(item)
    
    def add_benchmark_item(self, item: MEQBenchItem):
        """Add a new benchmark item"""
        self.benchmark_items.append(item)
    
    def generate_explanations(self, medical_content: str, model_func: callable) -> Dict[str, str]:
        """
        Generate audience-adaptive explanations using a model
        
        Args:
            medical_content: Medical information to adapt
            model_func: Function that takes prompt and returns model response
            
        Returns:
            Dictionary mapping audience to explanation
        """
        prompt = self.prompt_template.format_prompt(medical_content)
        response = model_func(prompt)
        
        return self.prompt_template.parse_response(response)
    
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
        """Save evaluation results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
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
            # Save sample dataset
            data_to_save = []
            for item in sample_items:
                data_to_save.append({
                    'id': item.id,
                    'medical_content': item.medical_content,
                    'complexity_level': item.complexity_level,
                    'source_dataset': item.source_dataset,
                    'reference_explanations': item.reference_explanations
                })
            
            with open(output_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        
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