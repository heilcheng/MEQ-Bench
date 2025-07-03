"""
Strategy pattern implementation for audience-specific scoring
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

from .config import config

logger = logging.getLogger(__name__)


class AudienceStrategy(ABC):
    """Abstract base class for audience-specific scoring strategies"""
    
    def __init__(self, audience: str):
        self.audience = audience
        self.eval_config = config.get_evaluation_config()
        
    @abstractmethod
    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """Calculate readability score for the audience"""
        pass
    
    @abstractmethod
    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """Calculate terminology appropriateness score for the audience"""
        pass
    
    @abstractmethod
    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Get expected explanation length range for the audience"""
        pass
    
    def get_readability_targets(self) -> Dict[str, float]:
        """Get readability targets for this audience"""
        return self.eval_config['readability_targets'][self.audience]
    
    def get_terminology_targets(self) -> Dict[str, float]:
        """Get terminology density targets for this audience"""
        return self.eval_config['terminology_density'][self.audience]


class PhysicianStrategy(AudienceStrategy):
    """Strategy for physician audience scoring"""
    
    def __init__(self):
        super().__init__('physician')
    
    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """
        Physicians expect graduate-level complexity (12-16 grade level)
        Higher complexity is better for physicians
        """
        targets = self.get_readability_targets()
        min_level = targets['min_grade_level']
        max_level = targets['max_grade_level']
        
        if grade_level < min_level:
            # Too simple for physicians
            return max(0.0, grade_level / min_level)
        elif grade_level > max_level:
            # Too complex even for physicians
            return max(0.0, 1.0 - (grade_level - max_level) / 4.0)
        else:
            # In the sweet spot
            return 1.0
    
    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """
        Physicians expect high medical terminology density
        """
        targets = self.get_terminology_targets()
        target = targets['target']
        tolerance = targets['tolerance']
        
        if abs(term_density - target) <= tolerance:
            return 1.0
        elif term_density < target:
            # Too little medical terminology
            return max(0.0, term_density / target)
        else:
            # Too much terminology (even for physicians)
            excess = term_density - target - tolerance
            return max(0.0, 1.0 - excess * 2.0)
    
    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Physicians can handle longer, detailed explanations"""
        scoring_config = config.get_scoring_config()
        max_length = scoring_config['parameters']['max_explanation_length']['physician']
        min_length = scoring_config['parameters']['min_explanation_length']
        
        return {'min': min_length, 'max': max_length}


class NurseStrategy(AudienceStrategy):
    """Strategy for nurse audience scoring"""
    
    def __init__(self):
        super().__init__('nurse')
    
    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """
        Nurses expect moderate complexity (10-14 grade level)
        Balance between technical accuracy and practical application
        """
        targets = self.get_readability_targets()
        min_level = targets['min_grade_level']
        max_level = targets['max_grade_level']
        
        if min_level <= grade_level <= max_level:
            return 1.0
        elif grade_level < min_level:
            return max(0.0, grade_level / min_level)
        else:
            return max(0.0, 1.0 - (grade_level - max_level) / 6.0)
    
    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """
        Nurses expect moderate medical terminology
        Should include technical terms but also practical language
        """
        targets = self.get_terminology_targets()
        target = targets['target']
        tolerance = targets['tolerance']
        
        if abs(term_density - target) <= tolerance:
            return 1.0
        else:
            deviation = abs(term_density - target) - tolerance
            return max(0.0, 1.0 - deviation * 3.0)
    
    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Nurses need practical, actionable explanations"""
        scoring_config = config.get_scoring_config()
        max_length = scoring_config['parameters']['max_explanation_length']['nurse']
        min_length = scoring_config['parameters']['min_explanation_length']
        
        return {'min': min_length, 'max': max_length}


class PatientStrategy(AudienceStrategy):
    """Strategy for patient audience scoring"""
    
    def __init__(self):
        super().__init__('patient')
    
    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """
        Patients need simple, accessible language (6-10 grade level)
        Lower complexity is better for patients
        """
        targets = self.get_readability_targets()
        min_level = targets['min_grade_level']
        max_level = targets['max_grade_level']
        
        if min_level <= grade_level <= max_level:
            return 1.0
        elif grade_level < min_level:
            # Very simple is still okay for patients
            return 0.8
        else:
            # Too complex for patients
            return max(0.0, 1.0 - (grade_level - max_level) / 4.0)
    
    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """
        Patients should have minimal medical terminology
        Jargon should be avoided or explained
        """
        targets = self.get_terminology_targets()
        target = targets['target']
        tolerance = targets['tolerance']
        
        if term_density <= target + tolerance:
            return 1.0
        else:
            # Penalty for too much medical terminology
            excess = term_density - target - tolerance
            return max(0.0, 1.0 - excess * 10.0)
    
    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Patients need concise, clear explanations"""
        scoring_config = config.get_scoring_config()
        max_length = scoring_config['parameters']['max_explanation_length']['patient']
        min_length = scoring_config['parameters']['min_explanation_length']
        
        return {'min': min_length, 'max': max_length}


class CaregiverStrategy(AudienceStrategy):
    """Strategy for caregiver audience scoring"""
    
    def __init__(self):
        super().__init__('caregiver')
    
    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """
        Caregivers need actionable, clear language (6-10 grade level)
        Focus on practical instructions
        """
        targets = self.get_readability_targets()
        min_level = targets['min_grade_level']
        max_level = targets['max_grade_level']
        
        if min_level <= grade_level <= max_level:
            return 1.0
        elif grade_level < min_level:
            return 0.9  # Simple is good for caregivers
        else:
            return max(0.0, 1.0 - (grade_level - max_level) / 4.0)
    
    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """
        Caregivers need minimal medical terminology
        Focus on observable symptoms and clear actions
        """
        targets = self.get_terminology_targets()
        target = targets['target']
        tolerance = targets['tolerance']
        
        if term_density <= target + tolerance:
            return 1.0
        else:
            excess = term_density - target - tolerance
            return max(0.0, 1.0 - excess * 8.0)
    
    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Caregivers need practical, step-by-step guidance"""
        scoring_config = config.get_scoring_config()
        max_length = scoring_config['parameters']['max_explanation_length']['caregiver']
        min_length = scoring_config['parameters']['min_explanation_length']
        
        return {'min': min_length, 'max': max_length}


class StrategyFactory:
    """Factory for creating audience strategies"""
    
    _strategies = {
        'physician': PhysicianStrategy,
        'nurse': NurseStrategy,
        'patient': PatientStrategy,
        'caregiver': CaregiverStrategy
    }
    
    @classmethod
    def create_strategy(cls, audience: str) -> AudienceStrategy:
        """
        Create strategy for given audience
        
        Args:
            audience: Target audience name
            
        Returns:
            AudienceStrategy instance
            
        Raises:
            ValueError: If audience not supported
        """
        if audience not in cls._strategies:
            supported = list(cls._strategies.keys())
            raise ValueError(f"Unsupported audience: {audience}. Supported: {supported}")
        
        strategy_class = cls._strategies[audience]
        return strategy_class()
    
    @classmethod
    def get_supported_audiences(cls) -> List[str]:
        """Get list of supported audiences"""
        return list(cls._strategies.keys())