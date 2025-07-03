"""Data loaders for external medical datasets.

This module provides data loading functionality for integrating external medical
datasets into the MEQ-Bench framework. It includes loaders for popular datasets
like MedQuAD, HealthSearchQA, and provides standardized conversion to MEQBenchItem objects.

The module ensures consistent data formatting and validation across different
dataset sources, making it easy to extend MEQ-Bench with new data sources.

Supported Datasets:
    - MedQuAD: Medical Question Answering Dataset
    - HealthSearchQA: Health Search Question Answering Dataset

Example:
    ```python
    from data_loaders import load_medquad, load_healthsearchqa
    
    # Load different datasets
    medquad_items = load_medquad('path/to/medquad.json')
    healthsearch_items = load_healthsearchqa('path/to/healthsearchqa.json')
    
    # Add to benchmark
    bench = MEQBench()
    for item in medquad_items + healthsearch_items:
        bench.add_benchmark_item(item)
    ```
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .benchmark import MEQBenchItem

logger = logging.getLogger('meq_bench.data_loaders')


def load_medquad(
    data_path: Union[str, Path], 
    max_items: Optional[int] = None,
    complexity_level: str = 'basic'
) -> List[MEQBenchItem]:
    """Load MedQuAD dataset and convert to MEQBenchItem objects.
    
    The MedQuAD (Medical Question Answering Dataset) contains consumer health
    questions and answers from various medical sources. This function loads
    the dataset and converts it to the MEQ-Bench format for evaluation.
    
    Args:
        data_path: Path to the MedQuAD JSON file. Can be a string or Path object.
        max_items: Maximum number of items to load. If None, loads all items.
        complexity_level: Complexity level to assign to all items. Defaults to 'basic'
            since MedQuAD primarily contains consumer health questions.
            
    Returns:
        List of MEQBenchItem objects converted from the MedQuAD dataset.
        Each item has the following structure:
        - id: Generated from original question ID or index
        - medical_content: Combined question and answer content
        - complexity_level: Set to the specified complexity level
        - source_dataset: Set to 'MedQuAD'
        
    Raises:
        FileNotFoundError: If the specified data file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the dataset format is invalid or missing required fields.
        
    Example:
        ```python
        # Load all MedQuAD items
        items = load_medquad('data/medquad.json')
        
        # Load only first 100 items
        items = load_medquad('data/medquad.json', max_items=100)
        
        # Load with different complexity level
        items = load_medquad('data/medquad.json', complexity_level='intermediate')
        ```
    """
    # Convert to Path object for consistent handling
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"MedQuAD data file not found: {data_file}")
    
    logger.info(f"Loading MedQuAD dataset from: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in MedQuAD data file: {e}",
            e.doc, e.pos
        )
    
    # Validate data format
    if not isinstance(data, list):
        raise ValueError("MedQuAD data must be a list of items")
    
    if not data:
        logger.warning("MedQuAD dataset is empty")
        return []
    
    # Convert to MEQBenchItem objects
    items = []
    items_to_process = data[:max_items] if max_items else data
    
    logger.info(f"Processing {len(items_to_process)} MedQuAD items")
    
    for i, item_data in enumerate(items_to_process):
        try:
            # Validate required fields in MedQuAD item
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping invalid item {i}: not a dictionary")
                continue
            
            # Extract question and answer - handle different possible formats
            question = _extract_field(item_data, ['question', 'Question', 'q', 'Q'], f"item {i}")
            answer = _extract_field(item_data, ['answer', 'Answer', 'a', 'A'], f"item {i}")
            
            if not question or not answer:
                logger.warning(f"Skipping item {i}: missing question or answer")
                continue
            
            # Create combined medical content
            medical_content = f"Question: {question.strip()}\n\nAnswer: {answer.strip()}"
            
            # Generate ID - use provided ID or create from index
            item_id = item_data.get('id') or item_data.get('ID') or f"medquad_{i:06d}"
            
            # Create MEQBenchItem
            item = MEQBenchItem(
                id=str(item_id),
                medical_content=medical_content,
                complexity_level=complexity_level,
                source_dataset='MedQuAD',
                reference_explanations=None  # MedQuAD doesn't have audience-specific explanations
            )
            
            # Validate the created item
            _validate_benchmark_item(item)
            
            items.append(item)
            
        except Exception as e:
            logger.error(f"Error processing MedQuAD item {i}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(items)} MEQBenchItem objects from MedQuAD")
    
    if len(items) == 0:
        logger.warning("No valid items were loaded from MedQuAD dataset")
    
    return items


def _extract_field(
    item_data: Dict[str, Any], 
    possible_keys: List[str], 
    item_identifier: str
) -> Optional[str]:
    """Extract a field from item data using multiple possible key names.
    
    Args:
        item_data: Dictionary containing the item data.
        possible_keys: List of possible key names to try.
        item_identifier: Identifier for the item (for error messages).
        
    Returns:
        The field value if found, None otherwise.
    """
    for key in possible_keys:
        if key in item_data:
            value = item_data[key]
            if isinstance(value, str) and value.strip():
                return value.strip()
    
    logger.debug(f"Could not find field with keys {possible_keys} in {item_identifier}")
    return None


def _validate_benchmark_item(item: MEQBenchItem) -> None:
    """Validate a MEQBenchItem for basic requirements.
    
    Args:
        item: MEQBenchItem to validate.
        
    Raises:
        ValueError: If the item doesn't meet basic requirements.
    """
    if not item.id or not isinstance(item.id, str):
        raise ValueError("Item ID must be a non-empty string")
    
    if not item.medical_content or not isinstance(item.medical_content, str):
        raise ValueError("Medical content must be a non-empty string")
    
    if len(item.medical_content.strip()) < 20:
        raise ValueError("Medical content is too short (less than 20 characters)")
    
    if item.complexity_level not in ['basic', 'intermediate', 'advanced']:
        raise ValueError("Complexity level must be 'basic', 'intermediate', or 'advanced'")
    
    if not item.source_dataset or not isinstance(item.source_dataset, str):
        raise ValueError("Source dataset must be a non-empty string")


def load_custom_dataset(
    data_path: Union[str, Path],
    source_name: str,
    field_mapping: Dict[str, str],
    default_complexity: str = 'basic',
    max_items: Optional[int] = None
) -> List[MEQBenchItem]:
    """Load a custom dataset with flexible field mapping.
    
    This function provides a generic way to load custom medical datasets
    by allowing users to specify how fields in their data map to the
    MEQBenchItem structure.
    
    Args:
        data_path: Path to the dataset JSON file.
        source_name: Name to use for the source_dataset field.
        field_mapping: Dictionary mapping MEQBenchItem fields to source fields.
            Required keys: 'id', 'medical_content'
            Optional keys: 'complexity_level'
        default_complexity: Default complexity level if not specified in mapping.
        max_items: Maximum number of items to load.
        
    Returns:
        List of MEQBenchItem objects.
        
    Example:
        ```python
        # Load custom dataset with field mapping
        mapping = {
            'id': 'question_id',
            'medical_content': 'content',
            'complexity_level': 'difficulty'
        }
        items = load_custom_dataset(
            'data/custom.json',
            'CustomDataset',
            mapping
        )
        ```
    """
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Custom dataset file not found: {data_file}")
    
    # Validate field mapping
    required_fields = {'id', 'medical_content'}
    if not required_fields.issubset(field_mapping.keys()):
        missing = required_fields - field_mapping.keys()
        raise ValueError(f"Field mapping missing required fields: {missing}")
    
    logger.info(f"Loading custom dataset from: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in custom dataset file: {e}",
            e.doc, e.pos
        )
    
    if not isinstance(data, list):
        raise ValueError("Custom dataset must be a list of items")
    
    items = []
    items_to_process = data[:max_items] if max_items else data
    
    logger.info(f"Processing {len(items_to_process)} custom dataset items")
    
    for i, item_data in enumerate(items_to_process):
        try:
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping invalid item {i}: not a dictionary")
                continue
            
            # Extract mapped fields
            item_id = item_data.get(field_mapping['id'])
            medical_content = item_data.get(field_mapping['medical_content'])
            
            if not item_id or not medical_content:
                logger.warning(f"Skipping item {i}: missing required fields")
                continue
            
            # Get complexity level
            complexity_field = field_mapping.get('complexity_level')
            if complexity_field and complexity_field in item_data:
                complexity = item_data[complexity_field]
                if complexity not in ['basic', 'intermediate', 'advanced']:
                    logger.warning(f"Invalid complexity level '{complexity}' in item {i}, using default")
                    complexity = default_complexity
            else:
                complexity = default_complexity
            
            # Create MEQBenchItem
            item = MEQBenchItem(
                id=str(item_id),
                medical_content=str(medical_content),
                complexity_level=complexity,
                source_dataset=source_name,
                reference_explanations=None
            )
            
            # Validate the created item
            _validate_benchmark_item(item)
            
            items.append(item)
            
        except Exception as e:
            logger.error(f"Error processing custom dataset item {i}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(items)} MEQBenchItem objects from custom dataset")
    return items


def save_benchmark_items(
    items: List[MEQBenchItem], 
    output_path: Union[str, Path]
) -> None:
    """Save MEQBenchItem objects to a JSON file.
    
    This function serializes a list of MEQBenchItem objects to JSON format
    for later use with the MEQ-Bench framework.
    
    Args:
        items: List of MEQBenchItem objects to save.
        output_path: Path where the JSON file should be saved.
        
    Raises:
        ValueError: If the items list is empty.
        Exception: If file writing fails.
        
    Example:
        ```python
        items = load_medquad('data/medquad.json')
        save_benchmark_items(items, 'data/benchmark_items.json')
        ```
    """
    if not items:
        raise ValueError("Cannot save empty items list")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert items to dictionaries
    items_data = []
    for item in items:
        items_data.append({
            'id': item.id,
            'medical_content': item.medical_content,
            'complexity_level': item.complexity_level,
            'source_dataset': item.source_dataset,
            'reference_explanations': item.reference_explanations
        })
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(items_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(items)} benchmark items to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save benchmark items to {output_file}: {e}")
        raise


def load_healthsearchqa(
    data_path: Union[str, Path], 
    max_items: Optional[int] = None,
    complexity_level: str = 'intermediate'
) -> List[MEQBenchItem]:
    """Load HealthSearchQA dataset and convert to MEQBenchItem objects.
    
    The HealthSearchQA dataset contains health-related search queries and answers
    from reputable medical sources. This function loads the dataset and converts
    it to the MEQ-Bench format for evaluation.
    
    Expected JSON format:
    [
        {
            "id": "healthsearchqa_1",
            "question": "What are the symptoms of a heart attack?",
            "answer": "Symptoms of a heart attack can include chest pain...",
            "source": "reputable medical source"
        }
    ]
    
    Args:
        data_path: Path to the HealthSearchQA JSON file. Can be a string or Path object.
        max_items: Maximum number of items to load. If None, loads all items.
        complexity_level: Complexity level to assign to all items. Defaults to 'intermediate'
            since HealthSearchQA contains moderately complex health information.
            
    Returns:
        List of MEQBenchItem objects converted from HealthSearchQA data.
        
    Raises:
        FileNotFoundError: If the data file does not exist.
        json.JSONDecodeError: If the JSON file is malformed.
        ValueError: If the data format is invalid.
        
    Example:
        ```python
        # Load HealthSearchQA dataset
        items = load_healthsearchqa('data/healthsearchqa.json', max_items=100)
        
        # Add to benchmark
        bench = MEQBench()
        for item in items:
            bench.add_benchmark_item(item)
        ```
    """
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"HealthSearchQA file not found: {data_file}")
    
    logger.info(f"Loading HealthSearchQA dataset from: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in HealthSearchQA file: {e}",
            e.doc, e.pos
        )
    
    if not isinstance(data, list):
        raise ValueError("HealthSearchQA data must be a list of items")
    
    # Validate complexity level
    if complexity_level not in ['basic', 'intermediate', 'advanced']:
        logger.warning(f"Invalid complexity level '{complexity_level}', using 'intermediate'")
        complexity_level = 'intermediate'
    
    items = []
    items_to_process = data[:max_items] if max_items else data
    
    logger.info(f"Processing {len(items_to_process)} HealthSearchQA items")
    
    for i, item_data in enumerate(items_to_process):
        try:
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping invalid HealthSearchQA item {i}: not a dictionary")
                continue
            
            # Extract required fields
            item_id = item_data.get('id')
            question = item_data.get('question', '')
            answer = item_data.get('answer', '')
            source = item_data.get('source', 'HealthSearchQA')
            
            # Validate required fields
            if not item_id:
                logger.warning(f"Skipping HealthSearchQA item {i}: missing 'id' field")
                continue
                
            if not question.strip():
                logger.warning(f"Skipping HealthSearchQA item {i}: empty 'question' field")
                continue
                
            if not answer.strip():
                logger.warning(f"Skipping HealthSearchQA item {i}: empty 'answer' field")
                continue
            
            # Combine question and answer to create medical content
            # This provides context for explanation generation
            medical_content = f"Question: {question.strip()}\n\nAnswer: {answer.strip()}"
            
            if source and source != 'HealthSearchQA':
                medical_content += f"\n\nSource: {source}"
            
            # Create MEQBenchItem
            item = MEQBenchItem(
                id=str(item_id),
                medical_content=medical_content,
                complexity_level=complexity_level,
                source_dataset='HealthSearchQA',
                reference_explanations=None  # No reference explanations in HealthSearchQA
            )
            
            # Validate the created item
            _validate_benchmark_item(item)
            
            items.append(item)
            
        except Exception as e:
            logger.error(f"Error processing HealthSearchQA item {i}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(items)} MEQBenchItem objects from HealthSearchQA")
    
    if len(items) == 0:
        logger.warning("No valid items were loaded from HealthSearchQA dataset")
    else:
        # Log some statistics
        avg_length = sum(len(item.medical_content) for item in items) / len(items)
        logger.info(f"HealthSearchQA dataset statistics:")
        logger.info(f"  - Total items: {len(items)}")
        logger.info(f"  - Average content length: {avg_length:.1f} characters")
        logger.info(f"  - Complexity level: {complexity_level}")
    
    return items