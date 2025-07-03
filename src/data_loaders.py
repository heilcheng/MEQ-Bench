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

from benchmark import MEQBenchItem

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
        List of MEQBenchItem objects converted from MedQuAD data.
        
    Raises:
        FileNotFoundError: If the data file does not exist.
        json.JSONDecodeError: If the JSON file is malformed.
        ValueError: If the data format is invalid.
        
    Example:
        ```python
        # Load all MedQuAD items
        items = load_medquad('data/medquad.json')
        
        # Load only first 100 items
        items = load_medquad('data/medquad.json', max_items=100)
        
        # Load with different complexity level
        items = load_medquad('data/medquad.json', complexity_level='intermediate')
        
        # Add to benchmark
        bench = MEQBench()
        for item in items:
            bench.add_benchmark_item(item)
        ```
    """
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"MedQuAD file not found: {data_file}")
    
    logger.info(f"Loading MedQuAD dataset from: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in MedQuAD file: {e}",
            e.doc, e.pos
        )
    
    if not isinstance(data, list):
        raise ValueError("MedQuAD data must be a list of items")
    
    # Validate complexity level
    if complexity_level not in ['basic', 'intermediate', 'advanced']:
        logger.warning(f"Invalid complexity level '{complexity_level}', using 'basic'")
        complexity_level = 'basic'
    
    # Convert to MEQBenchItem objects
    items = []
    items_to_process = data[:max_items] if max_items else data
    
    logger.info(f"Processing {len(items_to_process)} MedQuAD items")
    
    for i, item_data in enumerate(items_to_process):
        try:
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping invalid MedQuAD item {i}: not a dictionary")
                continue
            
            # Extract required fields - MedQuAD typically has 'question' and 'answer'
            question = item_data.get('question', '')
            answer = item_data.get('answer', '')
            item_id = item_data.get('id', f"medquad_{i}")
            
            if not question.strip() or not answer.strip():
                logger.warning(f"Skipping MedQuAD item {i}: empty question or answer")
                continue
            
            # Combine question and answer to create medical content
            medical_content = f"Question: {question.strip()}\\n\\nAnswer: {answer.strip()}"
            
            # Create MEQBenchItem
            item = MEQBenchItem(
                id=str(item_id),
                medical_content=medical_content,
                complexity_level=complexity_level,
                source_dataset='MedQuAD',
                reference_explanations=None  # No reference explanations in MedQuAD
            )
            
            # Basic validation
            _validate_benchmark_item(item)
            
            items.append(item)
            
        except Exception as e:
            logger.error(f"Error processing MedQuAD item {i}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(items)} MEQBenchItem objects from MedQuAD")
    
    if len(items) == 0:
        logger.warning("No valid items were loaded from MedQuAD dataset")
    else:
        # Log some statistics
        avg_length = sum(len(item.medical_content) for item in items) / len(items)
        logger.info(f"MedQuAD dataset statistics:")
        logger.info(f"  - Total items: {len(items)}")
        logger.info(f"  - Average content length: {avg_length:.1f} characters")
        logger.info(f"  - Complexity level: {complexity_level}")
    
    return items


def load_healthsearchqa(
    data_path: Union[str, Path], 
    max_items: Optional[int] = None,
    complexity_level: str = 'intermediate'
) -> List[MEQBenchItem]:
    """Load HealthSearchQA dataset and convert to MEQBenchItem objects.
    
    The HealthSearchQA dataset contains health-related search queries and answers
    from various health websites and search engines. This loader converts the dataset
    into MEQBenchItem objects for use in the benchmark.
    
    Args:
        data_path: Path to the HealthSearchQA JSON file.
        max_items: Maximum number of items to load. If None, loads all items.
        complexity_level: Complexity level to assign to all items. Defaults to 'intermediate'
            since HealthSearchQA contains more varied complexity levels.
            
    Returns:
        List of MEQBenchItem objects converted from HealthSearchQA data.
        
    Raises:
        FileNotFoundError: If the data file does not exist.
        json.JSONDecodeError: If the JSON file is malformed.
        ValueError: If the data format is invalid.
        
    Example:
        ```python
        # Load HealthSearchQA items
        items = load_healthsearchqa('data/healthsearchqa.json')
        
        # Load with custom complexity level
        items = load_healthsearchqa('data/healthsearchqa.json', complexity_level='advanced')
        
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
            
            # HealthSearchQA might have different field names
            query = item_data.get('query', item_data.get('question', ''))
            answer = item_data.get('answer', item_data.get('response', ''))
            item_id = item_data.get('id', f"healthsearch_{i}")
            
            if not query.strip() or not answer.strip():
                logger.warning(f"Skipping HealthSearchQA item {i}: empty query or answer")
                continue
            
            # Create medical content
            medical_content = f"Search Query: {query.strip()}\\n\\nAnswer: {answer.strip()}"
            
            # Create MEQBenchItem
            item = MEQBenchItem(
                id=str(item_id),
                medical_content=medical_content,
                complexity_level=complexity_level,
                source_dataset='HealthSearchQA',
                reference_explanations=None
            )
            
            # Basic validation
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


def load_custom_dataset(
    data_path: Union[str, Path],
    field_mapping: Optional[Dict[str, str]] = None,
    max_items: Optional[int] = None,
    complexity_level: str = 'basic'
) -> List[MEQBenchItem]:
    """Load custom dataset and convert to MEQBenchItem objects.
    
    Args:
        data_path: Path to the JSON file containing the dataset.
        field_mapping: Dictionary mapping dataset fields to MEQBenchItem fields.
                      Example: {'q': 'question', 'a': 'answer', 'topic': 'medical_content'}
        max_items: Maximum number of items to load.
        complexity_level: Complexity level to assign to all items.
        
    Returns:
        List of MEQBenchItem objects.
    """
    # Default field mapping
    if field_mapping is None:
        field_mapping = {
            'question': 'question',
            'answer': 'answer',
            'content': 'medical_content',
            'id': 'id'
        }
    
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Custom dataset file not found: {data_file}")
    
    logger.info(f"Loading custom dataset from: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Custom dataset must be a list of items")
    
    items = []
    items_to_process = data[:max_items] if max_items else data
    
    for i, item_data in enumerate(items_to_process):
        try:
            # Extract fields based on mapping
            question = item_data.get(field_mapping.get('question', 'question'), '')
            answer = item_data.get(field_mapping.get('answer', 'answer'), '')
            content = item_data.get(field_mapping.get('content', 'content'), '')
            item_id = item_data.get(field_mapping.get('id', 'id'), f"custom_{i}")
            
            # Create medical content
            if content:
                medical_content = content
            elif question and answer:
                medical_content = f"Question: {question.strip()}\\n\\nAnswer: {answer.strip()}"
            else:
                logger.warning(f"Skipping item {i}: no valid content found")
                continue
            
            item = MEQBenchItem(
                id=str(item_id),
                medical_content=medical_content,
                complexity_level=complexity_level,
                source_dataset='Custom',
                reference_explanations=None
            )
            
            _validate_benchmark_item(item)
            items.append(item)
            
        except Exception as e:
            logger.error(f"Error processing custom dataset item {i}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(items)} items from custom dataset")
    return items


def save_benchmark_items(
    items: List[MEQBenchItem],
    output_path: Union[str, Path],
    pretty_print: bool = True
) -> None:
    """Save MEQBenchItem objects to a JSON file.
    
    Args:
        items: List of MEQBenchItem objects to save.
        output_path: Path where to save the JSON file.
        pretty_print: Whether to format JSON with indentation.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert items to dictionaries
    items_data = []
    for item in items:
        item_dict = {
            'id': item.id,
            'medical_content': item.medical_content,
            'complexity_level': item.complexity_level,
            'source_dataset': item.source_dataset,
            'reference_explanations': item.reference_explanations
        }
        items_data.append(item_dict)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        if pretty_print:
            json.dump(items_data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(items_data, f, ensure_ascii=False)
    
    logger.info(f"Saved {len(items)} benchmark items to: {output_file}")


def _validate_benchmark_item(item: MEQBenchItem) -> None:
    """Validate a MEQBenchItem object for basic requirements.
    
    Args:
        item: MEQBenchItem to validate
        
    Raises:
        ValueError: If the item doesn't meet basic requirements
    """
    if not item.id or not isinstance(item.id, str):
        raise ValueError("Item ID must be a non-empty string")
    
    if not item.medical_content or not isinstance(item.medical_content, str):
        raise ValueError("Medical content must be a non-empty string")
    
    if len(item.medical_content.strip()) < 20:
        raise ValueError("Medical content is too short (less than 20 characters)")