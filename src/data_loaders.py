"""Data loaders for external medical datasets.

This module provides data loading functionality for integrating external medical
datasets into the MEQ-Bench framework. It includes loaders for popular medical datasets
and provides standardized conversion to MEQBenchItem objects with automatic complexity
stratification based on Flesch-Kincaid readability scores.

The module ensures consistent data formatting and validation across different
dataset sources, making it easy to extend MEQ-Bench with new data sources.

Supported Datasets:
    - MedQuAD: Medical Question Answering Dataset
    - HealthSearchQA: Health Search Question Answering Dataset
    - MedQA-USMLE: Medical Question Answering based on USMLE exams
    - iCliniq: Clinical question answering dataset
    - Cochrane Reviews: Evidence-based medical reviews

Complexity Stratification:
    - Uses Flesch-Kincaid Grade Level scores to automatically categorize content
    - Basic: FK score <= 8 (elementary/middle school level)
    - Intermediate: FK score 9-12 (high school level)
    - Advanced: FK score > 12 (college/professional level)

Example:
    ```python
    from data_loaders import load_medqa_usmle, load_icliniq, load_cochrane_reviews

    # Load different datasets with automatic complexity stratification
    medqa_items = load_medqa_usmle('path/to/medqa.json', max_items=300)
    icliniq_items = load_icliniq('path/to/icliniq.json', max_items=400)
    cochrane_items = load_cochrane_reviews('path/to/cochrane.json', max_items=300)

    # Combine and save as benchmark dataset
    all_items = medqa_items + icliniq_items + cochrane_items
    save_benchmark_items(all_items, 'data/benchmark_items.json')
    ```
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Union

try:
    import textstat  # type: ignore
except ImportError:
    textstat = None

from .benchmark import MEQBenchItem

logger = logging.getLogger("meq_bench.data_loaders")


def load_medquad(
    data_path: Union[str, Path], max_items: Optional[int] = None, complexity_level: str = "basic"
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
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in MedQuAD file: {e}", e.doc, e.pos)

    if not isinstance(data, list):
        raise ValueError("MedQuAD data must be a list of items")

    # Validate complexity level
    if complexity_level not in ["basic", "intermediate", "advanced"]:
        logger.warning(f"Invalid complexity level '{complexity_level}', using 'basic'")
        complexity_level = "basic"

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
            question = item_data.get("question", "")
            answer = item_data.get("answer", "")
            item_id = item_data.get("id", f"medquad_{i}")

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
                source_dataset="MedQuAD",
                reference_explanations=None,  # No reference explanations in MedQuAD
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
        logger.info("MedQuAD dataset statistics:")
        logger.info(f"  - Total items: {len(items)}")
        logger.info(f"  - Average content length: {avg_length:.1f} characters")
        logger.info(f"  - Complexity level: {complexity_level}")

    return items


def load_healthsearchqa(
    data_path: Union[str, Path], max_items: Optional[int] = None, complexity_level: str = "intermediate"
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
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in HealthSearchQA file: {e}", e.doc, e.pos)

    if not isinstance(data, list):
        raise ValueError("HealthSearchQA data must be a list of items")

    # Validate complexity level
    if complexity_level not in ["basic", "intermediate", "advanced"]:
        logger.warning(f"Invalid complexity level '{complexity_level}', using 'intermediate'")
        complexity_level = "intermediate"

    items = []
    items_to_process = data[:max_items] if max_items else data

    logger.info(f"Processing {len(items_to_process)} HealthSearchQA items")

    for i, item_data in enumerate(items_to_process):
        try:
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping invalid HealthSearchQA item {i}: not a dictionary")
                continue

            # HealthSearchQA might have different field names
            query = item_data.get("query", item_data.get("question", ""))
            answer = item_data.get("answer", item_data.get("response", ""))
            item_id = item_data.get("id", f"healthsearch_{i}")

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
                source_dataset="HealthSearchQA",
                reference_explanations=None,
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
        logger.info("HealthSearchQA dataset statistics:")
        logger.info(f"  - Total items: {len(items)}")
        logger.info(f"  - Average content length: {avg_length:.1f} characters")
        logger.info(f"  - Complexity level: {complexity_level}")

    return items


def load_custom_dataset(
    data_path: Union[str, Path],
    field_mapping: Optional[Dict[str, str]] = None,
    max_items: Optional[int] = None,
    complexity_level: str = "basic",
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
        field_mapping = {"question": "question", "answer": "answer", "content": "medical_content", "id": "id"}

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Custom dataset file not found: {data_file}")

    logger.info(f"Loading custom dataset from: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Custom dataset must be a list of items")

    items = []
    items_to_process = data[:max_items] if max_items else data

    for i, item_data in enumerate(items_to_process):
        try:
            # Extract fields based on mapping
            question = item_data.get(field_mapping.get("question", "question"), "")
            answer = item_data.get(field_mapping.get("answer", "answer"), "")
            content = item_data.get(field_mapping.get("content", "content"), "")
            item_id = item_data.get(field_mapping.get("id", "id"), f"custom_{i}")

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
                source_dataset="Custom",
                reference_explanations=None,
            )

            _validate_benchmark_item(item)
            items.append(item)

        except Exception as e:
            logger.error(f"Error processing custom dataset item {i}: {e}")
            continue

    logger.info(f"Successfully loaded {len(items)} items from custom dataset")
    return items


def save_benchmark_items(items: List[MEQBenchItem], output_path: Union[str, Path], pretty_print: bool = True) -> None:
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
            "id": item.id,
            "medical_content": item.medical_content,
            "complexity_level": item.complexity_level,
            "source_dataset": item.source_dataset,
            "reference_explanations": item.reference_explanations,
        }
        items_data.append(item_dict)

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        if pretty_print:
            json.dump(items_data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(items_data, f, ensure_ascii=False)

    logger.info(f"Saved {len(items)} benchmark items to: {output_file}")


def calculate_complexity_level(text: str) -> str:
    """Calculate complexity level based on Flesch-Kincaid Grade Level.

    Uses textstat library to compute Flesch-Kincaid Grade Level and categorizes
    the text into basic, intermediate, or advanced complexity levels.

    Args:
        text: Text content to analyze for complexity.

    Returns:
        Complexity level string: 'basic', 'intermediate', or 'advanced'

    Raises:
        ValueError: If text is empty or invalid.

    Example:
        ```python
        complexity = calculate_complexity_level("This is simple text.")
        # Returns: 'basic' (if FK score <= 8)
        ```
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")

    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty or whitespace only")

    # Clean text for analysis - remove extra whitespace and normalize
    cleaned_text = " ".join(text.split())

    # Fallback if textstat is not available
    if textstat is None:
        logger.warning("textstat library not available, using fallback complexity calculation")
        return _calculate_complexity_fallback(cleaned_text)

    try:
        # Calculate Flesch-Kincaid Grade Level
        fk_score = textstat.flesch_kincaid().grade(cleaned_text)

        # Categorize based on grade level
        if fk_score <= 8:
            return "basic"
        elif fk_score <= 12:
            return "intermediate"
        else:
            return "advanced"

    except Exception as e:
        logger.warning(f"Error calculating Flesch-Kincaid score: {e}, using fallback")
        return _calculate_complexity_fallback(cleaned_text)


def _calculate_complexity_fallback(text: str) -> str:
    """Fallback complexity calculation when textstat is unavailable.

    Uses simple heuristics based on sentence length, word length, and
    medical terminology density as approximations.

    Args:
        text: Cleaned text to analyze.

    Returns:
        Complexity level: 'basic', 'intermediate', or 'advanced'
    """
    # Count sentences (approximate)
    sentences = len(re.split(r"[.!?]+", text))
    if sentences == 0:
        sentences = 1

    # Count words
    words = len(text.split())
    if words == 0:
        return "basic"

    # Calculate average words per sentence
    avg_words_per_sentence = words / sentences

    # Count syllables (rough approximation)
    syllable_count = 0
    for word in text.split():
        # Simple syllable counting heuristic
        word = re.sub(r"[^a-zA-Z]", "", word)
        if word:
            syllables = len(re.findall(r"[aeiouyAEIOUY]+", word))
            if syllables == 0:
                syllables = 1
            syllable_count += syllables

    avg_syllables_per_word = syllable_count / words if words > 0 else 1

    # Check for medical terminology (indicator of higher complexity)
    medical_terms = [
        "diagnosis",
        "treatment",
        "therapy",
        "syndrome",
        "pathology",
        "etiology",
        "prognosis",
        "medication",
        "prescription",
        "dosage",
        "contraindication",
        "adverse",
        "efficacy",
        "pharmacology",
        "clinical",
        "therapeutic",
        "intervention",
    ]

    medical_term_count = sum(1 for term in medical_terms if term.lower() in text.lower())
    medical_density = medical_term_count / words * 100  # percentage

    # Simple scoring algorithm
    complexity_score: float = 0
    complexity_score += avg_words_per_sentence * 0.5
    complexity_score += avg_syllables_per_word * 3
    complexity_score += medical_density * 0.3

    if complexity_score <= 8:
        return "basic"
    elif complexity_score <= 15:
        return "intermediate"
    else:
        return "advanced"


def load_medqa_usmle(
    data_path: Union[str, Path], max_items: Optional[int] = None, auto_complexity: bool = True
) -> List[MEQBenchItem]:
    """Load MedQA-USMLE dataset and convert to MEQBenchItem objects.

    The MedQA-USMLE dataset contains medical questions based on USMLE exam format
    with multiple choice questions and explanations. This loader processes the
    dataset and optionally applies automatic complexity stratification.

    Args:
        data_path: Path to the MedQA-USMLE JSON file.
        max_items: Maximum number of items to load. If None, loads all items.
        auto_complexity: Whether to automatically calculate complexity levels
            using Flesch-Kincaid scores. If False, assigns 'intermediate' to all.

    Returns:
        List of MEQBenchItem objects converted from MedQA-USMLE data.

    Raises:
        FileNotFoundError: If the data file does not exist.
        json.JSONDecodeError: If the JSON file is malformed.
        ValueError: If the data format is invalid.

    Example:
        ```python
        # Load with automatic complexity calculation
        items = load_medqa_usmle('data/medqa_usmle.json', max_items=300)

        # Load without complexity calculation (all marked as intermediate)
        items = load_medqa_usmle('data/medqa_usmle.json', auto_complexity=False)
        ```
    """
    data_file = Path(data_path)

    if not data_file.exists():
        raise FileNotFoundError(f"MedQA-USMLE file not found: {data_file}")

    logger.info(f"Loading MedQA-USMLE dataset from: {data_file}")

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in MedQA-USMLE file: {e}", e.doc, e.pos)

    if not isinstance(data, list):
        raise ValueError("MedQA-USMLE data must be a list of items")

    items = []
    items_to_process = data[:max_items] if max_items else data

    logger.info(f"Processing {len(items_to_process)} MedQA-USMLE items")

    for i, item_data in enumerate(items_to_process):
        try:
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping invalid MedQA-USMLE item {i}: not a dictionary")
                continue

            # Extract required fields - MedQA typically has 'question', 'options', 'answer', 'explanation'
            question = item_data.get("question", "")
            options = item_data.get("options", {})
            answer = item_data.get("answer", "")
            explanation = item_data.get("explanation", "")
            item_id = item_data.get("id", f"medqa_usmle_{i}")

            if not question.strip():
                logger.warning(f"Skipping MedQA-USMLE item {i}: empty question")
                continue

            # Format options if available
            options_text = ""
            if isinstance(options, dict):
                options_text = "\n".join([f"{k}. {v}" for k, v in options.items() if v])
            elif isinstance(options, list):
                options_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(options) if opt])

            # Create comprehensive medical content
            medical_content_parts = [f"Question: {question.strip()}"]

            if options_text:
                medical_content_parts.append(f"Options:\n{options_text}")

            if answer.strip():
                medical_content_parts.append(f"Correct Answer: {answer.strip()}")

            if explanation.strip():
                medical_content_parts.append(f"Explanation: {explanation.strip()}")

            medical_content = "\n\n".join(medical_content_parts)

            # Calculate complexity level
            if auto_complexity:
                try:
                    complexity_level = calculate_complexity_level(medical_content)
                except Exception as e:
                    logger.warning(f"Error calculating complexity for item {i}: {e}, using 'intermediate'")
                    complexity_level = "intermediate"
            else:
                complexity_level = "intermediate"

            # Create MEQBenchItem
            item = MEQBenchItem(
                id=str(item_id),
                medical_content=medical_content,
                complexity_level=complexity_level,
                source_dataset="MedQA-USMLE",
                reference_explanations=None,
            )

            # Validate the item
            _validate_benchmark_item(item)

            items.append(item)

        except Exception as e:
            logger.error(f"Error processing MedQA-USMLE item {i}: {e}")
            continue

    logger.info(f"Successfully loaded {len(items)} MEQBenchItem objects from MedQA-USMLE")

    if len(items) == 0:
        logger.warning("No valid items were loaded from MedQA-USMLE dataset")
    else:
        # Log complexity distribution
        complexity_dist: Dict[str, int] = {}
        for item in items:
            complexity_dist[item.complexity_level] = complexity_dist.get(item.complexity_level, 0) + 1

        avg_length = sum(len(item.medical_content) for item in items) / len(items)
        logger.info("MedQA-USMLE dataset statistics:")
        logger.info(f"  - Total items: {len(items)}")
        logger.info(f"  - Average content length: {avg_length:.1f} characters")
        logger.info(f"  - Complexity distribution: {complexity_dist}")

    return items


def load_icliniq(
    data_path: Union[str, Path], max_items: Optional[int] = None, auto_complexity: bool = True
) -> List[MEQBenchItem]:
    """Load iCliniq dataset and convert to MEQBenchItem objects.

    The iCliniq dataset contains real clinical questions from patients and
    answers from medical professionals. This loader processes the dataset
    and optionally applies automatic complexity stratification.

    Args:
        data_path: Path to the iCliniq JSON file.
        max_items: Maximum number of items to load. If None, loads all items.
        auto_complexity: Whether to automatically calculate complexity levels.

    Returns:
        List of MEQBenchItem objects converted from iCliniq data.

    Raises:
        FileNotFoundError: If the data file does not exist.
        json.JSONDecodeError: If the JSON file is malformed.
        ValueError: If the data format is invalid.

    Example:
        ```python
        # Load iCliniq dataset with complexity stratification
        items = load_icliniq('data/icliniq.json', max_items=400)
        ```
    """
    data_file = Path(data_path)

    if not data_file.exists():
        raise FileNotFoundError(f"iCliniq file not found: {data_file}")

    logger.info(f"Loading iCliniq dataset from: {data_file}")

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in iCliniq file: {e}", e.doc, e.pos)

    if not isinstance(data, list):
        raise ValueError("iCliniq data must be a list of items")

    items = []
    items_to_process = data[:max_items] if max_items else data

    logger.info(f"Processing {len(items_to_process)} iCliniq items")

    for i, item_data in enumerate(items_to_process):
        try:
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping invalid iCliniq item {i}: not a dictionary")
                continue

            # Extract fields - iCliniq typically has 'patient_question', 'doctor_answer', 'speciality'
            patient_question = item_data.get("patient_question", item_data.get("question", ""))
            doctor_answer = item_data.get("doctor_answer", item_data.get("answer", ""))
            specialty = item_data.get("speciality", item_data.get("specialty", ""))
            item_id = item_data.get("id", f"icliniq_{i}")

            if not patient_question.strip() or not doctor_answer.strip():
                logger.warning(f"Skipping iCliniq item {i}: empty question or answer")
                continue

            # Create medical content
            medical_content_parts = [f"Patient Question: {patient_question.strip()}"]

            if specialty.strip():
                medical_content_parts.append(f"Medical Specialty: {specialty.strip()}")

            medical_content_parts.append(f"Doctor's Answer: {doctor_answer.strip()}")

            medical_content = "\n\n".join(medical_content_parts)

            # Calculate complexity level
            if auto_complexity:
                try:
                    complexity_level = calculate_complexity_level(medical_content)
                except Exception as e:
                    logger.warning(f"Error calculating complexity for item {i}: {e}, using 'basic'")
                    complexity_level = "basic"
            else:
                complexity_level = "basic"  # iCliniq tends to be more patient-focused

            # Create MEQBenchItem
            item = MEQBenchItem(
                id=str(item_id),
                medical_content=medical_content,
                complexity_level=complexity_level,
                source_dataset="iCliniq",
                reference_explanations=None,
            )

            # Validate the item
            _validate_benchmark_item(item)

            items.append(item)

        except Exception as e:
            logger.error(f"Error processing iCliniq item {i}: {e}")
            continue

    logger.info(f"Successfully loaded {len(items)} MEQBenchItem objects from iCliniq")

    if len(items) == 0:
        logger.warning("No valid items were loaded from iCliniq dataset")
    else:
        # Log complexity distribution
        complexity_dist: Dict[str, int] = {}
        for item in items:
            complexity_dist[item.complexity_level] = complexity_dist.get(item.complexity_level, 0) + 1

        avg_length = sum(len(item.medical_content) for item in items) / len(items)
        logger.info("iCliniq dataset statistics:")
        logger.info(f"  - Total items: {len(items)}")
        logger.info(f"  - Average content length: {avg_length:.1f} characters")
        logger.info(f"  - Complexity distribution: {complexity_dist}")

    return items


def load_cochrane_reviews(
    data_path: Union[str, Path], max_items: Optional[int] = None, auto_complexity: bool = True
) -> List[MEQBenchItem]:
    """Load Cochrane Reviews dataset and convert to MEQBenchItem objects.

    The Cochrane Reviews dataset contains evidence-based medical reviews and
    systematic analyses. This loader processes the dataset and optionally
    applies automatic complexity stratification.

    Args:
        data_path: Path to the Cochrane Reviews JSON file.
        max_items: Maximum number of items to load. If None, loads all items.
        auto_complexity: Whether to automatically calculate complexity levels.

    Returns:
        List of MEQBenchItem objects converted from Cochrane Reviews data.

    Raises:
        FileNotFoundError: If the data file does not exist.
        json.JSONDecodeError: If the JSON file is malformed.
        ValueError: If the data format is invalid.

    Example:
        ```python
        # Load Cochrane Reviews with complexity stratification
        items = load_cochrane_reviews('data/cochrane.json', max_items=300)
        ```
    """
    data_file = Path(data_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Cochrane Reviews file not found: {data_file}")

    logger.info(f"Loading Cochrane Reviews dataset from: {data_file}")

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in Cochrane Reviews file: {e}", e.doc, e.pos)

    if not isinstance(data, list):
        raise ValueError("Cochrane Reviews data must be a list of items")

    items = []
    items_to_process = data[:max_items] if max_items else data

    logger.info(f"Processing {len(items_to_process)} Cochrane Reviews items")

    for i, item_data in enumerate(items_to_process):
        try:
            if not isinstance(item_data, dict):
                logger.warning(f"Skipping invalid Cochrane Reviews item {i}: not a dictionary")
                continue

            # Extract fields - Cochrane typically has 'title', 'abstract', 'conclusions', 'background'
            title = item_data.get("title", "")
            abstract = item_data.get("abstract", "")
            conclusions = item_data.get("conclusions", item_data.get("main_results", ""))
            background = item_data.get("background", item_data.get("objectives", ""))
            item_id = item_data.get("id", f"cochrane_{i}")

            if not title.strip() and not abstract.strip():
                logger.warning(f"Skipping Cochrane Reviews item {i}: no title or abstract")
                continue

            # Create medical content from available fields
            medical_content_parts = []

            if title.strip():
                medical_content_parts.append(f"Title: {title.strip()}")

            if background.strip():
                medical_content_parts.append(f"Background: {background.strip()}")

            if abstract.strip():
                medical_content_parts.append(f"Abstract: {abstract.strip()}")

            if conclusions.strip():
                medical_content_parts.append(f"Conclusions: {conclusions.strip()}")

            if not medical_content_parts:
                logger.warning(f"Skipping Cochrane Reviews item {i}: no valid content")
                continue

            medical_content = "\n\n".join(medical_content_parts)

            # Calculate complexity level
            if auto_complexity:
                try:
                    complexity_level = calculate_complexity_level(medical_content)
                except Exception as e:
                    logger.warning(f"Error calculating complexity for item {i}: {e}, using 'advanced'")
                    complexity_level = "advanced"
            else:
                complexity_level = "advanced"  # Cochrane reviews tend to be more technical

            # Create MEQBenchItem
            item = MEQBenchItem(
                id=str(item_id),
                medical_content=medical_content,
                complexity_level=complexity_level,
                source_dataset="Cochrane Reviews",
                reference_explanations=None,
            )

            # Validate the item
            _validate_benchmark_item(item)

            items.append(item)

        except Exception as e:
            logger.error(f"Error processing Cochrane Reviews item {i}: {e}")
            continue

    logger.info(f"Successfully loaded {len(items)} MEQBenchItem objects from Cochrane Reviews")

    if len(items) == 0:
        logger.warning("No valid items were loaded from Cochrane Reviews dataset")
    else:
        # Log complexity distribution
        complexity_dist: Dict[str, int] = {}
        for item in items:
            complexity_dist[item.complexity_level] = complexity_dist.get(item.complexity_level, 0) + 1

        avg_length = sum(len(item.medical_content) for item in items) / len(items)
        logger.info("Cochrane Reviews dataset statistics:")
        logger.info(f"  - Total items: {len(items)}")
        logger.info(f"  - Average content length: {avg_length:.1f} characters")
        logger.info(f"  - Complexity distribution: {complexity_dist}")

    return items


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
