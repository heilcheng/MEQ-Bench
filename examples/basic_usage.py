"""
Basic usage example for MEQ-Bench with Hugging Face model integration
"""

import os
import sys
import logging
from typing import Optional
import torch
import warnings

# Suppress some common warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.benchmark import MEQBench
from src.evaluator import MEQBenchEvaluator
from src.config import config

# Set up logging
logger = logging.getLogger('meq_bench.examples')


def generate_with_huggingface(
    prompt: str, 
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
) -> str:
    """Generate text using a Hugging Face model.
    
    This function loads a pretrained language model from the Hugging Face Hub
    and generates a response to the given prompt. It's designed to work with
    instruction-tuned models that can follow the MEQ-Bench prompt format.
    
    Args:
        prompt: The input prompt containing medical content and audience instructions.
        model_name: Name of the Hugging Face model to use. Popular options include:
            - "mistralai/Mistral-7B-Instruct-v0.2" (recommended, ~7B parameters)
            - "microsoft/DialoGPT-medium" (smaller, faster)
            - "meta-llama/Llama-2-7b-chat-hf" (requires approval)
            - "google/flan-t5-large" (encoder-decoder architecture)
            
    Returns:
        Generated text response as a string.
        
    Note:
        This function requires the transformers library to be installed:
        pip install transformers torch
        
        For larger models, ensure you have sufficient GPU memory or use
        CPU inference (which will be slower). The function automatically
        detects available devices.
        
    Example:
        ```python
        # Generate explanation using Mistral-7B
        prompt = "Medical Information: Diabetes is high blood sugar..."
        response = generate_with_huggingface(prompt, "mistralai/Mistral-7B-Instruct-v0.2")
        
        # Use a smaller model for faster inference
        response = generate_with_huggingface(prompt, "microsoft/DialoGPT-medium")
        ```
    """
    try:
        # Import transformers here to make it optional
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from transformers import logging as hf_logging
        
        # Reduce transformers logging verbosity
        hf_logging.set_verbosity_error()
        
    except ImportError as e:
        logger.error("transformers library not installed. Install with: pip install transformers torch")
        raise ImportError(
            "transformers library is required for Hugging Face model integration. "
            "Install with: pip install transformers torch"
        ) from e
    
    logger.info(f"Loading Hugging Face model: {model_name}")
    
    try:
        # Determine device (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present (needed for some models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings
        logger.info("Loading model...")
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        # For CPU inference or limited GPU memory, use smaller precision
        if device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        ).to(device)
        
        # Create text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,  # 0 for first GPU, -1 for CPU
            return_full_text=False,  # Only return generated text, not input prompt
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Generation parameters - adjust these for different models/quality trade-offs
        generation_params = {
            "max_new_tokens": 800,  # Maximum tokens to generate
            "temperature": 0.7,     # Controls randomness (0.1 = deterministic, 1.0 = creative)
            "top_p": 0.9,          # Nucleus sampling parameter
            "do_sample": True,      # Enable sampling for more diverse outputs
            "num_return_sequences": 1,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        # For instruction-tuned models like Mistral, format the prompt appropriately
        if "instruct" in model_name.lower() or "chat" in model_name.lower():
            # Use instruction format for better results
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            # Use prompt as-is for base models
            formatted_prompt = prompt
        
        logger.info("Generating response...")
        
        # Generate response
        result = generator(formatted_prompt, **generation_params)
        
        # Extract generated text
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '')
        else:
            generated_text = str(result)
        
        # Clean up the response
        generated_text = generated_text.strip()
        
        # Remove potential instruction formatting artifacts
        if generated_text.startswith('[/INST]'):
            generated_text = generated_text[7:].strip()
        
        logger.info(f"Generated {len(generated_text)} characters")
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error during model generation: {e}")
        
        # Fallback to dummy response to keep the example working
        logger.warning("Falling back to dummy response due to model loading error")
        return dummy_model_function(prompt)
        
    finally:
        # Clean up GPU memory if used
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def dummy_model_function(prompt: str) -> str:
    """
    Dummy model function for demonstration when real models aren't available.
    In practice, this would call your actual LLM or use the Hugging Face function above.
    """
    return """
    For a Physician: The patient presents with essential hypertension, likely multifactorial etiology including genetic predisposition and lifestyle factors. Recommend ACE inhibitor initiation with monitoring of renal function and electrolytes. Consider cardiovascular risk stratification.

    For a Nurse: Monitor blood pressure readings twice daily, document trends. Educate patient on medication compliance, dietary sodium restriction, and importance of regular follow-up. Watch for signs of medication side effects.

    For a Patient: Your blood pressure is higher than normal, which means your heart is working harder than it should. We'll start you on medication to help lower it. It's important to take your medicine every day and eat less salt.

    For a Caregiver: Help ensure they take their blood pressure medication at the same time each day. Monitor for dizziness or fatigue. Encourage a low-salt diet and regular gentle exercise. Call the doctor if blood pressure readings are very high.
    """


def main():
    """Main example function demonstrating both dummy and Hugging Face models"""
    print("MEQ-Bench Basic Usage Example with Hugging Face Integration")
    print("=" * 60)
    
    # Check if running in non-interactive environment (CI/CD)
    is_interactive = os.isatty(sys.stdin.fileno())
    
    # Initialize benchmark
    bench = MEQBench()
    
    # Create sample dataset
    print("Creating sample dataset...")
    sample_items = bench.create_sample_dataset()
    
    # Add items to benchmark
    for item in sample_items:
        bench.add_benchmark_item(item)
    
    # Get benchmark statistics
    stats = bench.get_benchmark_stats()
    print(f"\nBenchmark Statistics:")
    print(f"Total items: {stats['total_items']}")
    print(f"Complexity distribution: {stats['complexity_distribution']}")
    print(f"Source distribution: {stats['source_distribution']}")
    
    # Test single explanation generation
    print("\nGenerating explanations for sample content...")
    medical_content = "Diabetes is a condition where blood sugar levels are too high. It requires careful management through diet, exercise, and sometimes medication."
    
    # Choose model based on environment
    if is_interactive:
        # Ask user which model to use (interactive mode)
        print("\nChoose model for explanation generation:")
        print("1. Dummy model (fast, for testing)")
        print("2. Hugging Face model (requires transformers library)")
        
        choice = input("Enter choice (1 or 2, default=1): ").strip()
        
        if choice == "2":
            print("\nUsing Hugging Face model...")
            print("Note: This requires 'transformers' and 'torch' libraries:")
            print("pip install transformers torch")
            
            # Option to specify model name
            model_choice = input("\nChoose model (or press Enter for default):\n"
                               "1. mistralai/Mistral-7B-Instruct-v0.2 (default, ~7B params)\n"
                               "2. microsoft/DialoGPT-medium (smaller, faster)\n"
                               "3. Custom model name\n"
                               "Choice: ").strip()
            
            if model_choice == "2":
                model_name = "microsoft/DialoGPT-medium"
            elif model_choice == "3":
                model_name = input("Enter model name: ").strip()
            else:
                model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            
            print(f"\nLoading model: {model_name}")
            print("This may take a few minutes on first run...")
            
            # Create model function with specified model
            def hf_model_function(prompt: str) -> str:
                return generate_with_huggingface(prompt, model_name)
            
            model_function = hf_model_function
            model_type = f"Hugging Face ({model_name})"
            
        else:
            print("\nUsing dummy model...")
            model_function = dummy_model_function
            model_type = "Dummy model"
    else:
        # Non-interactive mode (CI/CD) - use dummy model by default
        print("\nNon-interactive environment detected, using dummy model...")
        choice = "1"  # Set for later reference
        model_function = dummy_model_function
        model_type = "Dummy model"
    
    print(f"\nGenerating explanations with {model_type}...")
    explanations = bench.generate_explanations(medical_content, model_function)
    
    print("\nGenerated Explanations:")
    for audience, explanation in explanations.items():
        print(f"\n{audience.upper()}:")
        # Show more of the explanation for real models
        max_length = 300 if choice == "2" else 200
        if len(explanation) > max_length:
            print(explanation[:max_length] + "...")
        else:
            print(explanation)
    
    # Evaluate explanations
    print("\nEvaluating explanations...")
    try:
        evaluator = MEQBenchEvaluator()
        results = evaluator.evaluate_all_audiences(medical_content, explanations)
        
        print("\nEvaluation Results:")
        for audience, score in results.items():
            print(f"\n{audience.upper()}:")
            print(f"  Readability: {score.readability:.3f}")
            print(f"  Terminology: {score.terminology:.3f}")
            print(f"  Safety: {score.safety:.3f}")
            print(f"  Coverage: {score.coverage:.3f}")
            print(f"  Quality: {score.quality:.3f}")
            print(f"  Overall: {score.overall:.3f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("This might be due to missing dependencies or configuration.")
    
    # Option to run full benchmark evaluation
    if is_interactive:
        run_full = input("\nRun full benchmark evaluation? (y/N): ").strip().lower()
    else:
        # In non-interactive mode, run a minimal evaluation for testing
        run_full = 'y'
        print("\nNon-interactive mode: Running minimal benchmark evaluation...")
    
    if run_full == 'y':
        if is_interactive:
            print("\nRunning full benchmark evaluation...")
            print("Note: This may take several minutes with real models...")
            max_items = 2
        else:
            # In CI/CD, run a very minimal test
            max_items = 1
        
        try:
            full_results = bench.evaluate_model(model_function, max_items=max_items)
            
            print("\nFull Benchmark Results Summary:")
            summary = full_results['summary']
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.3f}")
            
            # Save results
            if is_interactive:
                output_path = f"sample_results_{model_type.replace(' ', '_').replace('(', '').replace(')', '')}.json"
                bench.save_results(full_results, output_path)
                print(f"\nResults saved to: {output_path}")
            else:
                print("Results generated successfully (not saved in CI/CD mode)")
            
        except Exception as e:
            print(f"Full evaluation failed: {e}")
            if not is_interactive:
                # In CI/CD mode, we want to fail if the evaluation doesn't work
                raise
    
    print("\n" + "=" * 60)
    if is_interactive:
        print("Example completed!")
        print("\nTo use Hugging Face models in your own code:")
        print("1. Install required libraries: pip install transformers torch")
        print("2. Use the generate_with_huggingface function")
        print("3. Choose models based on your hardware:")
        print("   - CPU: smaller models like DialoGPT-medium")
        print("   - GPU: larger models like Mistral-7B-Instruct-v0.2")
        print("4. Adjust generation parameters for quality vs speed")
    else:
        print("Integration test completed successfully!")
        print("The MEQ-Bench framework is working correctly in non-interactive mode.")


if __name__ == "__main__":
    main()