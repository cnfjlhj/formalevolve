#!/usr/bin/env python3
"""
Demo: Cycle Consistency with Real Models

This example shows how to use actual models for autoformalization.
You need to either:
1. Deploy models locally (e.g., with vLLM or Ollama)
2. Use an API endpoint

Modify the MODEL CONFIGURATION section below to match your setup.
"""

import sys
import logging
sys.path.insert(0, '..')

from src import (
    CycleConsistencyAutoformalization,
    OpenAICompatibleLLM,
    HuggingFaceLLM,
    Config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================
# MODEL CONFIGURATION - Modify this section for your setup
# ============================================================

# Option 1: Using vLLM or other OpenAI-compatible server
USE_API = True
FORMALIZER_BASE_URL = "http://localhost:8000/v1"  # Your vLLM/Ollama endpoint
FORMALIZER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

INFORMALIZER_BASE_URL = "http://localhost:8001/v1"  # Can be same or different
INFORMALIZER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Option 2: Using HuggingFace Transformers directly (requires GPU)
# USE_API = False
# FORMALIZER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-7B"  # Smaller for testing
# INFORMALIZER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# ============================================================


def create_models():
    """Create model instances based on configuration."""

    if USE_API:
        print(f"Using OpenAI-compatible API")
        print(f"  Formalizer: {FORMALIZER_BASE_URL}")
        print(f"  Informalizer: {INFORMALIZER_BASE_URL}")

        formalizer = OpenAICompatibleLLM(
            model=FORMALIZER_MODEL,
            base_url=FORMALIZER_BASE_URL,
        )
        informalizer = OpenAICompatibleLLM(
            model=INFORMALIZER_MODEL,
            base_url=INFORMALIZER_BASE_URL,
        )
    else:
        print(f"Using HuggingFace Transformers directly")
        print(f"  This requires significant GPU memory!")

        formalizer = HuggingFaceLLM(
            model_name=FORMALIZER_MODEL,
            device="auto",
        )
        informalizer = HuggingFaceLLM(
            model_name=INFORMALIZER_MODEL,
            device="auto",
        )

    return formalizer, informalizer


def main():
    print("=" * 70)
    print("Cycle Consistency Autoformalization - Real Models Demo")
    print("=" * 70)

    # Create models
    try:
        formalizer, informalizer = create_models()
    except Exception as e:
        print(f"\n❌ Failed to initialize models: {e}")
        print("\nPlease check:")
        print("  1. Your model server is running")
        print("  2. The URLs and model names are correct")
        print("  3. You have necessary dependencies installed")
        return

    # Create config
    config = Config()
    config.verbose = True
    config.model.num_candidates = 5
    config.model.temperature = 0.7

    # Initialize cycle consistency
    cc = CycleConsistencyAutoformalization(
        formalizer=formalizer,
        informalizer=informalizer,
        config=config,
    )

    # Test cases from miniF2F-style problems
    test_cases = [
        "For all positive integers n, n squared is greater than or equal to n.",
        "There exist integers x, y, z greater than 0 such that x squared plus y squared equals z squared.",
        "If a divides b and b divides c, then a divides c.",
        "The sum of the first n positive integers is n times n plus 1 divided by 2.",
    ]

    print("\n" + "=" * 70)
    print("Running Autoformalization")
    print("=" * 70)

    for i, informal in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}: {informal}")
        print("─" * 70)

        try:
            result = cc.autoformalize(informal)

            print(f"\n✓ Best formalization (score={result.best_score:.2f}):")
            print(f"  {result.best_formalization}")

            if len(result.all_candidates) > 1:
                print(f"\n  Other candidates:")
                for cand in result.all_candidates[1:3]:  # Show top 3
                    print(f"    [{cand.rank}] score={cand.normalized_score:.2f}")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
