                      
"""
Demo: Test the pipeline with dummy models

This demonstrates the flow without requiring actual models.
Run this first to verify the code structure works.
"""

import sys
sys.path.insert(0, '..')

from src import (
    CycleConsistencyAutoformalization,
    DummyLLM,
    Config,
)

def main():
    print("=" * 60)
    print("Cycle Consistency Autoformalization - Dummy Demo")
    print("=" * 60)

                                     
    formalizer = DummyLLM()
    informalizer = DummyLLM()

                   
    config = Config()
    config.verbose = True
    config.model.num_candidates = 3

                
    cc = CycleConsistencyAutoformalization(
        formalizer=formalizer,
        informalizer=informalizer,
        config=config,
    )

                             
    informal = "For all positive integers n, n squared is greater than or equal to n."

    print(f"\nInformal statement: {informal}")
    print("-" * 60)

                           
    result = cc.autoformalize(informal)

                   
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)

    print(f"\nBest formalization:")
    print(f"  {result.best_formalization}")
    print(f"\nBest score: {result.best_score:.2f}")

    print(f"\nAll candidates (ranked by cycle consistency score):")
    for cand in result.all_candidates:
        print(f"  [{cand.rank}] score={cand.normalized_score:.2f}: {cand.formal_statement[:50]}...")

    print("\n✓ Pipeline works correctly with dummy models!")
    print("  Now you can plug in real models.")


if __name__ == "__main__":
    main()
