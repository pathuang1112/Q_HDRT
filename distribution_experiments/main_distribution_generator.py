

import sys
import argparse
from typing import Dict
import random
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Generate supplemental test cases')
    parser.add_argument('--qubit_counts', type=str, default='4,7,8,9,10,12',
                        help='Qubit count list, Comma-separated')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()


    qubit_counts = [int(x.strip()) for x in args.qubit_counts.split(',')]


    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    print("=" * 60)
    print("Supplemental testcase generator")
    print("=" * 60)
    print(f"Qubit count: {qubit_counts}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 60)


    print("\n1. Analyzing existing test cases...")
    from testcase_analyzer import analyze_testcases
    analysis_results = analyze_testcases(qubit_counts)

    if not analysis_results:
        print("No testcase file found")
        return


    print("\n2. Compute distribution requirements...")
    from distribution_calculator import calculate_distribution_needs
    distribution_needs = calculate_distribution_needs(analysis_results)


    print("\n3. Compute supplemental requirements...")
    from testcase_generation_manager import TestcaseGenerationManager

    manager = TestcaseGenerationManager(qubit_counts)
    supplement_needs = manager.calculate_supplement_needs(analysis_results, distribution_needs)


    print("\n4. Generate supplemental test cases...")
    print("=" * 60)

    generation_results = {}

    for n_qubits in qubit_counts:
        if n_qubits not in supplement_needs:
            continue

        print(f"\nProcessing {n_qubits} qubits...")

        try:
            result = manager.generate_supplement_testcases(
                n_qubits=n_qubits,
                supplement_needs=supplement_needs[n_qubits],
                analysis_data=analysis_results[n_qubits]
            )
            generation_results[n_qubits] = result
        except Exception as e:
            print(f"  Failed to generate supplemental test cases: {e}")
            import traceback
            traceback.print_exc()


    print("\n" + "=" * 60)
    print("Generation summary")
    print("=" * 60)

    for n_qubits, result in generation_results.items():
        if result['generated'] > 0:
            print(f"{n_qubits} qubits:")
            print(f"  Generated file: {result['filepath']}")
            print(f"  Generated count: {result['generated']}")
            counts = result.get('counts', {})
            print(f"  Computational basis: {counts.get('classical', 0)}")
            print(f"  Superposition: {counts.get('superposition', 0)}")
            print(f"  Mixed states: {counts.get('mixed', 0)}")
        else:
            print(f"{n_qubits} qubits: No supplemental test cases required")

    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("Supplemental test cases saved to ./TestCases/supplements/")
    print("=" * 60)


    print("\nUsage:")
    print("1. in the experiment, Using get_testcases_for_distribution() function to obtain test cases")
    print("2. Example:")
    print("""
    from use_supplement_cases import get_testcases_for_distribution

    result = get_testcases_for_distribution(
        n_qubits=10,
        distribution_name="quantum_dominant",
        classical_needed=205,
        superposition_needed=819,
        mixed_needed=0
    )
    """)


if __name__ == "__main__":
    main()
