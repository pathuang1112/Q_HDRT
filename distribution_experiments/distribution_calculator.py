from typing import Dict, List, Tuple
import math


DISTRIBUTIONS = [
    {
        'name': 'quantum_dominant',
        'description': 'Quantum-state dominant',
        'classical_ratio': 0.2,
        'superposition_ratio': 0.8,
        'mixed_ratio': 0.0
    },


    {
        'name': 'only_classical',
        'description': 'Computational-basis states only',
        'classical_ratio': 1.0,
        'superposition_ratio': 0.0,
        'mixed_ratio': 0.0
    },
    {
        'name': 'classical_dominant',
        'description': 'Computational-basis dominant',
        'classical_ratio': 0.7,
        'superposition_ratio': 0.3,
        'mixed_ratio': 0.0
    },
    {
        'name': 'balanced',
        'description': 'Balanced distribution',
        'classical_ratio': 0.5,
        'superposition_ratio': 0.5,
        'mixed_ratio': 0.0
    },
    {
        'name': 'with_mixed',
        'description': 'Include mixed states',
        'classical_ratio': 1 / 3,
        'superposition_ratio': 1 / 3,
        'mixed_ratio': 1 / 3
    }
]


def calculate_distribution_needs(analysis_results: Dict) -> Dict:
    distribution_needs = {}

    for n_qubits, data in analysis_results.items():
        stats = data['stats']
        total_needed = data['total_needed']

        print(f"\n{n_qubits} qubits - Distribution requirements:")
        print("=" * 60)

        distribution_needs[n_qubits] = {}

        for dist in DISTRIBUTIONS:

            classical_needed = math.ceil(total_needed * dist['classical_ratio'])
            superposition_needed = math.ceil(total_needed * dist['superposition_ratio'])
            mixed_needed = math.ceil(total_needed * dist['mixed_ratio'])


            actual_total = classical_needed + superposition_needed + mixed_needed
            if actual_total > total_needed:

                if classical_needed >= superposition_needed and classical_needed >= mixed_needed:
                    classical_needed -= (actual_total - total_needed)
                elif superposition_needed >= classical_needed and superposition_needed >= mixed_needed:
                    superposition_needed -= (actual_total - total_needed)
                else:
                    mixed_needed -= (actual_total - total_needed)


            new_classical = max(0, classical_needed - stats['classical'])
            new_superposition = max(0, superposition_needed - stats['superposition'])
            new_mixed = max(0, mixed_needed - stats['mixed'])


            distribution_needs[n_qubits][dist['name']] = {
                'description': dist['description'],
                'needs': {
                    'classical': classical_needed,
                    'superposition': superposition_needed,
                    'mixed': mixed_needed,
                    'total': total_needed
                },
                'available': {
                    'classical': stats['classical'],
                    'superposition': stats['superposition'],
                    'mixed': stats['mixed']
                },
                'new_needed': {
                    'classical': new_classical,
                    'superposition': new_superposition,
                    'mixed': new_mixed
                }
            }


            print(f"\n{dist['description']} ({dist['name']}):")
            print(f"  Required: Computational basis{classical_needed}, Superposition{superposition_needed}, Mixed states{mixed_needed}")
            print(f"  Existing: Computational basis{stats['classical']}, Superposition{stats['superposition']}, Mixed states{stats['mixed']}")
            print(f"  Need to generate: Computational basis{new_classical}, Superposition{new_superposition}, Mixed states{new_mixed}")

    return distribution_needs