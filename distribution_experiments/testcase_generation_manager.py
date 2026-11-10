import os
import sys
import json
import random
from typing import Dict, List, Set, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
import hashlib
import pickle
import gzip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testcase_io


class TestcaseGenerationManager:
    def __init__(self, qubit_counts: List[int] = [4, 7, 8, 9, 10], compress: bool = True):
        self.qubit_counts = qubit_counts
        self.compress = compress


        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_script_dir)


        self.testcases_dir = os.path.join(self.project_root, "TestCases")


        self.supplement_dir = os.path.join(self.project_root, "TestCases", "supplements")
        os.makedirs(self.supplement_dir, exist_ok=True)

        print(f"Project root: {self.project_root}")
        print(f"Testcase directory: {self.testcases_dir}")
        print(f"Binary supplemental testcase directory: {self.supplement_dir}")

    def calculate_supplement_needs(self, analysis_results: Dict, distribution_needs: Dict) -> Dict:
        supplement_needs = {}

        for n_qubits in self.qubit_counts:
            if n_qubits not in analysis_results or n_qubits not in distribution_needs:
                continue

            stats = analysis_results[n_qubits]['stats']


            max_needs = {
                'classical': 0,
                'superposition': 0,
                'mixed': 0
            }


            for dist_name, dist_data in distribution_needs[n_qubits].items():
                for case_type in ['classical', 'superposition', 'mixed']:
                    if dist_data['new_needed'][case_type] > max_needs[case_type]:
                        max_needs[case_type] = dist_data['new_needed'][case_type]

            supplement_needs[n_qubits] = {
                'stats': stats,
                'max_supplement': max_needs,
                'total_needed': analysis_results[n_qubits]['total_needed']
            }

            print(f"\n{n_qubits} qubits - Supplement requirements:")
            print("-" * 40)
            print(f"  Additional computational-basis states required: {max_needs['classical']}")
            print(f"  Additional superposition states required: {max_needs['superposition']}")
            print(f"  Additional mixed states required: {max_needs['mixed']}")

        return supplement_needs

    def generate_supplement_testcases(self, n_qubits: int, supplement_needs: Dict,
                                      analysis_data: Dict) -> Dict:

        print(f"\n{'=' * 60}")
        print(f"Generating supplemental test cases for {n_qubits} qubits(binary format)")
        print(f"{'=' * 60}")

        max_needs = supplement_needs['max_supplement']


        if all(v == 0 for v in max_needs.values()):
            print("  No supplemental test cases needed")
            return {'generated': 0, 'filepath': None}


        existing_cases = self._load_existing_testcases(n_qubits)
        used_names = {case['name'] for case in existing_cases}


        generated_cases = []


        if max_needs['classical'] > 0:
            print(f"\n  Generate {max_needs['classical']} computational-basis states...")
            classical_cases = self.generate_new_classical_cases(
                n_qubits, max_needs['classical'], used_names
            )
            generated_cases.extend(classical_cases)


        if max_needs['superposition'] > 0:
            print(f"\n  Generate {max_needs['superposition']} superposition states...")
            superposition_cases = self.generate_new_superposition_cases(
                n_qubits, max_needs['superposition'], used_names
            )
            generated_cases.extend(superposition_cases)


        if max_needs['mixed'] > 0:
            print(f"\n  Generate {max_needs['mixed']} mixed states...")
            mixed_cases = self.generate_new_mixed_cases(
                n_qubits, max_needs['mixed'], used_names
            )
            generated_cases.extend(mixed_cases)


        output_file = os.path.join(self.supplement_dir, f"supplement_{n_qubits}qubits")

        if self.compress:
            output_file += ".pkl.gz"
        else:
            output_file += ".pkl"


        binary_data = []
        for name, state, attributes in generated_cases:

            if isinstance(state, Statevector):
                state_type = "statevector"
                data_real = state.data.real.astype(np.float32)
                data_imag = state.data.imag.astype(np.float32)
            elif isinstance(state, DensityMatrix):
                state_type = "density_matrix"
                data_real = state.data.real.astype(np.float32)
                data_imag = state.data.imag.astype(np.float32)
            else:
                continue


            binary_record = {
                'name': name,
                'attributes': attributes,
                'type': state_type,
                'data_real': data_real,
                'data_imag': data_imag
            }
            binary_data.append(binary_record)


        if self.compress:
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(binary_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(output_file, 'wb') as f:
                pickle.dump(binary_data, f, protocol=pickle.HIGHEST_PROTOCOL)


        file_size = os.path.getsize(output_file)
        print(f"\n  [OK] Generated binary supplemental testcase file: {output_file}")
        print(f"  [OK] Generated in total {len(generated_cases)} test cases")
        print(f"  [OK] File size: {file_size / 1024 / 1024:.2f} MB")
        print(f"  [OK] Computational basis: {max_needs['classical']}")
        print(f"  [OK] Superposition: {max_needs['superposition']}")
        print(f"  [OK] Mixed states: {max_needs['mixed']}")

        return {
            'generated': len(generated_cases),
            'filepath': output_file,
            'counts': max_needs,
            'size_mb': file_size / 1024 / 1024
        }

    def load_binary_testcases(self, filename: str) -> List[Tuple[str, any, Dict]]:
        if not os.path.exists(filename):
            print(f"File does not exist: {filename}")
            return []

        if filename.endswith('.gz'):
            with gzip.open(filename, 'rb') as f:
                binary_data = pickle.load(f)
        else:
            with open(filename, 'rb') as f:
                binary_data = pickle.load(f)

        test_cases = []
        for record in binary_data:
            name = record['name']
            attributes = record['attributes']
            state_type = record['type']
            data_real = record['data_real']
            data_imag = record['data_imag']


            if state_type == "statevector":
                data = data_real + 1j * data_imag
                state = Statevector(data)
            elif state_type == "density_matrix":
                data = data_real + 1j * data_imag
                state = DensityMatrix(data)
            else:
                continue

            test_cases.append((name, state, attributes))

        return test_cases

    def _load_existing_testcases(self, n_qubits: int) -> List[Dict]:
        filename = os.path.join(self.testcases_dir, f"quantum_test_cases_{n_qubits}qubits.json")

        try:
            filename = testcase_io.resolve_testcase_file(filename)
        except FileNotFoundError:
            return []

        return [
            {'name': name, 'attributes': attributes}
            for name, _state, attributes in testcase_io.load_test_cases(filename)
        ]

    def generate_new_classical_cases(self, n_qubits: int, n_needed: int,
                                     used_names: Set[str]) -> List[Tuple[str, Statevector, Dict]]:

        if n_needed == 0:
            return []

        new_cases = []
        generated_count = 0


        for i in range(2 ** n_qubits):
            if generated_count >= n_needed:
                break


            binary_str = format(i, f'0{n_qubits}b')
            name = f"classical_{binary_str}"


            if name in used_names:
                continue


            qc = QuantumCircuit(n_qubits)
            for j, bit in enumerate(binary_str):
                if bit == '1':
                    qc.x(j)

            state = Statevector(qc)

            attributes = {
                'type': 'classical',
                'subtype': 'computational_basis',
                'binary_state': binary_str,
                'num_qubits': n_qubits,
                'is_mixed': False
            }

            new_cases.append((name, state, attributes))
            used_names.add(name)
            generated_count += 1

        if generated_count < n_needed:
            print(f"    Warning: Insufficient computational-basis states, Required {n_needed}, Generated only {generated_count} ")

        return new_cases

    def generate_new_superposition_cases(self, n_qubits: int, n_needed: int,
                                         used_names: Set[str]) -> List[Tuple[str, Statevector, Dict]]:

        if n_needed == 0:
            return []

        new_cases = []
        attempt = 0
        max_attempts = n_needed * 10


        existing_hashes = set()

        while len(new_cases) < n_needed and attempt < max_attempts:
            attempt += 1


            base_state = np.random.randint(0, 2, n_qubits)
            qc = QuantumCircuit(n_qubits)

            for q, val in enumerate(base_state):
                if val == 1:
                    qc.x(q)


            superposed_qubits = random.sample(range(n_qubits), random.randint(1, n_qubits))
            gate_dict = {}
            rotation_angles = {}

            for q in superposed_qubits:
                if random.random() > 0.5:

                    qc.h(q)
                    gate_dict[q] = "h"
                else:

                    theta = np.random.uniform(0.1, np.pi - 0.1)
                    qc.ry(theta, q)
                    gate_dict[q] = f"ry({theta:.4f})"
                    rotation_angles[q] = theta


            state = Statevector(qc)


            state_hash = self._calculate_state_hash(state)
            if state_hash in existing_hashes:
                continue
            existing_hashes.add(state_hash)


            name = f"supplement_superposition_{len(used_names)}_{attempt}"
            if name in used_names:
                continue


            attributes = {
                'type': 'superposition',
                'subtype': 'supplement_random',
                'num_qubits': n_qubits,
                'superposed_qubits': sorted(superposed_qubits),
                'gate_distribution': gate_dict,
                'rotation_angles': rotation_angles,
                'is_mixed': False
            }
            new_cases.append((name, state, attributes))
            used_names.add(name)

        if len(new_cases) < n_needed:
            print(f"    Warning: Insufficient superposition states, Required {n_needed}, Generated only {len(new_cases)} ")

        return new_cases

    def generate_new_mixed_cases(self, n_qubits: int, n_needed: int,
                                 used_names: Set[str]) -> List[Tuple[str, DensityMatrix, Dict]]:

        if n_needed == 0:
            return []

        new_cases = []
        dim = 2 ** n_qubits

        print(f"    Starting generation {n_needed}  {n_qubits} qubit mixed states...")

        for i in range(n_needed):

            density_matrix = self._generate_random_density_matrix(dim)

            if density_matrix is None:
                continue


            purity = density_matrix.purity()


            if purity > 0.99:

                density_matrix = self._add_noise_to_density_matrix(density_matrix)
                purity = density_matrix.purity()


            name = f"supplement_mixed_{len(used_names)}_{i}"
            if name in used_names:
                continue


            attributes = {
                'type': 'mixed',
                'subtype': 'random_density_matrix',
                'num_qubits': n_qubits,
                'purity': float(purity),
                'dimension': dim,
                'is_mixed': True
            }
            state = density_matrix
            new_cases.append((name, state, attributes))
            used_names.add(name)

            if (i + 1) % 10 == 0:
                print(f"      Generated {i + 1}/{n_needed} mixed states")

        print(f"    Generated successfully {len(new_cases)} mixed states")
        return new_cases

    def _generate_random_density_matrix(self, dim: int) -> DensityMatrix:
        try:


            random_complex = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)


            hermitian = (random_complex + random_complex.conj().T) / 2


            positive_semidefinite = hermitian @ hermitian.conj().T


            trace = np.trace(positive_semidefinite)
            if abs(trace) < 1e-10:
                return None

            density_matrix_data = positive_semidefinite / trace


            density_matrix = DensityMatrix(density_matrix_data)


            if not self._is_valid_density_matrix(density_matrix):
                return None

            return density_matrix

        except Exception as e:
            print(f"      Failed to generate density matrix: {e}")
            return None

    def _add_noise_to_density_matrix(self, density_matrix: DensityMatrix) -> DensityMatrix:
        dim = density_matrix.dim
        data = density_matrix.data


        noise_level = 0.1
        noise = noise_level * (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))


        noise = (noise + noise.conj().T) / 2


        noisy_data = data + noise


        trace = np.trace(noisy_data)
        noisy_data = noisy_data / trace

        return DensityMatrix(noisy_data)

    def _is_valid_density_matrix(self, density_matrix: DensityMatrix, tolerance: float = 1e-10) -> bool:
        data = density_matrix.data


        trace = np.trace(data)
        if abs(trace - 1.0) > tolerance:
            return False


        if not np.allclose(data, data.conj().T, atol=tolerance):
            return False


        eigenvalues = np.linalg.eigvalsh(data)
        if np.any(eigenvalues < -tolerance):
            return False

        return True

    def _calculate_state_hash(self, state: Statevector) -> str:
        data_bytes = state.data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:16]

    def _calculate_density_hash(self, density: DensityMatrix) -> str:
        data_bytes = density.data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:16]


if __name__ == "__main__":

    manager = TestcaseGenerationManager(qubit_counts=[7, 10], compress=True)


    test_file = "./TestCases/supplements/supplement_7qubits.pkl.gz"
    if os.path.exists(test_file):
        test_cases = manager.load_binary_testcases(test_file)
        print(f"Loaded {len(test_cases)} test cases from {test_file}")
