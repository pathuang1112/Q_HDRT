import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, DensityMatrix, Operator
from qiskit.providers.aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error, pauli_error
import json
import itertools
import os
import random
import hashlib
import argparse
from typing import List, Tuple, Dict, Set

import testcase_io


def generate_classical_states(n_qubits):
    states = []
    for i in range(2 ** n_qubits):
        binary_str = format(i, f'0{n_qubits}b')
        qc = QuantumCircuit(n_qubits)
        for qubit, bit in enumerate(binary_str):
            if bit == '1':
                qc.x(qubit)
        state = Statevector(qc)
        attributes = {
            "type": "classical",
            "binary_state": binary_str,
            "num_qubits": n_qubits,
            "is_mixed": False
        }
        states.append((f"classical_{binary_str}", state, attributes))
    return states

def generate_random_base_state(n_qubits):
    base_state = np.random.choice([0, 1], size=n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i, bit in enumerate(base_state):
        if bit == 1:
            qc.x(i)
    return qc, base_state

def generate_superposition_states(n_qubits):
    states = []


    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    state = Statevector(qc)
    attributes = {
        "type": "superposition",
        "subtype": "uniform_all",
        "num_qubits": n_qubits,
        "superposed_qubits": list(range(n_qubits)),
        "gates_used": ["h"] * n_qubits,
        "is_mixed": False
    }
    states.append(("superposition_uniform_all", state, attributes))


    for k in range(1, min(4, n_qubits)):
        for qubits in itertools.combinations(range(n_qubits), k):
            base_qc, base_state = generate_random_base_state(n_qubits)
            qc = base_qc.copy()
            qc.h(qubits)
            state = Statevector(qc)
            attributes = {
                "type": "superposition",
                "subtype": f"uniform_partial_{k}qubits",
                "base_state": base_state.tolist(),
                "num_qubits": n_qubits,
                "superposed_qubits": list(qubits),
                "static_qubits": [q for q in range(n_qubits) if q not in qubits],
                "gates_used": ["h" if q in qubits else None for q in range(n_qubits)],
                "is_mixed": False
            }
            states.append((f"superposition_partial_{'_'.join(map(str, qubits))}", state, attributes))


    for k in range(1, min(4, n_qubits + 1)):
        for qubits in itertools.combinations(range(n_qubits), k):
            base_qc, base_state = generate_random_base_state(n_qubits)
            qc = base_qc.copy()
            angles = []
            for q in qubits:
                theta = np.random.uniform(0, 2 * np.pi)
                angles.append(theta)
                qc.ry(theta, q)
            state = Statevector(qc)

            angle_dict = {q: angles[i] for i, q in enumerate(qubits)}
            attributes = {
                "type": "superposition",
                "subtype": f"nonuniform_ry_{k}qubits",
                "base_state": base_state.tolist(),
                "num_qubits": n_qubits,
                "superposed_qubits": list(qubits),
                "rotation_angles": angle_dict,
                "gates_used": [f"ry({angle_dict[q]})" if q in qubits else None
                               for q in range(n_qubits)],
                "is_mixed": False
            }
            states.append((f"superposition_ry_{'_'.join(map(str, qubits))}", state, attributes))


    for k in range(1,  min(4, n_qubits)):
        for qubits in itertools.combinations(range(n_qubits), k):
            base_qc, base_state = generate_random_base_state(n_qubits)
            qc = base_qc.copy()
            gate_choices = []
            angles = []
            angle_dict = {}
            gate_dict = {}
            for i, q in enumerate(qubits):
                if np.random.rand() > 0.5:
                    qc.h(q)
                    gate_choices.append("h")
                    angles.append(None)
                    gate_dict[q] = "h"
                else:
                    theta = np.random.uniform(0, 2 * np.pi)
                    qc.ry(theta, q)
                    gate_choices.append("ry")
                    angles.append(theta)
                    angle_dict[q] = theta
                    gate_dict[q] = f"ry({theta})"
            state = Statevector(qc)
            attributes = {
                "type": "superposition",
                "subtype": f"mixed_h_ry_{k}qubits",
                "base_state": base_state.tolist(),
                "num_qubits": n_qubits,
                "superposed_qubits": list(qubits),
                "gate_distribution": gate_dict,
                "rotation_angles": angle_dict,
                "gates_used": [gate_dict.get(q, None) for q in range(n_qubits)],
                "is_mixed": False
            }
            states.append((f"superposition_mixed_{'_'.join(map(str, qubits))}", state, attributes))

    return states


def generate_entangled_states(n_qubits):
    states = []


    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qc = QuantumCircuit(n_qubits)
            qc.h(i)
            qc.cx(i, j)
            state = Statevector(qc)
            attributes = {
                "type": "entangled",
                "subtype": "bell",
                "num_qubits": n_qubits,
                "entangled_qubits": [i, j],
                "is_mixed": False
            }
            states.append((f"entangled_bell_{i}_{j}", state, attributes))


    if n_qubits >= 2:
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
        state = Statevector(qc)
        attributes = {
            "type": "entangled",
            "subtype": "ghz",
            "num_qubits": n_qubits,
            "entangled_qubits": list(range(n_qubits)),
            "is_mixed": False
        }
        states.append(("entangled_ghz", state, attributes))


        if n_qubits >= 3:

            qc = QuantumCircuit(n_qubits)
            entangled_qubits = random.sample(range(n_qubits), 3)
            independent_qubits = [q for q in range(n_qubits) if q not in entangled_qubits]
            qc.h(entangled_qubits[0])
            qc.cx(entangled_qubits[0], entangled_qubits[1])
            qc.cx(entangled_qubits[1], entangled_qubits[2])
            state = Statevector(qc)
            attributes = {
                "type": "entangled",
                "subtype": "layered",
                "entangled_qubits": [[entangled_qubits[0], entangled_qubits[1]], [entangled_qubits[1], entangled_qubits[2]]],
                "independent_qubits": independent_qubits,
                "num_qubits": n_qubits,
                "is_mixed": False
            }
            states.append(("entangled_partial_{entangled_qubits}", state, attributes))


        if n_qubits >= 3:

            target_qubits = random.sample(range(n_qubits), 3)
            target_qubits.sort()


            w_state = (
                    Statevector.from_label("001") +
                    Statevector.from_label("010") +
                    Statevector.from_label("100")
            )

            w_state = w_state / np.sqrt(3)


            qc = QuantumCircuit(n_qubits)
            qc.initialize(w_state, target_qubits)
            state = Statevector(qc)
            attributes = {
                "type": "entangled",
                "subtype": "w_state",
                "num_qubits": n_qubits,
                "entangled_qubits": target_qubits,
                "independent_qubits": [q for q in range(n_qubits) if q not in target_qubits],
                "is_mixed": False
            }
            states.append(("entangled_w", state, attributes))

    return states


# Finite-temperature damping can also mix the fixed all-zero input.
def _apply_local_damping_as_low_rank(
        state, target_qubit, gamma, phase_p, environment_ground_probability):

    if not 0 <= gamma <= 1 or not 0 <= phase_p <= 1:
        raise ValueError("Damping parameters gamma and phase_p must lie within [0, 1]")
    if not 0 < environment_ground_probability < 1:
        raise ValueError("Environment ground-state probability must lie strictly within (0, 1), to generate genuinely mixed states")


    ground = environment_ground_probability
    excited = 1.0 - ground
    amplitude_kraus = [
        np.sqrt(ground) * np.array(
            [[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex
        ),
        np.sqrt(ground) * np.array(
            [[0, np.sqrt(gamma)], [0, 0]], dtype=complex
        ),
        np.sqrt(excited) * np.array(
            [[np.sqrt(1 - gamma), 0], [0, 1]], dtype=complex
        ),
        np.sqrt(excited) * np.array(
            [[0, 0], [np.sqrt(gamma), 0]], dtype=complex
        ),
    ]
    phase_kraus = [
        np.array([[1, 0], [0, np.sqrt(1 - phase_p)]], dtype=complex),
        np.array([[0, 0], [0, np.sqrt(phase_p)]], dtype=complex),
    ]

    branches = [np.asarray(state.data, dtype=np.complex128)]
    for kraus_set in (amplitude_kraus, phase_kraus):
        next_branches = []
        for branch in branches:
            for operator in kraus_set:
                evolved = Statevector(branch).evolve(
                    Operator(operator), qargs=[target_qubit]
                ).data
                if np.linalg.norm(evolved) > 1e-12:
                    next_branches.append(evolved)
        branches = next_branches


    # SVD removes dependent Kraus branches while preserving the density matrix.
    factors = np.column_stack(branches)
    left_vectors, singular_values, _ = np.linalg.svd(factors, full_matrices=False)
    keep = singular_values > 1e-12
    compact_factors = left_vectors[:, keep] * singular_values[keep]
    return testcase_io.LowRankDensityMatrix(compact_factors)


def generate_mixed_states(
        n_qubits, gamma_list=(0.05, 0.1, 0.2),
        p_list=(0.03, 0.1, 0.15), num_random=3,
        environment_ground_probability=0.8):

    test_cases = []


    for gamma in gamma_list:
        for p in p_list:
            qc = QuantumCircuit(n_qubits)
            pure_state = Statevector(qc)
            state = _apply_local_damping_as_low_rank(
                pure_state, 0, gamma, p, environment_ground_probability
            )
            attributes = {
                "type": "mixed",
                "subtype": "fixed_basis",
                "num_qubits": n_qubits,
                "amplitude_damping": gamma,
                "phase_damping": p,
                "environment_ground_probability": environment_ground_probability,
                "noise_qubits": [0],
                "state_representation": testcase_io.MIXED_STATE_REPRESENTATION,
                "is_mixed": True
            }
            test_cases.append((f"mixed_base_gamma_{gamma}_p_{p}", state, attributes))


    for random_index in range(num_random):
        base_qc, base_state = generate_random_base_state(n_qubits)
        target_qubit = random_index % n_qubits

        for gamma in gamma_list:
            for p in p_list:
                state = _apply_local_damping_as_low_rank(
                    Statevector(base_qc), target_qubit, gamma, p,
                    environment_ground_probability
                )
                attributes = {
                    "type": "mixed",
                    "subtype": "random_base",
                    "num_qubits": n_qubits,
                    "base_state": base_state.tolist(),
                    "amplitude_damping": gamma,
                    "phase_damping": p,
                    "environment_ground_probability": environment_ground_probability,
                    "noise_qubits": [target_qubit],
                    "state_representation": testcase_io.MIXED_STATE_REPRESENTATION,
                    "is_mixed": True
                }
                test_cases.append((f"mixed_random_base_{'_'.join(map(str, base_state))}_gamma_{gamma}_p_{p}", state, attributes))


    for gamma in gamma_list:
        for p in p_list:
            qc = QuantumCircuit(n_qubits)
            qc.h(range(n_qubits))
            state = _apply_local_damping_as_low_rank(
                Statevector(qc), 0, gamma, p, environment_ground_probability
            )
            attributes = {
                "type": "mixed",
                "subtype": "superposition",
                "num_qubits": n_qubits,
                "amplitude_damping": gamma,
                "phase_damping": p,
                "environment_ground_probability": environment_ground_probability,
                "superposed_qubits": list(range(n_qubits)),
                "noise_qubits": [0],
                "state_representation": testcase_io.MIXED_STATE_REPRESENTATION,
                "is_mixed": True
            }
            test_cases.append((f"mixed_uniform_gamma_{gamma}_p_{p}", state, attributes))


    if n_qubits >= 2:
        for gamma in gamma_list:
            for p in p_list:

                qc = QuantumCircuit(n_qubits)
                qc.h(0)
                qc.cx(0, 1)

                state = _apply_local_damping_as_low_rank(
                    Statevector(qc), 0, gamma, p,
                    environment_ground_probability
                )
                attributes = {
                    "type": "mixed",
                    "subtype": "entangled",
                    "num_qubits": n_qubits,
                    "entangled_qubits": [0, 1],
                    "amplitude_damping": gamma,
                    "phase_damping": p,
                    "environment_ground_probability": environment_ground_probability,
                    "noise_qubits": [0],
                    "state_representation": testcase_io.MIXED_STATE_REPRESENTATION,
                    "is_mixed": True
                }
                test_cases.append((f"mixed_bell_gamma_{gamma}_p_{p}", state, attributes))
    return test_cases


def save_test_cases(test_cases, filename="test_cases.json", storage_format="auto"):
    return testcase_io.save_test_cases(test_cases, filename, storage_format=storage_format)


def load_test_cases(filename="test_cases.json"):
    return testcase_io.load_test_cases(filename)


def prepare_state_from_case(test_case, backend=None):
    name, state, attributes = test_case

    if isinstance(state, testcase_io.LowRankDensityMatrix):

        return state

    if backend is None:
        backend = Aer.get_backend('statevector_simulator')

    if isinstance(state, Statevector):
        qc = QuantumCircuit(attributes["num_qubits"])
        qc.initialize(state.data)
    elif isinstance(state, DensityMatrix):

        gamma = attributes.get("amplitude_damping", 0)
        p = attributes.get("phase_damping", 0)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            amplitude_damping_error(gamma),
            ['id', 'h', 'ry', 'rx']
        )
        noise_model.add_all_qubit_quantum_error(
            phase_damping_error(p),
            ['id', 'h', 'ry', 'rx']
        )


        qc = QuantumCircuit(attributes["num_qubits"])
        if attributes["subtype"] == "superposition":
            qc.h(range(attributes["num_qubits"]))

    result = execute(qc, backend, noise_model=noise_model if isinstance(state, DensityMatrix) else None).result()
    return result.get_statevector() if isinstance(state, Statevector) else result.get_statevector()


def generate_fresh_superposition_states(n_qubits: int, existing_hashes: Set[str] = None) -> List[Tuple[str, Statevector, Dict]]:
    if existing_hashes is None:
        existing_hashes = set()

    new_states = []
    max_attempts = 1000

    def calculate_state_hash(state: Statevector) -> str:
        data_bytes = state.data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:16]


    attempt = 0
    while len(new_states) < 5 and attempt < max_attempts:
        attempt += 1


        base_state = np.random.randint(0, 2, n_qubits)
        qc = QuantumCircuit(n_qubits)
        for q, val in enumerate(base_state):
            if val == 1:
                qc.x(q)


        h_qubits = random.sample(range(n_qubits), random.randint(1, n_qubits))
        for q in h_qubits:
            qc.h(q)


        state = Statevector(qc)
        state_hash = calculate_state_hash(state)


        if state_hash in existing_hashes:
            continue


        qubits_str = '_'.join(map(str, sorted(h_qubits)))
        name = f"fresh_superposition_h_{qubits_str}_attempt{attempt}"

        attributes = {
            "type": "superposition",
            "subtype": "fresh_random_h",
            "num_qubits": n_qubits,
            "superposed_qubits": h_qubits,
            "base_state": base_state.tolist(),
            "gates_used": ["h" if q in h_qubits else None for q in range(n_qubits)],
            "is_mixed": False
        }

        new_states.append((name, state, attributes))
        existing_hashes.add(state_hash)


    attempt = 0
    while len(new_states) < 10 and attempt < max_attempts:
        attempt += 1


        base_state = np.random.randint(0, 2, n_qubits)
        qc = QuantumCircuit(n_qubits)
        for q, val in enumerate(base_state):
            if val == 1:
                qc.x(q)


        ry_qubits = random.sample(range(n_qubits), random.randint(1, n_qubits))
        rotation_angles = {}

        for q in ry_qubits:
            theta = np.random.uniform(0.1, np.pi - 0.1)
            qc.ry(theta, q)
            rotation_angles[q] = theta


        state = Statevector(qc)
        state_hash = calculate_state_hash(state)


        if state_hash in existing_hashes:
            continue


        qubits_str = '_'.join(map(str, sorted(ry_qubits)))
        name = f"fresh_superposition_ry_{qubits_str}_attempt{attempt}"

        attributes = {
            "type": "superposition",
            "subtype": "fresh_random_ry",
            "num_qubits": n_qubits,
            "superposed_qubits": ry_qubits,
            "base_state": base_state.tolist(),
            "rotation_angles": rotation_angles,
            "gates_used": [f"ry({rotation_angles.get(q, 0)})" if q in ry_qubits else None for q in range(n_qubits)],
            "is_mixed": False
        }

        new_states.append((name, state, attributes))
        existing_hashes.add(state_hash)


    attempt = 0
    while len(new_states) < 15 and attempt < max_attempts:
        attempt += 1


        base_state = np.random.randint(0, 2, n_qubits)
        qc = QuantumCircuit(n_qubits)
        for q, val in enumerate(base_state):
            if val == 1:
                qc.x(q)


        superposed_qubits = random.sample(range(n_qubits), random.randint(1, n_qubits))
        gate_distribution = {}
        rotation_angles = {}

        for q in superposed_qubits:
            if random.random() > 0.5:
                qc.h(q)
                gate_distribution[q] = "h"
            else:
                theta = np.random.uniform(0.1, np.pi - 0.1)
                qc.ry(theta, q)
                gate_distribution[q] = f"ry({theta})"
                rotation_angles[q] = theta


        state = Statevector(qc)
        state_hash = calculate_state_hash(state)


        if state_hash in existing_hashes:
            continue


        qubits_str = '_'.join(map(str, sorted(superposed_qubits)))
        name = f"fresh_superposition_mixed_{qubits_str}_attempt{attempt}"

        attributes = {
            "type": "superposition",
            "subtype": "fresh_random_mixed",
            "num_qubits": n_qubits,
            "superposed_qubits": superposed_qubits,
            "base_state": base_state.tolist(),
            "gate_distribution": gate_distribution,
            "rotation_angles": rotation_angles,
            "gates_used": [gate_distribution.get(q, None) for q in range(n_qubits)],
            "is_mixed": False
        }

        new_states.append((name, state, attributes))
        existing_hashes.add(state_hash)

    return new_states


def generate_fresh_entangled_states(n_qubits: int, existing_hashes: Set[str] = None) -> List[Tuple[str, Statevector, Dict]]:
    if existing_hashes is None:
        existing_hashes = set()

    new_states = []
    max_attempts = 1000

    def calculate_state_hash(state: Statevector) -> str:
        data_bytes = state.data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:16]


    attempt = 0
    while len(new_states) < 5 and attempt < max_attempts and n_qubits >= 2:
        attempt += 1


        qubits = random.sample(range(n_qubits), 2)
        qubits.sort()

        qc = QuantumCircuit(n_qubits)
        qc.h(qubits[0])
        qc.cx(qubits[0], qubits[1])


        if random.random() > 0.7:
            qc.rx(np.random.uniform(0, 2 * np.pi), qubits[0])
        if random.random() > 0.7:
            qc.rz(np.random.uniform(0, 2 * np.pi), qubits[1])

        state = Statevector(qc)
        state_hash = calculate_state_hash(state)

        if state_hash in existing_hashes:
            continue

        name = f"fresh_entangled_bell_{qubits[0]}_{qubits[1]}_attempt{attempt}"
        attributes = {
            "type": "entangled",
            "subtype": "fresh_bell",
            "num_qubits": n_qubits,
            "entangled_qubits": qubits,
            "is_mixed": False
        }

        new_states.append((name, state, attributes))
        existing_hashes.add(state_hash)


    attempt = 0
    while len(new_states) < 10 and attempt < max_attempts and n_qubits >= 3:
        attempt += 1


        k = random.randint(3, min(5, n_qubits))
        entangled_qubits = random.sample(range(n_qubits), k)
        entangled_qubits.sort()

        qc = QuantumCircuit(n_qubits)


        qc.h(entangled_qubits[0])
        for i in range(1, k):

            if random.random() > 0.5:

                qc.cx(entangled_qubits[i - 1], entangled_qubits[i])
            else:

                qc.cx(entangled_qubits[0], entangled_qubits[i])


        for q in entangled_qubits:
            if random.random() > 0.8:
                qc.rz(np.random.uniform(0, 2 * np.pi), q)

        state = Statevector(qc)
        state_hash = calculate_state_hash(state)

        if state_hash in existing_hashes:
            continue

        qubits_str = '_'.join(map(str, entangled_qubits))
        name = f"fresh_entangled_multi_{qubits_str}_attempt{attempt}"
        attributes = {
            "type": "entangled",
            "subtype": "fresh_multi_entangled",
            "num_qubits": n_qubits,
            "entangled_qubits": entangled_qubits,
            "is_mixed": False
        }

        new_states.append((name, state, attributes))
        existing_hashes.add(state_hash)

    return new_states


def generate_fresh_mixed_states(
        n_qubits: int, existing_hashes: Set[str] = None,
        environment_ground_probability: float = 0.8):

    if existing_hashes is None:
        existing_hashes = set()

    new_states = []
    max_attempts = 1000

    def calculate_density_hash(density_matrix):
        data_bytes = density_matrix.factors.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:16]

    attempt = 0
    while len(new_states) < 5 and attempt < max_attempts:
        attempt += 1


        base_state = np.random.randint(0, 2, n_qubits)
        qc = QuantumCircuit(n_qubits)

        for q, val in enumerate(base_state):
            if val == 1:
                qc.x(q)


        for q in random.sample(range(n_qubits), random.randint(1, n_qubits)):
            if random.random() > 0.5:
                qc.h(q)
            else:
                qc.ry(np.random.uniform(0, 2 * np.pi), q)


        pure_state = Statevector(qc)
        gamma = np.random.uniform(0.05, 0.3)
        phase_p = np.random.uniform(0.05, 0.3)
        noise_qubit = random.randint(0, n_qubits - 1)
        density_matrix = _apply_local_damping_as_low_rank(
            pure_state, noise_qubit, gamma, phase_p,
            environment_ground_probability
        )

        density_hash = calculate_density_hash(density_matrix)

        if density_hash in existing_hashes:
            continue

        name = f"fresh_mixed_noise{gamma:.2f}_{phase_p:.2f}_attempt{attempt}"
        attributes = {
            "type": "mixed",
            "subtype": "fresh_mixed",
            "num_qubits": n_qubits,
            "amplitude_damping": gamma,
            "phase_damping": phase_p,
            "environment_ground_probability": environment_ground_probability,
            "noise_qubits": [noise_qubit],
            "state_representation": testcase_io.MIXED_STATE_REPRESENTATION,
            "is_mixed": True
        }

        new_states.append((name, density_matrix, attributes))
        existing_hashes.add(density_hash)

    return new_states

def generate_unique_mixed_states(n_qubits: int, count: int, existing_hashes: Set[str] = None) -> List[Tuple[str, Statevector, Dict]]:
    if existing_hashes is None:
        existing_hashes = set()

    all_fresh = generate_fresh_mixed_states(n_qubits, existing_hashes)


    while len(all_fresh) < count:
        additional = generate_fresh_mixed_states(n_qubits, existing_hashes)
        if not additional:
            break
        all_fresh.extend(additional)

    return all_fresh[:count]

def generate_unique_superposition_states(n_qubits: int, count: int, existing_hashes: Set[str] = None) -> List[Tuple[str, Statevector, Dict]]:
    if existing_hashes is None:
        existing_hashes = set()

    all_fresh = generate_fresh_superposition_states(n_qubits, existing_hashes)


    while len(all_fresh) < count:
        additional = generate_fresh_superposition_states(n_qubits, existing_hashes)
        if not additional:
            break
        all_fresh.extend(additional)

    return all_fresh[:count]


def _default_storage_format(n_qubits):
    return "binary" if n_qubits >= 11 else "json"


# Replace the destination only after the complete temporary file has been written.
def _atomic_save_test_cases(test_cases, filename, storage_format):
    if storage_format == "binary":
        if filename.endswith(".pkl.gz"):
            temporary = filename[:-7] + ".tmp.pkl.gz"
            compress = True
        elif filename.endswith(".pkl"):
            temporary = filename[:-4] + ".tmp.pkl"
            compress = False
        else:
            temporary = filename + ".tmp.pkl.gz"
            compress = True
        saved_temporary = testcase_io.save_test_cases(
            test_cases, temporary, storage_format="binary", compress=compress
        )
    else:
        temporary = filename + ".tmp"
        saved_temporary = testcase_io.save_test_cases(
            test_cases, temporary, storage_format="json"
        )

    os.replace(saved_temporary, filename)
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate quantum test cases")
    parser.add_argument("--n_qubits", type=int, default=12, help="Qubit count")
    parser.add_argument(
        "--format",
        choices=["auto", "json", "binary"],
        default="auto",
        help="Storage format. auto: binary for 11 or more qubits, otherwise JSON",
    )
    parser.add_argument(
        "--include_mixed",
        action="store_true",
        help="Generate low-rank GAD+PD mixed states as defined in the paper",
    )
    parser.add_argument(
        "--replace_mixed",
        action="store_true",
        help="Replace only existing mixed-state test cases, retain all other test cases",
    )
    parser.add_argument(
        "--environment_ground_probability",
        type=float,
        default=0.8,
        help="Environment ground-state probability for generalized amplitude damping, must lie strictly within (0, 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    n_qubits = args.n_qubits
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not 0 < args.environment_ground_probability < 1:
        parser.error("--environment_ground_probability must lie strictly within (0, 1)")

    storage_format = _default_storage_format(n_qubits) if args.format == "auto" else args.format
    default_ext = "pkl.gz" if storage_format == "binary" else "json"
    output_file = args.output or f"./TestCases/quantum_test_cases_{n_qubits}qubits.{default_ext}"

    if args.replace_mixed:
        try:
            source_file = testcase_io.resolve_testcase_file(output_file)
        except FileNotFoundError as error:
            parser.error(f"--replace_mixed Existing testcase file not found: {error}")
        if args.output is None:
            output_file = source_file
            storage_format = "binary" if source_file.endswith((".pkl", ".pkl.gz")) else "json"

        all_test_cases = testcase_io.load_test_cases(
            source_file, exclude_types={"mixed"}
        )
        print(f"Retained {len(all_test_cases)} non-mixed test cases from {source_file}")
    else:
        all_test_cases = []
        all_test_cases.extend(generate_classical_states(n_qubits))
        all_test_cases.extend(generate_superposition_states(n_qubits))
        all_test_cases.extend(generate_entangled_states(n_qubits))

    if args.include_mixed or args.replace_mixed:
        mixed_test_cases = generate_mixed_states(
            n_qubits,
            environment_ground_probability=args.environment_ground_probability,
        )
        all_test_cases.extend(mixed_test_cases)
        mixed_purities = [state.purity() for _, state, _ in mixed_test_cases]
        mixed_ranks = [state.rank for _, state, _ in mixed_test_cases]
        if any(purity >= 1 - 1e-10 for purity in mixed_purities):
            raise RuntimeError("Generated states include a pure state, Check the noise parameters")
        print(
            f"Generated {len(mixed_test_cases)}  GAD+PD low-rank mixed states; "
            f"rankrange={min(mixed_ranks)}-{max(mixed_ranks)}, "
            f"purityrange={min(mixed_purities):.6f}-{max(mixed_purities):.6f}"
        )
    else:
        print("Skipping mixed-state generation; to include mixed states, Add --include_mixed")


    if args.replace_mixed:
        saved_path = _atomic_save_test_cases(
            all_test_cases, output_file, storage_format=storage_format
        )
    else:
        saved_path = save_test_cases(
            all_test_cases, output_file, storage_format=storage_format
        )
    print(f"Generated and saved {len(all_test_cases)} test cases to {saved_path}")


    loaded_cases = load_test_cases(saved_path)
    print(f"\nLoaded {len(loaded_cases)} test cases")


    print("\nPreparing the first test case...")
    first_case = loaded_cases[0]
    print(f"Test-case name: {first_case[0]}")
    print(f"Attributes: {json.dumps(first_case[2], indent=2)}")
    prepared_state = prepare_state_from_case(first_case)
    print(f"Prepared quantum state:\n{prepared_state}")
