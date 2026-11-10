import random
import time
import ast
import hashlib
import math
import sys
import os
import re
import pandas as pd
import numpy as np
from qiskit.quantum_info import state_fidelity, Statevector, DensityMatrix
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import hamming
import json
from qiskit.quantum_info import partial_trace
from collections import defaultdict
import openpyxl
import io
import argparse
import testcase_io
from experiment_metrics import (
    elapsed_since,
    format_time_costs,
    get_memory_usage,
    start_memory_tracing,
)

def trace_distance(rho1, rho2):
    if (testcase_io.density_factors(rho1) is not None and
            testcase_io.density_factors(rho2) is not None):
        return testcase_io.low_rank_trace_distance(rho1, rho2)


    delta = rho1 - rho2


    if hasattr(delta, 'data'):
        svd = np.linalg.svd(delta.data, compute_uv=False)
    else:
        svd = np.linalg.svd(np.array(delta), compute_uv=False)


    return 0.5 * np.sum(svd)


def hamming_distance(statevec1, statevec2):
    idx1 = np.argmax(np.abs(statevec1))
    idx2 = np.argmax(np.abs(statevec2))


    n_qubits = int(np.log2(len(statevec1)))
    bin1 = format(idx1, f'0{n_qubits}b')
    bin2 = format(idx2, f'0{n_qubits}b')


    distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


    return distance / n_qubits

class DRT_D:
    def __init__(self, number_of_variables, input_qbit_count, output_qbit_count, group_name, program_name, algorithm,
                 iter_num, distance_metric="hybrid", n_clusters=5, use_mixstate=False,
                 progress_interval=100):
        self.number_of_variables = number_of_variables
        self.input_qbit_count = input_qbit_count
        self.output_qbit_count = output_qbit_count
        self.group_name = group_name
        self.program_name = program_name
        self.algorithm = algorithm
        self.iter_num = iter_num
        self.distance_metric = distance_metric
        self.n_clusters = n_clusters
        self.use_mixstate = use_mixstate
        self.progress_interval = max(0, int(progress_interval))

        self.test_cases = []
        self.max_fail_count = -1
        self.best_candidates = []
        self.lower_bound = 0


        oracle_path = os.path.join("./oracle/OpenQASM/" + self.program_name + "_oracle.xlsx")
        oracle_data = pd.read_excel(oracle_path, sheet_name=self.group_name, dtype=str)
        if self.use_mixstate:
            mixed_oracle_attributes = []
            for raw_attributes in oracle_data.iloc[:, 2].dropna():
                try:
                    attributes = json.loads(raw_attributes)
                except (TypeError, json.JSONDecodeError):
                    continue
                if attributes.get("type") == "mixed":
                    mixed_oracle_attributes.append(attributes)
            if (not mixed_oracle_attributes or any(
                    attributes.get("state_representation") !=
                    testcase_io.MIXED_STATE_REPRESENTATION
                    for attributes in mixed_oracle_attributes)):
                raise ValueError(
                    "Oracle labels do not match the current GAD+PD mixed states. "
                    "Regenerate matching labels with generate_oracles.py."
                )
        self.All_candidates = []
        tcname_oracle_col = oracle_data.iloc[:, 0]
        reslult_oracle_col = oracle_data.iloc[:, 3]
        pvalue_oracle_col = oracle_data.iloc[:, 4]
        self.oracle = dict(zip(tcname_oracle_col, reslult_oracle_col))
        self.pvalues = dict(zip(tcname_oracle_col, pvalue_oracle_col))


        self.json_data = self._load_test_cases(f"./TestCases/quantum_test_cases_{self.input_qbit_count}qubits.json")


        if not self.use_mixstate:
            filtered_test_cases = {}
            for name, data in self.json_data.items():
                type_str = name.split('_', 1)[0]
                if type_str not in ['classical', 'superposition', 'entangled', 'mixed']:
                    if 'attributes' in data and 'type' in data['attributes']:
                        type_str = data['attributes']['type']
                    else:
                        type_str = 'classical'

                if type_str != 'mixed':
                    filtered_test_cases[name] = data
            self.json_data = filtered_test_cases

        self.dataset_version = next((
            data.get('attributes', {}).get('state_representation')
            for data in self.json_data.values()
            if data.get('attributes', {}).get('state_representation')
        ), "legacy")

        self.All_candidates = [v["state"] for v in self.json_data.values()]
        self.upper_bound = len(self.All_candidates)


        self.classical_basis_indices = {
            name: int(np.argmax(np.abs(data["state"])))
            for name, data in self.json_data.items()
            if name.split('_', 1)[0] == 'classical'
        }
        self.classical_bits = {
            name: tuple(int(bit) for bit in format(
                basis_index, f'0{self.input_qbit_count}b'
            ))
            for name, basis_index in self.classical_basis_indices.items()
        }
        self.superposition_marginal_cache = {}
        self.hamming_weights = np.fromiter(
            (bin(index).count('1') for index in range(1 << self.input_qbit_count)),
            dtype=np.uint8,
        )


        self.cluster_cache_dir = f"./cluster_cache/{self.input_qbit_count}qubits"
        os.makedirs(self.cluster_cache_dir, exist_ok=True)


        cached_data = self._load_cluster_cache()
        if cached_data is not None:
            print("Loading clustering results from cache...")
            self.clustered_cases = cached_data['clustered_cases']
            self.cluster_assignments = cached_data['cluster_assignments']
            self.k = cached_data['k']
            self.medoids = cached_data['medoids']
        else:
            print("Computing new clustering results...")

            self.distance_cache = DistanceMatrixCache(f"./distance_cache/{self.input_qbit_count}_distance_cache")

            assignment_checkpoint = self._load_assignment_checkpoint()
            if assignment_checkpoint is not None:
                self.cluster_assignments = np.asarray(
                    assignment_checkpoint['cluster_assignments'], dtype=int
                )
                self.k = int(assignment_checkpoint['k'])
                self.clustered_cases = self._clustered_cases_from_assignments(
                    self.cluster_assignments, self.k
                )
                print("Restored QDRT_D cluster labels from checkpoint; skipping K-Medoids.")
            else:

                (self.clustered_cases, self.cluster_assignments,
                 self.k) = self._cluster_cases_by_fidelity(k=self.n_clusters)
                self._save_assignment_checkpoint()

            self.medoids = self._load_medoid_checkpoint()
            for cluster_id, cases in self.clustered_cases.items():
                if cluster_id in self.medoids:
                    continue
                print(f"Computing QDRT_D medoid: {cluster_id} ({len(cases)} cases)...")
                self.medoids[cluster_id] = self._select_cluster_medoid(
                    cluster_id, cases
                )
                self._save_medoid_checkpoint()
                print(f"QDRT_D medoid complete: {cluster_id}")

            self._save_cluster_cache()


        self.base_case_to_subcluster = {
            case[0]: cluster_id
            for cluster_id, cases in self.clustered_cases.items()
            for case in cases
        }
        self.base_medoid_dist_matrix = self._build_medoid_distance_matrix()


        self.all_runs_fail_history = []
        self.all_runs_test_cases = []

    def _get_cluster_cache_key(self):
        case_names = sorted(self.json_data.keys())
        params_str = (
            f"QDRT_D_v3_{self.input_qbit_count}_fidelity_cluster_"
            f"hybrid_medoids_{self.n_clusters}_{self.use_mixstate}_"
            f"data{self.dataset_version}"
        )
        content_str = f"{params_str}:{':'.join(case_names)}"
        return hashlib.md5(content_str.encode()).hexdigest()
    def _get_cluster_cache_path(self):
        cache_key = self._get_cluster_cache_key()
        return os.path.join(self.cluster_cache_dir, f"clustering_{cache_key}.json")

    def _get_assignment_checkpoint_path(self):
        return self._get_cluster_cache_path().replace(
            '.json', '.assignments.json'
        )

    def _cache_params(self):
        return {
            'input_qbit_count': self.input_qbit_count,
            'n_clusters': self.n_clusters,
            'use_mixstate': self.use_mixstate,
            'test_case_count': len(self.json_data),
            'algorithm': 'QDRT_D',
            'dataset_version': self.dataset_version,
            'cache_version': 3,
        }

    def _load_assignment_checkpoint(self):
        path = self._get_assignment_checkpoint_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            assignments = data.get('cluster_assignments', [])
            if (data.get('cache_params') != self._cache_params() or
                    len(assignments) != len(self.json_data)):
                return None
            return data
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    # Preserve cluster labels independently of medoids for interruption recovery.
    def _save_assignment_checkpoint(self):
        path = self._get_assignment_checkpoint_path()
        temporary = f"{path}.{os.getpid()}.tmp"
        data = {
            'cluster_assignments': np.asarray(
                self.cluster_assignments, dtype=int
            ).tolist(),
            'k': int(self.k),
            'cache_params': self._cache_params(),
        }
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False)
        os.replace(temporary, path)
        print(f"QDRT_D cluster-label checkpoint saved: {path}")

    def _oracle_case_record(self, case_name):
        return [
            case_name,
            ast.literal_eval(self.oracle[case_name]),
            self.pvalues[case_name],
        ]

    def _clustered_cases_from_assignments(self, assignments, k):
        clustered_cases = {f"cluster_{i}": [] for i in range(k)}
        for case_name, cluster_id in zip(self.json_data.keys(), assignments):
            clustered_cases[f"cluster_{int(cluster_id)}"].append(
                self._oracle_case_record(case_name)
            )
        return clustered_cases

    def _load_cluster_cache(self):
        cache_path = self._get_cluster_cache_path()

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)


            required_keys = ['clustered_cases', 'cluster_assignments', 'k', 'medoids', 'cache_params']
            if not all(key in data for key in required_keys):
                print("Cache data is incomplete, Recompute clustering")
                return None


            cache_params = data['cache_params']
            current_params = self._cache_params()

            if cache_params != current_params:
                print("Cache parameters do not match, Recompute clustering")
                return None

            assignments = data['cluster_assignments']
            data['clustered_cases'] = self._clustered_cases_from_assignments(
                assignments, int(data['k'])
            )
            refreshed_medoids = {}
            for cluster_id, cached_medoid in data['medoids'].items():
                medoid_name = (
                    cached_medoid[0]
                    if isinstance(cached_medoid, list) else cached_medoid
                )
                refreshed_medoids[cluster_id] = self._oracle_case_record(
                    medoid_name
                )
            data['medoids'] = refreshed_medoids

            print(f"Loading clustering results from cache: {cache_path}")
            return data

        except Exception as e:
            print(f"Failed to load clustering cache: {e}, Recompute clustering")
            return None
    def _save_cluster_cache(self):
        cache_path = self._get_cluster_cache_path()

        try:

            cache_data = {
                'clustered_cases': self.clustered_cases,
                'cluster_assignments': self.cluster_assignments.tolist() if hasattr(self.cluster_assignments,
                                                                                    'tolist') else self.cluster_assignments,
                'k': self.k,
                'medoids': self.medoids,
                'cache_params': self._cache_params()
            }

            temporary = f"{cache_path}.{os.getpid()}.tmp"
            with open(temporary, 'w') as f:
                json.dump(cache_data, f, indent=2)
            os.replace(temporary, cache_path)

            print(f"Clustering results saved to cache: {cache_path}")

        except Exception as e:
            print(f"Failed to save clustering cache: {e}")

    def _get_medoid_checkpoint_path(self):
        return self._get_cluster_cache_path().replace(
            '.json', '.medoids.json'
        )

    def _load_medoid_checkpoint(self):
        path = self._get_medoid_checkpoint_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            if data.get('cache_params') != self._cache_params():
                return {}
            medoids = data.get('medoids', {})
            valid_clusters = set(self.clustered_cases)
            if not set(medoids).issubset(valid_clusters):
                return {}
            print(f"Restored {len(medoids)} QDRT_D medoids from checkpoint: {path}")
            return {
                cluster_id: self._oracle_case_record(case_name)
                for cluster_id, case_name in medoids.items()
            }
        except (OSError, ValueError, json.JSONDecodeError, KeyError):
            return {}

    def _save_medoid_checkpoint(self):
        path = self._get_medoid_checkpoint_path()
        temporary = f"{path}.{os.getpid()}.tmp"
        data = {
            'medoids': {
                cluster_id: medoid[0]
                for cluster_id, medoid in self.medoids.items()
            },
            'cache_params': self._cache_params(),
        }
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False)
        os.replace(temporary, path)

    def _classical_hamming_distance(self, case1, case2):
        differing_bits = self.hamming_weights[
            self.classical_basis_indices[case1] ^
            self.classical_basis_indices[case2]
        ]
        return float(differing_bits) / self.input_qbit_count

    def _select_cluster_medoid(self, cluster_id, cases):
        if len(cases) == 1:
            return cases[0]

        case_names = [case[0] for case in cases]
        if all(name in self.classical_basis_indices for name in case_names):


            indices = np.asarray(
                [self.classical_basis_indices[name] for name in case_names],
                dtype=np.uint64,
            )
            distance_sums = np.zeros(len(indices), dtype=np.float64)
            for bit in range(self.input_qbit_count):
                values = (indices >> bit) & 1
                ones = int(np.sum(values))
                distance_sums += np.where(values == 0, ones, len(indices) - ones)
            medoid_index = int(np.argmin(distance_sums))
            return cases[medoid_index]


        n_cases = len(cases)
        distance_sums = np.zeros(n_cases, dtype=np.float64)
        classical_positions = [
            index for index, name in enumerate(case_names)
            if name in self.classical_basis_indices
        ]
        if len(classical_positions) > 1:
            classical_indices = np.asarray([
                self.classical_basis_indices[case_names[index]]
                for index in classical_positions
            ], dtype=np.uint64)
            classical_sums = np.zeros(
                len(classical_positions), dtype=np.float64
            )
            for bit in range(self.input_qbit_count):
                values = (classical_indices >> bit) & 1
                ones = int(np.sum(values))
                classical_sums += np.where(
                    values == 0, ones, len(values) - ones
                )
            distance_sums[classical_positions] += (
                classical_sums / self.input_qbit_count
            )

        classical_names = set(self.classical_basis_indices)
        all_pairs = n_cases * (n_cases - 1) // 2
        classical_pairs = len(classical_positions) * (len(classical_positions) - 1) // 2
        total_pairs = all_pairs - classical_pairs
        report_every = max(10000, total_pairs // 20)
        completed = 0
        started = time.perf_counter()
        for i in range(n_cases - 1):
            for j in range(i + 1, n_cases):
                if (case_names[i] in classical_names and
                        case_names[j] in classical_names):
                    continue
                distance = self._compute_single_distance(
                    case_names[i], case_names[j]
                )
                distance_sums[i] += distance
                distance_sums[j] += distance
                completed += 1
                if completed % report_every == 0:
                    elapsed = time.perf_counter() - started
                    remaining = (
                        elapsed * (total_pairs - completed) / completed
                        if completed else 0.0
                    )
                    print(
                        f"  [{cluster_id}] {completed}/{total_pairs} "
                        f"({completed / total_pairs:.2%}), "
                        f"Elapsed time {elapsed:.0f}s, ETA {remaining:.0f}s"
                    )
        return cases[int(np.argmin(distance_sums))]

    def _compute_single_distance(self, case1, case2):
        raw_type1 = case1.split('_', 1)[0]
        raw_type2 = case2.split('_', 1)[0]
        type1 = 'superposition' if raw_type1 == 'entangled' else raw_type1
        type2 = 'superposition' if raw_type2 == 'entangled' else raw_type2

        state1 = self.json_data[case1]['state']
        state2 = self.json_data[case2]['state']
        attr1 = self.json_data[case1]['attributes']
        attr2 = self.json_data[case2]['attributes']

        if type1 == 'classical' and type2 == 'classical':
            return self._classical_hamming_distance(case1, case2)
        elif type1 == 'superposition' and type2 == 'superposition':
            return self._compute_superposition_superposition_distance(state1, state2, attr1, attr2)
        elif type1 == 'mixed' and type2 == 'mixed' and self.use_mixstate:
            return trace_distance(state1, state2)
        elif {type1, type2} == {'classical', 'superposition'}:
            if 'entangled' not in {raw_type1, raw_type2}:
                classical_name = case1 if type1 == 'classical' else case2
                superposition_name = case2 if type1 == 'classical' else case1
                return self._compute_classical_superposition_distance_by_name(
                    classical_name, superposition_name
                )
            classical_state = state1 if type1 == 'classical' else state2
            superposition_state = state2 if type1 == 'classical' else state1
            classical_attr = attr1 if type1 == 'classical' else attr2
            superposition_attr = attr2 if type1 == 'classical' else attr1
            return self._compute_classical_superposition_distance(
                classical_state, superposition_state,
                classical_attr, superposition_attr
            )
        elif {type1, type2} == {'classical', 'mixed'} and self.use_mixstate:
            classical_state = state1 if type1 == 'classical' else state2
            mixed_state = state2 if type1 == 'classical' else state1
            return testcase_io.low_rank_trace_distance(classical_state, mixed_state)
        elif {type1, type2} == {'superposition', 'mixed'} and self.use_mixstate:
            superposition_state = state1 if type1 == 'superposition' else state2
            mixed_state = state2 if type1 == 'superposition' else state1
            return testcase_io.low_rank_trace_distance(superposition_state, mixed_state)
        else:
            raise ValueError(
                f"Unsupported test-case type combination: {raw_type1}, {raw_type2}"
            )

    def _load_test_cases(self, filename):
        exclude_types = {"mixed"} if not self.use_mixstate else None
        return testcase_io.load_test_cases(
            filename,
            as_dict=True,
            exclude_types=exclude_types,
        )

    def _compute_fidelity_distance_matrix(self, test_cases):
        print("Computing fidelity-based distance matrix...")
        n = len(test_cases)
        distance_matrix = np.zeros((n, n))
        case_names = list(test_cases.keys())


        states = [test_cases[name]["state"] for name in case_names]


        for i in range(n):
            for j in range(i + 1, n):
                try:
                    fidelity_val = testcase_io.low_rank_fidelity(
                        states[i], states[j]
                    )
                    distance = 1 - fidelity_val
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
                except Exception as e:
                    print(f"Failed to compute fidelity: {e}")
                    distance_matrix[i, j] = 1.0
                    distance_matrix[j, i] = 1.0

        return distance_matrix, case_names

    def _cluster_cases_by_fidelity(self, k=5):
        print("Clustering using fidelity-based dissimilarity...")


        distance_matrix = self.distance_cache.compute_or_load_fidelity_matrix(
            self.json_data, self.dataset_version
        )
        case_names = list(self.json_data.keys())


        print(f"Starting K-Medoids (PAM) clustering: {len(case_names)} cases, k={k}...")
        clustering_started = time.perf_counter()
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', method='pam', init='random', random_state=42)
        clusters = kmedoids.fit_predict(distance_matrix)
        print(f"K-Medoids clustering complete, Elapsed time {time.perf_counter() - clustering_started:.0f}s")


        clustered_cases = {}
        for i in range(k):
            cluster_name = f"cluster_{i}"
            clustered_cases[cluster_name] = []


        for i, case_name in enumerate(case_names):
            cluster_id = clusters[i]

            result = ast.literal_eval(self.oracle[case_name])
            p_value = self.pvalues[case_name]

            clustered_cases[f"cluster_{cluster_id}"].append([
                case_name, result, p_value
            ])

        return clustered_cases, clusters, k


    def _initialize_probabilities(self):
        self.subclusters = {}
        self.case_to_subcluster = self.base_case_to_subcluster
        self.subcluster_probs = {}
        self.medoid_dist_matrix = None
        self.subcluster_indices = {}


        all_subclusters = {}
        total_cases = 0

        for cluster_id, cases in self.clustered_cases.items():
            case_list = list(cases)
            all_subclusters[cluster_id] = {
                'cases': case_list,
                'medoid': self.medoids[cluster_id]
            }
            total_cases += len(case_list)

        self.subclusters = all_subclusters


        total_clusters = len(self.subclusters)
        self.subcluster_probs = {cluster_id: 1.0 / total_clusters for cluster_id in self.subclusters}


        self.medoid_dist_matrix = self.base_medoid_dist_matrix.copy()


        self.subcluster_indices = {
            sc_id: idx for idx, sc_id in enumerate(self.subclusters)
        }

    def _build_medoid_distance_matrix(self):
        medoid_cases = [self.medoids[key] for key in self.clustered_cases]


        dist_matrix = np.zeros((len(medoid_cases), len(medoid_cases)))
        for i, medoid1 in enumerate(medoid_cases):
            for j, medoid2 in enumerate(medoid_cases[i + 1:], i + 1):
                dist = self.medoid_distance(medoid1[0], medoid2[0])
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist


        max_distance = np.max(dist_matrix)
        if max_distance > 0:
            dist_matrix /= max_distance
        return dist_matrix

    def build_distance_matrix(self):
        self.medoid_dist_matrix = self._build_medoid_distance_matrix()

    def medoid_distance(self, medoid1, medoid2):
        type1 = medoid1.split('_', 1)[0]
        type2 = medoid2.split('_', 1)[0]
        if type1 == 'entangled':
            type1 = 'superposition'
        if type2 == 'entangled':
            type2 = 'superposition'

        state1 = self.json_data[medoid1]['state']
        state2 = self.json_data[medoid2]['state']
        attr1 = self.json_data[medoid1]['attributes']
        attr2 = self.json_data[medoid2]['attributes']


        if type1 == 'classical' and type2 == 'classical':
            return self._classical_hamming_distance(medoid1, medoid2)
        elif type1 == 'superposition' and type2 == 'superposition':
            return self._compute_superposition_superposition_distance(state1, state2, attr1, attr2)
        elif type1 == 'mixed' and type2 == 'mixed' and self.use_mixstate:
            return trace_distance(state1, state2)
        elif {type1, type2} == {'classical', 'superposition'}:
            classical_state = state1 if type1 == 'classical' else state2
            superposition_state = state2 if type1 == 'classical' else state1
            classical_attr = attr1 if type1 == 'classical' else attr2
            superposition_attr = attr2 if type1 == 'classical' else attr1

            return self._compute_classical_superposition_distance(classical_state, superposition_state,
                                                                  classical_attr, superposition_attr)
        elif {type1, type2} == {'classical', 'mixed'} and self.use_mixstate:
            classical_state = state1 if type1 == 'classical' else state2
            mixed_state = state2 if type1 == 'classical' else state1
            return testcase_io.low_rank_trace_distance(classical_state, mixed_state)
        elif {type1, type2} == {'superposition', 'mixed'} and self.use_mixstate:
            superposition_state = state1 if type1 == 'superposition' else state2
            mixed_state = state2 if type1 == 'superposition' else state1
            return testcase_io.low_rank_trace_distance(superposition_state, mixed_state)
        else:

            return hamming_distance(state1, state2)


    def _compute_superposition_superposition_distance(self, state1, state2, attr1, attr2):
        n_qubits = attr1['num_qubits']


        superposed_qubits1 = set(attr1.get('superposed_qubits', []))
        superposed_qubits2 = set(attr2.get('superposed_qubits', []))


        common_classical_qubits = set(range(n_qubits)) - superposed_qubits1 - superposed_qubits2


        any_superposed_qubits = set(range(n_qubits)) - common_classical_qubits


        if common_classical_qubits:

            base_state1 = attr1.get('base_state', [0] * n_qubits)
            base_state2 = attr2.get('base_state', [0] * n_qubits)


            classical_distance = 0
            for q in common_classical_qubits:
                if base_state1[q] != base_state2[q]:
                    classical_distance += 1
            classical_distance = classical_distance / len(common_classical_qubits) if common_classical_qubits else 0
        else:
            classical_distance = 0


        if any_superposed_qubits:

            superposition_part1 = self._extract_qubits_state(state1, any_superposed_qubits, n_qubits)
            superposition_part2 = self._extract_qubits_state(state2, any_superposed_qubits, n_qubits)
            superposition_distance = 1 - state_fidelity(superposition_part1, superposition_part2)
        else:
            superposition_distance = 0


        total_qubits = n_qubits
        classical_weight = len(common_classical_qubits) / total_qubits
        superposition_weight = len(any_superposed_qubits) / total_qubits


        if classical_weight + superposition_weight > 0:
            total_distance = (
                                         classical_weight * classical_distance + superposition_weight * superposition_distance) / (
                                     classical_weight + superposition_weight)
        else:
            total_distance = 0

        return total_distance

    def _compute_classical_superposition_distance(self, classical_state, superposition_state, classical_attr,
                                                  superposition_attr):

        n_qubits = classical_attr['num_qubits']


        superposed_qubits = set(superposition_attr.get('superposed_qubits', []))


        classical_part_qubits = set(range(n_qubits)) - superposed_qubits


        superposed_part_qubits = superposed_qubits


        if classical_part_qubits:

            classical_base = self._extract_classical_state(classical_state, n_qubits)


            superposition_base = superposition_attr.get('base_state', [0] * n_qubits)

            classical_distance = 0
            for q in classical_part_qubits:
                if classical_base[q] != superposition_base[q]:
                    classical_distance += 1
            classical_distance = classical_distance / len(classical_part_qubits) if classical_part_qubits else 0
        else:
            classical_distance = 0


        if superposed_part_qubits:

            classical_part_state = self._extract_qubits_state(classical_state, superposed_part_qubits, n_qubits)


            superposition_part_state = self._extract_qubits_state(superposition_state, superposed_part_qubits,
                                                                  n_qubits)

            superposition_distance = 1 - state_fidelity(classical_part_state, superposition_part_state)
        else:
            superposition_distance = 0


        total_qubits = n_qubits
        classical_weight = len(classical_part_qubits) / total_qubits
        superposition_weight = len(superposed_part_qubits) / total_qubits


        if classical_weight + superposition_weight > 0:
            total_distance = (
                                         classical_weight * classical_distance + superposition_weight * superposition_distance) / (
                                     classical_weight + superposition_weight)
        else:
            total_distance = 0

        return total_distance

    def _get_superposition_marginal(self, case_name, target_qubits):
        key = (case_name, tuple(sorted(target_qubits)))
        cached = self.superposition_marginal_cache.get(key)
        if cached is not None:
            return cached

        state = self.json_data[case_name]['state']
        amplitudes = np.asarray(getattr(state, 'data', state)).reshape(-1)
        probabilities = np.abs(amplitudes) ** 2
        full_indices = np.arange(probabilities.size, dtype=np.uint64)
        reduced_indices = np.zeros(probabilities.size, dtype=np.int64)
        for position, qubit in enumerate(key[1]):
            reduced_indices |= (
                ((full_indices >> qubit) & 1).astype(np.int64) << position
            )
        marginal = np.bincount(
            reduced_indices,
            weights=probabilities,
            minlength=1 << len(key[1]),
        )
        self.superposition_marginal_cache[key] = marginal
        return marginal

    def _compute_classical_superposition_distance_by_name(
            self, classical_name, superposition_name):

        attributes = self.json_data[superposition_name]['attributes']
        n_qubits = attributes['num_qubits']
        superposed_qubits = tuple(sorted(attributes.get('superposed_qubits', [])))
        superposed_set = set(superposed_qubits)
        classical_qubits = [
            qubit for qubit in range(n_qubits)
            if qubit not in superposed_set
        ]

        classical_base = self.classical_bits[classical_name]
        superposition_base = attributes.get('base_state', [0] * n_qubits)
        classical_mismatches = sum(
            classical_base[qubit] != superposition_base[qubit]
            for qubit in classical_qubits
        )

        if superposed_qubits:
            reduced_index = sum(
                ((self.classical_basis_indices[classical_name] >> qubit) & 1)
                << position
                for position, qubit in enumerate(superposed_qubits)
            )
            fidelity = self._get_superposition_marginal(
                superposition_name, superposed_qubits
            )[reduced_index]
            superposition_contribution = len(superposed_qubits) * (1 - fidelity)
        else:
            superposition_contribution = 0.0

        return float(
            (classical_mismatches + superposition_contribution) / n_qubits
        )

    def _extract_classical_state(self, classical_state, n_qubits):
        idx = np.argmax(np.abs(classical_state))
        bin_str = format(idx, f'0{n_qubits}b')
        return [int(bit) for bit in bin_str]

    def _extract_qubits_state(self, full_state, target_qubits, n_qubits):
        if len(target_qubits) == 0:
            return full_state


        target_qubits_sorted = sorted(target_qubits)

        if len(target_qubits) == n_qubits:

            return full_state
        else:
            try:


                trace_qubits = [q for q in range(n_qubits) if q not in target_qubits_sorted]

                if len(trace_qubits) == 0:
                    return full_state

                state_data = np.asarray(getattr(full_state, 'data', full_state))
                if state_data.ndim == 1:


                    target_axes = [
                        n_qubits - 1 - qubit
                        for qubit in reversed(target_qubits_sorted)
                    ]
                    environment_qubits = [
                        qubit for qubit in range(n_qubits)
                        if qubit not in target_qubits_sorted
                    ]
                    environment_axes = [
                        n_qubits - 1 - qubit
                        for qubit in reversed(environment_qubits)
                    ]
                    tensor = state_data.reshape((2,) * n_qubits)
                    matrix = np.transpose(
                        tensor, target_axes + environment_axes
                    ).reshape(1 << len(target_qubits_sorted), -1)
                    return DensityMatrix(matrix @ matrix.conj().T)

                density_matrix = full_state


                reduced_dm = partial_trace(density_matrix, trace_qubits)
                return reduced_dm

            except Exception as e:

                print(f"Partial trace failed: {e}, Computing fidelity using the full state")
                return full_state

    def select_test_case(self):
        clusters, probs = zip(*self.subcluster_probs.items())
        selected_cluster = np.random.choice(clusters, p=probs)


        cases_in_cluster = self.available_cases_by_cluster[selected_cluster]
        if not cases_in_cluster:
            raise ValueError(f"No remaining cases in cluster {selected_cluster}")

        selected_case = cases_in_cluster[np.random.randint(len(cases_in_cluster))]

        return selected_cluster, selected_case

    def update_probabilities(self, cluster, test_passed):
        if test_passed:
            adjust_direction = -self.delta
        else:
            adjust_direction = self.epsilon


        self.distance_scale = 1
        cluster_idx = self.subcluster_indices[cluster]
        cluster_distances = self.medoid_dist_matrix[cluster_idx]
        weights = cluster_distances * self.distance_scale

        self.subcluster_probs[cluster] += adjust_direction
        if self.subcluster_probs[cluster] > 1:
            self.subcluster_probs[cluster] = 1
        if self.subcluster_probs[cluster] < 0:
            adjust_direction = -self.subcluster_probs[cluster]
            self.subcluster_probs[cluster] = 0

        for i, (clu, prob) in enumerate(self.subcluster_probs.items()):
            if clu == cluster:
                continue
            else:
                adjust = adjust_direction * (weights[i] / np.sum(weights))
                if self.subcluster_probs[cluster] == 1:
                    self.subcluster_probs[clu] = 0
                else:
                    self.subcluster_probs[clu] -= adjust


        for clu in self.subcluster_probs:
            self.subcluster_probs[clu] = max(0, self.subcluster_probs[clu])


        self._normalize_probs()

    def _normalize_probs(self):
        total = sum(max(0, p) for p in self.subcluster_probs.values())
        for cluster in self.subcluster_probs:
            self.subcluster_probs[cluster] = max(0, self.subcluster_probs[cluster]) / total

    def _update_failure_stats(self, test_passed):
        if not test_passed:
            progress = len(self.executed_order) / self.upper_bound
            for threshold in sorted(self.count_fail.keys()):
                if progress <= threshold:
                    self.count_fail[threshold] += 1

    def _clean_empty_structures(self, case, cluster, test_passed):
        cases = self.available_cases_by_cluster[cluster]
        positions = self.available_case_positions[cluster]
        removed_index = positions.pop(case)
        last_case = cases.pop()
        if removed_index < len(cases):
            cases[removed_index] = last_case
            positions[last_case] = removed_index

        if not cases:

            deleted_idx = self.subcluster_indices[cluster]
            self.medoid_dist_matrix = np.delete(
                np.delete(self.medoid_dist_matrix, deleted_idx, axis=0),
                deleted_idx, axis=1
            )


            del self.subclusters[cluster]
            del self.subcluster_probs[cluster]
            del self.available_cases_by_cluster[cluster]
            del self.available_case_positions[cluster]


            idx = self.subcluster_indices.pop(cluster)
            for s, i in list(self.subcluster_indices.items()):
                if i > idx:
                    self.subcluster_indices[s] = i - 1


            if self.subcluster_probs:
                total = sum(self.subcluster_probs.values())
                deleted_prob = 1 - total
                for c in self.subcluster_probs:
                    if total != 0:
                        self.subcluster_probs[c] /= total
                    else:
                        self.subcluster_probs[c] = 1 / len(self.subcluster_probs)

    def _print_current_state(self, case, test_passed):
        print(f"\nStep {self.current_step}:")
        print(f"Executed: {case} (Cluster {self.case_to_subcluster[case]})")
        print(f"Result: {'FAIL' if not test_passed else 'PASS'}")
        print(f"Failing test cases: {self.fault_counts[-1]}")
        print(f"Current p-value: {self.pvalue_history[-1]}")

        print("\nCumulative failing test cases by budget:")
        for thresh, count in sorted(self.count_fail.items()):
            print(f"  <={int(thresh * 100)}%: {count} failing test cases")
        print(f"Clusters: {list(self.subclusters.keys())}")

    def run(self):
        self.start_computing_time = time.perf_counter()

        self.epsilon = 0.05
        self.delta = 0.05
        max_try = self.upper_bound

        stats = defaultdict(int)

        for iter in range(self.iter_num):
            self.current_step = 0
            self.remaining_cases = set(v for v in self.json_data)
            self.available_cases_by_cluster = {
                cluster_id: [case[0] for case in cases]
                for cluster_id, cases in self.clustered_cases.items()
            }
            self.available_case_positions = {
                cluster_id: {
                    case_name: index
                    for index, case_name in enumerate(case_names)
                }
                for cluster_id, case_names in self.available_cases_by_cluster.items()
            }
            self.executed_order = []
            self.fault_counts = []
            self.pvalue_history = []
            self.count_fail = {k: 0 for k in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}
            run_fail_history = []


            self._initialize_probabilities()

            print('Iteration = ' + str(iter))

            for n_total in range(self.upper_bound):

                cluster, case = self.select_test_case()
                self.current_step += 1


                test_passed = ast.literal_eval(self.oracle[case])
                pvalue = self.pvalues[case]


                stats[(cluster, test_passed)] += 1
                self.executed_order.append(case)
                self.pvalue_history.append(pvalue)

                prev_count = self.fault_counts[-1] if self.fault_counts else 0
                self.fault_counts.append(prev_count + (0 if test_passed else 1))

                self._update_failure_stats(test_passed)
                run_fail_history.append({
                    'test_count': n_total + 1,
                    'fail_count': self.fault_counts[-1]
                })

                if n_total >= max_try:
                    break


                self.remaining_cases.remove(case)


                self.update_probabilities(cluster, test_passed)


                self._clean_empty_structures(case, cluster, test_passed)


                if (self.progress_interval and
                        (self.current_step == 1 or
                         self.current_step % self.progress_interval == 0 or
                         self.current_step == self.upper_bound)):
                    self._print_current_state(case, test_passed)

            if 0.3 in self.count_fail:
                if self.count_fail[0.3] > self.max_fail_count:
                    self.max_fail_count = self.count_fail[0.3]
                    self.best_candidates = self.executed_order


            self.all_runs_test_cases.append(self.executed_order)
            self.all_runs_fail_history.append(run_fail_history)


            mixstate_suffix = "MixState" if self.use_mixstate else "NoMixState"
            filename_suffix = f"fidelity_cluster_hybrid_adjust_n{self.n_clusters}_{mixstate_suffix}"


            result_dir = f"./result/{self.group_name}"
            os.makedirs(result_dir, exist_ok=True)

            result_file = os.path.join(result_dir, f"{self.program_name}_{self.algorithm}_{filename_suffix}.txt")

            with open(result_file, 'a') as f:
                f.write(' '.join(map(str, self.executed_order)) + ' \n')
                f.write(' '.join(map(str, self.fault_counts)) + ' \n')
                f.write(' '.join(map(str, self.pvalue_history)) + ' \n')
                json.dump(self.count_fail, f)
                f.write('\n')

        self.preprocessing_time = getattr(self, "preprocessing_time", 0.0)
        self.online_selection_time = elapsed_since(self.start_computing_time)
        self.total_computing_time = self.preprocessing_time + self.online_selection_time
        self.total_memory_usage = get_memory_usage()


        time_file = f'./result/{self.group_name}/{self.program_name}_{self.algorithm}_{filename_suffix}_time.txt'
        with open(time_file, 'a') as f:
            f.write(format_time_costs(
                self.iter_num,
                self.total_computing_time,
                self.preprocessing_time,
                self.online_selection_time,
                getattr(self, "preprocessing_memory_usage", None),
                self.total_memory_usage,
            ))

    def _select_medoids(self, testsuit, cluster_names):
        medoid_cases = []

        for cluster_name in cluster_names:
            cases = testsuit[cluster_name]
            if not cases:
                continue


            medoid_cases.append(cases[0][0])

        return medoid_cases

    def _update_probabilities_with_distance(self, testsuit, current_cluster, adjust_direction,
                                            distance_matrix, cluster_names):

        n_clusters = len(cluster_names)

        if n_clusters <= 1:
            return


        current_distances = distance_matrix[current_cluster]


        other_distances = np.delete(current_distances, current_cluster)
        other_indices = [i for i in range(n_clusters) if i != current_cluster]


        if np.sum(other_distances) > 0:
            weights = other_distances / np.sum(other_distances)
        else:
            weights = np.ones(len(other_indices)) / len(other_indices)


        testsuit["probability_profile"][current_cluster] += adjust_direction


        if testsuit["probability_profile"][current_cluster] > 1:
            testsuit["probability_profile"][current_cluster] = 1
        elif testsuit["probability_profile"][current_cluster] < 0:

            excess = -testsuit["probability_profile"][current_cluster]
            testsuit["probability_profile"][current_cluster] = 0


            for i, idx in enumerate(other_indices):
                testsuit["probability_profile"][idx] += excess * weights[i]
        else:

            adjust_per_cluster = adjust_direction / (n_clusters - 1)

            for i, idx in enumerate(other_indices):
                testsuit["probability_profile"][idx] -= adjust_per_cluster * weights[i]


        for i in range(n_clusters):
            testsuit["probability_profile"][i] = max(0, testsuit["probability_profile"][i])


        total_prob = sum(testsuit["probability_profile"])
        if total_prob > 0:
            for i in range(n_clusters):
                testsuit["probability_profile"][i] /= total_prob

    def get_result(self):
        return self.best_candidates

    def reset(self):
        self.test_cases = []


class DistanceMatrixCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, cases, metric="fidelity"):
        case_ids = sorted([case for case in cases])
        key_str = f"{metric}:{':'.join(case_ids)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.npy")

    def compute_or_load_fidelity_matrix(self, test_cases, dataset_version="legacy"):
        cache_metric = f"QDRT_D_fidelity_v2_data{dataset_version}"
        cache_key = self._get_cache_key(test_cases, cache_metric)
        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            print(f"Loading distance matrix from cache: {cache_path}")
            try:

                data = np.load(cache_path, mmap_mode='r')
                expected_shape = (len(test_cases), len(test_cases))
                if data.shape != expected_shape:
                    raise ValueError(
                        f"Cached matrix shape {data.shape}, expected {expected_shape}"
                    )
                return data
            except Exception as e:
                print(f"Failed to load cache: {e}")


        print("Computing a new fidelity-based distance matrix...")
        n = len(test_cases)
        distance_matrix = np.zeros((n, n))
        case_names = list(test_cases.keys())

        states = [test_cases[name]["state"] for name in case_names]
        total_pairs = n * (n - 1) // 2
        report_every = max(10000, total_pairs // 20)
        completed = 0
        started = time.perf_counter()

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    fidelity_val = testcase_io.low_rank_fidelity(
                        states[i], states[j]
                    )
                    distance = 1 - fidelity_val
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
                except Exception as e:
                    raise RuntimeError(
                        f"Compute {case_names[i]} and {case_names[j]} fidelity computation failed"
                    ) from e
                completed += 1
                if completed % report_every == 0 or completed == total_pairs:
                    elapsed = time.perf_counter() - started
                    remaining = (
                        elapsed * (total_pairs - completed) / completed
                        if completed else 0.0
                    )
                    print(
                        f"  [fidelity-based dissimilarity] {completed}/{total_pairs} "
                        f"({completed / total_pairs:.2%}), "
                        f"Elapsed time {elapsed:.0f}s, ETA {remaining:.0f}s"
                    )


        try:
            temporary = f"{cache_path}.{os.getpid()}.tmp.npy"
            np.save(temporary, distance_matrix)
            os.replace(temporary, cache_path)
            print(f"Distance matrix saved to cache: {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return distance_matrix

def save_t_measure_data(algorithm, group_name, program_name, algorithm_name, distance_metric, n_clusters,
                        use_mixstate):


    result_dir = f"./result/{group_name}/t_measure_data"
    os.makedirs(result_dir, exist_ok=True)


    mixstate_suffix = "MixState" if use_mixstate else "NoMixState"
    filename = f"{program_name}_{algorithm_name}_{distance_metric}_n{n_clusters}_{mixstate_suffix}.json"

    filepath = os.path.join(result_dir, filename)


    data = {
        "algorithm": algorithm_name,
        "group_name": group_name,
        "program_name": program_name,
        "distance_metric": distance_metric,
        "n_clusters": n_clusters,
        "use_mixstate": use_mixstate,
        "t_measure": [],
        "params_update": []
    }


    for i, fail_history in enumerate(algorithm.all_runs_fail_history):
        data["t_measure"].append({
            "run_id": i,
            "fail_history": fail_history
        })
    if hasattr(algorithm, 'all_runs_params_history'):
        for j, params_history in enumerate(algorithm.all_runs_params_history):
            data["params_update"].append({
                "step_id": j,
                "params_history": params_history
            })


    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved T-measure and params updating data to {filepath}")

def process_results(alg, group_name, program_name, distance_metric, n_clusters, use_mixstate):
    mixstate_suffix = "MixState" if use_mixstate else "NoMixState"
    filename_suffix = f"fidelity_cluster_hybrid_adjust_n{n_clusters}_{mixstate_suffix}"

    file_path = f'./result/{group_name}/{program_name}_{alg}_{filename_suffix}.txt'

    count_fail = {}

    try:
        with open(file_path, 'r') as f:
            data = f.readlines()
            nums = []
            for i in range(3, len(data), 4):
                d = json.load(io.StringIO(data[i].strip()))
                for k, v in d.items():
                    if k not in count_fail:
                        count_fail[k] = [v]
                    else:
                        count_fail[k].append(v)
    except FileNotFoundError:
        print(f"Result file not found: {file_path}")
        return

    avg_failCount = {}
    max_failCount = {}

    for k, v in count_fail.items():
        avg_failCount[k] = sum(v) / len(v)
        max_failCount[k] = max(v)

    print(avg_failCount)
    print(max_failCount)


    stats_file = f'./result/{group_name}/{program_name}_{alg}_{filename_suffix}_failCount.txt'
    with open(stats_file, "w") as f2:
        f2.write("count_fail:\n")
        f2.write(json.dumps(count_fail) + "\n")
        f2.write("avg_failCount:\n")
        f2.write(json.dumps(avg_failCount) + "\n")
        f2.write("max_failCount:\n")
        f2.write(json.dumps(max_failCount) + "\n")


    from datetime import date


    avg_filename = f"{program_name}_avg_faultNum.xlsx"
    max_filename = f"{program_name}_max_faultNum.xlsx"


    try:
        workbook1 = openpyxl.load_workbook('./result/' + avg_filename)
    except FileNotFoundError:
        workbook1 = openpyxl.Workbook()

        if 'Sheet' in workbook1.sheetnames:
            del workbook1['Sheet']


    sheet_name = f"{group_name}"
    if sheet_name not in workbook1.sheetnames:
        worksheet = workbook1.create_sheet(sheet_name)
        worksheet.cell(row=1, column=2, value=f"{alg}")
        for index, key in enumerate(avg_failCount.keys()):
            worksheet.cell(row=index + 2, column=1, value=key)
        empty_column = 2
    else:
        worksheet = workbook1[sheet_name]
        empty_column = worksheet.max_column + 1
        worksheet.cell(row=1, column=empty_column, value=f"{alg}")


    for index, key in enumerate(avg_failCount.keys()):
        worksheet.cell(row=index + 2, column=empty_column, value=avg_failCount[key])

    worksheet.cell(row=22, column=empty_column, value=f"{filename_suffix}")
    worksheet.cell(row=23, column=empty_column, value=f"{date.today()}")


    workbook1.save('./result/' + avg_filename)


    try:
        workbook2 = openpyxl.load_workbook('./result/' + max_filename)
    except FileNotFoundError:
        workbook2 = openpyxl.Workbook()
        if 'Sheet' in workbook2.sheetnames:
            del workbook2['Sheet']


    if sheet_name not in workbook2.sheetnames:
        worksheet = workbook2.create_sheet(sheet_name)
        worksheet.cell(row=1, column=2, value=f"{alg}")
        for index, key in enumerate(max_failCount.keys()):
            worksheet.cell(row=index + 2, column=1, value=key)
        empty_column = 2
    else:
        worksheet = workbook2[sheet_name]
        empty_column = worksheet.max_column + 1
        worksheet.cell(row=1, column=empty_column, value=f"{alg}")


    for index, key in enumerate(max_failCount.keys()):
        worksheet.cell(row=index + 2, column=empty_column, value=max_failCount[key])


    worksheet.cell(row=22, column=empty_column, value=f"{filename_suffix}")
    worksheet.cell(row=23, column=empty_column, value=f"{date.today()}")


    workbook2.save('./result/' + max_filename)

def main():
    parser = argparse.ArgumentParser(description='DRT-DAlgorithm: Fidelity-based clustering+Hybrid distance-based adjustment')


    parser.add_argument('input_qbit_count', type=int, help='Input qubit count')
    parser.add_argument('output_qbit_count', type=int, help='Output qubit count')
    parser.add_argument('group_name', type=str, help='Benchmark group name (e.g., adder_n7_1)')
    parser.add_argument('program_name', type=str, help='Canonical program name (e.g., adder_n7)')
    parser.add_argument('alg', type=str, help='Algorithm name(DRT_D)')
    parser.add_argument('--iter_num', type=int, default=50, help='Repetition count')


    parser.add_argument('--distance_metric', type=str, default='hybrid',
                        choices=['hybrid'], help='Dissimilarity method')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--use_mixstate', action='store_true',
                        help='Include mixed-state test cases')
    parser.add_argument('--progress_interval', type=int, default=100,
                        help='Online-selection progress interval in steps; 0 disables output')

    args = parser.parse_args()
    start_memory_tracing()


    algorithm = "DRT_D"


    num_vars = -1
    if args.input_qbit_count == 10:
        num_vars = 485
    elif args.input_qbit_count == 11:
        num_vars = 845
    elif args.input_qbit_count == 12:
        num_vars = 5114
    elif args.input_qbit_count == 7:
        num_vars = 108
    elif args.input_qbit_count == 9:
        num_vars = 287
    elif args.input_qbit_count == 8:
        num_vars = 174
    elif args.input_qbit_count == 4:
        num_vars = 26
    elif args.input_qbit_count == 5:
        num_vars = 41
    else:
        num_vars = -1


    result_dir = f"./result/{args.group_name}"
    os.makedirs(result_dir, exist_ok=True)


    experiment_start_time = time.perf_counter()
    drt_d = DRT_D(
        number_of_variables=num_vars,
        input_qbit_count=args.input_qbit_count,
        output_qbit_count=args.output_qbit_count,
        group_name=args.group_name,
        program_name=args.program_name,
        algorithm=algorithm,
        iter_num=args.iter_num,
        distance_metric=args.distance_metric,
        n_clusters=args.n_clusters,
        use_mixstate=args.use_mixstate,
        progress_interval=args.progress_interval
    )
    drt_d.preprocessing_time = elapsed_since(experiment_start_time)
    drt_d.preprocessing_memory_usage = get_memory_usage()

    drt_d.run()


    save_t_measure_data(drt_d, args.group_name, args.program_name, args.alg,
                        args.distance_metric, args.n_clusters, args.use_mixstate)


    process_results(args.alg, args.group_name, args.program_name,
                    args.distance_metric, args.n_clusters, args.use_mixstate)

    print(f'Algorithm: {args.alg}')
    print(f'FaultVersion: {args.group_name}')
    print(f'Distance_Metric: {args.distance_metric}')
    print(f'Number of Clusters: {args.n_clusters}')
    print(f'Use MixState: {args.use_mixstate}')
    print(f'Computing time: {drt_d.total_computing_time}')


if __name__ == '__main__':
    main()
