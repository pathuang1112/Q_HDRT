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
from qiskit.quantum_info import state_fidelity
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import hamming
import json
from qiskit.quantum_info import Statevector, DensityMatrix
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict, OrderedDict
import copy
from qiskit.quantum_info import partial_trace
import openpyxl
import io
import argparse
import tempfile
from datetime import datetime
import testcase_io
from experiment_metrics import (
    elapsed_since,
    format_time_costs,
    get_memory_usage,
    PeakMemoryMonitor,
)

print(sys.getdefaultencoding())
print(sys.getfilesystemencoding())


HYBRID_DISTANCE_VERSION = "entangled_qubits_v1"


def _nonclassical_qubits(attributes, include_entangled_qubits=True):
    qubits = set()
    n_qubits = attributes['num_qubits']

    def collect(indices):
        for index in indices:
            if isinstance(index, (list, tuple, set, np.ndarray)):
                collect(index)
            elif isinstance(index, (int, np.integer)) and 0 <= index < n_qubits:
                qubits.add(int(index))
            else:
                raise ValueError(f"Invalid qubit index {index!r} for {n_qubits} qubits")

    collect(attributes.get('superposed_qubits', []))
    if include_entangled_qubits:
        collect(attributes.get('entangled_qubits', []))
    return qubits


def _new_run_directory(root):
    run_root = os.path.join(os.fspath(root), 'runs')
    os.makedirs(run_root, exist_ok=True)
    return tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=run_root)


def _density_factors(state):
    if isinstance(state, testcase_io.LowRankDensityMatrix):
        return state.factors
    data = np.asarray(getattr(state, 'data', state))
    if data.ndim == 1:
        return data[:, np.newaxis]
    return None


class _ReducedDensityFactors(testcase_io.LowRankDensityMatrix):
    def __init__(self, factors):
        self.factors = np.asarray(factors, dtype=np.complex128)


def _factor_partial_trace(state, target_qubits, n_qubits):
    factors = _density_factors(state)
    if factors is None:
        return None
    if factors.shape[0] != 2 ** n_qubits:
        raise ValueError('State dimension does not match n_qubits')
    targets = sorted(target_qubits)
    traced = [qubit for qubit in range(n_qubits) if qubit not in targets]

    axes = ([n_qubits - 1 - qubit for qubit in reversed(targets)]
            + [n_qubits - 1 - qubit for qubit in reversed(traced)] + [n_qubits])
    tensor = factors.reshape([2] * n_qubits + [factors.shape[1]])
    reduced = tensor.transpose(axes).reshape(2 ** len(targets), -1)
    if reduced.shape[1] > reduced.shape[0]:


        reduced = np.linalg.qr(reduced.conj().T, mode='r').conj().T
    return _ReducedDensityFactors(reduced)


def _extract_cached_subsystem(engine, full_state, target_qubits, n_qubits):
    targets = tuple(sorted(target_qubits))
    if not targets or len(targets) == n_qubits:
        return full_state
    if len(set(targets)) != len(targets) or any(q < 0 or q >= n_qubits for q in targets):
        raise ValueError('Invalid target qubits')
    if not hasattr(engine, '_reduced_state_cache'):
        engine._reduced_state_cache = OrderedDict()
        engine._reduced_state_cache_bytes = 0
        engine._reduced_state_cache_limit = 256 * 1024 * 1024
    key = (id(full_state), targets)
    cached = engine._reduced_state_cache.get(key)
    if cached is not None:
        engine._reduced_state_cache.move_to_end(key)
        return cached
    reduced = _factor_partial_trace(full_state, targets, n_qubits)
    if reduced is None:
        traced = [q for q in range(n_qubits) if q not in targets]
        reduced = partial_trace(full_state, traced)
    data = reduced.factors if isinstance(reduced, testcase_io.LowRankDensityMatrix) else reduced.data
    size = np.asarray(data).nbytes
    limit = engine._reduced_state_cache_limit
    if size <= limit:

        reduced._source_state = full_state
        while engine._reduced_state_cache and (
                engine._reduced_state_cache_bytes + size > limit
                or len(engine._reduced_state_cache) >= 4096):
            _, evicted = engine._reduced_state_cache.popitem(last=False)
            data = evicted.factors if isinstance(evicted, testcase_io.LowRankDensityMatrix) else evicted.data
            engine._reduced_state_cache_bytes -= np.asarray(data).nbytes
        engine._reduced_state_cache[key] = reduced
        engine._reduced_state_cache_bytes += size
    return reduced


def trace_distance(rho1, rho2):
    factors1 = _density_factors(rho1)
    factors2 = _density_factors(rho2)
    if factors1 is not None and factors2 is not None:
        if factors1.shape[1] == 1 and factors2.shape[1] == 1:
            vector1 = factors1[:, 0]
            vector2 = factors2[:, 0]
            fidelity = float(abs(np.vdot(vector1, vector2)) ** 2)
            return math.sqrt(max(0.0, 1.0 - min(1.0, fidelity)))


        combined = np.column_stack((factors1, factors2))
        gram = combined.conj().T @ combined
        gram = (gram + gram.conj().T) / 2
        gram_values, gram_vectors = np.linalg.eigh(gram)
        gram_values = np.clip(gram_values, 0.0, None)
        sqrt_gram = ((gram_vectors * np.sqrt(gram_values)) @
                     gram_vectors.conj().T)
        signs = np.concatenate((
            np.ones(factors1.shape[1]),
            -np.ones(factors2.shape[1]),
        ))
        compressed_delta = (sqrt_gram * signs[np.newaxis, :]) @ sqrt_gram
        compressed_delta = (compressed_delta + compressed_delta.conj().T) / 2
        eigvals = np.linalg.eigvalsh(compressed_delta)
        distance = float(0.5 * np.sum(np.abs(eigvals)))
        return min(1.0, max(0.0, distance))

    data1 = np.asarray(getattr(rho1, 'data', rho1))
    data2 = np.asarray(getattr(rho2, 'data', rho2))
    if data1.ndim == 1 and data2.ndim == 1:

        fidelity = float(abs(np.vdot(data1, data2)) ** 2)
        return math.sqrt(max(0.0, 1.0 - min(1.0, fidelity)))

    if data1.ndim == 1:
        rho1 = DensityMatrix(rho1)
    if data2.ndim == 1:
        rho2 = DensityMatrix(rho2)


    delta = rho1 - rho2


    eigvals = np.linalg.eigvalsh(delta)

    return 0.5 * np.sum(np.abs(eigvals))


def normalized_density_euclidean_distance(state1, state2):
    factors1 = _density_factors(state1)
    factors2 = _density_factors(state2)
    if factors1 is not None and factors2 is not None:
        gram1 = factors1.conj().T @ factors1
        gram2 = factors2.conj().T @ factors2
        cross = factors1.conj().T @ factors2
        squared_norm = (
            np.linalg.norm(gram1, 'fro') ** 2
            + np.linalg.norm(gram2, 'fro') ** 2
            - 2 * np.linalg.norm(cross, 'fro') ** 2
        )
        return math.sqrt(max(0.0, float(np.real(squared_norm))) / 2.0)

    rho1 = state1 if isinstance(state1, DensityMatrix) else DensityMatrix(state1)
    rho2 = state2 if isinstance(state2, DensityMatrix) else DensityMatrix(state2)
    return float(np.linalg.norm(rho1.data - rho2.data) / np.sqrt(2))


def hamming_distance(statevec1, statevec2):
    idx1 = np.argmax(np.abs(statevec1))
    idx2 = np.argmax(np.abs(statevec2))


    n_qubits = int(np.log2(len(statevec1)))
    bin1 = format(idx1, f'0{n_qubits}b')
    bin2 = format(idx2, f'0{n_qubits}b')


    distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


    return distance / n_qubits


def fast_state_fidelity(state1, state2):
    factors1 = _density_factors(state1)
    factors2 = _density_factors(state2)
    if factors1 is not None and factors2 is not None:
        if factors1.shape[1] == factors2.shape[1] == 1:
            return min(1.0, max(0.0, float(abs(np.vdot(factors1[:, 0], factors2[:, 0])) ** 2)))
        overlap = factors1.conj().T @ factors2
        if factors1.shape[1] == 1 or factors2.shape[1] == 1:
            return min(1.0, max(0.0, float(np.real(np.vdot(overlap, overlap)))))
        root_fidelity = float(np.sum(np.linalg.svd(overlap, compute_uv=False)))
        fidelity = root_fidelity ** 2
        return min(1.0, max(0.0, fidelity))

    data1 = getattr(state1, 'data', state1)
    data2 = getattr(state2, 'data', state2)
    array1 = np.asarray(data1)
    array2 = np.asarray(data2)

    if array1.ndim == 1 and array2.ndim == 1:
        fidelity = float(abs(np.vdot(array1, array2)) ** 2)

        return min(1.0, max(0.0, fidelity))

    return state_fidelity(state1, state2)


class QuantumTestClusterer:
    def __init__(self, json_data, cache_dir, classification_file, distance_metric="hybrid", n_clusters=5,
                 use_mixstate=False, classification_method='direct', type_k=3,
                 include_entangled_qubits=False):
        self.json_data = json_data
        self.include_entangled_qubits = include_entangled_qubits
        self.hybrid_distance_version = (
            HYBRID_DISTANCE_VERSION if include_entangled_qubits else 'legacy'
        )
        if include_entangled_qubits and distance_metric == 'hybrid':

            cache_dir = os.path.join(cache_dir, HYBRID_DISTANCE_VERSION)
            classification_file = os.path.splitext(classification_file)[0] + f'_{HYBRID_DISTANCE_VERSION}.json'
        os.makedirs(os.path.dirname(classification_file) or '.', exist_ok=True)
        self.cache = DistanceMatrixCache(cache_dir, preserve_existing=True)
        self.classification_file = classification_file
        self.distance_metric = distance_metric
        self.n_clusters = n_clusters
        if type(type_k) is not int or type_k < 1:
            raise ValueError('type_k must be a positive integer')
        self.type_k = type_k
        self.use_mixstate = use_mixstate
        self.classification_method = classification_method
        self.dataset_version = next((
            data.get('attributes', {}).get('state_representation')
            for data in self.json_data.values()
            if data.get('attributes', {}).get('state_representation')
        ), "legacy")


        self.loaded_classifications = self._load_classifications()


        self.allcases = []

        self.medoids = {}


        self._reduced_state_cache = OrderedDict()
        self._reduced_state_cache_bytes = 0
        self._reduced_state_cache_limit = 256 * 1024 * 1024

    def _load_classifications(self):
        if os.path.exists(self.classification_file):
            try:
                with open(self.classification_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_classifications(self):
        save_data = {
            'cases': self.allcases,
            'medoids': self.medoids,
            'case_classifications': {name: data.get('cluster_id', -1)
                                     for name, data in self.json_data.items()},
            'distance_metric': self.distance_metric,
            'n_clusters': self.n_clusters,
            'type_k': self.type_k,
            'use_mixstate': self.use_mixstate,
            'classification_method': self.classification_method,
            'dataset_version': self.dataset_version,
            'hybrid_distance_version': self.hybrid_distance_version,
        }


        if os.path.exists(self.classification_file):
            root, extension = os.path.splitext(self.classification_file)
            descriptor, self.classification_file = tempfile.mkstemp(
                prefix=os.path.basename(root) + '_', suffix=extension,
                dir=os.path.dirname(root) or '.',
            )
            os.close(descriptor)
        with open(self.classification_file, 'w') as f:
            json.dump(save_data, f, indent=2)

    def cluster_cases(self):
        if (self.loaded_classifications and
                self.loaded_classifications.get('classification_method') == self.classification_method and
                self.loaded_classifications.get('distance_metric') == self.distance_metric and
                self.loaded_classifications.get('n_clusters') == self.n_clusters and
                self.loaded_classifications.get('use_mixstate') == self.use_mixstate and
                (self.distance_metric != 'hybrid' or
                 self.loaded_classifications.get('hybrid_distance_version', 'legacy') ==
                 self.hybrid_distance_version) and
                (self.classification_method != 'type_clustering' or
                 self.loaded_classifications.get('type_k', 3) == self.type_k) and
                self.loaded_classifications.get('dataset_version', 'legacy') == self.dataset_version):

            self.allcases = self.loaded_classifications.get('cases', [])
            self.medoids = {int(k): v for k, v in self.loaded_classifications.get('medoids', {}).items()}


            case_classifications = self.loaded_classifications.get('case_classifications', {})
            for name, cluster_id in case_classifications.items():
                if name in self.json_data:
                    self.json_data[name]["cluster_id"] = cluster_id
            if all(
                    'cluster_id' in self.json_data[name] and self.json_data[name]['cluster_id'] != -1
                    for name in self.allcases
            ):
                return [self.json_data[name]['cluster_id'] for name in self.allcases]


        for name, data in self.json_data.items():
            self.allcases.append(name)


        if self.distance_metric == "hybrid":
            dist_func = self._compute_hybrid_distance
        elif self.distance_metric == "fidelity":
            dist_func = self._compute_fidelity_distance
        elif self.distance_metric == "euclidean":
            dist_func = self._compute_euclidean_distance
        elif self.distance_metric == "trace":
            dist_func = self._compute_trace_distance
        else:
            dist_func = self._compute_hybrid_distance

        data_suffix = "" if self.dataset_version == "legacy" else f"_data{self.dataset_version}"
        if self.distance_metric == 'hybrid' and self.include_entangled_qubits:
            data_suffix += f"_{HYBRID_DISTANCE_VERSION}"

        """Select clustering strategy by partitioning method"""
        if self.classification_method == 'direct':

            dist_matrix = self.cache.compute_or_load(
                {name: self.json_data[name] for name in self.allcases},
                f'{self.distance_metric}_n{self.n_clusters}_mix{self.use_mixstate}{data_suffix}',
                dist_func
            )


            dist_matrix = np.maximum(dist_matrix, 0)
            assert np.all(dist_matrix >= 0), "Distance matrix still contains negative values!"
            return self._cluster_direct(dist_matrix)
        elif self.classification_method == 'adaptive':

            dist_matrix = self.cache.compute_or_load(
                {name: self.json_data[name] for name in self.allcases},
                f'{self.distance_metric}_n{self.n_clusters}_mix{self.use_mixstate}{data_suffix}',
                dist_func
            )


            dist_matrix = np.maximum(dist_matrix, 0)
            assert np.all(dist_matrix >= 0), "Distance matrix still contains negative values!"
            return self._cluster_adaptive(dist_matrix)
        elif self.classification_method == 'by_type':
            return self._cluster_by_type()
        elif self.classification_method == 'detailed_type1':
            return self._cluster_detailed_type1()
        elif self.classification_method == 'detailed_type2':
            return self._cluster_detailed_type2()
        elif self.classification_method == 'type_clustering':
            return self._cluster_type_clustering()
        else:


            dist_matrix = self.cache.compute_or_load(
                {name: self.json_data[name] for name in self.allcases},
                f'{self.distance_metric}_n{self.n_clusters}_mix{self.use_mixstate}{data_suffix}',
                dist_func
            )


            dist_matrix = np.maximum(dist_matrix, 0)
            assert np.all(dist_matrix >= 0), "Distance matrix still contains negative values!"
            return self._cluster_direct(dist_matrix)


    def _cluster_direct(self, dist_matrix):
        print("Using direct clustering...")


        kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            metric='precomputed',
            method='pam',
            init='random',
            random_state=42
        )
        clusters = kmedoids.fit_predict(dist_matrix)


        for i, name in enumerate(self.allcases):
            self.json_data[name]["cluster_id"] = int(clusters[i])


        self._find_medoids(self.allcases, dist_matrix, clusters)


        self._save_classifications()

        return clusters

    def _cluster_adaptive(self, dist_matrix):
        print("Using adaptive clustering...")


        best_k = self._find_optimal_clusters(dist_matrix)
        print(f"Cluster count selected by adaptive clustering: {best_k}")


        kmedoids = KMedoids(
            n_clusters=best_k,
            metric='precomputed',
            method='pam',
            init='random',
            random_state=42
        )
        clusters = kmedoids.fit_predict(dist_matrix)


        for i, name in enumerate(self.allcases):
            self.json_data[name]["cluster_id"] = int(clusters[i])


        self._find_medoids(self.allcases, dist_matrix, clusters)


        self._save_classifications()

        return clusters

    def _cluster_by_type(self):
        print("Partitioning by input-state type...")

        type_mapping = {
            'classical': 0,
            'superposition': 1,
            'entangled': 2,
        }


        if self.use_mixstate:
            type_mapping['mixed'] = 3


        type_cases = {type_name: [] for type_name in type_mapping.values()}

        for name in self.allcases:

            type_str = name.split('_', 1)[0]
            if type_str not in type_mapping:

                if 'attributes' in self.json_data[name] and 'type' in self.json_data[name]['attributes']:
                    type_str = self.json_data[name]['attributes']['type']
                else:

                    type_str = 'classical'


            if type_str in type_mapping:
                type_id = type_mapping[type_str]
                self.json_data[name]["cluster_id"] = type_id
                type_cases[type_id].append(name)


        self._select_type_medoids(type_cases)


        self._save_classifications()

        return [self.json_data[name]['cluster_id'] for name in self.allcases]

    def _cluster_detailed_type1(self):
        print("Using detailed type partitioning 1...")

        type_mapping = {}
        type_id_counter = 0

        for name, data in self.json_data.items():

            type_str = name.split('_', 1)[0]
            if type_str not in ['classical', 'superposition', 'entangled', 'mixed']:
                if 'attributes' in data and 'type' in data['attributes']:
                    type_str = data['attributes']['type']
                else:
                    type_str = 'classical'


            if not self.use_mixstate and type_str == 'mixed':
                continue


            if type_str == 'superposition':

                superposed_qubits = data['attributes'].get('superposed_qubits', [])
                superposition_level = len(superposed_qubits)


                subtype = f"superposition_{superposition_level}"
                if subtype not in type_mapping:
                    type_mapping[subtype] = type_id_counter
                    type_id_counter += 1

                cluster_id = type_mapping[subtype]
            else:

                if type_str not in type_mapping:
                    type_mapping[type_str] = type_id_counter
                    type_id_counter += 1

                cluster_id = type_mapping[type_str]


            self.json_data[name]["cluster_id"] = cluster_id


        self._select_type_medoids_detailed(type_mapping)


        self._save_classifications()


        return [self.json_data[name]['cluster_id'] for name in self.allcases]

    def _cluster_detailed_type2(self):
        print("Using detailed type partitioning 2...")

        type_mapping = {}
        type_id_counter = 0


        classical_cases = []
        other_cases = {}

        for name, data in self.json_data.items():

            type_str = name.split('_', 1)[0]
            if type_str not in ['classical', 'superposition', 'entangled', 'mixed']:
                if 'attributes' in data and 'type' in data['attributes']:
                    type_str = data['attributes']['type']
                else:
                    type_str = 'classical'


            if not self.use_mixstate and type_str == 'mixed':
                continue

            if type_str == 'classical':
                classical_cases.append(name)
            else:
                if type_str not in other_cases:
                    other_cases[type_str] = []
                other_cases[type_str].append(name)


        random.shuffle(classical_cases)
        classical_clusters = [[] for _ in range(3)]
        for i, case in enumerate(classical_cases):
            cluster_idx = i % 3
            classical_clusters[cluster_idx].append(case)
            self.json_data[case]["cluster_id"] = cluster_idx


        type_id_counter = 3


        for type_str, cases in other_cases.items():
            if type_str == 'superposition':

                superposition_groups = {
                    1: [],
                    2: [],
                    3: [],
                    4: []
                }
                for case in cases:
                    data = self.json_data[case]
                    superposed_qubits = data['attributes'].get('superposed_qubits', [])
                    superposition_level = len(superposed_qubits)


                    if superposition_level == 1:
                        group_key = 1
                    elif superposition_level == 2:
                        group_key = 2
                    elif superposition_level == 3:
                        group_key = 3
                    else:
                        group_key = 4
                    superposition_groups[group_key].append(case)

                if not superposition_groups[4]:
                    del superposition_groups[4]

                for level, group_cases in superposition_groups.items():
                    type_mapping_key = f"superposition_{level}"
                    if type_mapping_key not in type_mapping:
                        type_mapping[type_mapping_key] = type_id_counter
                        type_id_counter += 1

                    cluster_id = type_mapping[type_mapping_key]
                    for case in group_cases:
                        self.json_data[case]["cluster_id"] = cluster_id

            else:

                if type_str not in type_mapping:
                    type_mapping[type_str] = type_id_counter
                    type_id_counter += 1

                cluster_id = type_mapping[type_str]
                for case in cases:
                    self.json_data[case]["cluster_id"] = cluster_id


        self._select_type_medoids_detailed(type_mapping, classical_clusters)


        self._save_classifications()


        return [self.json_data[name]['cluster_id'] for name in self.allcases]

    def _cluster_type_clustering(self):
        print("Using within-type clustering...")


        cluster_id_counter = 0


        classical_cases = []
        superposition_cases = []
        entangled_cases = []
        mixed_cases = []

        for name, data in self.json_data.items():

            type_str = name.split('_', 1)[0]
            if type_str not in ['classical', 'superposition', 'entangled', 'mixed']:
                if 'attributes' in data and 'type' in data['attributes']:
                    type_str = data['attributes']['type']
                else:
                    type_str = 'classical'


            if not self.use_mixstate and type_str == 'mixed':
                continue


            if type_str == 'classical':
                classical_cases.append(name)
            elif type_str == 'superposition':
                superposition_cases.append(name)
            elif type_str == 'entangled':
                entangled_cases.append(name)
            elif type_str == 'mixed':
                mixed_cases.append(name)


        all_medoids = {}


        if classical_cases:
            print(f"  Computational-basis test cases: {len(classical_cases)}")

            if len(classical_cases) <= self.type_k:

                for i, case in enumerate(classical_cases):
                    self.json_data[case]["cluster_id"] = cluster_id_counter
                    all_medoids[cluster_id_counter] = case
                    cluster_id_counter += 1
                print(f"  Insufficient computational-basis cases, assigning directly to {len(classical_cases)} clusters")
            else:

                class_clusters = self._cluster_subset(
                    classical_cases, self.type_k, cluster_id_counter, cache_label="classical"
                )


                for cluster_id in range(cluster_id_counter, cluster_id_counter + self.type_k):
                    if cluster_id in class_clusters['medoids']:
                        all_medoids[cluster_id] = class_clusters['medoids'][cluster_id]
                cluster_id_counter += self.type_k


        if superposition_cases:
            print(f"  Superposition test cases: {len(superposition_cases)}")

            if len(superposition_cases) <= self.type_k:

                for i, case in enumerate(superposition_cases):
                    self.json_data[case]["cluster_id"] = cluster_id_counter
                    all_medoids[cluster_id_counter] = case
                    cluster_id_counter += 1
                print(f"  Insufficient superposition cases, assigning directly to {len(superposition_cases)} clusters")
            else:

                super_clusters = self._cluster_subset(
                    superposition_cases, self.type_k, cluster_id_counter, cache_label="superposition"
                )


                for cluster_id in range(cluster_id_counter, cluster_id_counter + self.type_k):
                    if cluster_id in super_clusters['medoids']:
                        all_medoids[cluster_id] = super_clusters['medoids'][cluster_id]
                cluster_id_counter += self.type_k


        if entangled_cases:
            print(f"  Entangled test cases: {len(entangled_cases)}")


            for case in entangled_cases:
                self.json_data[case]["cluster_id"] = cluster_id_counter


            if len(entangled_cases) == 1:
                all_medoids[cluster_id_counter] = entangled_cases[0]
            else:

                medoid = self._select_medoid_for_subset(entangled_cases, cache_label="entangled")
                all_medoids[cluster_id_counter] = medoid

            cluster_id_counter += 1


        if mixed_cases and self.use_mixstate:
            print(f"  Mixed-state test cases: {len(mixed_cases)}")


            for case in mixed_cases:
                self.json_data[case]["cluster_id"] = cluster_id_counter


            if len(mixed_cases) == 1:
                all_medoids[cluster_id_counter] = mixed_cases[0]
            else:

                medoid = self._select_medoid_for_subset(mixed_cases, cache_label="mixed")
                all_medoids[cluster_id_counter] = medoid

            cluster_id_counter += 1


        self.medoids = all_medoids


        self._save_classifications()


        return [self.json_data[name]['cluster_id'] for name in self.allcases]

    def _cluster_subset(self, case_subset, n_clusters, start_cluster_id, cache_label="subset"):
        if len(case_subset) <= n_clusters:

            result = {'clusters': {}, 'medoids': {}}
            for i, case in enumerate(case_subset):
                cluster_id = start_cluster_id + i
                self.json_data[case]["cluster_id"] = cluster_id
                result['clusters'][cluster_id] = [case]
                result['medoids'][cluster_id] = case
            return result


        dist_matrix = self._compute_subset_distance_matrix(case_subset, cache_label)


        dist_matrix = np.maximum(dist_matrix, 0)
        assert np.all(dist_matrix >= 0), "Distance matrix still contains negative values!"


        actual_n_clusters = min(n_clusters, len(case_subset))
        kmedoids = KMedoids(
            n_clusters=actual_n_clusters,
            metric='precomputed',
            method='pam',
            init='random',
            random_state=42
        )
        clusters = kmedoids.fit_predict(dist_matrix)


        result = {'clusters': {}, 'medoids': {}}
        for i, case in enumerate(case_subset):
            cluster_label = int(clusters[i])
            cluster_id = start_cluster_id + cluster_label
            self.json_data[case]["cluster_id"] = cluster_id

            if cluster_id not in result['clusters']:
                result['clusters'][cluster_id] = []
            result['clusters'][cluster_id].append(case)


        for cluster_id, cases in result['clusters'].items():
            if len(cases) == 1:
                result['medoids'][cluster_id] = cases[0]
            else:

                cluster_indices = [i for i, case in enumerate(case_subset) if case in cases]
                sub_matrix = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
                avg_distances = np.mean(sub_matrix, axis=1)
                medoid_idx = cluster_indices[np.argmin(avg_distances)]
                result['medoids'][cluster_id] = case_subset[medoid_idx]

        return result

    def _compute_subset_distance_matrix(self, case_subset, cache_label="subset"):
        metric = self.distance_metric if self.distance_metric in {
            "hybrid", "fidelity", "euclidean", "trace"
        } else "hybrid"

        states = [self.json_data[name]['state'] for name in case_subset]
        attributes = [self.json_data[name]['attributes'] for name in case_subset]

        if metric == "hybrid" and cache_label == "classical":


            basis_indices = [int(np.argmax(np.abs(state))) for state in states]
            n_qubits = attributes[0]['num_qubits'] if attributes else 1

            def compute_pair(i, j):
                different_bits = bin(basis_indices[i] ^ basis_indices[j]).count('1')
                return different_bits / n_qubits
        elif metric == "hybrid" and cache_label in {"superposition", "entangled"}:


            n_qubits = attributes[0]['num_qubits'] if attributes else 0
            full_mask = (1 << n_qubits) - 1
            superposed_masks = []
            for attr in attributes:
                mask = 0
                for qubit in _nonclassical_qubits(attr, self.include_entangled_qubits):
                    mask |= 1 << qubit
                superposed_masks.append(mask)

            full_union_pairs = sum(
                1
                for i in range(len(states))
                for j in range(i + 1, len(states))
                if superposed_masks[i] | superposed_masks[j] == full_mask
            )
            full_fidelities = None
            total_subset_pairs = len(states) * (len(states) - 1) // 2


            if full_union_pairs >= max(1, total_subset_pairs // 4):
                state_arrays = [np.asarray(getattr(state, 'data', state)) for state in states]
                if all(array.ndim == 1 for array in state_arrays):
                    print(
                        f"  [Distance matrix] {cache_label}: Batch-computing "
                        f"{full_union_pairs} full-state pure-state fidelities...",
                        flush=True,
                    )
                    state_matrix = np.vstack(state_arrays)
                    overlaps = state_matrix @ state_matrix.conj().T
                    full_fidelities = np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)
                    print(
                        f"  [Distance matrix] {cache_label}: Precomputed "
                        f"{full_union_pairs} full-state pure-state fidelities"
                    )

            def compute_pair(i, j):
                if (full_fidelities is not None and
                        superposed_masks[i] | superposed_masks[j] == full_mask):
                    return 1.0 - float(full_fidelities[i, j])
                return self._compute_superposition_superposition_distance(
                    states[i], states[j], attributes[i], attributes[j]
                )
        elif metric == "hybrid" and cache_label == "mixed":
            def compute_pair(i, j):
                return trace_distance(states[i], states[j])
        elif metric == "fidelity":
            def compute_pair(i, j):
                return 1 - fast_state_fidelity(states[i], states[j])
        else:
            def compute_pair(i, j):
                return self._compute_single_distance(case_subset[i], case_subset[j], metric)

        cache_metric = (
            f"type_clustering_{cache_label}_{metric}_"
            f"mix{self.use_mixstate}_v2_data{self.dataset_version}"
        )
        if (metric == 'hybrid' and self.include_entangled_qubits and
                cache_label not in {'classical', 'mixed'}):
            cache_metric += f"_{HYBRID_DISTANCE_VERSION}"
        return self.cache.compute_or_load_resumable(
            case_subset,
            cache_metric,
            compute_pair,
            label=f"{cache_label} ({len(case_subset)} cases)",
        )

    def _select_medoid_for_subset(self, case_subset, cache_label="subset"):
        if len(case_subset) == 1:
            return case_subset[0]


        dist_matrix = self._compute_subset_distance_matrix(case_subset, cache_label)


        avg_distances = np.mean(dist_matrix, axis=1)
        medoid_idx = np.argmin(avg_distances)
        return case_subset[medoid_idx]


    def _select_type_medoids_detailed(self, type_mapping, classical_clusters=None):
        cluster_groups = {}
        for name in self.allcases:
            cluster_id = self.json_data[name]['cluster_id']
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(name)


        self.medoids = {}
        for cluster_id, cases in cluster_groups.items():
            if len(cases) == 1:
                self.medoids[cluster_id] = cases[0]
            else:

                dist_matrix = np.zeros((len(cases), len(cases)))
                for i, case1 in enumerate(cases):
                    for j, case2 in enumerate(cases):
                        if i < j:
                            dist = self._compute_single_distance(case1, case2, self.distance_metric)
                            dist_matrix[i, j] = dist
                            dist_matrix[j, i] = dist


                avg_distances = np.mean(dist_matrix, axis=1)
                medoid_idx = np.argmin(avg_distances)
                self.medoids[cluster_id] = cases[medoid_idx]


    def _compute_single_distance(self, case1, case2, distance_metric):
        type1 = case1.split('_', 1)[0]
        type2 = case2.split('_', 1)[0]

        if type1 == 'supplement':
            type1 = case1.split('_', 2)[1]
        if type2 == 'supplement':
            type2 = case2.split('_', 2)[1]

        state1 = self.json_data[case1]['state']
        state2 = self.json_data[case2]['state']
        attr1 = self.json_data[case1]['attributes']
        attr2 = self.json_data[case2]['attributes']

        if distance_metric == 'euclidean':
            return normalized_density_euclidean_distance(state1, state2)

        elif distance_metric == 'fidelity':
            return 1 - fast_state_fidelity(state1, state2)

        elif distance_metric == 'trace':
            if type1 == 'classical' and type2 == 'classical':
                dist = 1
            else:
                dist = trace_distance(state1, state2)

            return dist

        elif distance_metric == 'hybrid':
            if type1 == 'classical' and type2 == 'classical':
                return hamming_distance(state1, state2)
            elif (type1 == 'superposition' or type1 == 'entangled') and (type2 == 'superposition' or type2 == 'entangled'):
                return self._compute_superposition_superposition_distance(state1, state2, attr1, attr2)
            elif type1 == 'mixed' and type2 == 'mixed' and self.use_mixstate:
                return trace_distance(state1, state2)

    def _select_type_medoids(self, type_cases):
        for type_name, cases in type_cases.items():
            if not cases:
                continue


            if len(cases) == 1:
                self.medoids[type_name] = cases[0]
                continue


            case_subset = {name: self.json_data[name] for name in cases}
            dist_matrix = self._compute_type_internal_distance(case_subset, type_name)


            avg_distances = []
            for i, case1 in enumerate(cases):
                distances = []
                for j, case2 in enumerate(cases):
                    if i != j:
                        distances.append(dist_matrix[i, j])
                avg_distance = np.mean(distances) if distances else 0
                avg_distances.append(avg_distance)


            min_avg_idx = np.argmin(avg_distances)
            self.medoids[type_name] = cases[min_avg_idx]

    def _compute_type_internal_distance(self, case_subset, type_name):
        cases_list = list(case_subset.keys())
        n = len(cases_list)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                case1 = cases_list[i]
                case2 = cases_list[j]
                dist = self._compute_single_distance(case1, case2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def _find_optimal_clusters(self, dist_matrix, max_k=10):
        inertias = []
        k_range = range(2, min(max_k + 1, len(self.allcases)))

        for k in k_range:
            kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
            kmedoids.fit(dist_matrix)
            inertias.append(kmedoids.inertia_)


        if len(inertias) < 2:
            return 2


        second_diff = []
        for i in range(1, len(inertias) - 1):
            diff = (inertias[i + 1] - inertias[i]) - (inertias[i] - inertias[i - 1])
            second_diff.append(abs(diff))

        if second_diff:

            best_k_index = np.argmax(second_diff) + 2
            return min(best_k_index + 2, max_k)
        else:
            return 3

    def _compute_hybrid_distance(self, cases):
        state_vectors = []
        types = []
        attributes_list = []
        for name in self.allcases:
            type_str = name.split('_', 1)[0]
            if type_str == 'entangled':
                types.append('superposition')
            else:
                types.append(type_str)
            data = self.json_data[name]['state']
            state_vectors.append(data)
            attributes_list.append(self.json_data[name]['attributes'])

        n = len(state_vectors)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                type1, type2 = types[i], types[j]
                state1, state2 = state_vectors[i], state_vectors[j]
                attr1, attr2 = attributes_list[i], attributes_list[j]


                if type1 == 'classical' and type2 == 'classical':
                    dist = hamming_distance(state1, state2)
                elif type1 == 'superposition' and type2 == 'superposition':
                    dist = self._compute_superposition_superposition_distance(state1, state2, attr1, attr2)
                elif type1 == 'mixed' and type2 == 'mixed' and self.use_mixstate:
                    dist = trace_distance(state1, state2)
                elif {type1, type2} == {'classical', 'superposition'}:

                    classical_state = state1 if type1 == 'classical' else state2
                    superposition_state = state2 if type1 == 'classical' else state1
                    classical_attr = attr1 if type1 == 'classical' else attr2
                    superposition_attr = attr2 if type1 == 'classical' else attr1

                    dist = self._compute_classical_superposition_distance(
                        classical_state, superposition_state, classical_attr, superposition_attr)
                elif {type1, type2} == {'classical', 'mixed'} and self.use_mixstate:
                    classical_state = state1 if type1 == 'classical' else state2
                    mixed_state = state2 if type1 == 'classical' else state1
                    dist = trace_distance(classical_state, mixed_state)
                elif {type1, type2} == {'superposition', 'mixed'} and self.use_mixstate:
                    superposition_state = state1 if type1 == 'superposition' else state2
                    mixed_state = state2 if type1 == 'superposition' else state1
                    dist = trace_distance(superposition_state, mixed_state)
                else:

                    dist = hamming_distance(state1, state2)

                dist_matrix[i, j] = dist_matrix[j, i] = dist

        return dist_matrix

    def _compute_fidelity_distance(self, cases):
        state_vectors = []
        for name in self.allcases:
            data = self.json_data[name]['state']
            state_vectors.append(data)

        n = len(state_vectors)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = 1 - fast_state_fidelity(state_vectors[i], state_vectors[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        return dist_matrix

    def _compute_trace_distance(self, cases):
        state_vectors = [self.json_data[name]['state'] for name in self.allcases]

        n = len(state_vectors)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = trace_distance(state_vectors[i], state_vectors[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        return dist_matrix


    def _compute_euclidean_distance(self, cases):
        states = [self.json_data[name]['state'] for name in self.allcases]
        if any(isinstance(state, testcase_io.LowRankDensityMatrix) for state in states):
            n = len(states)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    dist = normalized_density_euclidean_distance(states[i], states[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = dist
            return dist_matrix

        state_vectors = []
        for name in self.allcases:
            data = self.json_data[name]['state']

            if hasattr(data, 'data'):
                real_vec = np.concatenate([data.data.real, data.data.imag])
            else:
                real_vec = np.concatenate([data.real, data.imag])
            state_vectors.append(real_vec)

        n = len(state_vectors)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(state_vectors[i] - state_vectors[j])

                max_dist = np.sqrt(2)
                dist = dist / max_dist
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        return dist_matrix


    def _compute_superposition_superposition_distance(self, state1, state2, attr1, attr2):
        n_qubits = attr1['num_qubits']


        include_entangled = getattr(self, 'include_entangled_qubits', False)
        superposed_qubits1 = _nonclassical_qubits(attr1, include_entangled)
        superposed_qubits2 = _nonclassical_qubits(attr2, include_entangled)


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
            superposition_distance = 1 - fast_state_fidelity(superposition_part1, superposition_part2)
        else:
            superposition_distance = 0


        total_qubits = n_qubits
        classical_weight = len(common_classical_qubits) / total_qubits
        superposition_weight = len(any_superposed_qubits) / total_qubits


        if classical_weight + superposition_weight > 0:
            total_distance = (classical_weight * classical_distance + superposition_weight * superposition_distance) / (
                    classical_weight + superposition_weight)
        else:
            total_distance = 0

        return total_distance

    def _compute_classical_superposition_distance(self, classical_state, superposition_state, classical_attr,
                                                  superposition_attr):

        n_qubits = classical_attr['num_qubits']


        superposed_qubits = _nonclassical_qubits(
            superposition_attr, getattr(self, 'include_entangled_qubits', False)
        )


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


            superposition_part_state = self._extract_qubits_state(superposition_state, superposed_part_qubits, n_qubits)

            superposition_distance = 1 - fast_state_fidelity(classical_part_state, superposition_part_state)
        else:
            superposition_distance = 0


        total_qubits = n_qubits
        classical_weight = len(classical_part_qubits) / total_qubits
        superposition_weight = len(superposed_part_qubits) / total_qubits


        if classical_weight + superposition_weight > 0:
            total_distance = (classical_weight * classical_distance + superposition_weight * superposition_distance) / (
                    classical_weight + superposition_weight)
        else:
            total_distance = 0

        return total_distance

    def _extract_classical_state(self, classical_state, n_qubits):
        idx = np.argmax(np.abs(classical_state))
        bin_str = format(idx, f'0{n_qubits}b')
        return [int(bit) for bit in bin_str]

    def _extract_qubits_state(self, full_state, target_qubits, n_qubits):
        return _extract_cached_subsystem(self, full_state, target_qubits, n_qubits)

    def _find_medoids(self, case_names, dist_matrix, clusters):
        unique_clusters = set(clusters)
        medoids = {}

        for cluster_id in unique_clusters:
            indices = np.where(clusters == cluster_id)[0]
            sub_matrix = dist_matrix[np.ix_(indices, indices)]
            avg_distances = np.mean(sub_matrix, axis=1)
            medoid_idx = indices[np.argmin(avg_distances)]
            medoids[int(cluster_id)] = case_names[medoid_idx]


        self.medoids = medoids

    def build_clusters(self):
        clusters = {}


        for name, data in self.json_data.items():
            if 'cluster_id' in data:
                cluster_id = data['cluster_id']
                if cluster_id not in clusters:
                    clusters[cluster_id] = [name]
                else:
                    clusters[cluster_id].append(name)

        return clusters


# Checkpoint partial matrices so interrupted distance calculations can resume.
class DistanceMatrixCache:
    def __init__(self, cache_dir, preserve_existing=False):
        self.cache_dir = cache_dir
        self.preserve_existing = preserve_existing
        os.makedirs(cache_dir, exist_ok=True)

    def _new_cache(self):
        return DistanceMatrixCache(os.path.join(self.cache_dir, 'new_computations'))

    def _get_cache_key(self, cases, metric):
        case_ids = sorted([case for case in cases])
        key_str = f"{metric}:{':'.join(case_ids)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _get_ordered_cache_key(self, cases, metric):
        key_str = f"resumable-v1:{metric}:{':'.join(cases)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    @staticmethod
    def _format_duration(seconds):
        if seconds is None or not math.isfinite(seconds):
            return "Unknown"
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes}min{secs}s"
        if minutes:
            return f"{minutes}min{secs}s"
        return f"{secs}s"

    @staticmethod
    def _write_json_atomic(path, data):
        temp_path = f"{path}.{os.getpid()}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)

    def compute_or_load(self, cases, metric, compute_func):
        key = self._get_cache_key(cases, metric)
        cache_path = self._get_cache_path(key)

        if os.path.exists(cache_path):
            return np.load(cache_path)

        if self.preserve_existing:
            return self._new_cache().compute_or_load(cases, metric, compute_func)

        dist_matrix = compute_func(cases)
        np.save(cache_path, dist_matrix)
        return dist_matrix

    @staticmethod
    def _process_is_running(pid):
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False

    def compute_or_load_resumable(self, cases, metric, compute_pair, label,
                                  progress_interval=60, checkpoint_interval=60):

        ordered_cases = list(cases)
        key = self._get_ordered_cache_key(ordered_cases, metric)
        cache_path = self._get_cache_path(key)
        if self.preserve_existing:
            if os.path.exists(cache_path):
                print(f"  [Distance matrix] {label}: Loaded complete cache {cache_path}")
                return np.load(cache_path, mmap_mode='r')

            return self._new_cache().compute_or_load_resumable(
                ordered_cases, metric, compute_pair, label,
                progress_interval=progress_interval,
                checkpoint_interval=checkpoint_interval,
            )
        lock_path = os.path.join(self.cache_dir, f"{key}.lock")
        last_wait_report = 0.0

        while True:
            if os.path.exists(cache_path):
                print(f"  [Distance matrix] {label}: Loaded complete cache {cache_path}")
                return np.load(cache_path, mmap_mode='r')
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(lock_fd, 'w', encoding='utf-8') as lock_file:
                    json.dump({'pid': os.getpid(), 'created_at': time.time()}, lock_file)
                break
            except FileExistsError:
                owner_pid = None
                try:
                    with open(lock_path, 'r', encoding='utf-8') as lock_file:
                        owner_pid = json.load(lock_file).get('pid')
                except (OSError, ValueError, json.JSONDecodeError):

                    pass

                if owner_pid is None:
                    try:
                        lock_age = time.time() - os.path.getmtime(lock_path)
                    except OSError:
                        lock_age = 0
                    if lock_age > 120:
                        try:
                            os.remove(lock_path)
                        except FileNotFoundError:
                            pass
                        continue

                if owner_pid is not None and not self._process_is_running(owner_pid):
                    try:
                        os.remove(lock_path)
                    except FileNotFoundError:
                        pass
                    continue

                now = time.monotonic()
                if now - last_wait_report >= 60:
                    print(
                        f"  [Distance matrix] {label}: Process {owner_pid or 'unknown'} is computing this cache; waiting to reuse its results...",
                        flush=True,
                    )
                    last_wait_report = now
                time.sleep(2)

        try:
            return self._compute_or_load_resumable_unlocked(
                ordered_cases,
                metric,
                compute_pair,
                label,
                progress_interval,
                checkpoint_interval,
            )
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

    def _compute_or_load_resumable_unlocked(self, cases, metric, compute_pair, label,
                                            progress_interval, checkpoint_interval):

        ordered_cases = list(cases)
        n = len(ordered_cases)
        key = self._get_ordered_cache_key(ordered_cases, metric)
        cache_path = self._get_cache_path(key)
        partial_path = os.path.join(self.cache_dir, f"{key}.partial.npy")
        progress_path = os.path.join(self.cache_dir, f"{key}.progress.json")

        if os.path.exists(cache_path):
            print(f"  [Distance matrix] {label}: Loaded complete cache {cache_path}")
            return np.load(cache_path, mmap_mode='r')

        start_row = 0
        dist_matrix = None
        if os.path.exists(partial_path) and os.path.exists(progress_path):
            try:
                with open(progress_path, 'r', encoding='utf-8') as file:
                    progress = json.load(file)
                if (progress.get('version') == 1 and
                        progress.get('metric') == metric and
                        progress.get('cases') == ordered_cases and
                        progress.get('shape') == [n, n]):
                    dist_matrix = np.load(partial_path, mmap_mode='r+')
                    start_row = max(0, min(int(progress.get('next_row', 0)), n))
            except (OSError, ValueError, json.JSONDecodeError):

                dist_matrix = None
                start_row = 0

        if dist_matrix is None:
            dist_matrix = np.lib.format.open_memmap(
                partial_path, mode='w+', dtype=np.float64, shape=(n, n)
            )
            dist_matrix[:] = 0.0
            dist_matrix.flush()
            self._write_json_atomic(progress_path, {
                'version': 1,
                'metric': metric,
                'cases': ordered_cases,
                'shape': [n, n],
                'next_row': 0,
            })

        total_pairs = n * (n - 1) // 2
        completed_pairs = start_row * (2 * n - start_row - 1) // 2
        initial_completed_pairs = completed_pairs
        run_started = time.perf_counter()
        last_report = run_started
        last_checkpoint = run_started

        def report_progress(current_completed, now, force=False):
            nonlocal last_report
            if not force and now - last_report < progress_interval:
                return
            elapsed = now - run_started
            new_pairs = current_completed - initial_completed_pairs
            rate = new_pairs / elapsed if elapsed > 0 else 0
            remaining = total_pairs - current_completed
            eta = remaining / rate if rate > 0 else None
            percent = 100.0 if total_pairs == 0 else current_completed * 100.0 / total_pairs
            print(
                f"  [Distance matrix] {label}: Completed {current_completed}/{total_pairs} "
                f"({percent:.2f}%), Elapsed "
                f"{self._format_duration(elapsed)}, ETA "
                f"{self._format_duration(eta)}",
                flush=True,
            )
            last_report = now

        if start_row:
            percent = 100.0 if total_pairs == 0 else completed_pairs * 100.0 / total_pairs
            print(f"  [Distance matrix] {label}: Resuming from cache, Completed "
                  f"{completed_pairs}/{total_pairs} ({percent:.2f}%)")
        else:
            print(f"  [Distance matrix] {label}: Starting computation, total {total_pairs} distances")

        try:
            for i in range(start_row, n):
                row_completed = 0
                for j in range(i + 1, n):
                    distance = compute_pair(i, j)
                    dist_matrix[i, j] = distance
                    dist_matrix[j, i] = distance
                    row_completed += 1

                    if row_completed % 10 == 0:
                        report_progress(completed_pairs + row_completed, time.perf_counter())

                completed_pairs += row_completed
                now = time.perf_counter()
                report_progress(completed_pairs, now, force=(i == n - 1))

                if now - last_checkpoint >= checkpoint_interval or i == n - 1:
                    dist_matrix.flush()
                    self._write_json_atomic(progress_path, {
                        'version': 1,
                        'metric': metric,
                        'cases': ordered_cases,
                        'shape': [n, n],
                        'next_row': i + 1,
                    })
                    last_checkpoint = now
        except BaseException:

            dist_matrix.flush()
            raise

        dist_matrix.flush()
        del dist_matrix
        os.replace(partial_path, cache_path)
        if os.path.exists(progress_path):
            os.remove(progress_path)
        print(f"  [Distance matrix] {label}: Computation complete, Cache saved to {cache_path}")
        return np.load(cache_path, mmap_mode='r')


def parse_oracle_attributes(raw_attributes):
    """Accept JSON attributes and legacy Python dict literals without using eval."""
    if isinstance(raw_attributes, dict):
        return raw_attributes
    try:
        attributes = json.loads(raw_attributes)
    except (TypeError, ValueError):
        try:
            attributes = ast.literal_eval(raw_attributes)
        except (TypeError, ValueError, SyntaxError) as exc:
            raise ValueError("Oracle attributes must be a JSON object or Python dict literal") from exc
    if not isinstance(attributes, dict):
        raise ValueError("Oracle attributes must decode to a dictionary")
    return attributes


def validate_mixed_oracle(oracle_data, test_cases):
    mixed_cases = {
        name: data['attributes'] for name, data in test_cases.items()
        if data['attributes'].get('type') == 'mixed'
        or data['attributes'].get('is_mixed') is True
    }

    oracle_attributes = dict(zip(oracle_data.iloc[:, 0], oracle_data.iloc[:, 2]))
    for case_name, attributes in mixed_cases.items():
        if case_name not in oracle_attributes:
            raise ValueError(f"Oracle is missing a mixed-state test case: {case_name}")
        try:
            oracle_attr = parse_oracle_attributes(oracle_attributes[case_name])
        except ValueError as exc:
            raise ValueError(f"Cannot parse mixed-state oracle attributes: {case_name}; {exc}") from exc
        if oracle_attr.get('type') != 'mixed' and oracle_attr.get('is_mixed') is not True:
            raise ValueError(f"Mixed-state types in the test suite and oracle do not match: {case_name}")
        expected = attributes.get('state_representation') or 'legacy'
        actual = oracle_attr.get('state_representation') or 'legacy'
        if actual != expected:
            raise ValueError(
                f"Mixed-state test-suite and oracle versions do not match: {case_name}; "
                f"test cases={expected}, Oracle={actual}. Use the matching oracle. "
            )


# Keep one-time preprocessing separate from repeated online selection.
class QHDRT:
    def __init__(self, number_of_variables, input_qbit_count, output_qbit_count, group_name, program_name, algorithm,
                 iter_num, classification_method='direct', distance_metric="hybrid", n_clusters=5, use_mixstate=False,
                 verbose=False, epsilon=0.05, delta=0.05, seed=None, result_dir=None,
                 type_k=3, experiment_cache_dir=None, oracle_path=None,
                 validate_mixed_oracle_version=True, include_entangled_qubits=False):
        self.number_of_variables = number_of_variables
        self.input_qbit_count = input_qbit_count
        self.output_qbit_count = output_qbit_count
        self.group_name = group_name
        self.program_name = program_name
        self.algorithm = algorithm
        self.iter_num = iter_num
        self.distance_metric = distance_metric
        self.include_entangled_qubits = include_entangled_qubits
        self.n_clusters = n_clusters
        self.use_mixstate = use_mixstate
        self.classification_method = classification_method
        self.verbose = verbose
        if not (math.isfinite(epsilon) and math.isfinite(delta)
                and 0 <= epsilon <= 1 and 0 <= delta <= 1):
            raise ValueError("epsilon and delta must be finite values in [0, 1]")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.seed = seed
        if result_dir is None:
            result_root = './result'
            if include_entangled_qubits:
                result_root = os.path.join(result_root, HYBRID_DISTANCE_VERSION)
            result_dir = _new_run_directory(os.path.join(result_root, self.group_name))
        self.result_dir = result_dir

        self.test_cases = []
        self.max_fail_count = -1
        self.best_candidates = []


        oracle_path = oracle_path or os.path.join("./oracle/OpenQASM/" + self.program_name + "_oracle.xlsx")
        oracle_data = pd.read_excel(oracle_path, sheet_name=self.group_name, dtype=str)
        tcname_oracle_col = oracle_data.iloc[:, 0]
        reslult_oracle_col = oracle_data.iloc[:, 3]
        pvalue_oracle_col = oracle_data.iloc[:, 4]
        self.oracle = dict(zip(tcname_oracle_col, reslult_oracle_col))
        self.pvalues = dict(zip(tcname_oracle_col, pvalue_oracle_col))
        self.All_candidates = []


        self.all_runs_fail_history = []
        self.all_runs_test_cases = []


        loaded = self._load_test_cases(f"./TestCases/quantum_test_cases_{self.input_qbit_count}qubits.json")
        all_test_cases = loaded["test_cases"]


        if not self.use_mixstate:
            filtered_test_cases = {}
            for name, data in all_test_cases.items():

                type_str = name.split('_', 1)[0]


                if type_str not in ['classical', 'superposition', 'entangled', 'mixed']:
                    if 'attributes' in data and 'type' in data['attributes']:
                        type_str = data['attributes']['type']
                    else:

                        type_str = 'classical'


                if type_str != 'mixed':
                    filtered_test_cases[name] = data
            self.json_data = filtered_test_cases
        else:
            self.json_data = all_test_cases

        if self.use_mixstate and validate_mixed_oracle_version:
            validate_mixed_oracle(oracle_data, self.json_data)

        self.All_candidates = [v["state"] for v in self.json_data.values()]
        self.all_cases = {v for v in self.json_data}


        cache_dir = './distance_cache/' + str(self.input_qbit_count) + '_distance_cache'
        classification_file = f'./TestCases/classification/{self.distance_metric}/{self.input_qbit_count}_quantum_classifications_{self.classification_method}_{"MixState" if self.use_mixstate else "NoMixState"}_n{self.n_clusters}.json'
        if experiment_cache_dir is not None:
            cache_root = os.fspath(experiment_cache_dir)
            cache_dir = os.path.join(cache_root, 'distances', str(self.input_qbit_count))
            classification_file = os.path.join(
                cache_root, 'classifications',
                f'{self.input_qbit_count}_{self.classification_method}_{self.distance_metric}_'
                f'mix{self.use_mixstate}_n{self.n_clusters}_typek{type_k}.json')
            os.makedirs(os.path.dirname(classification_file), exist_ok=True)
        elif self.classification_method == 'type_clustering' and type_k != 3:
            classification_file = classification_file.replace('.json', f'_typek{type_k}.json')
        self.clusterer = QuantumTestClusterer(self.json_data, cache_dir, classification_file,
                                              distance_metric, n_clusters, use_mixstate, classification_method,
                                              type_k=type_k,
                                              include_entangled_qubits=include_entangled_qubits)
        self.perform_clustering()
        self._prepare_online_templates()


        self.upper_bound = len(self.All_candidates)
        self.current_step = 0

    def _prepare_online_templates(self):
        self.base_subclusters = {}
        self.base_case_to_subcluster = {}

        for cluster_id, cases in self.clusters.items():
            case_list = list(cases)
            self.base_subclusters[cluster_id] = {
                'cases': case_list,
                'medoid': self.clusterer.medoids[cluster_id]
            }
            for case in case_list:
                self.base_case_to_subcluster[case] = cluster_id

        total_clusters = len(self.base_subclusters)
        self.base_subcluster_probs = {cluster_id: 1.0 / total_clusters for cluster_id in self.base_subclusters}
        self.base_subcluster_indices = {
            sc_id: idx for idx, sc_id in enumerate(self.base_subclusters)
        }
        self.base_medoid_dist_matrix = self._build_medoid_distance_matrix(self.base_subclusters)

    def _initialize_probabilities(self):
        self.subclusters = {
            cluster_id: {
                'cases': list(data['cases']),
                'medoid': data['medoid']
            }
            for cluster_id, data in self.base_subclusters.items()
        }
        self.case_to_subcluster = self.base_case_to_subcluster
        self.subcluster_probs = dict(self.base_subcluster_probs)
        self.medoid_dist_matrix = self.base_medoid_dist_matrix.copy()
        self.subcluster_indices = dict(self.base_subcluster_indices)

    def _build_medoid_distance_matrix(self, subclusters):
        medoid_cases = [data['medoid'] for data in subclusters.values()]


        dist_matrix = np.zeros((len(medoid_cases), len(medoid_cases)))
        for i, medoid1 in enumerate(medoid_cases):
            for j, medoid2 in enumerate(medoid_cases[i + 1:], i + 1):
                dist = self.medoid_distance(medoid1, medoid2, self.distance_metric)
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist


        if np.max(dist_matrix) > 0:
            dist_matrix /= np.max(dist_matrix)
        return dist_matrix

    def build_distance_matrix(self):
        self.medoid_dist_matrix = self._build_medoid_distance_matrix(self.subclusters)

    def _extract_features(self, case_name):
        data = self.json_data[case_name]
        state = data['state']
        attributes = data['attributes']


        if hasattr(state, 'data'):

            state_data = state.data
        else:

            state_data = state


        real_features = np.real(state_data)
        imag_features = np.imag(state_data)


        features = np.concatenate([real_features, imag_features])


        attr_features = []
        if 'num_qubits' in attributes:
            attr_features.append(attributes['num_qubits'])
        if 'base_state' in attributes:
            attr_features.extend(attributes['base_state'])
        if 'superposed_qubits' in attributes:
            superposed = [1 if i in attributes['superposed_qubits'] else 0
                          for i in range(attributes['num_qubits'])]
            attr_features.extend(superposed)


        all_features = np.concatenate([features, np.array(attr_features)])

        return all_features

    def medoid_distance(self, medoid1, medoid2, distance_metric):
        type1 = self.json_data[medoid1]['attributes'].get('type', medoid1.split('_', 1)[0])
        type2 = self.json_data[medoid2]['attributes'].get('type', medoid2.split('_', 1)[0])
        if type1 == 'entangled':
            type1 = 'superposition'
        if type2 == 'entangled':
            type2 = 'superposition'

        state1 = self.json_data[medoid1]['state']
        state2 = self.json_data[medoid2]['state']
        attr1 = self.json_data[medoid1]['attributes']
        attr2 = self.json_data[medoid2]['attributes']

        if distance_metric == 'euclidean':
            return normalized_density_euclidean_distance(state1, state2)

        elif distance_metric == 'fidelity':
            return 1 - fast_state_fidelity(state1, state2)

        elif distance_metric == 'trace':
            return trace_distance(state1, state2)

        elif distance_metric == 'hybrid':

            if type1 == 'classical' and type2 == 'classical':
                return hamming_distance(state1, state2)
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
                return trace_distance(classical_state, mixed_state)
            elif {type1, type2} == {'superposition', 'mixed'} and self.use_mixstate:
                superposition_state = state1 if type1 == 'superposition' else state2
                mixed_state = state2 if type1 == 'superposition' else state1
                return trace_distance(superposition_state, mixed_state)
            else:

                return hamming_distance(state1, state2)


    def _compute_superposition_superposition_distance(self, state1, state2, attr1, attr2):
        n_qubits = attr1['num_qubits']


        include_entangled = getattr(self, 'include_entangled_qubits', False)
        superposed_qubits1 = _nonclassical_qubits(attr1, include_entangled)
        superposed_qubits2 = _nonclassical_qubits(attr2, include_entangled)


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
            superposition_distance = 1 - fast_state_fidelity(superposition_part1, superposition_part2)
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


        superposed_qubits = _nonclassical_qubits(
            superposition_attr, getattr(self, 'include_entangled_qubits', False)
        )


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

            superposition_distance = 1 - fast_state_fidelity(classical_part_state, superposition_part_state)
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

    def _extract_classical_state(self, classical_state, n_qubits):
        idx = np.argmax(np.abs(classical_state))
        bin_str = format(idx, f'0{n_qubits}b')
        return [int(bit) for bit in bin_str]

    def _extract_qubits_state(self, full_state, target_qubits, n_qubits):
        return _extract_cached_subsystem(self, full_state, target_qubits, n_qubits)

    def select_test_case(self):
        clusters, probs = zip(*self.subcluster_probs.items())
        selected_cluster = np.random.choice(clusters, p=probs)

        cases_in_cluster = self.subclusters[selected_cluster]['cases']
        if not cases_in_cluster:
            raise ValueError(f"No remaining cases in cluster {selected_cluster}")

        selected_case = np.random.choice(cases_in_cluster)
        cases_in_cluster.remove(selected_case)

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
        weight_sum = np.sum(weights)

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
                if weight_sum > 0:
                    adjust = adjust_direction * (weights[i] / weight_sum)
                else:
                    adjust = adjust_direction * (1 / (len(self.subcluster_probs) - 1))
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
        if not self.subclusters[cluster]['cases']:

            deleted_idx = self.subcluster_indices[cluster]
            self.medoid_dist_matrix = np.delete(
                np.delete(self.medoid_dist_matrix, deleted_idx, axis=0),
                deleted_idx, axis=1
            )


            del self.subclusters[cluster]
            del self.subcluster_probs[cluster]


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

        max_try = self.upper_bound

        stats = defaultdict(int)
        result_chunks = []
        mixstate_suffix = "MixState" if self.use_mixstate else "NoMixState"
        filename_suffix = f"{self.classification_method}_{self.distance_metric}_n{self.n_clusters}_{mixstate_suffix}"

        for iter in range(self.iter_num):
            if self.seed is not None:
                random.seed(self.seed + iter)
                np.random.seed(self.seed + iter)
            self.current_step = 0
            self.executed_order = []
            self.fault_counts = []
            self.pvalue_history = []
            self.count_fail = {k: 0 for k in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}
            run_fail_history = []


            self._initialize_probabilities()

            if self.verbose:
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


                self.update_probabilities(cluster, test_passed)


                self._clean_empty_structures(case, cluster, test_passed)

                if self.verbose:
                    self._print_current_state(case, test_passed)

            if 0.3 in self.count_fail:
                if self.count_fail[0.3] > self.max_fail_count:
                    self.max_fail_count = self.count_fail[0.3]
                    self.best_candidates = self.executed_order


            self.all_runs_test_cases.append(self.executed_order)
            self.all_runs_fail_history.append(run_fail_history)

            result_chunks.append(' '.join(map(str, self.executed_order)) + ' \n')
            result_chunks.append(' '.join(map(str, self.fault_counts)) + ' \n')
            result_chunks.append(' '.join(map(str, self.pvalue_history)) + ' \n')
            result_chunks.append(json.dumps(self.count_fail) + '\n')

        os.makedirs(self.result_dir, exist_ok=True)
        result_file = os.path.join(self.result_dir, f'{self.program_name}_{self.algorithm}_{filename_suffix}.txt')
        with open(result_file, 'a') as f:
            f.writelines(result_chunks)

        self.preprocessing_time = getattr(self, "preprocessing_time", 0.0)
        self.online_selection_time = elapsed_since(self.start_computing_time)
        self.total_computing_time = self.preprocessing_time + self.online_selection_time
        memory_monitor = getattr(self, "memory_monitor", None)
        self.total_memory_usage = (
            memory_monitor.snapshot() if memory_monitor is not None else get_memory_usage()
        )


        time_file = os.path.join(self.result_dir, f'{self.program_name}_{self.algorithm}_{filename_suffix}_time.txt')
        with open(time_file, 'a') as f:
            f.write(format_time_costs(
                self.iter_num,
                self.total_computing_time,
                self.preprocessing_time,
                self.online_selection_time,
                getattr(self, "preprocessing_memory_usage", None),
                self.total_memory_usage,
            ))

    def perform_clustering(self):
        print("Clustering cases...")
        self.clusterer.cluster_cases()
        self.clusters = self.clusterer.build_clusters()

    def _load_test_cases(self, filename):
        exclude_types = {"mixed"} if not self.use_mixstate else None
        return {
            "test_cases": testcase_io.load_test_cases(
                filename,
                as_dict=True,
                exclude_types=exclude_types,
                compact_pure_density=self.use_mixstate,
            )
        }

    def reset(self):
        self.test_cases = []

    def get_result(self):
        return self.best_candidates

def save_t_measure_data(algorithm, group_name, program_name, algorithm_name, distance_metric, n_clusters,
                        use_mixstate, classification_method):


    result_dir = os.path.join(
        getattr(algorithm, 'result_dir', f'./result/{group_name}'), 't_measure_data'
    )
    os.makedirs(result_dir, exist_ok=True)


    mixstate_suffix = "MixState" if use_mixstate else "NoMixState"
    filename = f"{program_name}_{algorithm_name}_{classification_method}_{distance_metric}_n{n_clusters}_{mixstate_suffix}.json"

    filepath = os.path.join(result_dir, filename)


    data = {
        "algorithm": algorithm_name,
        "group_name": group_name,
        "program_name": program_name,
        "distance_metric": distance_metric,
        "n_clusters": n_clusters,
        "use_mixstate": use_mixstate,
        "classification_method": classification_method,
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

def process_results(alg, group_name, program_name, distance_metric, n_clusters, use_mixstate,
                    classification_method, cand_num=1, result_dir=None, summary_dir=None):


    mixstate_suffix = "MixState" if use_mixstate else "NoMixState"
    filename_suffix = f"{classification_method}_{distance_metric}_n{n_clusters}_{mixstate_suffix}"
    result_dir = result_dir or f'./result/{group_name}'
    summary_dir = summary_dir or './result'
    os.makedirs(summary_dir, exist_ok=True)

    if alg == 'ART_1':
        file_path = os.path.join(result_dir, f'{program_name}_{alg}_{filename_suffix}(cand_num={cand_num}).txt')
    else:
        file_path = os.path.join(result_dir, f'{program_name}_{alg}_{filename_suffix}.txt')

    count_fail = {}

    try:
        with open(file_path, 'r') as f:
            data = f.readlines()
            nums = []
            if alg in ['DRT_1', 'RPT', 'DDRT', 'DDRT_F', 'QHDRT', 'DDRT_E']:
                for i in range(3, len(data), 4):
                    d = json.load(io.StringIO(data[i].strip()))
                    for k, v in d.items():
                        if k not in count_fail:
                            count_fail[k] = [v]
                        else:
                            count_fail[k].append(v)
            else:
                for i in range(2, len(data), 3):
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


    stats_file = os.path.join(result_dir, f'{program_name}_{alg}_{filename_suffix}_failCount.txt')
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
        workbook1 = openpyxl.load_workbook(os.path.join(summary_dir, avg_filename))
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


    workbook1.save(os.path.join(summary_dir, avg_filename))


    try:
        workbook2 = openpyxl.load_workbook(os.path.join(summary_dir, max_filename))
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


    workbook2.save(os.path.join(summary_dir, max_filename))

def main():
    parser = argparse.ArgumentParser(description='Quantum test selection algorithm')


    parser.add_argument('input_qbit_count', type=int, help='Input qubit count')
    parser.add_argument('output_qbit_count', type=int, help='Output qubit count')
    parser.add_argument('group_name', type=str, help='Benchmark group name (e.g., adder_n7_1)')
    parser.add_argument('program_name', type=str, help='Canonical program name (e.g., adder_n7)')
    parser.add_argument('alg', type=str, help='Algorithm name(QHDRT)')


    parser.add_argument('--distance_metric', type=str, default='hybrid',
                        choices=['hybrid', 'fidelity', 'euclidean', 'trace'],
                        help='Dissimilarity measure: hybrid, fidelity, euclidean, or trace')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters')
    parser.add_argument('--use_mixstate', action='store_true',
                        help='Include mixed-state test cases')
    parser.add_argument('--iter_num', type=int, default=50,
                        help='Repetition count')
    parser.add_argument('--cand_num', type=int, default=10,
                        help='Number of candidate test cases(Compatibility option)')

    parser.add_argument('--classification_method', type=str, default='detailed_type2',
                        choices=['direct', 'adaptive', 'by_type', 'detailed_type1', 'detailed_type2', 'type_clustering'],
                        help='Test-case partitioning method: direct(Direct clustering), adaptive(Adaptive clustering), by_type(Partition by type), detailed_type1(Detailed partitioning1), detailed_type2(Detailed partitioning2), type_clustering(Partition by type, then cluster within each type)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print every online-selection step')
    parser.add_argument('--cost_only', action='store_true',
                        help='Collect time and memory costs only, without modifying main-experiment results')
    parser.add_argument('--cost_output_root', type=str, default='./cost_measurements',
                        help='Isolated output root for cost-only measurements')
    parser.add_argument('--experiment_cache_dir', type=str, default=None,
                        help='Isolated cache directory for this cost measurement')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Explicit result directory, Isolate outputs of dissimilarity and other subexperiments')
    parser.add_argument('--allow_legacy_mixed_oracle', action='store_true',
                        help='Allow legacy mixed-state oracle labels in cost-only mode')
    parser.add_argument('--include_entangled_qubits', action='store_true',
                        help='Include entangled qubits in the hybrid fidelity subsystem; Use the legacy algorithm and caches by default')

    args = parser.parse_args()
    if args.allow_legacy_mixed_oracle and not args.cost_only:
        parser.error('--allow_legacy_mixed_oracle may only be used with --cost_only')
    memory_monitor = PeakMemoryMonitor().start()


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


    result_root = args.cost_output_root if args.cost_only else './result'

    if args.include_entangled_qubits:
        result_root = os.path.join(result_root, HYBRID_DISTANCE_VERSION)
    result_path = args.result_dir or _new_run_directory(os.path.join(result_root, args.group_name))
    mixstate_suffix = "MixState" if args.use_mixstate else "NoMixState"
    filename_suffix = f"{args.classification_method}_{args.distance_metric}_n{args.n_clusters}_{mixstate_suffix}"

    print(f'Result directory: {os.path.abspath(result_path)}')
    print('Hybrid distance mode: ' + (
        HYBRID_DISTANCE_VERSION if args.include_entangled_qubits else 'legacy'
    ))


    experiment_start_time = time.perf_counter()
    if args.alg == "QHDRT":
        algorithm = QHDRT(num_vars, args.input_qbit_count, args.output_qbit_count,
                           args.group_name, args.program_name, args.alg, args.iter_num, args.classification_method,
                           args.distance_metric, args.n_clusters, args.use_mixstate, args.verbose,
                           result_dir=result_path,
                           experiment_cache_dir=args.experiment_cache_dir,
                           validate_mixed_oracle_version=not args.allow_legacy_mixed_oracle,
                           include_entangled_qubits=args.include_entangled_qubits)
    else:
        print(f"Algorithm {args.alg} not supported!")
        return
    algorithm.preprocessing_time = elapsed_since(experiment_start_time)
    algorithm.memory_monitor = memory_monitor
    algorithm.preprocessing_memory_usage = memory_monitor.snapshot()

    algorithm.run()
    memory_monitor.stop()

    if args.cost_only:

        raw_result_file = os.path.join(
            result_path,
            f'{args.program_name}_{args.alg}_{filename_suffix}.txt',
        )
        if os.path.exists(raw_result_file):
            os.remove(raw_result_file)
    else:

        save_t_measure_data(algorithm, args.group_name, args.program_name, args.alg,
                            args.distance_metric, args.n_clusters, args.use_mixstate, args.classification_method)
        process_results(args.alg, args.group_name, args.program_name,
                        args.distance_metric, args.n_clusters, args.use_mixstate,
                        args.classification_method, args.cand_num,
                        result_dir=result_path, summary_dir=result_path)

    print(f'Algorithm: {args.alg}')
    print(f'FaultVersion: {args.group_name}')
    print(f'Classification Method: {args.classification_method}')
    print(f'Distance_Metric: {args.distance_metric}')
    print(f'Number of Clusters: {args.n_clusters}')
    print(f'Use MixState: {args.use_mixstate}')
    print(f'Computing time: {algorithm.total_computing_time}')

if __name__ == '__main__':
    main()
