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
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import hamming
import json
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import copy
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

print(sys.getdefaultencoding())
print(sys.getfilesystemencoding())


class DRT_C:
    def __init__(self, number_of_variables, input_qbit_count, output_qbit_count, group_name, program_name, algorithm,
                 iter_num, classification_method='direct', distance_metric="hybrid", n_clusters=5,
                 use_mixstate=False):
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
        self.classification_method = classification_method

        self.test_cases = []
        self.max_fail_count = -1
        self.best_candidates = []


        oracle_path = os.path.join("./oracle/OpenQASM/" + self.program_name + "_oracle.xlsx")
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

        self._validate_oracle_compatibility(oracle_data)

        self.oracle = {
            name: self._parse_oracle_result(self.oracle[name])
            for name in self.json_data
        }

        self.All_candidates = [v["state"] for v in self.json_data.values()]
        self.all_cases = {v for v in self.json_data}


        cache_dir = './distance_cache/' + str(self.input_qbit_count) + '_distance_cache'
        classification_file = f'./TestCases/classification/{self.distance_metric}/{self.input_qbit_count}_quantum_classifications_{self.classification_method}_{"MixState" if self.use_mixstate else "NoMixState"}_n{self.n_clusters}.json'
        self.clusterer = ClassicalTestClusterer(self.json_data, cache_dir, classification_file,
                                                distance_metric, n_clusters, use_mixstate, classification_method)
        self.perform_clustering()


        self.upper_bound = len(self.All_candidates)
        self.current_step = 0

    @staticmethod
    def _parse_oracle_result(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {'true', '1', '1.0'}:
            return True
        if normalized in {'false', '0', '0.0'}:
            return False
        raise ValueError(f"oracle Test result is not a valid boolean: {value!r}")

    def _validate_oracle_compatibility(self, oracle_data):
        missing = [name for name in self.json_data if name not in self.oracle]
        if missing:
            raise ValueError(
                f"oracle Missing current {self.input_qbit_count} qubit test suite "
                f"{len(missing)} cases, for example {missing[:3]}. Check input qubit count, "
                f"program_name={self.program_name} and sheet={self.group_name} Match. "
            )
        if not self.use_mixstate:
            return
        attributes_by_name = {}
        for name, raw in zip(oracle_data.iloc[:, 0], oracle_data.iloc[:, 2]):
            try:
                attributes = json.loads(raw)
            except (TypeError, ValueError):
                try:
                    attributes = ast.literal_eval(str(raw))
                except (ValueError, SyntaxError):
                    attributes = {}
            attributes_by_name[name] = attributes if isinstance(attributes, dict) else {}
        unversioned = 0
        for name, data in self.json_data.items():
            attributes = data.get('attributes', {})
            if not (name.startswith('mixed_') or attributes.get('type') == 'mixed'):
                continue
            testcase_version = attributes.get('state_representation')
            oracle_version = attributes_by_name.get(name, {}).get('state_representation')
            if testcase_version and oracle_version and testcase_version != oracle_version:
                raise ValueError(f"Mixed states {name} test-suite version and oracle Versions do not match. ")
            if not testcase_version or not oracle_version:
                unversioned += 1
        if unversioned:
            print(
                f"Read legacy mixed oracle: {unversioned} cases missing a version tag. "
                "Verify oracle come from the same set of test states; matching names alone do not verify state identity. "
            )

    def _initialize_probabilities(self):
        self.subclusters = {}
        self.case_to_subcluster = {}
        self.subcluster_probs = {}
        self.subcluster_indices = {}


        all_subclusters = {}
        total_cases = 0

        for cluster_id, cases in self.clusters.items():
            case_list = list(cases)
            all_subclusters[cluster_id] = {
                'cases': case_list,
                'medoid': self.clusterer.medoids[cluster_id]
            }
            total_cases += len(case_list)

        self.subclusters = all_subclusters


        for cluster_id, data in self.subclusters.items():
            for case in data['cases']:
                self.case_to_subcluster[case] = cluster_id


        total_clusters = len(self.subclusters)
        self.subcluster_probs = {cluster_id: 1.0 / total_clusters for cluster_id in self.subclusters}


        self.subcluster_indices = {
            sc_id: idx for idx, sc_id in enumerate(self.subclusters)
        }


    def medoid_distance(self, medoid1, medoid2):
        features1 = self._extract_features(medoid1)
        features2 = self._extract_features(medoid2)


        dist = np.linalg.norm(features1 - features2)


        max_dist = np.sqrt(len(features1) * 4)
        return dist / max_dist if max_dist > 0 else 0

    def _extract_features(self, case_name):
        return self.clusterer._extract_features(case_name)

    def _extract_legacy_features(self, case_name):
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

    def select_test_case(self):
        clusters, probs = zip(*self.subcluster_probs.items())
        selected_cluster = np.random.choice(clusters, p=probs)


        cases_in_cluster = [
            name for name in self.subclusters[selected_cluster]['cases']
            if name in self.remaining_cases
        ]
        if not cases_in_cluster:
            raise ValueError(f"No remaining cases in cluster {selected_cluster}")

        selected_case = np.random.choice(cases_in_cluster)

        return selected_cluster, selected_case

    def update_probabilities(self, cluster, test_passed):
        if test_passed:
            adjust_direction = -self.delta
        else:
            adjust_direction = self.epsilon


        self.distance_scale = 1
        cluster_idx = self.subcluster_indices[cluster]

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
        remaining_in_cluster = [
            name for name in self.subclusters[cluster]['cases']
            if name in self.remaining_cases
        ]

        if not remaining_in_cluster:


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

        self.epsilon = 0.05
        self.delta = 0.05
        max_try = self.upper_bound

        stats = defaultdict(int)

        for iter in range(self.iter_num):
            self.current_step = 0
            self.remaining_cases = set(v for v in self.json_data)
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


                test_passed = self.oracle[case]
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


                self._print_current_state(case, test_passed)

            if 0.3 in self.count_fail:
                if self.count_fail[0.3] > self.max_fail_count:
                    self.max_fail_count = self.count_fail[0.3]
                    self.best_candidates = self.executed_order


            self.all_runs_test_cases.append(self.executed_order)
            self.all_runs_fail_history.append(run_fail_history)


            mixstate_suffix = "MixState" if self.use_mixstate else "NoMixState"
            filename_suffix = f"{self.classification_method}_{self.distance_metric}_n{self.n_clusters}_{mixstate_suffix}"


            result_file = f'./result/{self.group_name}/{self.program_name}_{self.algorithm}_{filename_suffix}.txt'
            f = open(result_file, 'a')
            for i in range(len(self.executed_order)):
                f.write(str(self.executed_order[i]))
                f.write(' ')
            f.write('\n')
            for i in range(len(self.fault_counts)):
                f.write(str(self.fault_counts[i]))
                f.write(' ')
            f.write('\n')
            for i in range(len(self.pvalue_history)):
                f.write(str(self.pvalue_history[i]))
                f.write(' ')
            f.write('\n')
            json.dump(self.count_fail, f)
            f.write('\n')
            f.close()

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

                as_numpy=False,
                exclude_types=exclude_types,
            )
        }

    def reset(self):
        self.test_cases = []

    def get_result(self):
        return self.best_candidates


class ClassicalTestClusterer:
    def __init__(self, json_data, cache_dir, classification_file, distance_metric="hybrid", n_clusters=5,
                 use_mixstate=False, classification_method='direct'):
        self.json_data = json_data
        self.cache = DistanceMatrixCache(cache_dir)
        self.classification_file = classification_file
        self.distance_metric = distance_metric
        self.n_clusters = n_clusters
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
            'use_mixstate': self.use_mixstate,
            'classification_method': self.classification_method,
            'algorithm': 'QDRT_C',
            'dataset_version': self.dataset_version,
            'feature_version': 3,
        }


        os.makedirs(os.path.dirname(self.classification_file), exist_ok=True)
        temporary = f"{self.classification_file}.{os.getpid()}.tmp"
        with open(temporary, 'w') as f:
            json.dump(save_data, f, indent=2)
        os.replace(temporary, self.classification_file)

    def cluster_cases(self):
        if self.classification_method == 'detailed_type2':
            return self._cluster_detailed_type2()
        return self._cluster_with_qhdrt()

    def _cluster_detailed_type2(self):
        return self._cluster_with_qhdrt(classification_method='detailed_type2')

    # Reuse the main partitioning and cache validation for a controlled ablation.
    def _cluster_with_qhdrt(self, classification_method=None):
        from QHDRT import QuantumTestClusterer

        shared_clusterer = QuantumTestClusterer(
            self.json_data, self.cache.cache_dir, self.classification_file,
            distance_metric=self.distance_metric,
            n_clusters=self.n_clusters,
            use_mixstate=self.use_mixstate,
            classification_method=classification_method or self.classification_method,
        )
        labels = shared_clusterer.cluster_cases()
        self.allcases = shared_clusterer.allcases
        self.medoids = shared_clusterer.medoids
        self.classification_file = shared_clusterer.classification_file
        return labels

    def _cluster_cases_legacy(self):
        if (self.loaded_classifications and
                self.loaded_classifications.get('classification_method') == self.classification_method and
                self.loaded_classifications.get('distance_metric') == self.distance_metric and
                self.loaded_classifications.get('n_clusters') == self.n_clusters and
                self.loaded_classifications.get('use_mixstate') == self.use_mixstate and
                self.loaded_classifications.get('algorithm') == 'QDRT_C' and
                self.loaded_classifications.get('dataset_version', 'legacy') == self.dataset_version and
                self.loaded_classifications.get('feature_version') == 3):

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


        if self.distance_metric == "euclidean":
            dist_func = self._compute_euclidean_distance
        else:
            dist_func = self._compute_euclidean_distance


        if self.classification_method == 'direct':

            dist_matrix = self.cache.compute_or_load(
                {name: self.json_data[name] for name in self.allcases},
                f'QDRT_C_feature_v3_{self.distance_metric}_n{self.n_clusters}_'
                f'mix{self.use_mixstate}_data{self.dataset_version}',
                dist_func
            )


            dist_matrix = np.maximum(dist_matrix, 0)
            assert np.all(dist_matrix >= 0), "Distance matrix still contains negative values!"
            return self._cluster_direct(dist_matrix)
        elif self.classification_method == 'detailed_type2':

            return self._cluster_detailed_type2()
        else:

            dist_matrix = self.cache.compute_or_load(
                {name: self.json_data[name] for name in self.allcases},
                f'QDRT_C_feature_v3_{self.distance_metric}_n{self.n_clusters}_'
                f'mix{self.use_mixstate}_data{self.dataset_version}',
                dist_func
            )


            dist_matrix = np.maximum(dist_matrix, 0)
            assert np.all(dist_matrix >= 0), "Distance matrix still contains negative values!"
            return self._cluster_direct(dist_matrix)

    def _cluster_direct(self, dist_matrix):
        print("Using direct clustering...")

        clusters = self._load_cluster_labels()
        if clusters is None:


            kmedoids = KMedoids(
                n_clusters=self.n_clusters,
                metric='precomputed',
                method='pam',
                init='random',
                random_state=42
            )
            clusters = kmedoids.fit_predict(dist_matrix)
            self._save_cluster_labels(clusters)
        else:
            print("from QDRT_C Cluster labels checkpoint Resume, Skip K-Medoids. ")


        for i, name in enumerate(self.allcases):
            self.json_data[name]["cluster_id"] = int(clusters[i])


        self._find_medoids(self.allcases, dist_matrix, clusters)


        self._save_classifications()

        return clusters

    def _cluster_labels_path(self):
        identity = (
            f"QDRT_C_labels_feature_v3:{self.distance_metric}:{self.n_clusters}:"
            f"{self.use_mixstate}:{self.dataset_version}:"
            f"{':'.join(self.allcases)}"
        )
        key = hashlib.md5(identity.encode()).hexdigest()
        return os.path.join(self.cache.cache_dir, f"{key}.clusters.json")

    def _load_cluster_labels(self):
        path = self._cluster_labels_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            labels = np.asarray(data.get('labels', []), dtype=int)
            if (data.get('version') != 1 or
                    data.get('cases') != self.allcases or
                    data.get('dataset_version') != self.dataset_version or
                    labels.shape != (len(self.allcases),)):
                return None
            return labels
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _save_cluster_labels(self, clusters):
        path = self._cluster_labels_path()
        temporary = f"{path}.{os.getpid()}.tmp"
        data = {
            'version': 1,
            'algorithm': 'QDRT_C',
            'dataset_version': self.dataset_version,
            'cases': self.allcases,
            'labels': np.asarray(clusters, dtype=int).tolist(),
        }
        with open(temporary, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False)
        os.replace(temporary, path)
        print(f"QDRT_C Cluster labels checkpoint Saved: {path}")

    def _compute_euclidean_distance(self, cases):
        feature_vectors = []
        for name in self.allcases:
            features = self._extract_features(name)
            feature_vectors.append(features)


        dist_matrix = pairwise_distances(feature_vectors, metric='euclidean')


        max_dist = np.max(dist_matrix)
        if max_dist > 0:
            dist_matrix = dist_matrix / max_dist

        return dist_matrix

    def _extract_features(self, case_name):
        data = self.json_data[case_name]
        state = data['state']
        attributes = data['attributes']


        factors = testcase_io.density_factors(state)
        if isinstance(state, testcase_io.LowRankDensityMatrix):
            probabilities = np.sum(np.abs(factors) ** 2, axis=1)
            real_features = probabilities
            imag_features = np.zeros_like(probabilities)
        elif hasattr(state, 'data'):
            state_data = np.asarray(state.data)
            if state_data.ndim == 2:

                real_features = np.real(np.diag(state_data))
                imag_features = np.zeros_like(real_features)
            else:
                real_features = np.real(state_data).reshape(-1)
                imag_features = np.imag(state_data).reshape(-1)
        else:
            state_data = np.asarray(state)
            if state_data.ndim == 2:
                real_features = np.real(np.diag(state_data))
                imag_features = np.zeros_like(real_features)
            else:
                real_features = np.real(state_data).reshape(-1)
                imag_features = np.imag(state_data).reshape(-1)


        features = np.concatenate([real_features, imag_features])


        n_qubits = attributes.get('num_qubits', self.json_data[case_name]
                                  ['attributes'].get('num_qubits', 0))
        base_state = list(attributes.get('base_state', [0] * n_qubits))
        if len(base_state) != n_qubits:
            base_state = (base_state + [0] * n_qubits)[:n_qubits]
        superposed_qubits = set(attributes.get('superposed_qubits', []))
        superposed = [1 if i in superposed_qubits else 0
                      for i in range(n_qubits)]

        attr_features = [n_qubits] + base_state + superposed


        all_features = np.concatenate([features, np.array(attr_features)])

        return all_features

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


class DistanceMatrixCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, cases, metric):
        case_ids = sorted([case for case in cases])
        key_str = f"{metric}:{':'.join(case_ids)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.npy")

    def compute_or_load(self, cases, metric, compute_func):
        key = self._get_cache_key(cases, metric)
        cache_path = self._get_cache_path(key)

        if os.path.exists(cache_path):
            return np.load(cache_path)

        dist_matrix = compute_func(cases)
        np.save(cache_path, dist_matrix)
        return dist_matrix


def save_t_measure_data(algorithm, group_name, program_name, algorithm_name, distance_metric, n_clusters,
                        use_mixstate, classification_method):


    result_dir = f"./result/{group_name}/t_measure_data"
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


    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved T-measure and params updating data to {filepath}")


def process_results(alg, group_name, program_name, distance_metric, n_clusters, use_mixstate, classification_method,
                    cand_num=1):


    mixstate_suffix = "MixState" if use_mixstate else "NoMixState"
    filename_suffix = f"{classification_method}_{distance_metric}_n{n_clusters}_{mixstate_suffix}"

    if alg == 'ART_1':
        file_path = f'./result/{group_name}/{program_name}_{alg}_{filename_suffix}(cand_num={cand_num}).txt'
    else:
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
    parser = argparse.ArgumentParser(description='Classical test selection algorithms DRT-C')


    parser.add_argument('input_qbit_count', type=int, help='Input qubit count')
    parser.add_argument('output_qbit_count', type=int, help='Output qubit count')
    parser.add_argument('group_name', type=str, help='Benchmark group name (e.g., adder_n7_1)')
    parser.add_argument('program_name', type=str, help='Canonical program name (e.g., adder_n7)')
    parser.add_argument('alg', type=str, help='Algorithm name(DRT_C)')


    parser.add_argument('--distance_metric', type=str, default='hybrid',
                        choices=['hybrid', 'fidelity', 'euclidean'],
                        help='Dissimilarity measure: hybrid, fidelity, euclidean, or trace')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters')
    parser.add_argument('--use_mixstate', action='store_true',
                        help='Include mixed-state test cases')
    parser.add_argument('--iter_num', type=int, default=50,
                        help='Repetition count')
    parser.add_argument('--cand_num', type=int, default=1,
                        help='Number of candidate test cases(Compatibility option)')
    parser.add_argument('--classification_method', type=str, default='direct',
                        choices=['direct', 'adaptive', 'by_type', 'detailed_type1', 'detailed_type2',
                                 'type_clustering'],
                        help='Test-case partitioning method: direct(Direct clustering), adaptive(Adaptive clustering), by_type(Partition by type), detailed_type1(Detailed partitioning1), detailed_type2(Detailed partitioning2), type_clustering(Partition by type, then cluster within each type)')

    args = parser.parse_args()
    start_memory_tracing()


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


    result_path = './result/' + args.group_name
    mixstate_suffix = "MixState" if args.use_mixstate else "NoMixState"
    filename_suffix = f"{args.classification_method}_{args.distance_metric}_n{args.n_clusters}_{mixstate_suffix}"

    pattern1 = re.compile(f"{args.program_name}_{args.alg}_{filename_suffix}\\.txt$")
    pattern2 = re.compile(f"{args.program_name}_{args.alg}_{filename_suffix}_failCount\\.txt$")
    pattern3 = re.compile(f"{args.program_name}_{args.alg}_{filename_suffix}_time\\.txt$")

    if os.path.exists(result_path):
        for filename in os.listdir(result_path):
            if (pattern1.match(filename) or pattern2.match(filename) or
                    pattern3.match(filename)):
                os.remove(os.path.join(result_path, filename))
                print("Delete File: " + os.path.join(result_path, filename))


    folder = './result/' + args.group_name
    if not os.path.exists(folder):
        os.makedirs(folder)


    experiment_start_time = time.perf_counter()
    if args.alg == "DRT_C":
        algorithm = DRT_C(num_vars, args.input_qbit_count, args.output_qbit_count,
                          args.group_name, args.program_name, args.alg, args.iter_num, args.classification_method,
                          args.distance_metric, args.n_clusters, args.use_mixstate)
    else:
        print(f"Algorithm {args.alg} not supported!")
        return
    algorithm.preprocessing_time = elapsed_since(experiment_start_time)
    algorithm.preprocessing_memory_usage = get_memory_usage()

    algorithm.run()


    save_t_measure_data(algorithm, args.group_name, args.program_name, args.alg,
                        args.distance_metric, args.n_clusters, args.use_mixstate, args.classification_method)


    process_results(args.alg, args.group_name, args.program_name,
                    args.distance_metric, args.n_clusters, args.use_mixstate, args.classification_method, args.cand_num)

    print(f'Algorithm: {args.alg}')
    print(f'FaultVersion: {args.group_name}')
    print(f'Classification Method: {args.classification_method}')
    print(f'Distance_Metric: {args.distance_metric}')
    print(f'Number of Clusters: {args.n_clusters}')
    print(f'Use MixState: {args.use_mixstate}')
    print(f'Computing time: {algorithm.total_computing_time}')


if __name__ == '__main__':
    main()
