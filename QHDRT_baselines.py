import argparse
import ast
import hashlib
import io
import json
import os
import random
import re
import time
from collections import defaultdict

import numpy as np
import openpyxl
import pandas as pd
from qiskit.quantum_info import DensityMatrix, state_fidelity
from sklearn_extra.cluster import KMedoids

import testcase_io
from experiment_metrics import (
    elapsed_since,
    format_time_costs,
    get_memory_usage,
    PeakMemoryMonitor,
)


THRESHOLDS = [round(i * 0.05, 2) for i in range(1, 21)]


def testcase_type(name, data):
    type_str = name.split("_", 1)[0]
    if type_str in {"classical", "superposition", "entangled", "mixed"}:
        return type_str
    return data.get("attributes", {}).get("type", "classical")


def as_density_matrix(state):
    if isinstance(state, DensityMatrix):
        return state
    return DensityMatrix(state)


def hamming_distance(statevec1, statevec2):
    data1 = statevec1.data if hasattr(statevec1, "data") else statevec1
    data2 = statevec2.data if hasattr(statevec2, "data") else statevec2
    idx1 = np.argmax(np.abs(data1))
    idx2 = np.argmax(np.abs(data2))
    n_qubits = int(np.log2(len(data1)))
    bin1 = format(idx1, f"0{n_qubits}b")
    bin2 = format(idx2, f"0{n_qubits}b")
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2)) / n_qubits


def trace_distance(rho1, rho2):
    if (testcase_io.density_factors(rho1) is not None and
            testcase_io.density_factors(rho2) is not None):
        return testcase_io.low_rank_trace_distance(rho1, rho2)
    data1 = rho1.data if hasattr(rho1, "data") else np.asarray(rho1)
    data2 = rho2.data if hasattr(rho2, "data") else np.asarray(rho2)
    eigvals = np.linalg.eigvalsh(data1 - data2)
    return 0.5 * np.sum(np.abs(eigvals))


def quantum_distance(name1, name2, json_data, use_mixstate=False):
    type1 = testcase_type(name1, json_data[name1])
    type2 = testcase_type(name2, json_data[name2])
    if type1 == "entangled":
        type1 = "superposition"
    if type2 == "entangled":
        type2 = "superposition"

    state1 = json_data[name1]["state"]
    state2 = json_data[name2]["state"]

    if type1 == "classical" and type2 == "classical":
        return hamming_distance(state1, state2)
    if type1 == "mixed" and type2 == "mixed" and use_mixstate:
        return trace_distance(state1, state2)
    if "mixed" in {type1, type2} and use_mixstate:
        return testcase_io.low_rank_trace_distance(state1, state2)
    return 1 - testcase_io.low_rank_fidelity(state1, state2)


class MatrixCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def path_for(self, names, metric):
        key = hashlib.md5(f"{metric}:{':'.join(sorted(names))}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.npy")

    def load_or_compute(self, names, metric, compute_func):
        path = self.path_for(names, metric)
        if os.path.exists(path):
            return np.load(path)
        matrix = compute_func()
        np.save(path, matrix)
        return matrix


class BaselineBase:
    def __init__(self, number_of_variables, input_qbit_count, output_qbit_count, group_name, program_name, algorithm,
                 iter_num, cand_num=10, use_mixstate=False, result_dir=None,
                 experiment_cache_dir=None, validate_mixed_oracle_version=True):
        self.number_of_variables = number_of_variables
        self.input_qbit_count = input_qbit_count
        self.output_qbit_count = output_qbit_count
        self.group_name = group_name
        self.program_name = program_name
        self.algorithm = algorithm
        self.iter_num = iter_num
        self.cand_num = cand_num
        self.use_mixstate = use_mixstate
        self.max_fail_count = -1
        self.best_candidates = []
        self.all_runs_fail_history = []
        self.all_runs_test_cases = []

        oracle_path = os.path.join("./oracle/OpenQASM", f"{self.program_name}_oracle.xlsx")
        oracle_data = pd.read_excel(oracle_path, sheet_name=self.group_name, dtype=str)
        self.oracle_data = oracle_data
        self.oracle = dict(zip(oracle_data.iloc[:, 0], oracle_data.iloc[:, 3]))
        self.pvalues = dict(zip(oracle_data.iloc[:, 0], oracle_data.iloc[:, 4]))

        all_test_cases = testcase_io.load_test_cases(
            f"./TestCases/quantum_test_cases_{self.input_qbit_count}qubits.json",
            as_dict=True,
            exclude_types=None if use_mixstate else {"mixed"},
        )
        if use_mixstate:
            self.json_data = all_test_cases
        else:
            self.json_data = {
                name: data
                for name, data in all_test_cases.items()
                if testcase_type(name, data) != "mixed"
            }
        self.dataset_version = next((
            data.get("attributes", {}).get("state_representation")
            for data in self.json_data.values()
            if data.get("attributes", {}).get("state_representation")
        ), "legacy")

        if use_mixstate and validate_mixed_oracle_version:
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

        self.candidates = [
            name for name in oracle_data.iloc[:, 0].tolist()
            if name in self.json_data
        ]
        self.upper_bound = len(self.candidates)
        self.result_dir = result_dir or f"./result/{self.group_name}"
        self.experiment_cache_dir = experiment_cache_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def _empty_count_fail(self):
        return {k: 0 for k in THRESHOLDS}

    def _update_count_fail(self, count_fail, test_index, test_passed):
        if test_passed:
            return
        for threshold in THRESHOLDS:
            if test_index <= self.upper_bound * threshold:
                count_fail[threshold] += 1

    def _active_prob_array(self, active, probabilities):
        values = np.array([max(0, probabilities.get(cid, 0)) for cid in active], dtype=float)
        total = values.sum()
        if total <= 0:
            return np.ones(len(active), dtype=float) / len(active)
        return values / total

    def _renormalize_probabilities(self, probabilities, active):
        if not active:
            return {}
        total = sum(max(0, probabilities.get(cid, 0)) for cid in active)
        if total <= 0:
            uniform = 1 / len(active)
            return {cid: uniform for cid in active}
        return {cid: max(0, probabilities.get(cid, 0)) / total for cid in active}

    def _record_run(self, executed_order, fault_counts, pvalue_history, count_fail, run_fail_history):
        if count_fail.get(0.3, 0) > self.max_fail_count:
            self.max_fail_count = count_fail[0.3]
            self.best_candidates = executed_order

        self.all_runs_test_cases.append(executed_order)
        self.all_runs_fail_history.append(run_fail_history)

        with open(self.result_path(), "a") as f:
            f.write(" ".join(map(str, executed_order)) + "\n")
            f.write(" ".join(map(str, fault_counts)) + "\n")
            f.write(" ".join(map(str, pvalue_history)) + "\n")
            json.dump(count_fail, f)
            f.write("\n")

    def _write_time_costs(self):
        self.preprocessing_time = getattr(self, "preprocessing_time", 0.0)
        self.online_selection_time = elapsed_since(self.start_computing_time)
        self.total_computing_time = self.preprocessing_time + self.online_selection_time
        memory_monitor = getattr(self, "memory_monitor", None)
        self.total_memory_usage = (
            memory_monitor.snapshot() if memory_monitor is not None else get_memory_usage()
        )
        with open(self.time_path(), "a") as f:
            f.write(format_time_costs(
                self.iter_num,
                self.total_computing_time,
                self.preprocessing_time,
                self.online_selection_time,
                getattr(self, "preprocessing_memory_usage", None),
                self.total_memory_usage,
            ))

    def result_path(self):
        return os.path.join(self.result_dir, f"{self.program_name}_{self.algorithm}.txt")

    def time_path(self):
        return os.path.join(self.result_dir, f"{self.program_name}_{self.algorithm}_time.txt")

    def get_result(self):
        return self.best_candidates


class RT(BaselineBase):
    def run(self):
        self.start_computing_time = time.perf_counter()

        for _ in range(self.iter_num):
            remaining = list(self.candidates)
            random.shuffle(remaining)
            executed_order = []
            fault_counts = []
            pvalue_history = []
            count_fail = self._empty_count_fail()
            run_fail_history = []
            current_fail_count = 0

            for idx, case in enumerate(remaining):
                test_passed = ast.literal_eval(self.oracle[case])
                pvalue = self.pvalues[case]
                executed_order.append(case)
                pvalue_history.append(pvalue)
                if not test_passed:
                    current_fail_count += 1
                fault_counts.append(current_fail_count)
                self._update_count_fail(count_fail, idx, test_passed)
                run_fail_history.append({"test_count": idx + 1, "fail_count": current_fail_count})

            self._record_run(executed_order, fault_counts, pvalue_history, count_fail, run_fail_history)

        self._write_time_costs()


class RPT(BaselineBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        clusters = defaultdict(list)
        for name in self.candidates:
            type_id = {"classical": 0, "superposition": 1, "entangled": 2, "mixed": 3}.get(
                testcase_type(name, self.json_data[name]),
                0,
            )
            clusters[type_id].append(name)
        self.base_clusters = {cluster_id: cases for cluster_id, cases in clusters.items() if cases}

    def run(self):
        self.start_computing_time = time.perf_counter()

        for _ in range(self.iter_num):
            clusters = {cid: list(cases) for cid, cases in self.base_clusters.items()}
            probabilities = {cid: 1 / len(clusters) for cid in clusters}
            executed_order = []
            fault_counts = []
            pvalue_history = []
            count_fail = self._empty_count_fail()
            run_fail_history = []
            current_fail_count = 0

            for idx in range(self.upper_bound):
                active = [cid for cid, cases in clusters.items() if cases]
                if not active:
                    break
                active_probs = self._active_prob_array(active, probabilities)
                cluster_id = np.random.choice(active, p=active_probs)
                case = random.choice(clusters[cluster_id])
                clusters[cluster_id].remove(case)

                test_passed = ast.literal_eval(self.oracle[case])
                pvalue = self.pvalues[case]
                executed_order.append(case)
                pvalue_history.append(pvalue)
                if not test_passed:
                    current_fail_count += 1
                fault_counts.append(current_fail_count)
                self._update_count_fail(count_fail, idx, test_passed)
                run_fail_history.append({"test_count": idx + 1, "fail_count": current_fail_count})

                if not clusters[cluster_id]:
                    probabilities.pop(cluster_id, None)
                    if probabilities:
                        uniform = 1 / len(probabilities)
                        probabilities = {cid: uniform for cid in probabilities}

            self._record_run(executed_order, fault_counts, pvalue_history, count_fail, run_fail_history)

        self._write_time_costs()


class ART_1(BaselineBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache_dir = self.experiment_cache_dir or f"./distance_cache/{self.input_qbit_count}_distance_cache"
        cache = MatrixCache(cache_dir)

        def compute():
            n = len(self.candidates)
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    distance = quantum_distance(self.candidates[i], self.candidates[j], self.json_data, self.use_mixstate)
                    matrix[i, j] = matrix[j, i] = distance
            return matrix

        self.distance_matrix = cache.load_or_compute(
            self.candidates,
            f"ART_hybrid_mix{self.use_mixstate}_data{self.dataset_version}",
            compute,
        )
        self.case_to_index = {case: i for i, case in enumerate(self.candidates)}

    def result_path(self):
        return os.path.join(self.result_dir, f"{self.program_name}_{self.algorithm}(cand_num={self.cand_num}).txt")

    def _select_by_distance(self, executed_order, candidates):
        if not executed_order:
            return random.choice(candidates)
        executed_indices = [self.case_to_index[case] for case in executed_order]
        best_case = None
        best_distance = -1
        for case in candidates:
            candidate_index = self.case_to_index[case]
            min_distance = np.min(self.distance_matrix[candidate_index, executed_indices])
            if min_distance >= best_distance:
                best_distance = min_distance
                best_case = case
        return best_case

    def run(self):
        self.start_computing_time = time.perf_counter()

        for _ in range(self.iter_num):
            remaining = set(self.candidates)
            executed_order = []
            fault_counts = []
            pvalue_history = []
            count_fail = self._empty_count_fail()
            run_fail_history = []
            current_fail_count = 0

            for idx in range(self.upper_bound):
                sample_size = min(self.cand_num, len(remaining))
                sampled_candidates = random.sample(list(remaining), sample_size)
                case = self._select_by_distance(executed_order, sampled_candidates)
                remaining.remove(case)

                test_passed = ast.literal_eval(self.oracle[case])
                pvalue = self.pvalues[case]
                executed_order.append(case)
                pvalue_history.append(pvalue)
                if not test_passed:
                    current_fail_count += 1
                fault_counts.append(current_fail_count)
                self._update_count_fail(count_fail, idx, test_passed)
                run_fail_history.append({"test_count": idx + 1, "fail_count": current_fail_count})

            self._record_run(executed_order, fault_counts, pvalue_history, count_fail, run_fail_history)

        self._write_time_costs()


class DRT_1(BaselineBase):
    def __init__(self, *args, n_clusters=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        cache_dir = self.experiment_cache_dir or f"./distance_cache/{self.input_qbit_count}_distance_cache"
        cache = MatrixCache(cache_dir)

        def compute():
            n = len(self.candidates)
            matrix = np.zeros((n, n))
            states = [self.json_data[name]["state"] for name in self.candidates]
            for i in range(n):
                for j in range(i + 1, n):
                    distance = 1 - testcase_io.low_rank_fidelity(states[i], states[j])
                    matrix[i, j] = matrix[j, i] = distance
            return np.maximum(matrix, 0)

        self.distance_matrix = cache.load_or_compute(
            self.candidates,
            f"DRT_fidelity_n{n_clusters}_mix{self.use_mixstate}_data{self.dataset_version}",
            compute,
        )
        kmedoids = KMedoids(
            n_clusters=min(self.n_clusters, len(self.candidates)),
            metric="precomputed",
            method="pam",
            init="random",
            random_state=42,
        )
        assignments = kmedoids.fit_predict(self.distance_matrix)

        clusters = {cid: [] for cid in range(kmedoids.n_clusters)}
        for case, cluster_id in zip(self.candidates, assignments):
            clusters[int(cluster_id)].append(case)
        self.base_clusters = dict(clusters)

    @staticmethod
    def _original_random_index(rate, acc=5):
        weights = np.array(rate) * (10 ** acc)
        for index in range(len(weights)):
            weights[index] = round(weights[index])
        draw = random.randint(1, int(sum(weights)))
        cumulative = 0
        index = 0
        for index, weight in enumerate(weights):
            cumulative += weight
            if draw <= cumulative:
                break
        return index

    @staticmethod
    def _update_original_profile(probabilities, cluster_id, test_passed,
                                 epsilon=0.05, delta=0.05):

        others = [cid for cid in probabilities if cid != cluster_id]
        if not test_passed:
            probabilities[cluster_id] += epsilon
            if probabilities[cluster_id] > 1:
                probabilities[cluster_id] = 1
            for other in others:
                if probabilities[cluster_id] == 1:
                    probabilities[other] = 0
                else:
                    probabilities[other] -= epsilon / len(others)
        elif probabilities[cluster_id] < delta:
            for other in others:
                probabilities[other] += probabilities[cluster_id] / len(others)
            probabilities[cluster_id] = 0
        else:
            for other in others:
                probabilities[other] += delta / len(others)
            probabilities[cluster_id] -= delta

    @staticmethod
    def _redistribute_original_empty_cluster(probabilities, cluster_id):
        remaining_prob = probabilities[cluster_id]
        for other in probabilities:
            if other != cluster_id:
                probabilities[other] += remaining_prob / (len(probabilities) - 1)
        probabilities[cluster_id] = 0

    def run(self):
        self.start_computing_time = time.perf_counter()
        epsilon = 0.05
        delta = 0.05

        for _ in range(self.iter_num):
            clusters = {cid: list(cases) for cid, cases in self.base_clusters.items()}
            probabilities = {cid: 1 / len(clusters) for cid in clusters}
            executed_order = []
            fault_counts = []
            pvalue_history = []
            count_fail = self._empty_count_fail()
            run_fail_history = []
            current_fail_count = 0

            for idx in range(self.upper_bound):
                cluster_ids = list(probabilities)
                cluster_index = (
                    self._original_random_index(list(probabilities.values()))
                    if len(cluster_ids) > 1 else 0
                )
                cluster_id = cluster_ids[cluster_index]
                if not clusters[cluster_id]:

                    self._redistribute_original_empty_cluster(probabilities, cluster_id)
                    continue
                cases = clusters[cluster_id]
                case_index = (
                    self._original_random_index([1 / len(cases)] * len(cases))
                    if len(cases) > 1 else 0
                )
                case = cases[case_index]
                clusters[cluster_id].remove(case)

                test_passed = ast.literal_eval(self.oracle[case])
                pvalue = self.pvalues[case]
                executed_order.append(case)
                pvalue_history.append(pvalue)
                if not test_passed:
                    current_fail_count += 1
                fault_counts.append(current_fail_count)
                self._update_count_fail(count_fail, idx, test_passed)
                run_fail_history.append({"test_count": idx + 1, "fail_count": current_fail_count})

                self._update_original_profile(
                    probabilities, cluster_id, test_passed, epsilon, delta
                )

                if not clusters[cluster_id]:
                    self._redistribute_original_empty_cluster(probabilities, cluster_id)

            self._record_run(executed_order, fault_counts, pvalue_history, count_fail, run_fail_history)

        self._write_time_costs()


def save_t_measure_data(algorithm):
    result_dir = f"./result/{algorithm.group_name}/t_measure_data"
    os.makedirs(result_dir, exist_ok=True)
    if algorithm.algorithm in {"ART_1", "DRT_1"}:
        filename = f"{algorithm.program_name}_{algorithm.algorithm}_cand_num={algorithm.cand_num}.json"
    else:
        filename = f"{algorithm.program_name}_{algorithm.algorithm}.json"
    filepath = os.path.join(result_dir, filename)
    data = {
        "algorithm": algorithm.algorithm,
        "group_name": algorithm.group_name,
        "program_name": algorithm.program_name,
        "cand_num": algorithm.cand_num,
        "use_mixstate": algorithm.use_mixstate,
        "t_measure": [
            {"run_id": run_id, "fail_history": fail_history}
            for run_id, fail_history in enumerate(algorithm.all_runs_fail_history)
        ],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def process_results(algorithm):
    count_fail = {}
    with open(algorithm.result_path(), "r") as f:
        data = f.readlines()
        for i in range(3, len(data), 4):
            values = json.load(io.StringIO(data[i].strip()))
            for key, value in values.items():
                count_fail.setdefault(key, []).append(value)

    avg_fail_count = {key: sum(values) / len(values) for key, values in count_fail.items()}
    max_fail_count = {key: max(values) for key, values in count_fail.items()}

    stats_file = os.path.join(algorithm.result_dir, f"{algorithm.program_name}_{algorithm.algorithm}_failCount.txt")
    with open(stats_file, "w") as f:
        f.write("count_fail:\n")
        f.write(json.dumps(count_fail) + "\n")
        f.write("avg_failCount:\n")
        f.write(json.dumps(avg_fail_count) + "\n")
        f.write("max_failCount:\n")
        f.write(json.dumps(max_fail_count) + "\n")

    from datetime import date

    for suffix, values in [("avg", avg_fail_count), ("max", max_fail_count)]:
        path = f"./result/{algorithm.program_name}_{suffix}_faultNum.xlsx"
        try:
            workbook = openpyxl.load_workbook(path)
        except FileNotFoundError:
            workbook = openpyxl.Workbook()
            if "Sheet" in workbook.sheetnames:
                del workbook["Sheet"]

        if algorithm.group_name not in workbook.sheetnames:
            worksheet = workbook.create_sheet(algorithm.group_name)
            worksheet.cell(row=1, column=2, value=algorithm.algorithm)
            for row, key in enumerate(values.keys(), start=2):
                worksheet.cell(row=row, column=1, value=key)
            column = 2
        else:
            worksheet = workbook[algorithm.group_name]
            column = worksheet.max_column + 1
            worksheet.cell(row=1, column=column, value=algorithm.algorithm)

        for row, key in enumerate(values.keys(), start=2):
            worksheet.cell(row=row, column=column, value=values[key])
        worksheet.cell(row=22, column=column, value=f"cand_num={algorithm.cand_num}; mixstate={algorithm.use_mixstate}")
        worksheet.cell(row=23, column=column, value=f"{date.today()}")
        workbook.save(path)


def number_of_variables(input_qbit_count):
    return {
        4: 26,
        5: 41,
        7: 108,
        8: 174,
        9: 287,
        10: 485,
        11: 845,
        12: 5114,
    }.get(input_qbit_count, -1)


def remove_old_outputs(args, result_path=None):
    result_path = result_path or f"./result/{args.group_name}"
    if not os.path.exists(result_path):
        return
    patterns = [
        re.compile(f"{args.program_name}_{args.alg}.*\\.txt$"),
        re.compile(f"{args.program_name}_{args.alg}.*_time\\.txt$"),
        re.compile(f"{args.program_name}_{args.alg}.*_failCount\\.txt$"),
    ]
    for filename in os.listdir(result_path):
        if any(pattern.match(filename) for pattern in patterns):
            os.remove(os.path.join(result_path, filename))


def build_algorithm(args):
    common = dict(
        number_of_variables=number_of_variables(args.input_qbit_count),
        input_qbit_count=args.input_qbit_count,
        output_qbit_count=args.output_qbit_count,
        group_name=args.group_name,
        program_name=args.program_name,
        algorithm=args.alg,
        iter_num=args.iter_num,
        cand_num=args.cand_num,
        use_mixstate=args.use_mixstate,
        result_dir=args.result_dir,
        experiment_cache_dir=args.experiment_cache_dir,
        validate_mixed_oracle_version=not args.allow_legacy_mixed_oracle,
    )
    if args.alg == "RT":
        return RT(**common)
    if args.alg == "RPT":
        return RPT(**common)
    if args.alg == "ART_1":
        return ART_1(**common)
    if args.alg == "DRT_1":
        return DRT_1(**common, n_clusters=args.n_clusters)
    raise ValueError(f"Algorithm {args.alg} is not supported")


def main():
    parser = argparse.ArgumentParser(description="Main-experiment baseline strategies for QHDRT")
    parser.add_argument("input_qbit_count", type=int)
    parser.add_argument("output_qbit_count", type=int)
    parser.add_argument("group_name", type=str)
    parser.add_argument("program_name", type=str)
    parser.add_argument("alg", choices=["RT", "RPT", "ART_1", "DRT_1"])
    parser.add_argument("--iter_num", type=int, default=50)
    parser.add_argument("--cand_num", type=int, default=10)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--use_mixstate", action="store_true")
    parser.add_argument("--cost_only", action="store_true",
                        help="Collect time and memory costs only, without modifying main-experiment results")
    parser.add_argument("--cost_output_root", default="./cost_measurements",
                        help="Isolated output root for cost-only measurements")
    parser.add_argument("--experiment_cache_dir", default=None,
                        help="Isolated distance-cache directory for this cost measurement")
    parser.add_argument("--allow_legacy_mixed_oracle", action="store_true",
                        help="Allow legacy mixed-state oracle labels in cost-only mode")
    args = parser.parse_args()
    if args.allow_legacy_mixed_oracle and not args.cost_only:
        parser.error("--allow_legacy_mixed_oracle may only be used with --cost_only")

    memory_monitor = PeakMemoryMonitor().start()
    args.result_dir = (
        os.path.join(args.cost_output_root, args.group_name)
        if args.cost_only else os.path.join("./result", args.group_name)
    )
    os.makedirs(args.result_dir, exist_ok=True)
    remove_old_outputs(args, args.result_dir)

    preprocessing_start = time.perf_counter()
    algorithm = build_algorithm(args)
    algorithm.preprocessing_time = elapsed_since(preprocessing_start)
    algorithm.memory_monitor = memory_monitor
    algorithm.preprocessing_memory_usage = memory_monitor.snapshot()

    algorithm.run()
    memory_monitor.stop()
    if args.cost_only:

        raw_result_file = algorithm.result_path()
        if os.path.exists(raw_result_file):
            os.remove(raw_result_file)
    else:
        save_t_measure_data(algorithm)
        process_results(algorithm)

    print(f"Algorithm: {args.alg}")
    print(f"FaultVersion: {args.group_name}")
    print(f"Use MixState: {args.use_mixstate}")
    print(f"Preprocessing time: {algorithm.preprocessing_time}")
    print(f"Online selection time: {algorithm.online_selection_time}")
    print(f"Computing time: {algorithm.total_computing_time}")


if __name__ == "__main__":
    main()
