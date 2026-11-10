
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
import time
import argparse
from datetime import datetime
import shutil
import subprocess
from typing import List, Dict, Any
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testcase_io
from experiment_config import ROOT, SUBJECTS


def setup_experiment_directories(experiment_dir: str = "./"):
    directories = [
        experiment_dir,
        os.path.join(experiment_dir, "configs"),
        os.path.join(experiment_dir, "results"),
        os.path.join(experiment_dir, "summaries"),
        os.path.join(experiment_dir, "logs")
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Creating directory: {directory}")

    return experiment_dir


def find_available_qubit_counts(base_dir: str = str(ROOT / "TestCases")):
    """Find qubit counts with either legacy JSON or binary testcase files."""
    qubit_counts = set()
    if not os.path.exists(base_dir):
        return qubit_counts

    for file in os.listdir(base_dir):
        match = re.match(r"quantum_test_cases_(\d+)qubits\.(json|pkl|pkl\.gz)", file)
        if match:
            qubit_counts.add(int(match.group(1)))

    return qubit_counts


def find_all_programs_and_qubits(base_dir: str = str(ROOT / "TestCases")):
    test_case_files = {}
    qubit_counts = set()


    for file in os.listdir(base_dir):
        match = re.match(r"quantum_test_cases_(\d+)qubits\.(json|pkl|pkl\.gz)", file)
        if match:
            qubit_counts.add(int(match.group(1)))


    oracle_dir = str(ROOT / "oracle/OpenQASM")
    for oracle_file in os.listdir(oracle_dir):
        if oracle_file.endswith("_oracle.xlsx"):
            program_name = oracle_file.replace("_oracle.xlsx", "")
            for n_qubits in sorted(qubit_counts):
                test_case_path = os.path.join(base_dir, f"quantum_test_cases_{n_qubits}qubits.json")
                try:
                    testcase_io.resolve_testcase_file(test_case_path)
                except FileNotFoundError:
                    continue
                if program_name not in test_case_files:
                    test_case_files[program_name] = []
                if n_qubits not in test_case_files[program_name]:
                    test_case_files[program_name].append(n_qubits)


    return {program: [SUBJECTS[program]['qubits']]
            for program in test_case_files if program in SUBJECTS
            and SUBJECTS[program]['qubits'] in test_case_files[program]}


def get_fault_versions_for_program(program_name: str) -> Dict[int, List[int]]:
    oracle_file = str(ROOT / "oracle/OpenQASM" / f"{program_name}_oracle.xlsx")

    if not os.path.exists(oracle_file):
        return {}

    fault_versions_by_qubits = {}

    try:
        excel_file = pd.ExcelFile(oracle_file)
        for sheet_name in excel_file.sheet_names:

            if sheet_name.startswith(f"{program_name}_"):
                parts = sheet_name.split('_')
                if len(parts) >= 2:
                    try:
                        fault_version = int(parts[-1])


                        for n_qubits in sorted(find_available_qubit_counts()):
                            try:
                                testcase_io.resolve_testcase_file(ROOT / "TestCases" / f"quantum_test_cases_{n_qubits}qubits.json")
                            except FileNotFoundError:
                                continue
                            else:
                                if n_qubits not in fault_versions_by_qubits:
                                    fault_versions_by_qubits[n_qubits] = []
                                if fault_version not in fault_versions_by_qubits[n_qubits]:
                                    fault_versions_by_qubits[n_qubits].append(fault_version)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Failed to read oracle file: {e}")

    if program_name not in SUBJECTS:
        raise ValueError('Unknown subject: ' + program_name)
    width = SUBJECTS[program_name]['qubits']
    return {width: [version for version in fault_versions_by_qubits.get(width, [])
                    if version in range(1, 6)]}


def create_experiment_config(program_name: str, n_qubits: int,
                             distance_metrics: List[str] = None,
                             n_clusters: int = 5,
                             iter_num: int = 50) -> Dict[str, Any]:

    if distance_metrics is None:
        distance_metrics = ['hybrid', 'fidelity', 'euclidean']

    config = {
        'experiment_id': f"{program_name}_{n_qubits}qubits_distance_metric",
        'program_name': program_name,
        'n_qubits': n_qubits,
        'distance_metrics': distance_metrics,
        'n_clusters': n_clusters,
        'iter_num': iter_num,
        'classification_method': 'type_clustering',
        'use_mixstate': True,
        'test_case_file': f"../TestCases/quantum_test_cases_{n_qubits}qubits.json",
        'oracle_file': f"../oracle/OpenQASM/{program_name}_oracle.xlsx",
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }

    return config


def save_experiment_config(config: Dict[str, Any], experiment_dir: str):
    config_file = os.path.join(
        experiment_dir,
        "configs",
        f"{config['experiment_id']}_{config['timestamp']}.json"
    )

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Experiment configuration saved to: {config_file}")
    return config_file


def run_single_experiment(config: Dict[str, Any], experiment_dir: str,
                          distance_metric: str, fault_version: int) -> Dict[str, Any]:

    program_name = config['program_name']
    n_qubits = config['n_qubits']
    n_clusters = config['n_clusters']
    iter_num = config['iter_num']
    classification_method = config['classification_method']
    use_mixstate = config['use_mixstate']

    group_name = f"{program_name}_{fault_version}"

    print(f"\n  Run: {program_name} (Qubit count: {n_qubits}, Dissimilarity measure: {distance_metric}, Faulty version: {fault_version})")
    print("  " + "-" * 50)


    result_dir = os.path.join(
        experiment_dir,
        "results",
        f"{program_name}_{n_qubits}qubits",
        distance_metric,
        group_name
    )
    os.makedirs(result_dir, exist_ok=True)


    classification_dir = os.path.join(
        "../TestCases/classification",
        distance_metric
    )
    os.makedirs(classification_dir, exist_ok=True)


    cache_dir = os.path.join(
        "../distance_cache",
        f"{n_qubits}_distance_cache"
    )
    os.makedirs(cache_dir, exist_ok=True)


    cmd = [
        sys.executable, str(ROOT / "QHDRT.py"),
        str(n_qubits), str(n_qubits),
        group_name, program_name, "QHDRT",
        "--distance_metric", distance_metric,
        "--n_clusters", str(n_clusters),
        "--iter_num", str(iter_num),
        "--classification_method", classification_method,
        "--result_dir", str(Path(result_dir).resolve())
    ]

    if use_mixstate:
        cmd.append("--use_mixstate")

    print(f"  Command: {' '.join(cmd)}")

    start_time = time.time()

    try:

        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=ROOT
        )

        execution_time = time.time() - start_time


        log_file = os.path.join(
            experiment_dir,
            "logs",
            f"{program_name}_{n_qubits}qubits_{distance_metric}_{group_name}.log"
        )



        if result.returncode == 0:
            print(f"  [OK] Success (Elapsed time: {execution_time:.2f}s)")


            source_result_dir = result_dir
            if os.path.exists(source_result_dir):

                for file in os.listdir(source_result_dir):
                    source_file = os.path.join(source_result_dir, file)
                    if (os.path.isfile(source_file) and distance_metric in os.path.basename(source_file)
                            and Path(source_file).resolve().parent != Path(result_dir).resolve()):
                        shutil.copy2(source_file, result_dir)
                        print(f"    Copy result files: {file}")

            return {
                'success': True,
                'distance_metric': distance_metric,
                'fault_version': fault_version,
                'execution_time': execution_time,
                'result_dir': result_dir,
                'log_file': log_file
            }
        else:
            print(f"  [FAIL] Failed (Exit code: {result.returncode})")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")

            return {
                'success': False,
                'distance_metric': distance_metric,
                'fault_version': fault_version,
                'error': result.stderr[:500] if result.stderr else "Unknown error",
                'execution_time': execution_time,
                'log_file': log_file
            }

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'distance_metric': distance_metric,
            'fault_version': fault_version,
            'error': str(e),
            'execution_time': time.time() - start_time
        }


def collect_results(experiment_dir: str, config: Dict[str, Any],
                    all_results: List[Dict[str, Any]]):

    program_name = config['program_name']
    n_qubits = config['n_qubits']

    summary = {
        'experiment_id': config['experiment_id'],
        'program_name': program_name,
        'n_qubits': n_qubits,
        'distance_metrics': config['distance_metrics'],
        'total_experiments': 0,
        'successful_experiments': 0,
        'failed_experiments': 0,
        'results_by_distance_metric': {},
        'execution_times': {}
    }

    for result in all_results:
        distance_metric = result['distance_metric']
        fault_version = result['fault_version']

        summary['total_experiments'] += 1
        if result['success']:
            summary['successful_experiments'] += 1
        else:
            summary['failed_experiments'] += 1

        if distance_metric not in summary['results_by_distance_metric']:
            summary['results_by_distance_metric'][distance_metric] = {
                'successful': 0,
                'failed': 0,
                'total': 0,
                'fault_versions': []
            }

        metric_summary = summary['results_by_distance_metric'][distance_metric]
        metric_summary['total'] += 1
        if result['success']:
            metric_summary['successful'] += 1
            if 'execution_time' in result:
                if distance_metric not in summary['execution_times']:
                    summary['execution_times'][distance_metric] = []
                summary['execution_times'][distance_metric].append(result['execution_time'])
        else:
            metric_summary['failed'] += 1

        metric_summary['fault_versions'].append(fault_version)


    summary_file = os.path.join(
        experiment_dir,
        "summaries",
        f"{config['experiment_id']}_{config['timestamp']}_summary.json"
    )

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_file}")

    return summary


def generate_comparison_report(summary: Dict[str, Any], experiment_dir: str):
    if not summary.get('results_by_distance_metric'):
        return

    report_file = os.path.join(
        experiment_dir,
        "summaries",
        f"{summary['experiment_id']}_comparison_report.md"
    )

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Dissimilarity comparison report\n\n")
        f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"## Experiment overview\n\n")
        f.write(f"- Program: {summary['program_name']}\n")
        f.write(f"- Qubit count: {summary['n_qubits']}\n")
        f.write(f"- Total experiments: {summary['total_experiments']}\n")
        f.write(f"- Successful experiments: {summary['successful_experiments']}\n")
        f.write(f"- Failed experiments: {summary['failed_experiments']}\n")
        f.write(f"- Success rate: {summary['successful_experiments'] / summary['total_experiments'] * 100:.1f}%\n\n")

        f.write(f"## Performance comparison by dissimilarity measure\n\n")
        f.write("| Dissimilarity measure | Passing test cases | Failures | Total | Success rate | Mean execution time(s) |\n")
        f.write("|----------|--------|--------|------|--------|------------------|\n")

        for distance_metric, metric_summary in summary['results_by_distance_metric'].items():
            success_rate = metric_summary['successful'] / metric_summary['total'] * 100
            avg_time = 0
            if distance_metric in summary.get('execution_times', {}):
                times = summary['execution_times'][distance_metric]
                if times:
                    avg_time = sum(times) / len(times)

            f.write(f"| {distance_metric} | {metric_summary['successful']} | {metric_summary['failed']} | ")
            f.write(f"{metric_summary['total']} | {success_rate:.1f}% | {avg_time:.2f} |\n")

        f.write("\n")

        f.write(f"## Result directory layout\n\n")
        f.write(f"All experiment results are stored in:\n")
        f.write(f"```\n")
        f.write(f"{experiment_dir}/\n")
        f.write(f"├── results/\n")
        f.write(f"│   ├── {summary['program_name']}_{summary['n_qubits']}qubits/\n")
        f.write(f"│   │   ├── hybrid/\n")
        f.write(f"│   │   │   ├── {summary['program_name']}_1/\n")
        f.write(f"│   │   │   │   ├── Result file...\n")
        f.write(f"│   │   │   └── ...\n")
        f.write(f"│   │   ├── fidelity/\n")
        f.write(f"│   │   └── euclidean/\n")
        f.write(f"├── configs/\n")
        f.write(f"├── summaries/\n")
        f.write(f"└── logs/\n")
        f.write(f"```\n")

        f.write(f"\n## Dissimilarity measure descriptions\n\n")
        f.write(f"### 1. Hybrid (Hybrid dissimilarity)\n")
        f.write(f"- Combines Hamming distance, fidelity and trace distance\n")
        f.write(f"- Select dissimilarity computation by test-case type\n")
        f.write(f"- Using Hamming distance for computational-basis pairs\n")
        f.write(f"- Using specialized dissimilarity for superposition pairs\n")
        f.write(f"- Using trace distance for mixed-state pairs\n")

        f.write(f"\n### 2. Fidelity (fidelity-based dissimilarity)\n")
        f.write(f"- Based on fidelity between quantum states\n")
        f.write(f"- Distance = 1 - Fidelity(state1, state2)\n")
        f.write(f"- Applicable to all quantum-state types\n")

        f.write(f"\n### 3. Euclidean (Euclidean distance)\n")
        f.write(f"- Represent quantum states as complex vectors\n")
        f.write(f"- Compute Euclidean distance between state vectors\n")
        f.write(f"- normalized to[0,1]range\n")

        f.write(f"\n## Experiment parameters\n\n")
        f.write(f"- Cluster count: 5\n")
        f.write(f"- Repetition count: 20\n")
        f.write(f"- Partitioning method: detailed_type2\n")
        f.write(f"- Include mixed states: No(Use only original test cases)\n")

    print(f"Comparison report saved to: {report_file}")


def run_distance_metric_experiments(program_name: str, n_qubits: int,
                                    distance_metrics: List[str] = None,
                                    fault_versions: List[int] = None,
                                    n_clusters: int = 5,
                                    iter_num: int = 50,
                                    experiment_dir: str = "./"):

    print("=" * 60)
    print("Dissimilarity comparison experiment")
    print("=" * 60)


    setup_experiment_directories(experiment_dir)

    if distance_metrics is None:
        distance_metrics = ['hybrid', 'fidelity', 'euclidean']


    test_case_file = f"../TestCases/quantum_test_cases_{n_qubits}qubits.json"
    try:
        testcase_io.resolve_testcase_file(test_case_file)
    except FileNotFoundError:
        print(f"Error: Testcase file does not exist: {test_case_file}")
        return


    oracle_file = f"../oracle/OpenQASM/{program_name}_oracle.xlsx"
    if not os.path.exists(oracle_file):
        print(f"Error: Oracle file does not exist: {oracle_file}")
        return


    if fault_versions is None:
        fault_versions_by_qubits = get_fault_versions_for_program(program_name)
        if n_qubits in fault_versions_by_qubits:
            fault_versions = fault_versions_by_qubits[n_qubits]
        else:
            print(f"Warning: No faulty versions found for {program_name} ({n_qubits} qubits)")
            fault_versions = [1]

    print(f"Program: {program_name}")
    print(f"Qubit count: {n_qubits}")
    print(f"Dissimilarity measure: {', '.join(distance_metrics)}")
    print(f"Faulty version: {fault_versions}")
    print(f"Cluster count: {n_clusters}")
    print(f"Repetition count: {iter_num}")
    print("=" * 60)


    config = create_experiment_config(
        program_name=program_name,
        n_qubits=n_qubits,
        distance_metrics=distance_metrics,
        n_clusters=n_clusters,
        iter_num=iter_num
    )

    save_experiment_config(config, experiment_dir)

    all_results = []


    for distance_metric in distance_metrics:
        print(f"\nRun dissimilarity measure: {distance_metric}")
        print("-" * 40)

        for fault_version in fault_versions:
            result = run_single_experiment(
                config=config,
                experiment_dir=experiment_dir,
                distance_metric=distance_metric,
                fault_version=fault_version
            )
            all_results.append(result)


    summary = collect_results(experiment_dir, config, all_results)


    generate_comparison_report(summary, experiment_dir)

    print("\n" + "=" * 60)
    print("Dissimilarity comparison experiment complete!")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run dissimilarity comparison experiments')

    parser.add_argument('program_name', type=str, help='Program name')
    parser.add_argument('n_qubits', type=int, help='Qubit count')
    parser.add_argument('--distance_metrics', type=str, default='hybrid,fidelity,euclidean,trace',
                        help='Dissimilarity measure list, Comma-separated')
    parser.add_argument('--fault_versions', type=str, default=None,
                        help='Faulty version list, Comma-separated or use a range such as"1-5"')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters')
    parser.add_argument('--iter_num', type=int, default=50,
                        help='Repetition count')
    parser.add_argument('--experiment_dir', type=str, default='./',
                        help='Experiment directory')

    args = parser.parse_args()


    distance_metrics = args.distance_metrics.split(',')


    fault_versions = None
    if args.fault_versions:
        if '-' in args.fault_versions:
            start, end = map(int, args.fault_versions.split('-'))
            fault_versions = list(range(start, end + 1))
        else:
            fault_versions = list(map(int, args.fault_versions.split(',')))


    summary = run_distance_metric_experiments(
        program_name=args.program_name,
        n_qubits=args.n_qubits,
        distance_metrics=distance_metrics,
        fault_versions=fault_versions,
        n_clusters=args.n_clusters,
        iter_num=args.iter_num,
        experiment_dir=args.experiment_dir
    )


if __name__ == "__main__":
    main()
