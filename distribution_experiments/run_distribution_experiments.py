"""Run the production QHDRT on fixed distribution pools in the historical layout."""

import argparse
import hashlib
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import testcase_io
from QHDRT import QHDRT, save_t_measure_data
from experiment_metrics import PeakMemoryMonitor
from distribution_experiments.workflow_utils import (
    DEFAULT_VERSIONS, atomic_json, category, distribution_sizes, file_digest,
    load_pool, parse_versions, validate_oracle,
)


class DistributionExperimentQHDRT(QHDRT):
    """Override only testcase loading; selection/update/reset are inherited intact."""

    def __init__(self, experiment_dir, program_name, n_qubits, distribution_name,
                 fault_version, iter_num=50, n_clusters=5, original_cases=None,
                 supplement_cases=None, seed=42, include_entangled_qubits=False):
        self.n_qubits = n_qubits
        self.distribution_name = distribution_name
        self.fault_version = fault_version
        self.experiment_dir = Path(experiment_dir).resolve()
        self.subject_dir = self.experiment_dir / f'{program_name}_{n_qubits}qubits'
        oracle_file = self.subject_dir / 'oracle' / distribution_name / f'{program_name}_oracle.xlsx'
        frame = validate_oracle(pd.read_excel(oracle_file, sheet_name=f'{program_name}_{fault_version}'))
        if len(frame) != 2 ** n_qubits:
            raise ValueError(f'Expected {2 ** n_qubits} distinct cases, found {len(frame)}.')
        self._selected_names = frame['input'].tolist()
        self._original_cases = original_cases or ROOT / f'TestCases/quantum_test_cases_{n_qubits}qubits.json'
        candidate = Path(supplement_cases or ROOT / f'TestCases/supplements/supplement_{n_qubits}qubits.pkl.gz')
        if supplement_cases is not None and not candidate.exists():
            raise FileNotFoundError(candidate)
        self._supplement_cases = candidate if candidate.exists() else None
        use_mixed = distribution_sizes(n_qubits)[distribution_name]['mixed'] > 0
        result_dir = self.subject_dir / 'results' / distribution_name / f'{program_name}_{fault_version}'
        result_dir.mkdir(parents=True, exist_ok=True)
        cache_identity = {'original': file_digest(testcase_io.resolve_testcase_file(str(self._original_cases))),
                          'supplement': file_digest(self._supplement_cases) if self._supplement_cases else None,
                          'names': self._selected_names}
        cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:20]
        super().__init__(
            len(frame), n_qubits, n_qubits, f'{program_name}_{fault_version}', program_name,
            'QHDRT', iter_num, classification_method='type_clustering', distance_metric='hybrid',
            n_clusters=n_clusters, use_mixstate=use_mixed, seed=seed, result_dir=str(result_dir),
            experiment_cache_dir=str(self.subject_dir / 'cache' / distribution_name / cache_key),
            oracle_path=str(oracle_file), include_entangled_qubits=include_entangled_qubits)
        # Normalize legacy Excel Boolean spelling without changing the core algorithm.
        self.oracle = dict(zip(frame['input'], frame['result'].map(lambda value: str(bool(value)))))

    def _load_test_cases(self, _filename):
        pool = load_pool(self._original_cases, self._supplement_cases)
        missing = set(self._selected_names) - set(pool)
        if missing:
            raise ValueError('Oracle cases missing from testcase pool: ' + str(sorted(missing)[:5]))
        selected = {name: pool[name] for name in self._selected_names}
        counts = {kind: sum(category(data['attributes']) == kind for data in selected.values())
                  for kind in ('classical', 'superposition', 'mixed')}
        expected = distribution_sizes(self.n_qubits)[self.distribution_name]
        if counts != expected:
            raise ValueError(f'Wrong distribution sizes: actual={counts}, expected={expected}')

        frame = pd.read_excel(self.subject_dir / 'oracle' / self.distribution_name
                              / f'{self.program_name}_oracle.xlsx', sheet_name=self.group_name)
        for name, encoded in zip(frame['input'], frame['attributes']):
            expected_attrs = json.loads(json.dumps(selected[name]['attributes']))
            if json.loads(encoded) != expected_attrs:
                raise ValueError('Oracle/testcase attributes differ: ' + name)
        return {'test_cases': selected}


def find_all_distribution_experiments(base_dir=ROOT / 'distribution_experiments'):
    experiments = []
    for directory in sorted(Path(base_dir).iterdir()):
        match = re.fullmatch(r'(.+)_(\d+)qubits', directory.name)
        if not directory.is_dir() or not match:
            continue
        program, n_qubits = match.group(1), int(match.group(2))
        for oracle in sorted(directory.glob(f'oracle/*/{program}_oracle.xlsx')):
            with pd.ExcelFile(oracle) as workbook:
                for sheet in workbook.sheet_names:
                    version_match = re.fullmatch(re.escape(program) + r'_(\d+)', sheet)
                    if version_match:
                        experiments.append({'experiment_dir': str(Path(base_dir).resolve()),
                                            'program_name': program, 'n_qubits': n_qubits,
                                            'distribution_name': oracle.parent.name,
                                            'fault_version': int(version_match.group(1)),
                                            'oracle_file': str(oracle)})
    return experiments


def save_statistics(algorithm):
    """Keep statistics1.txt readable by the existing distribution plotter."""

    thresholds = [i / 20 for i in range(1, 20)] + [1]
    count_fail = {}
    for threshold in thresholds:
        values = []
        for history in algorithm.all_runs_fail_history:

            index = max((i for i in range(len(history)) if (i + 1) / len(history) <= threshold), default=-1)
            values.append(history[index]['fail_count'] if index >= 0 else 0)
        count_fail[str(threshold)] = values
    averages = {key: sum(values) / len(values) for key, values in count_fail.items()}
    maxima = {key: max(values) for key, values in count_fail.items()}
    contents = ('count_fail:\n' + json.dumps(count_fail) + '\navg_failCount:\n'
                + json.dumps(averages) + '\nmax_failCount:\n' + json.dumps(maxima) + '\n')
    contents += (f'program: {algorithm.program_name}\ndistribution: {algorithm.distribution_name}\n'
                 f'iterations: {algorithm.iter_num}\ncases: {algorithm.upper_bound}\n')
    directory = Path(algorithm.result_dir)
    (directory / f'{algorithm.program_name}_QHDRT_statistics1.txt').write_text(contents, encoding='utf-8')
    suffix = f'type_clustering_hybrid_n{algorithm.n_clusters}_' + ('MixState' if algorithm.use_mixstate else 'NoMixState')
    (directory / f'{algorithm.program_name}_QHDRT_{suffix}_failCount.txt').write_text(contents, encoding='utf-8')


def run_one(exp, args):
    directory = Path(exp['experiment_dir']) / f"{exp['program_name']}_{exp['n_qubits']}qubits" / 'results' / exp['distribution_name'] / f"{exp['program_name']}_{exp['fault_version']}"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / 'run_metadata.json'
    original = Path(testcase_io.resolve_testcase_file(str(args.original_cases or
                    ROOT / f"TestCases/quantum_test_cases_{exp['n_qubits']}qubits.json")))
    candidate = args.supplement_cases or ROOT / f"TestCases/supplements/supplement_{exp['n_qubits']}qubits.pkl.gz"
    signature = {'oracle_sha256': file_digest(exp['oracle_file']),
                 'original_sha256': file_digest(original),
                 'supplement_sha256': file_digest(candidate) if Path(candidate).exists() else None,
                 'iterations': args.iter_num, 'n_clusters': args.n_clusters, 'seed': args.seed,
                 'include_entangled_qubits': args.include_entangled_qubits,
                 'algorithm_sha256': file_digest(ROOT / 'QHDRT.py'),
                 'runner_sha256': file_digest(__file__), 'experiment': exp}
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding='utf-8'))
        if saved.get('status') == 'completed' and saved.get('signature') == signature and not args.rerun:
            outputs = saved.get('outputs', {})
            if outputs and all((directory / name).is_file() and file_digest(directory / name) == checksum
                               for name, checksum in outputs.items()):
                print('Skip completed: ' + str(directory), flush=True)
                return
    existing = [p for p in directory.iterdir() if p.name != 'previous']
    if existing:
        if not args.rerun:
            raise FileExistsError('Existing/incomplete results require --rerun (archives old files): ' + str(directory))
        archive = directory / 'previous' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        archive.mkdir(parents=True)
        for path in existing:
            shutil.move(str(path), str(archive / path.name))
    atomic_json(metadata, {'status': 'running', 'signature': signature})
    monitor = PeakMemoryMonitor().start()
    started = time.perf_counter()
    try:
        algorithm = DistributionExperimentQHDRT(
            exp['experiment_dir'], exp['program_name'], exp['n_qubits'], exp['distribution_name'],
            exp['fault_version'], args.iter_num, args.n_clusters, original, args.supplement_cases,
            args.seed, args.include_entangled_qubits)
        algorithm.preprocessing_time = time.perf_counter() - started
        algorithm.memory_monitor = monitor
        algorithm.preprocessing_memory_usage = monitor.snapshot()
        algorithm.run()
        monitor.stop()
        save_t_measure_data(algorithm, algorithm.group_name, algorithm.program_name, 'QHDRT',
                            'hybrid', args.n_clusters, algorithm.use_mixstate, 'type_clustering')
        save_statistics(algorithm)
        atomic_json(metadata, {'status': 'completed', 'signature': signature,
                               'preprocessing_time': algorithm.preprocessing_time,
                               'online_selection_time': algorithm.online_selection_time,
                               'result_dir': str(directory),
                               'outputs': {str(p.relative_to(directory)): file_digest(p)
                                           for p in directory.rglob('*') if p.is_file()
                                           and p != metadata and 'previous' not in p.relative_to(directory).parts}})
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(metadata, {'status': 'failed', 'signature': signature, 'error': str(exc)})
        raise
    finally:
        monitor.stop()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_dir', '--base-dir', type=Path, default=ROOT / 'distribution_experiments')
    parser.add_argument('--program', default=None)
    parser.add_argument('--n_qubits', '--n-qubits', type=int)
    parser.add_argument('--distribution', help='One distribution; default all five.')
    parser.add_argument('--fault_versions', '--fault-versions', default=None)
    parser.add_argument('--iter_num', '--iter-num', type=int, default=50)
    parser.add_argument('--n_clusters', '--n-clusters', type=int, default=5)
    parser.add_argument('--classification_method', choices=['type_clustering'], default='type_clustering')
    parser.add_argument('--distance_metric', choices=['hybrid'], default='hybrid')
    parser.add_argument('--use_mixstate', action='store_true', help='Compatibility flag; Mixed is enabled according to each distribution.')
    parser.add_argument('--cand_num', type=int, default=1, help='Compatibility parameter; production QHDRT does not use it.')
    parser.add_argument('--original-cases', type=Path)
    parser.add_argument('--supplement-cases', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include_entangled_qubits', '--include-entangled-qubits', action='store_true')
    parser.add_argument('--rerun', action='store_true', help='Archive old files before rerunning; never append old rounds.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.iter_num < 1 or args.n_clusters < 1 or args.seed < 0:
        parser.error('Iterations/clusters must be positive, seed nonnegative.')
    experiments = find_all_distribution_experiments(args.base_dir)
    versions = parse_versions(args.fault_versions) if args.fault_versions else (
        parse_versions(DEFAULT_VERSIONS) if args.program == 'QAOA_12' else None)
    experiments = [exp for exp in experiments
                   if (args.program is None or exp['program_name'] == args.program)
                   and (args.n_qubits is None or exp['n_qubits'] == args.n_qubits)
                   and (args.distribution is None or exp['distribution_name'] == args.distribution)
                   and (versions is None or exp['fault_version'] in versions)]
    if not experiments:
        raise ValueError('No matching distribution oracles. Generate them first.')
    if args.program and versions is not None:
        qubit_counts = {args.n_qubits} if args.n_qubits else {exp['n_qubits'] for exp in experiments}
        for n_qubits in qubit_counts:
            distributions = [args.distribution] if args.distribution else list(distribution_sizes(n_qubits))
            expected = {(distribution, version) for distribution in distributions for version in versions}
            actual = {(exp['distribution_name'], exp['fault_version']) for exp in experiments
                      if exp['n_qubits'] == n_qubits}
            if expected - actual:
                raise ValueError('Missing requested distribution/version sheets: ' + str(sorted(expected - actual)))
    if args.single:
        if len(experiments) != 1:
            parser.error('--single requires exactly one experiment.')
        run_one(experiments[0], args)
        return
    failures = []
    for index, exp in enumerate(experiments, 1):
        print(f"[{index}/{len(experiments)}] {exp['program_name']}_{exp['fault_version']} / {exp['distribution_name']}", flush=True)
        if args.dry_run:
            continue

        command = [sys.executable, str(Path(__file__).resolve()), '--single',
                   '--base_dir', str(args.base_dir.resolve()), '--program', exp['program_name'],
                   '--n_qubits', str(exp['n_qubits']), '--distribution', exp['distribution_name'],
                   '--fault_versions', str(exp['fault_version']), '--iter_num', str(args.iter_num),
                   '--n_clusters', str(args.n_clusters), '--seed', str(args.seed)]
        for option, value in [('--original-cases', args.original_cases), ('--supplement-cases', args.supplement_cases)]:
            if value:
                command.extend([option, str(value.resolve())])
        for option, enabled in [('--rerun', args.rerun), ('--include_entangled_qubits', args.include_entangled_qubits)]:
            if enabled:
                command.append(option)
        if subprocess.run(command, cwd=str(ROOT)).returncode:
            failures.append(exp)
    if failures:
        raise RuntimeError(f'{len(failures)} experiment(s) failed. Completed runs will be skipped on restart.')


if __name__ == '__main__':
    main()
