"""Full factorial profile adjustment experiment; run from any directory."""
import argparse
import ast
import contextlib
from decimal import Decimal, InvalidOperation
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GRID = [0.01, 0.03, 0.05, 0.10, 0.20, 0.25]

LEGACY_RUNNER_SHA256 = 'dfc8388f8e008caa725e0327238ae718229f36887b552fa306206d500e7e8a4a'
sys.path.insert(0, str(ROOT))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def merge_manifests(existing, requested, verified_input_updates=None):
    """Union subjects/inputs while retaining strict checks on experiment settings."""
    conflicts = [key for key in ('settings', 'epsilon', 'delta', 'repeats', 'seed',
                                 'algorithm_sha256')
                 if existing.get(key) != requested.get(key)]
    if existing.get('runner_sha256') not in (
            requested['runner_sha256'], LEGACY_RUNNER_SHA256):
        conflicts.append('runner_sha256')
    subjects = {s['group']: s for s in existing['subjects']}
    for subject in requested['subjects']:
        group = subject['group']
        if group in subjects and subjects[group] != subject:
            conflicts.append(f'subjects[{group}]')
        subjects[group] = subject
    inputs = dict(existing['input_sha256'])
    inputs.update(verified_input_updates or {})
    for path, fingerprint in requested['input_sha256'].items():
        if path in inputs and inputs[path] != fingerprint:
            conflicts.append(f'input_sha256[{path}]')
        inputs[path] = fingerprint
    if conflicts:
        raise ValueError('Cannot append experiments: incompatible ' + ', '.join(conflicts)
                         + '; use a new --output for different settings/data/code')
    merged = dict(existing)
    merged.update(subjects=list(subjects.values()), input_sha256=inputs,
                  runner_sha256=requested['runner_sha256'])
    history = list(existing.get('runner_sha256_history', []))
    for fingerprint in (existing['runner_sha256'], requested['runner_sha256']):
        if fingerprint not in history:
            history.append(fingerprint)
    merged['runner_sha256_history'] = history
    if verified_input_updates:
        input_history = {path: list(values) for path, values in
                         existing.get('input_sha256_history', {}).items()}
        for path, fingerprint in verified_input_updates.items():
            versions = input_history.setdefault(path, [])
            for value in (existing['input_sha256'][path], fingerprint):
                if value not in versions:
                    versions.append(value)
        merged['input_sha256_history'] = input_history
    return merged


def revalidate_oracle_results(output, subjects, oracle_path):
    import pandas as pd

    def same_value(old, current):
        if old == current:
            return True
        try:
            return Decimal(old) == Decimal(current)
        except (InvalidOperation, TypeError, ValueError):
            return False

    verified = []
    with pd.ExcelFile(oracle_path) as workbook:
        for subject in subjects:
            group = subject['group']
            paths = sorted((output / 'raw' / group).glob('eps_*_delta_*.json'))
            if not paths:
                continue
            raw = json.loads(paths[0].read_text(encoding='utf-8'))
            run = raw['runs'][0]
            order, counts = run['executed_order'], run['cumulative_failures']
            if (len(order) != raw['n_cases'] or len(set(order)) != len(order)
                    or len(counts) != len(order) or not counts
                    or counts[-1] != raw['total_failures']):
                raise ValueError(f'Cannot verify changed Oracle: incomplete raw history for {group}')
            frame = pd.read_excel(workbook, sheet_name=group, dtype=str)
            labels = dict(zip(frame.iloc[:, 0], frame.iloc[:, 3]))
            pvalues = dict(zip(frame.iloc[:, 0], frame.iloc[:, 4]))
            previous = 0
            for name, count in zip(order, counts):
                failed = count - previous
                previous = count
                passed = ast.literal_eval(labels[name]) if name in labels else None
                if failed not in (0, 1) or type(passed) is not bool or passed != (failed == 0):
                    raise ValueError(f'Oracle result changed: {group}/{name}; use a new --output')

            legacy_dir = output / 'raw' / group / 'legacy' / paths[0].stem
            text_files = [path for path in legacy_dir.glob('*.txt')
                          if not path.name.endswith('_time.txt')]
            if len(text_files) != 1:
                raise ValueError(f'Cannot verify changed Oracle: missing original p-value log for {group}')
            with text_files[0].open(encoding='utf-8') as stream:
                logged_order = stream.readline().split()
                logged_counts = [int(x) for x in stream.readline().split()]
                logged_pvalues = stream.readline().split()
            if logged_order != order or logged_counts != counts or len(logged_pvalues) != len(order):
                raise ValueError(f'Cannot verify changed Oracle: inconsistent original log for {group}')
            for name, value in zip(order, logged_pvalues):
                if name not in pvalues or not same_value(value, pvalues[name]):
                    raise ValueError(f'Oracle p-value changed: {group}/{name}; use a new --output')
            verified.append(group)
    return verified


def register_manifest(output, requested):
    """Validate and register a batch; never remove existing subjects or raw data."""
    manifest_path = output / 'manifest.json'
    if not manifest_path.exists():
        atomic_json(manifest_path, requested)
        return requested
    existing = json.loads(manifest_path.read_text(encoding='utf-8'))

    verified_updates = {}
    revalidations = list(existing.get('oracle_revalidations', []))
    for path, fingerprint in existing['input_sha256'].items():
        current = requested['input_sha256'].get(path)
        if current is None:
            if not Path(path).is_file():
                raise ValueError(f'Existing experiment input is missing: {path}')
            current = digest(path)
        if current != fingerprint:
            oracle_subjects = [subject for subject in existing['subjects']
                               if Path(path).resolve() == (ROOT / 'oracle' / 'OpenQASM' /
                                  (subject['program'] + '_oracle.xlsx')).resolve()]
            if not oracle_subjects:
                raise ValueError(f'Existing experiment input changed: {path}; use a new --output')
            groups = revalidate_oracle_results(output, oracle_subjects, path)
            verified_updates[path] = current
            revalidations.append(dict(path=path, previous_sha256=fingerprint, current_sha256=current,
                                      verified_groups=groups, evidence='saved_complete_order_labels_and_pvalues'))
            print('Oracle workbook changed; saved labels and p-values verified for: '
                  + ', '.join(groups), flush=True)
    merged = merge_manifests(existing, requested, verified_updates)
    if revalidations:
        merged['oracle_revalidations'] = revalidations
    if merged != existing:
        backup = output / 'manifest_history' / (digest(manifest_path) + '.json')
        if not backup.exists():
            atomic_json(backup, existing)
        atomic_json(manifest_path, merged)
    return merged


def progress_state(output, manifest):
    pairs = list(itertools.product(manifest['epsilon'], manifest['delta']))
    completed = sum(
        (output / 'raw' / subject['group'] / f'eps_{e:.6g}_delta_{d:.6g}.json').is_file()
        for subject in manifest['subjects'] for e, d in pairs)
    expected = len(manifest['subjects']) * len(pairs)
    return dict(completed_pairs=completed, expected_pairs=expected,
                status='complete' if completed == expected else 'partial')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=HERE / 'subjects.json')
    p.add_argument('--output', type=Path, default=HERE / 'results')
    p.add_argument('--groups', nargs='+', help='Optional subset of configured oracle sheets')
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261009)
    p.add_argument('--epsilon', nargs='+', type=float, default=GRID)
    p.add_argument('--delta', nargs='+', type=float, default=GRID)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--prepare-only', action='store_true',
                   help='Validate/register this batch in the output directory without running it')
    args = p.parse_args()
    if args.repeats < 2 or args.seed < 0 or args.seed + args.repeats >= 2**32:
        p.error('Use at least two repeats and valid NumPy seeds')
    if any(not 0 < x <= 1 for x in args.epsilon + args.delta):
        p.error('All profile adjustment values must be in (0, 1]')
    if len(set(args.epsilon)) != len(args.epsilon) or len(set(args.delta)) != len(args.delta):
        p.error('Duplicate grid values are not allowed')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    subjects = config['subjects']
    if args.groups:
        subjects = [s for s in subjects if s['group'] in args.groups]
        if set(args.groups) != {s['group'] for s in subjects}:
            p.error('Some requested groups are absent from the configuration')
    if not subjects or len({s['group'] for s in subjects}) != len(subjects):
        p.error('Configure unique, nonempty subjects')
    pairs = list(itertools.product(args.epsilon, args.delta))

    pairs.sort(key=lambda pair: (pair != (0.05, 0.05), pair))
    print(f'{len(subjects)} subjects x {len(pairs)} pairs x {args.repeats} repeats', flush=True)
    if args.dry_run:
        for s in subjects:
            print(s['group'], s['input_qubits'], s['output_qubits'])
        return

    import numpy as np
    import pandas as pd
    import testcase_io
    from QHDRT import QHDRT

    output = args.output.resolve()
    os.chdir(ROOT)
    settings = config['settings']
    inputs = {}
    for s in subjects:
        oracle = ROOT / 'oracle' / 'OpenQASM' / (s['program'] + '_oracle.xlsx')
        dataset = Path(testcase_io.resolve_testcase_file(
            ROOT / 'TestCases' / f"quantum_test_cases_{s['input_qubits']}qubits.json"))
        for path in (oracle, dataset):
            if str(path) not in inputs:
                inputs[str(path)] = digest(path)
        with pd.ExcelFile(oracle) as workbook:
            if s['group'] not in workbook.sheet_names:
                raise ValueError(f"Missing oracle sheet: {s['group']}")
    manifest = dict(subjects=subjects, settings=settings, epsilon=args.epsilon,
                    delta=args.delta, repeats=args.repeats, seed=args.seed,
                    input_sha256=inputs, algorithm_sha256=digest(ROOT / 'QHDRT.py'),
                    runner_sha256=digest(HERE / 'run_sensitivity.py'))
    manifest = register_manifest(output, manifest)
    progress = progress_state(output, manifest)
    atomic_json(output / 'progress.json', progress)
    completed, expected = progress['completed_pairs'], progress['expected_pairs']
    print(f'Output directory: {len(manifest["subjects"])} registered subjects, '
          f'{completed}/{expected} completed pairs', flush=True)
    if args.prepare_only:
        return

    for s in subjects:
        group_dir = output / 'raw' / s['group']
        group_dir.mkdir(parents=True, exist_ok=True)
        pending = [(e, d) for e, d in pairs
                   if not (group_dir / f'eps_{e:.6g}_delta_{d:.6g}.json').exists()]
        if not pending:
            continue
        with open(group_dir / 'execution.log', 'a', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log):
                random.seed(args.seed)
                np.random.seed(args.seed)
                started = time.perf_counter()
                alg = QHDRT(-1, s['input_qubits'], s['output_qubits'], s['group'],
                             s['program'], 'QHDRT', args.repeats,
                             settings['classification_method'], settings['distance_metric'],
                             settings['n_clusters'], settings['use_mixstate'],
                             seed=args.seed, result_dir=str(group_dir / 'legacy'))
                alg.preprocessing_time = time.perf_counter() - started

                clustering = {
                    'clusters': {str(k): v for k, v in alg.base_subclusters.items()},
                    'distance_matrix': alg.base_medoid_dist_matrix.tolist()}
                clustering_path = group_dir / 'clustering.json'
                if clustering_path.exists():
                    if json.loads(clustering_path.read_text(encoding='utf-8')) != clustering:
                        raise ValueError('Clustering changed since the previous run; use a new --output')
                else:
                    atomic_json(clustering_path, clustering)
                if set(alg.json_data) - set(alg.oracle):
                    raise ValueError('Oracle does not cover every selected test case')
                labels = [ast.literal_eval(alg.oracle[c]) for c in alg.json_data]
                if any(type(x) is not bool for x in labels):
                    raise ValueError('Oracle labels must be boolean pass/fail values')
                total_failures = sum(not x for x in labels)
                for epsilon, delta in pending:
                    alg.epsilon, alg.delta = epsilon, delta
                    alg.all_runs_test_cases = []
                    alg.all_runs_fail_history = []
                    alg.max_fail_count = -1
                    alg.best_candidates = []
                    alg.result_dir = str(group_dir / 'legacy' / f'eps_{epsilon:.6g}_delta_{delta:.6g}')
                    alg.run()
                    runs = []
                    for i, (order, history) in enumerate(zip(
                            alg.all_runs_test_cases, alg.all_runs_fail_history)):
                        counts = [int(point['fail_count']) for point in history]
                        if len(set(order)) != alg.upper_bound or counts[-1] != total_failures:
                            raise AssertionError('Incomplete test ordering or inconsistent total failures')
                        runs.append(dict(run_id=i, seed=args.seed + i, executed_order=order,
                                         cumulative_failures=counts))
                    atomic_json(group_dir / f'eps_{epsilon:.6g}_delta_{delta:.6g}.json', dict(
                        subject=s, epsilon=epsilon, delta=delta,
                        n_cases=alg.upper_bound, total_failures=total_failures,
                        preprocessing_seconds=alg.preprocessing_time,
                        online_seconds=alg.online_selection_time,
                        runs=runs))
                    completed += 1
                    atomic_json(output / 'progress.json', dict(
                        completed_pairs=completed, expected_pairs=expected, status='running',
                        last_group=s['group'], epsilon=epsilon, delta=delta))
                    print(f'[{completed}/{expected}] {s["group"]} '
                          f'epsilon={epsilon:g}, delta={delta:g}', file=sys.__stdout__, flush=True)
    atomic_json(output / 'progress.json', progress_state(output, manifest))
    from analyze_results import analyze
    analyze(output)


if __name__ == '__main__':
    main()
