"""Shared within-type k sensitivity for QHDRT; runnable from any directory."""
import argparse
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
DEFAULT_K = [1, 2, 3, 4, 5, 7, 10]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def workbook_groups(path, program):
    # Read sheet names without importing the quantum runtime or loading cells.
    with ZipFile(path) as book:
        root = ET.fromstring(book.read('xl/workbook.xml'))
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    pattern = re.compile(re.escape(program) + r'_\d+$')
    groups = [sheet.attrib['name'] for sheet in root.findall('x:sheets/x:sheet', ns)
              if pattern.fullmatch(sheet.attrib['name'])]
    return sorted(groups, key=lambda name: int(name.rsplit('_', 1)[1]))


def discover_subjects(registry, programs, groups, root=ROOT, oracle_file=None):
    unknown = set(programs) - set(registry)
    if unknown:
        raise ValueError(f'Unknown programs: {sorted(unknown)}; update programs.json')
    owners = {}
    for group in groups:
        owner = next((p for p in registry if re.fullmatch(re.escape(p) + r'_\d+', group)), None)
        if owner is None:
            raise ValueError(f'Unknown group: {group}; expected PROGRAM_VERSION')
        owners[group] = owner
    selected_programs = sorted(set(programs) | set(owners.values()))
    if oracle_file is not None and len(selected_programs) != 1:
        raise ValueError('--oracle-file requires selecting exactly one program')
    selected = []
    for program in selected_programs:
        oracle = (Path(oracle_file) if oracle_file is not None else
                  root / 'oracle' / 'OpenQASM' / f'{program}_oracle.xlsx')
        if not oracle.is_file():
            raise ValueError(f'Missing oracle workbook: {oracle}')
        available = workbook_groups(oracle, program)
        requested = {g for g, p in owners.items() if p == program}
        if requested - set(available):
            raise ValueError(f'Groups absent from {oracle.name}: {sorted(requested - set(available))}')
        chosen = available if program in programs else [g for g in available if g in requested]
        if not chosen:
            raise ValueError(f'No version sheets found for {program}')
        selected.extend(dict(program=program, group=g, **registry[program]) for g in chosen)
    return selected


def require_manifest(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Experiment settings or inputs changed: {path}; use a new --output')
    else:
        atomic_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', '--programs', nargs='+', default=[], dest='programs',
                        help='Run all numbered oracle versions of each program')
    parser.add_argument('--groups', nargs='+', default=[], help='Run selected oracle version sheets')
    parser.add_argument('--config', type=Path, default=HERE / 'programs.json')
    parser.add_argument('--oracle-file', type=Path,
                        help='Explicit oracle workbook for a single selected program')
    parser.add_argument('--k-values', nargs='+', type=int, default=DEFAULT_K)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--output', type=Path, default=HERE / 'result')
    parser.add_argument('--distance-metric', choices=['hybrid', 'fidelity', 'trace', 'euclidean'],
                        default='hybrid')
    parser.add_argument('--no-mixstate', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='List jobs without running experiments')
    parser.add_argument('--list-subjects', action='store_true', help='List available programs and versions')
    parser.add_argument('--no-analysis', action='store_true')
    args = parser.parse_args()
    if args.oracle_file is not None:
        args.oracle_file = args.oracle_file.resolve()
    if (args.repeats < 1 or args.seed < 0 or args.seed + args.repeats - 1 >= 2**32
            or any(k < 1 for k in args.k_values) or len(set(args.k_values)) != len(args.k_values)):
        parser.error('Use positive unique k values, positive repeats and valid NumPy seeds')
    registry = json.loads(args.config.read_text(encoding='utf-8'))
    if args.list_subjects:
        if args.oracle_file is not None:
            parser.error('--oracle-file is used with --program/--groups, not --list-subjects')
        for program in registry:
            oracle = ROOT / 'oracle' / 'OpenQASM' / f'{program}_oracle.xlsx'
            if oracle.exists():
                print(program + ': ' + ', '.join(workbook_groups(oracle, program)))
        return
    if not args.programs and not args.groups:
        parser.error('Specify --program or --groups (or both; selection is their union)')
    try:
        subjects = discover_subjects(registry, args.programs, args.groups,
                                     oracle_file=args.oracle_file)
    except ValueError as error:
        parser.error(str(error))
    ks = sorted(args.k_values, key=lambda k: (k != 3, k))
    print(f'{len(subjects)} groups x {len(ks)} k values x {args.repeats} repeats', flush=True)
    for subject in subjects:
        print(f"  {subject['group']}: input={subject['input_qubits']}, output={subject['output_qubits']}")
    if args.oracle_file is not None:
        print(f'Oracle: {args.oracle_file}')
    if args.dry_run:
        return

    import numpy as np
    import testcase_io
    from QHDRT import QHDRT

    output = args.output.resolve()
    os.chdir(ROOT)
    manifest = dict(schema_version=1, classification_method='type_clustering',
                    k_values=sorted(ks), repeats=args.repeats, seed=args.seed,
                    distance_metric=args.distance_metric, use_mixstate=not args.no_mixstate,
                    epsilon=0.05, delta=0.05, clustering_seed=42,
                    budget_rule='floor(0.30 * n_cases)',
                    algorithm_sha256=digest(ROOT / 'QHDRT.py'),
                    runner_sha256=digest(Path(__file__)),
                    testcase_io_sha256=digest(ROOT / 'testcase_io.py'))


    input_manifests = []
    dataset_hashes = {}
    oracle_hashes = {}
    for subject in subjects:
        dataset = Path(testcase_io.resolve_testcase_file(
            ROOT / 'TestCases' / f"quantum_test_cases_{subject['input_qubits']}qubits.json"))
        oracle = (args.oracle_file if args.oracle_file is not None else
                  ROOT / 'oracle' / 'OpenQASM' / f"{subject['program']}_oracle.xlsx")
        if dataset not in dataset_hashes:
            dataset_hashes[dataset] = digest(dataset)
        if oracle not in oracle_hashes:
            oracle_hashes[oracle] = digest(oracle)
        input_manifests.append(dict(subject=subject, dataset=str(dataset),
                                    dataset_sha256=dataset_hashes[dataset],
                                    oracle=str(oracle), oracle_sha256=oracle_hashes[oracle]))
    require_manifest(output / 'manifest.json', manifest)
    for item in input_manifests:
        group = item['subject']['group']
        require_manifest(output / 'raw' / group / 'manifest.json', item)
    completed = 0
    expected = len(subjects) * len(ks)
    for item in input_manifests:
        subject = item['subject']
        group_dir = output / 'raw' / subject['group']

        cache = output / 'cache' / item['dataset_sha256']
        for k in ks:
            target = group_dir / f'k_{k}.json'
            if target.exists():
                completed += 1
                print(f"[{completed}/{expected}] {subject['group']} k={k}: already complete", flush=True)
                continue
            print(f"[{completed + 1}/{expected}] {subject['group']} k={k}: running", flush=True)
            with (group_dir / f'k_{k}.log').open('w', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    started = time.perf_counter()
                    alg = QHDRT(-1, subject['input_qubits'], subject['output_qubits'],
                                 subject['group'], subject['program'], 'QHDRT', args.repeats,
                                 classification_method='type_clustering',
                                 distance_metric=args.distance_metric, use_mixstate=not args.no_mixstate,
                                 seed=args.seed, type_k=k, experiment_cache_dir=cache,
                                 oracle_path=item['oracle'],
                                 result_dir=str(group_dir / 'legacy' / f'k_{k}'))
                    preprocessing = time.perf_counter() - started
                    alg.preprocessing_time = preprocessing
                    if set(alg.json_data) - set(alg.oracle):
                        raise ValueError('Oracle does not cover the full test-case pool')
                    labels = {case: ast.literal_eval(alg.oracle[case]) for case in alg.json_data}
                    if any(type(label) is not bool for label in labels.values()):
                        raise ValueError('Oracle pass/fail labels must be boolean')
                    total_failures = sum(not label for label in labels.values())
                    cluster_types = {}
                    type_counts = {}
                    for cluster_id, cluster in alg.base_subclusters.items():
                        names = cluster['cases']
                        types = set()
                        for name in names:
                            kind = name.split('_', 1)[0]
                            if kind not in {'classical', 'superposition', 'entangled', 'mixed'}:
                                kind = alg.json_data[name]['attributes'].get('type', 'classical')
                            types.add(kind)
                        if len(types) != 1:
                            raise AssertionError('Type clustering mixed different state types')
                        kind = types.pop()
                        type_counts[kind] = type_counts.get(kind, 0) + len(names)
                        cluster_types[kind] = cluster_types.get(kind, 0) + 1
                    for kind in ('classical', 'superposition'):
                        if type_counts.get(kind, 0) < k:
                            raise ValueError(f'{kind} has fewer than k={k} cases; choose a smaller k')
                        if cluster_types.get(kind, 0) != k:
                            raise ValueError(f'k={k} produced only {cluster_types.get(kind, 0)} '
                                             f'nonempty {kind} clusters; inspect the clustering log')
                    atomic_json(group_dir / f'clustering_k_{k}.json', dict(
                        k=k, actual_clusters=len(alg.base_subclusters),
                        clusters_by_type=cluster_types,
                        clusters={str(cid): cluster for cid, cluster in alg.base_subclusters.items()},
                        medoid_distance_matrix=alg.base_medoid_dist_matrix.tolist()))
                    alg.run()
                    runs = []
                    if len(alg.all_runs_test_cases) != args.repeats:
                        raise AssertionError('Missing repeat histories')
                    for index, (order, history) in enumerate(zip(
                            alg.all_runs_test_cases, alg.all_runs_fail_history)):
                        counts = [int(point['fail_count']) for point in history]
                        expected_counts = np.cumsum([not labels[case] for case in order]).tolist()
                        if (len(order) != alg.upper_bound or set(order) != set(labels)
                                or counts != expected_counts or counts[-1] != total_failures):
                            raise AssertionError('Invalid ordering or cumulative failure history')
                        runs.append(dict(run_id=index, seed=args.seed + index,
                                         executed_order=order, cumulative_failures=counts))
                    atomic_json(target, dict(subject=subject, k=k,
                        actual_clusters=len(alg.base_subclusters), clusters_by_type=cluster_types,
                        n_cases=alg.upper_bound, total_failures=total_failures,
                        budget_30=alg.upper_bound * 30 // 100,
                        preprocessing_seconds=preprocessing,
                        online_seconds=alg.online_selection_time, runs=runs))
            completed += 1
            atomic_json(output / 'progress.json', dict(selected_subjects=subjects,
                        completed_combinations=completed, expected_combinations=expected,
                        last_group=subject['group'], last_k=k))
    atomic_json(output / 'progress.json', dict(selected_subjects=subjects,
                completed_combinations=completed, expected_combinations=expected, status='complete'))
    if not args.no_analysis:
        from analyze_results import analyze
        analyze(output)


if __name__ == '__main__':
    main()
