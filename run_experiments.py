"""Run selected main experiments and subexperiments with shared settings."""
import argparse
import subprocess
import sys

from experiment_config import ROOT, SUBJECTS, SELECTED_GROUPS, REPEATS

# Edit this list to choose versions without a long command line.
selected_groups = list(SELECTED_GROUPS)
ALGORITHMS = ['RT', 'RPT', 'DRT_1', 'ART_1', 'QHDRT']


def jobs(experiment, groups, repeats):
    if experiment == 'testcases':
        for qubits in sorted({SUBJECTS[g.rsplit('_', 1)[0]]['qubits'] for g in groups}):
            yield ['TestcaseGenerator.py', '--n_qubits', str(qubits), '--include_mixed']
        return
    for group in groups:
        program, version = group.rsplit('_', 1)
        qubits = SUBJECTS[program]['qubits']
        prefix = [str(qubits), str(qubits), group, program]
        shared = ['--iter_num', str(repeats), '--use_mixstate', '--n_clusters', '5']
        if experiment in ('main', 'ablation'):
            algorithms = ALGORITHMS if experiment == 'main' else ['DRT_C', 'DRT_D']
            for algorithm in algorithms:
                if algorithm in ('QHDRT', 'DRT_C'):
                    script = 'QHDRT.py' if algorithm == 'QHDRT' else 'QDRT_C.py'
                    yield [script] + prefix + [algorithm] + shared + ['--classification_method', 'type_clustering', '--distance_metric', 'hybrid']
                else:
                    script = 'QDRT_D.py' if algorithm == 'DRT_D' else 'QHDRT_baselines.py'
                    yield [script] + prefix + [algorithm] + shared
        elif experiment == 'distance':
            for metric in ('hybrid', 'fidelity', 'euclidean', 'trace'):
                output = ROOT / 'distance_metric_experiments/results' / f'{program}_{qubits}qubits' / metric / group
                yield ['QHDRT.py'] + prefix + ['QHDRT'] + shared + [
                    '--classification_method', 'type_clustering', '--distance_metric', metric,
                    '--result_dir', str(output)]
        elif experiment == 'k':
            yield ['k_sensitivity_experiments/run_sensitivity.py', '--groups', group, '--repeats', str(repeats)]
        elif experiment == 'profile':
            yield ['profile_adjustment_experiments/run_sensitivity.py', '--groups', group, '--repeats', str(repeats)]
        elif experiment == 'distribution':
            yield ['distribution_experiments/run_distribution_experiments.py', '--program', program,
                   '--fault-versions', version, '--iter_num', str(repeats)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['main', 'ablation', 'distance', 'distribution', 'k', 'profile', 'testcases'])
    parser.add_argument('--groups', nargs='+', default=None)
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    groups = args.groups or selected_groups
    if not groups or set(groups) - set(SELECTED_GROUPS) or args.repeats < 1:
        parser.error('Choose canonical version names 1 through 5 and positive repeats.')
    for arguments in jobs(args.experiment, groups, args.repeats):
        command = [sys.executable] + arguments
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
