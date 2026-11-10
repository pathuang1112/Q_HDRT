"""Generate fresh oracle workbooks for the selected subject programs."""
import argparse
import subprocess
import sys

from experiment_config import ROOT, SUBJECTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--programs', nargs='+', choices=list(SUBJECTS), default=list(SUBJECTS))
    parser.add_argument('--swap-shots', type=int, default=50)
    parser.add_argument('--distribution-shots', type=int, default=None)
    parser.add_argument('--distribution-resamples', type=int, default=9999)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--rerun', action='store_true', help='Recompute distribution-based sheets explicitly.')
    args = parser.parse_args()
    if (args.swap_shots < 1 or not 0 < args.alpha < 1 or args.distribution_resamples < 1
            or (args.distribution_shots is not None and args.distribution_shots < 1)):
        parser.error('Invalid shots or alpha')
    for program in args.programs:
        if program in ('adder_n7', 'rand_cliff_8_AG', 'QAOA_12'):
            command = ['oracle_distribution_based.py',
                       '--program', program, '--fault-versions', 'all', '--alpha', str(args.alpha),
                       '--output-dir', str(ROOT / 'oracle/OpenQASM')]
            command += ['--resamples', str(args.distribution_resamples)]
            if args.distribution_shots is not None:
                command += ['--shots', str(args.distribution_shots)]
            if args.rerun:
                command.append('--rerun')
        else:
            command = ['oracle_swaptest_based.py', '--program', program, '--shots', str(args.swap_shots)]
        command.insert(0, sys.executable)
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
