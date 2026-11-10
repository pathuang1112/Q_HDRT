
import os
import sys
import json
import argparse
from distance_metric_experiments import find_all_programs_and_qubits, run_distance_metric_experiments
from experiment_config import ROOT


def run_all_experiments(distance_metrics=None, n_clusters=5, iter_num=50,
                        experiment_dir="./distance_metric_experiments"):

    if distance_metrics is None:
        distance_metrics = ['hybrid', 'fidelity', 'euclidean']

    print("=" * 60)
    print("Run dissimilarity comparison experiments in batch")
    print("=" * 60)


    programs_and_qubits = find_all_programs_and_qubits(str(ROOT / "TestCases"))

    if not programs_and_qubits:
        print("No usable testcase file found")
        return

    print(f"Found {len(programs_and_qubits)} programs")

    all_summaries = []

    for program_name, qubits_list in programs_and_qubits.items():
        print(f"\nProgram: {program_name}")
        print(f"Available qubit counts: {qubits_list}")

        for n_qubits in qubits_list:
            print(f"\n{'=' * 40}")
            print(f"Run: {program_name} (Qubit count: {n_qubits})")
            print(f"{'=' * 40}")

            try:
                summary = run_distance_metric_experiments(
                    program_name=program_name,
                    n_qubits=n_qubits,
                    distance_metrics=distance_metrics,
                    n_clusters=n_clusters,
                    iter_num=iter_num,
                    experiment_dir=experiment_dir
                )
                all_summaries.append(summary)
            except Exception as e:
                print(f"Run {program_name}_{n_qubits}qubits Failed: {e}")
                import traceback
                traceback.print_exc()


    if all_summaries:
        generate_overall_summary(all_summaries, experiment_dir)

    print("\n" + "=" * 60)
    print("All dissimilarity comparison experiments completed!")
    print("=" * 60)


def generate_overall_summary(all_summaries, experiment_dir):
    report_file = os.path.join(experiment_dir, "overall_summary.md")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Overall dissimilarity comparison report\n\n")

        f.write("## Experiment statistics\n\n")
        f.write(f"- Total experiment combinations: {len(all_summaries)}\n")


        distance_metric_stats = {}
        for summary in all_summaries:
            for distance_metric, metric_summary in summary.get('results_by_distance_metric', {}).items():
                if distance_metric not in distance_metric_stats:
                    distance_metric_stats[distance_metric] = {
                        'total': 0,
                        'successful': 0,
                        'failed': 0
                    }
                stats = distance_metric_stats[distance_metric]
                stats['total'] += metric_summary['total']
                stats['successful'] += metric_summary['successful']
                stats['failed'] += metric_summary['failed']

        f.write("\n## Overall performance by dissimilarity measure\n\n")
        f.write("| Dissimilarity measure | Total experiments | Passing test cases | Failures | Success rate |\n")
        f.write("|----------|----------|--------|--------|--------|\n")

        for distance_metric, stats in distance_metric_stats.items():
            if stats['total'] > 0:
                success_rate = stats['successful'] / stats['total'] * 100
                f.write(f"| {distance_metric} | {stats['total']} | {stats['successful']} | ")
                f.write(f"{stats['failed']} | {success_rate:.1f}% |\n")

        f.write("\n## Experimental results by program\n\n")
        for summary in all_summaries:
            f.write(f"### {summary['program_name']} ({summary['n_qubits']}qubits)\n")
            for distance_metric, metric_summary in summary.get('results_by_distance_metric', {}).items():
                success_rate = metric_summary['successful'] / metric_summary['total'] * 100
                f.write(f"- {distance_metric}: {metric_summary['successful']}/{metric_summary['total']} ")
                f.write(f"({success_rate:.1f}%)\n")
            f.write("\n")

    print(f"\nOverall summary report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Run all dissimilarity comparison experiments')

    parser.add_argument('--distance_metrics', type=str, default='fidelity,euclidean',
                        help='Dissimilarity measure list, Comma-separated')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='Number of clusters')
    parser.add_argument('--iter_num', type=int, default=50,
                        help='Repetition count')
    parser.add_argument('--experiment_dir', type=str, default='./',
                        help='Experiment directory')

    args = parser.parse_args()


    distance_metrics = args.distance_metrics.split(',')


    run_all_experiments(
        distance_metrics=distance_metrics,
        n_clusters=args.n_clusters,
        iter_num=args.iter_num,
        experiment_dir=args.experiment_dir
    )


if __name__ == "__main__":
    main()
