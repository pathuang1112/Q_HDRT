"""Verify selection, actual k, cache isolation, metrics and exported analysis."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from QHDRT import QuantumTestClusterer, QHDRT
from run_sensitivity import discover_subjects, require_manifest
from analyze_results import analyze, curve_metrics


def sample_cases():
    return {f'{kind}_{i}': {'attributes': {'type': kind}, 'state': None}
            for kind in ['classical', 'superposition', 'entangled', 'mixed']
            for i in range(6)}


class SensitivityTests(unittest.TestCase):
    def test_explicit_oracle_selects_its_versions(self):
        registry = json.loads((HERE / 'programs.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temporary:
            oracle = Path(temporary) / 'explicit.xlsx'
            with pd.ExcelWriter(oracle) as writer:
                for version in range(1, 6):
                    pd.DataFrame({'result': [True]}).to_excel(writer, sheet_name=f'QAOA_12_{version}')
            subjects = discover_subjects(registry, ['QAOA_12'], [], oracle_file=oracle)
            self.assertEqual([s['group'] for s in subjects], [f'QAOA_12_{i}' for i in range(1, 6)])
            with self.assertRaisesRegex(ValueError, 'exactly one program'):
                discover_subjects(registry, ['QAOA_12', 'adder_n7'], [], oracle_file=oracle)

    def test_algorithm_reads_explicit_oracle_path(self):
        frame = pd.DataFrame([['classical_0', '', '{}', 'True', '1']])
        cases = {'classical_0': {'state': None, 'attributes': {'type': 'classical'}}}
        with patch('QHDRT.pd.read_excel', return_value=frame) as read, \
             patch.object(QHDRT, '_load_test_cases', return_value={'test_cases': cases}), \
             patch('QHDRT.QuantumTestClusterer'), \
             patch.object(QHDRT, 'perform_clustering'), \
             patch.object(QHDRT, '_prepare_online_templates'):
            QHDRT(-1, 12, 12, 'QAOA_12_2', 'QAOA_12', 'QHDRT', 1,
                   oracle_path='explicit_oracle.xlsx')
            read.assert_called_once_with('explicit_oracle.xlsx', sheet_name='QAOA_12_2', dtype=str)

    def test_program_all_versions_and_group_union(self):
        registry = json.loads((HERE / 'programs.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'oracle/OpenQASM'
            folder.mkdir(parents=True)
            for program in ('adder_n7', 'qft_8', 'pe_9'):
                with pd.ExcelWriter(folder / f'{program}_oracle.xlsx') as writer:
                    for version in range(1, 6):
                        pd.DataFrame({'result': [True]}).to_excel(writer, sheet_name=f'{program}_{version}')
            subjects = discover_subjects(registry, ['adder_n7'], ['adder_n7_1', 'qft_8_3'], root=root)
            self.assertEqual({s['group'] for s in subjects},
                             {f'adder_n7_{i}' for i in range(1, 6)} | {'qft_8_3'})
            selected = discover_subjects(registry, [], ['pe_9_1'], root=root)
            self.assertEqual(selected[0]['input_qubits'], 9)
            with self.assertRaisesRegex(ValueError, 'absent'):
                discover_subjects(registry, [], ['adder_n7_99'], root=root)

    def test_type_k_reaches_both_kmedoids_calls(self):
        for k in [1, 3, 5]:
            with self.subTest(k=k), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                clusterer = QuantumTestClusterer(sample_cases(), str(root / 'distance'),
                    str(root / 'classification.json'), use_mixstate=True,
                    classification_method='type_clustering', type_k=k)
                clusterer.allcases = list(clusterer.json_data)
                with patch.object(clusterer, '_compute_subset_distance_matrix',
                                  side_effect=lambda cases, label: np.ones((len(cases), len(cases)))
                                  - np.eye(len(cases))), \
                     patch('QHDRT.KMedoids') as medoids:
                    medoids.return_value.fit_predict.side_effect = (
                        lambda distances: np.arange(len(distances)) % k)
                    with contextlib.redirect_stdout(io.StringIO()):
                        clusterer._cluster_type_clustering()
                self.assertEqual([call.kwargs['n_clusters'] for call in medoids.call_args_list], [k, k])
                self.assertEqual(len(clusterer.medoids), 2*k + 2)
                saved = json.loads((root / 'classification.json').read_text())
                self.assertEqual(saved['type_k'], k)

    def test_cache_with_different_k_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'classification.json'
            cases = sample_cases()
            loaded = dict(cases=list(cases), medoids={'0': next(iter(cases))},
                case_classifications={case: 0 for case in cases}, classification_method='type_clustering',
                distance_metric='hybrid', n_clusters=5, use_mixstate=True, dataset_version='legacy')
            # Existing files without type_k represent the original k=3.
            path.write_text(json.dumps(loaded))
            same = QuantumTestClusterer(cases, str(root / 'cache'), str(path),
                use_mixstate=True, classification_method='type_clustering')
            with patch.object(same, '_cluster_type_clustering') as fresh:
                same.cluster_cases()
                fresh.assert_not_called()
            changed = QuantumTestClusterer(sample_cases(), str(root / 'cache'), str(path),
                use_mixstate=True, classification_method='type_clustering', type_k=5)
            with patch.object(changed, '_cluster_type_clustering', return_value=[]) as fresh:
                changed.cluster_cases()
                fresh.assert_called_once()

    def test_budget_rounding_and_partial_auc(self):
        metrics = curve_metrics([0, 1, 1, 2, 2, 2, 2, 2, 2, 2], 2)
        self.assertEqual(metrics['failures_30'], 1)
        self.assertEqual(metrics['recall_30'], .5)
        self.assertEqual(metrics['normalized_auc_30'], .25)
        self.assertEqual(curve_metrics([1, 1, 1], 1)['failures_30'], 0)
        self.assertTrue(np.isnan(curve_metrics([0]*10, 0)['recall_30']))
        with self.assertRaises(ValueError):
            curve_metrics([0, 2], 2)

    def test_manifest_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'manifest.json'
            require_manifest(path, {'k_values': [1, 3]})
            require_manifest(path, {'k_values': [1, 3]})
            with self.assertRaisesRegex(ValueError, 'changed'):
                require_manifest(path, {'k_values': [1, 5]})

    def test_analysis_exports_curves_tables_and_figures(self):
        # Synthetic fixtures verify exports; they never enter the real result directory.
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'manifest.json').write_text(json.dumps(dict(k_values=[1, 3], repeats=2, seed=10)))
            group_dir = output / 'raw' / 'fixture_1'
            group_dir.mkdir(parents=True)
            for k in [1, 3]:
                value = dict(subject=dict(program='fixture', group='fixture_1'), k=k,
                    actual_clusters=2*k+2, n_cases=10, total_failures=2, budget_30=3,
                    preprocessing_seconds=.1, online_seconds=.2,
                    runs=[dict(seed=10+i, run_id=i,
                               cumulative_failures=[0, 1, 1, 2, 2, 2, 2, 2, 2, 2])
                          for i in range(2)])
                (group_dir / f'k_{k}.json').write_text(json.dumps(value))
            with contextlib.redirect_stdout(io.StringIO()):
                analyze(output)
            for path in ['tables/run_level.csv', 'tables/cumulative_curves.csv',
                         'tables/paired_vs_k3.csv', 'tables/program_summary.csv',
                         'plots/fixture_1_budget30.png', 'plots/fixture_1_curves_full.pdf', 'RESULTS.md']:
                self.assertTrue((output / path).is_file(), path)


if __name__ == '__main__':
    unittest.main()
