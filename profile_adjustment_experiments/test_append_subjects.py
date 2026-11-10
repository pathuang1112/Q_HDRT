"""Append/resume regression tests; fixtures stay inside temporary directories."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_sensitivity import (LEGACY_RUNNER_SHA256, atomic_json, digest,
                             merge_manifests, progress_state, register_manifest)


def manifest(*groups, runner=LEGACY_RUNNER_SHA256):
    return dict(subjects=[dict(program='QAOA_12', group=g, input_qubits=12,
                               output_qubits=12) for g in groups],
                settings={'use_mixstate': True}, epsilon=[.05], delta=[.05],
                repeats=20, seed=20261009, input_sha256={},
                algorithm_sha256='same-algorithm', runner_sha256=runner)


class AppendTests(unittest.TestCase):
    def test_append_and_subset_resume_preserve_subjects(self):
        old = manifest('QAOA_12_1', 'QAOA_12_3')
        untouched = copy.deepcopy(old)
        requested = manifest('QAOA_12_4', 'QAOA_12_5', runner='current-runner')
        merged = merge_manifests(old, requested)
        self.assertEqual([s['group'] for s in merged['subjects']],
                         ['QAOA_12_1', 'QAOA_12_3', 'QAOA_12_4', 'QAOA_12_5'])
        self.assertEqual(old, untouched)
        self.assertEqual(merged['runner_sha256_history'], [LEGACY_RUNNER_SHA256, 'current-runner'])
        resumed = merge_manifests(merged, manifest('QAOA_12_4', runner='current-runner'))
        self.assertEqual(resumed, merged)

    def test_settings_algorithm_and_unknown_runner_changes_are_rejected(self):
        old = manifest('QAOA_12_1')
        for field, changed in [('settings', {'use_mixstate': False}), ('epsilon', [.1]),
                               ('delta', [.1]), ('repeats', 30), ('seed', 1),
                               ('algorithm_sha256', 'different-algorithm')]:
            requested = manifest('QAOA_12_4', runner='current-runner')
            requested[field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                merge_manifests(old, requested)
        old['runner_sha256'] = 'unknown-old-runner'
        with self.assertRaisesRegex(ValueError, 'runner_sha256'):
            merge_manifests(old, manifest('QAOA_12_4', runner='current-runner'))

    def test_conflicting_subject_or_shared_input_is_rejected(self):
        old = manifest('QAOA_12_1')
        requested = manifest('QAOA_12_1')
        requested['subjects'][0]['input_qubits'] = 8
        with self.assertRaisesRegex(ValueError, 'subjects'):
            merge_manifests(old, requested)
        old['input_sha256'] = {'shared-oracle': 'original'}
        requested = manifest('QAOA_12_4')
        requested['input_sha256'] = {'shared-oracle': 'changed'}
        with self.assertRaisesRegex(ValueError, 'input_sha256'):
            merge_manifests(old, requested)

    def test_registration_backs_up_manifest_and_preserves_raw_results(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            output = Path(temp)
            old = manifest('QAOA_12_1')
            atomic_json(output / 'manifest.json', old)
            raw = output / 'raw' / 'QAOA_12_1' / 'eps_0.05_delta_0.05.json'
            atomic_json(raw, {'fixture': 'existing data must remain unchanged'})
            before = raw.read_bytes()
            merged = register_manifest(output, manifest('QAOA_12_4', runner='current-runner'))
            self.assertEqual(raw.read_bytes(), before)
            backups = list((output / 'manifest_history').glob('*.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text(encoding='utf-8')), old)
            self.assertEqual(progress_state(output, merged),
                             dict(completed_pairs=1, expected_pairs=2, status='partial'))
            register_manifest(output, manifest('QAOA_12_4', runner='current-runner'))
            self.assertEqual(len(list((output / 'manifest_history').glob('*.json'))), 1)
            atomic_json(output / 'raw' / 'QAOA_12_4' / raw.name, {'fixture': 'new result'})
            self.assertEqual(progress_state(output, merged),
                             dict(completed_pairs=2, expected_pairs=2, status='complete'))

    def test_old_inputs_are_verified_when_appending_another_program(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            output = Path(temp)
            source = output / 'source.txt'
            source.write_text('original', encoding='utf-8')
            old = manifest('QAOA_12_1')
            old['input_sha256'] = {str(source): digest(source)}
            atomic_json(output / 'manifest.json', old)
            original_manifest = (output / 'manifest.json').read_bytes()
            source.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                register_manifest(output, manifest('QAOA_12_4', runner='current-runner'))
            self.assertEqual((output / 'manifest.json').read_bytes(), original_manifest)
            self.assertFalse((output / 'manifest_history').exists())

    def test_workbook_changes_require_matching_saved_results_and_pvalues(self):
        import pandas as pd
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            workspace = Path(temp).resolve()
            output = workspace / 'output'
            source = workspace / 'oracle' / 'OpenQASM' / 'QAOA_12_oracle.xlsx'
            source.parent.mkdir(parents=True)

            def write_workbook(first_result='True', pvalue='0.8', other_value='first'):
                frame = pd.DataFrame([
                    ['case1', '', '{}', first_result, pvalue],
                    ['case2', '', '{}', 'False', '0.01']],
                    columns=['input', 'input_state', 'attributes', 'result', 'p'])
                with pd.ExcelWriter(source, engine='openpyxl') as book:
                    frame.to_excel(book, sheet_name='QAOA_12_1', index=False)
                    pd.DataFrame({'extra': [other_value]}).to_excel(
                        book, sheet_name='QAOA_12_4', index=False)

            write_workbook()
            old = manifest('QAOA_12_1')
            old['input_sha256'] = {str(source): digest(source)}
            atomic_json(output / 'manifest.json', old)
            raw_path = output / 'raw' / 'QAOA_12_1' / 'eps_0.05_delta_0.05.json'
            atomic_json(raw_path, dict(n_cases=2, total_failures=1, runs=[dict(
                executed_order=['case1', 'case2'], cumulative_failures=[0, 1])]))
            log = raw_path.parent / 'legacy' / raw_path.stem / 'run.txt'
            log.parent.mkdir(parents=True)
            log.write_text('case1 case2\n0 1\n0.8 0.01\n', encoding='utf-8')
            before_raw = raw_path.read_bytes()
            write_workbook(other_value='other sheet changed')
            requested = manifest('QAOA_12_4', runner='current-runner')
            requested['input_sha256'] = {str(source): digest(source)}
            with patch('run_sensitivity.ROOT', workspace):
                merged = register_manifest(output, requested)
                self.assertEqual(merged['oracle_revalidations'][0]['verified_groups'], ['QAOA_12_1'])
                self.assertEqual(merged['input_sha256_history'][str(source)],
                                 [old['input_sha256'][str(source)], digest(source)])
                self.assertEqual(raw_path.read_bytes(), before_raw)
                before_manifest = (output / 'manifest.json').read_bytes()
                for kwargs, message in [({'first_result': 'False'}, 'Oracle result changed'),
                                        ({'pvalue': '0.3'}, 'Oracle p-value changed')]:
                    write_workbook(**kwargs)
                    requested['input_sha256'] = {str(source): digest(source)}
                    with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                        register_manifest(output, requested)
                    self.assertEqual((output / 'manifest.json').read_bytes(), before_manifest)


if __name__ == '__main__':
    unittest.main()
