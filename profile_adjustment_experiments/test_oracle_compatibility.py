"""Regression tests for legacy and current mixed-state oracle metadata."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from QHDRT import QHDRT, parse_oracle_attributes, validate_mixed_oracle
from testcase_io import MIXED_STATE_REPRESENTATION


def attributes(version=None):
    result = {'type': 'mixed', 'is_mixed': True, 'amplitude_damping': .05}
    if version is not None:
        result['state_representation'] = version
    return result


def oracle(raw, name='mixed_case'):
    return pd.DataFrame([[name, '', raw, 'False', '0.01']],
                        columns=['input', 'input_state', 'attributes', 'result', 'p'])


def cases(attrs, name='mixed_case'):
    return {name: {'attributes': attrs, 'state': None}}


class OracleCompatibilityTests(unittest.TestCase):
    def test_accepts_json_and_python_dict_literals(self):
        for attrs in [attributes(), attributes(MIXED_STATE_REPRESENTATION)]:
            for raw in [json.dumps(attrs), repr(attrs)]:
                with self.subTest(raw=raw):
                    self.assertEqual(parse_oracle_attributes(raw), attrs)
                    validate_mixed_oracle(oracle(raw), cases(attrs))

    def test_constructor_accepts_matching_old_and_new_metadata(self):
        for attrs in [attributes(), attributes(MIXED_STATE_REPRESENTATION)]:
            with self.subTest(attrs=attrs), \
                    patch('QHDRT.pd.read_excel', return_value=oracle(repr(attrs))), \
                    patch.object(QHDRT, '_load_test_cases', return_value={'test_cases': cases(attrs)}), \
                    patch('QHDRT.QuantumTestClusterer'), \
                    patch.object(QHDRT, 'perform_clustering'), \
                    patch.object(QHDRT, '_prepare_online_templates'):
                alg = QHDRT(-1, 8, 8, 'example_1', 'example', 'QHDRT', 2, use_mixstate=True)
                self.assertEqual(alg.upper_bound, 1)

    def test_rejects_version_mismatch_in_both_directions(self):
        old, new = attributes(), attributes(MIXED_STATE_REPRESENTATION)
        for expected, actual in [(old, new), (new, old)]:
            with self.assertRaisesRegex(ValueError, 'mixed_case'):
                validate_mixed_oracle(oracle(json.dumps(actual)), cases(expected))

    def test_rejects_missing_wrong_type_and_malformed_rows(self):
        invalid_frames = [
            oracle(repr(attributes()), 'other_case'),
            oracle("{'type': 'classical'}"),
            oracle('not a dictionary'),
        ]
        for frame in invalid_frames:
            with self.subTest(frame=frame.to_dict()), self.assertRaises(ValueError):
                validate_mixed_oracle(frame, cases(attributes()))

    def test_duplicate_names_follow_existing_last_row_wins_policy(self):
        old = oracle(repr(attributes()))
        new = oracle(json.dumps(attributes(MIXED_STATE_REPRESENTATION)))
        validate_mixed_oracle(pd.concat([new, old], ignore_index=True), cases(attributes()))
        with self.assertRaises(ValueError):
            validate_mixed_oracle(pd.concat([old, new], ignore_index=True), cases(attributes()))

    def test_ignores_oracle_rows_outside_loaded_mixed_pool(self):
        extra = oracle(json.dumps(attributes(MIXED_STATE_REPRESENTATION)), 'unused_mixed')
        validate_mixed_oracle(pd.concat([oracle(repr(attributes())), extra]), cases(attributes()))
        validate_mixed_oracle(extra, {})

    def test_rejects_non_dict_and_executable_input(self):
        for raw in ['[]', 'null', "__import__('os').getcwd()"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_oracle_attributes(raw)

    def test_actual_qft_iqft_and_qaoa_oracle_formats(self):
        if not all((ROOT / 'oracle/OpenQASM' / f'{program}_oracle.xlsx').exists()
                   for program in ('qft_8', 'IQFT', 'QAOA_12')):
            self.skipTest('Fresh checkout: generate oracle workbooks before this integration check.')
        for program, expected_version in [('qft_8', MIXED_STATE_REPRESENTATION), ('IQFT', MIXED_STATE_REPRESENTATION),
                                           ('QAOA_12', MIXED_STATE_REPRESENTATION)]:
            frame = pd.read_excel(ROOT / 'oracle' / 'OpenQASM' / f'{program}_oracle.xlsx',
                                  sheet_name=f'{program}_1', dtype=str)
            mixed = [parse_oracle_attributes(raw) for raw in frame.iloc[:, 2].dropna()]
            mixed = [attr for attr in mixed if attr.get('type') == 'mixed']
            with self.subTest(program=program):
                self.assertTrue(mixed)
                self.assertTrue(all(attr.get('state_representation') == expected_version for attr in mixed))


if __name__ == '__main__':
    unittest.main()
