"""Shared paths, distribution sizes and validation for distribution experiments."""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import testcase_io
from distribution_experiments.distribution_calculator import DISTRIBUTIONS

COLUMNS = ['input', 'input_state', 'attributes', 'result', 'p']
DEFAULT_VERSIONS = '1,2,3,4,5'
KS_SOURCE = ROOT / 'oracle/OpenQASM/QAOA_12_oracle.xlsx'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def parse_versions(value):
    values = []
    for item in value.split(','):
        if '-' in item:
            first, last = map(int, item.split('-'))
            values.extend(range(first, last + 1))
        else:
            values.append(int(item))
    if not values or min(values) < 1 or max(values) > 5:
        raise ValueError('Fault versions must be integers from 1 through 5.')
    return list(dict.fromkeys(values))


def distribution_sizes(n_qubits):
    """Keep the historical floor-and-remainder-to-classical sampling rule."""
    total = 2 ** n_qubits
    sizes = {}
    for distribution in DISTRIBUTIONS:
        counts = {kind: int(total * distribution[kind + '_ratio'])
                  for kind in ('classical', 'superposition', 'mixed')}
        target = 'classical' if distribution['classical_ratio'] else 'mixed'
        counts[target] += total - sum(counts.values())
        sizes[distribution['name']] = counts
    return sizes


def category(attributes):
    kind = attributes.get('type')
    if kind == 'entangled':
        return 'superposition'
    if kind not in ('classical', 'superposition', 'mixed'):
        raise ValueError('Unknown testcase type: ' + str(kind))
    return kind


def load_pool(original, supplement=None):
    """Never allow duplicate names to silently replace different quantum states."""
    cases = {}
    paths = [original] + ([supplement] if supplement is not None else [])
    for path in paths:
        for name, state, attributes in testcase_io.load_test_cases(str(path), compact_pure_density=True):
            if name in cases:
                raise ValueError('Duplicate testcase name: ' + name)
            cases[name] = {'state': state, 'attributes': attributes}
    return cases


def parse_result(value):
    text = str(value).strip().lower()
    if text not in ('true', 'false'):
        raise ValueError('Invalid oracle Boolean: ' + str(value))
    return text == 'true'


def validate_oracle(frame):
    if not set(COLUMNS).issubset(frame.columns):
        raise ValueError('Oracle must contain columns: ' + ', '.join(COLUMNS))
    if frame['input'].isna().any() or frame['input'].duplicated().any():
        raise ValueError('Oracle contains empty or duplicate testcase names.')
    frame = frame[COLUMNS].copy()
    frame['result'] = frame['result'].map(parse_result)
    import pandas as pd
    probabilities = pd.to_numeric(frame['p'], errors='raise')
    if not probabilities.between(0, 1).all():
        raise ValueError('Oracle p-values must be finite and between zero and one.')
    frame['p'] = probabilities
    return frame


def atomic_cases(cases, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.pkl.gz')
    testcase_io.save_test_cases_binary(cases, str(temporary), compress=True)
    os.replace(temporary, path)
