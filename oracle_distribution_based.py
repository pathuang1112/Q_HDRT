"""Generate main-experiment oracle labels from sampled output distributions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from scipy.stats import chi2_contingency, random_table

from experiment_config import ROOT, SUBJECTS, get_fault_versions
import testcase_io

PROGRAMS = ('adder_n7', 'rand_cliff_8_AG', 'QAOA_12')
COLUMNS = ['input', 'input_state', 'attributes', 'result', 'p']
METHOD = 'pearson_output_homogeneity_v1'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def select_versions(value, program):
    available = list(get_fault_versions(program))
    if value == 'all':
        return available
    versions = sorted(set(int(part.strip()) for part in value.split(',')))
    if not versions or not set(versions).issubset(available):
        raise ValueError('Fault versions must be integers from 1 through 5.')
    return versions


def pearson_test(correct, faulty, resamples=9999, seed=0, batch_size=256):
    """Test categorical homogeneity using independent samples with aligned bins."""
    left, right = np.asarray(correct), np.asarray(faulty)
    if left.ndim != 1 or right.shape != left.shape or left.size == 0:
        raise ValueError('Counts must be nonempty vectors with equal shapes.')
    for values in (left, right):
        if (not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values != np.floor(values))):
            raise ValueError('Counts must be finite nonnegative integers.')
    if resamples < 1 or batch_size < 1:
        raise ValueError('Resamples and batch size must be positive.')
    table = np.stack((left, right)).astype(np.int64)
    if np.any(table.sum(axis=1) == 0):
        raise ValueError('Both distributions must have positive shot totals.')
    # Bins absent from both samples contribute no information to the test.
    table = table[:, table.sum(axis=0) > 0]
    result = dict(active_categories=int(table.shape[1]), mc_resamples=0,
                  mc_standard_error=None, min_expected_count=None)
    if table.shape[1] == 1:
        return dict(result, p=1.0, statistic=0.0, degrees_of_freedom=0,
                    p_method='single_category')
    statistic, p, dof, expected = chi2_contingency(table, correction=False)
    result.update(p=float(p), statistic=float(statistic), degrees_of_freedom=int(dof),
                  p_method='pearson_asymptotic', min_expected_count=float(expected.min()))
    if np.all(expected >= 5):
        return result
    # Sparse tables retain Pearson's statistic but use conditional calibration.
    rng = np.random.default_rng(seed)
    extreme = 0
    tolerance = 1e-12 * max(1.0, abs(float(statistic)))
    for start in range(0, resamples, batch_size):
        simulated = random_table.rvs(table.sum(axis=1), table.sum(axis=0),
                                     size=min(batch_size, resamples - start),
                                     method='patefield', random_state=rng)
        null_statistics = np.sum((simulated - expected) ** 2 / expected, axis=(1, 2))
        extreme += int(np.count_nonzero(null_statistics >= statistic - tolerance))
    calibrated = (extreme + 1) / (resamples + 1)
    result.update(p=float(calibrated), p_method='pearson_conditional_monte_carlo',
                  mc_resamples=resamples,
                  mc_standard_error=float(np.sqrt(calibrated * (1 - calibrated) / (resamples + 1))))
    return result


def load_circuit(program, version=None):
    suffix = '' if version is None else f'_b{version}'
    path = ROOT / 'programs_OpenQASM' / program / f'{program}{suffix}.qasm'
    circuit = QuantumCircuit.from_qasm_file(str(path))
    if circuit.num_qubits != SUBJECTS[program]['qubits']:
        raise ValueError(f'Circuit qubit count does not match the subject: {path}')
    for instruction, _, _ in circuit.data:
        if instruction.name == 'barrier':
            continue
        if instruction.condition is not None or instruction.name in {'measure', 'reset', 'initialize'}:
            raise ValueError(f'Expected a unitary subject circuit: {path}')
        instruction.inverse()
    return circuit, path


def output_probabilities(state, circuit):
    factors = testcase_io.density_factors(state)
    if factors is not None:
        factors = np.asarray(factors, dtype=complex)
        if factors.ndim != 2 or factors.shape[0] != 2 ** circuit.num_qubits:
            raise ValueError('Input-state dimensions do not match the circuit.')
        probabilities = np.zeros(factors.shape[0])
        # Evolve each low-rank branch exactly without constructing a dense matrix.
        for column in factors.T:
            weight = float(np.vdot(column, column).real)
            if weight > 0:
                evolved = Statevector(column / np.sqrt(weight)).evolve(circuit)
                probabilities += weight * np.abs(evolved.data) ** 2
    else:
        density = state if isinstance(state, DensityMatrix) else DensityMatrix(state)
        probabilities = np.asarray(density.evolve(circuit).probabilities(), dtype=float)
    if (probabilities.size != 2 ** circuit.num_qubits or not np.isfinite(probabilities).all()
            or probabilities.min() < -1e-10 or not np.isclose(probabilities.sum(), 1, atol=1e-6)):
        raise ValueError('Output probabilities are not a normalized distribution.')
    probabilities = np.clip(probabilities, 0, None)
    return probabilities / probabilities.sum()


def write_workbook(path, sheet, rows, columns):
    temporary = path.with_name(path.stem + '.tmp.xlsx')
    try:
        options = {}
        if path.exists():
            shutil.copyfile(path, temporary)
            options = dict(mode='a', if_sheet_exists='replace')
        with pd.ExcelWriter(temporary, engine='openpyxl', **options) as writer:
            pd.DataFrame(rows, columns=columns).to_excel(writer, sheet_name=sheet, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint(path):
    if not path.exists():
        return {}
    rows = {}
    with path.open('rb+') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if stream.read(1):
                    raise ValueError(f'Corrupted checkpoint before its last row: {path}')
                # Discard only a torn final write so appending remains safe.
                stream.truncate(start)
                break
            rows[row['input']] = row
            if not line.endswith(b'\n'):
                stream.write(b'\n')
    return rows


def run(args):
    if args.program not in PROGRAMS:
        raise ValueError('Select a supported distribution-based subject.')
    if (not 0 < args.alpha < 1 or args.resamples < 1 or args.batch_size < 1
            or args.progress_interval <= 0 or args.seed < 0
            or (args.limit is not None and args.limit < 1)):
        raise ValueError('Invalid alpha, resampling, progress, or limit setting.')
    versions = select_versions(args.fault_versions, args.program)
    qubits = SUBJECTS[args.program]['qubits']
    shots = args.shots if args.shots is not None else 100 * 2 ** qubits
    if shots < 1:
        raise ValueError('Shots must be positive.')
    testfile = args.testcases or ROOT / 'TestCases' / f'quantum_test_cases_{qubits}qubits.json'
    testfile = Path(testcase_io.resolve_testcase_file(str(testfile)))
    cases = testcase_io.load_test_cases(str(testfile))
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases or len({case[0] for case in cases}) != len(cases):
        raise ValueError('Test cases must be nonempty and have unique names.')
    output = args.output_dir.resolve()
    if args.limit is not None:
        output = output / 'smoke' / f'{args.program}_limit_{args.limit}'
    output.mkdir(parents=True, exist_ok=True)
    state_dir = output / 'distribution_state' / args.program
    state_dir.mkdir(parents=True, exist_ok=True)
    correct, correct_path = load_circuit(args.program)
    base_signature = dict(method=METHOD, program=args.program, shots=shots, alpha=args.alpha,
                          resamples=args.resamples, batch_size=args.batch_size, seed=args.seed,
                          limit=args.limit, case_count=len(cases), testcase_sha256=file_digest(testfile),
                          correct_sha256=file_digest(correct_path), runner_sha256=file_digest(__file__),
                          state_io_sha256=file_digest(testcase_io.__file__))
    workbook = output / f'{args.program}_oracle.xlsx'
    details = output / f'{args.program}_distribution_details.xlsx'
    for version in versions:
        faulty, faulty_path = load_circuit(args.program, version)
        group = f'{args.program}_{version}'
        checkpoint = state_dir / f'{group}.jsonl'
        metadata = state_dir / f'{group}.json'
        signature = dict(base_signature, fault_version=version, faulty_sha256=file_digest(faulty_path))
        if metadata.exists():
            previous = json.loads(metadata.read_text(encoding='utf-8'))
            if previous['signature'] != signature and not args.rerun:
                raise ValueError(f'Settings or inputs changed for {group}; use --rerun or a new output directory.')
        elif checkpoint.exists() and not args.rerun:
            raise ValueError(f'Checkpoint metadata missing for {group}; use --rerun or a new output directory.')
        elif workbook.exists() and not args.rerun:
            sheets = pd.ExcelFile(workbook, engine='openpyxl')
            try:
                if group in sheets.sheet_names:
                    raise ValueError(f'Existing labels for {group} have no matching distribution metadata; '
                                     'use --rerun to replace this sheet explicitly.')
            finally:
                sheets.close()
        if args.rerun and checkpoint.exists():
            checkpoint.unlink()
        atomic_json(metadata, dict(signature=signature, oracle=str(workbook),
                                   sampling='Independent multinomial samples of exact output probabilities; '
                                            'Qiskit little-endian computational-basis ordering.'))
        saved = read_checkpoint(checkpoint)
        print(f'{group}: {len(saved)}/{len(cases)} cases restored; shots={shots}, alpha={args.alpha}', flush=True)
        started = last_report = time.monotonic()
        with checkpoint.open('a', encoding='utf-8') as stream:
            for index, (name, state, attributes) in enumerate(cases):
                if name not in saved:
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed, qubits, version, index]))
                    expected = rng.multinomial(shots, output_probabilities(state, correct))
                    observed = rng.multinomial(shots, output_probabilities(state, faulty))
                    statistics = pearson_test(expected, observed, args.resamples,
                                              seed=int(rng.integers(0, 2 ** 32)), batch_size=args.batch_size)
                    row = dict(input=name, input_state='', attributes=json.dumps(attributes, default=json_default),
                               result=bool(statistics['p'] >= args.alpha), **statistics,
                               shots=shots, alpha=args.alpha, fault_version=version)
                    stream.write(json.dumps(row, default=json_default) + '\n')
                    stream.flush()
                    saved[name] = row
                now = time.monotonic()
                if now - last_report >= args.progress_interval or index + 1 == len(cases):
                    print(f'{group}: completed {index + 1}/{len(cases)} '
                          f'({100 * (index + 1) / len(cases):.1f}%), elapsed={now - started:.1f}s', flush=True)
                    last_report = now
        rows = [saved[name] for name, _, _ in cases]
        write_workbook(workbook, group, rows, COLUMNS)
        write_workbook(details, group, rows, list(rows[0]))
        print(f'{group}: saved {workbook}', flush=True)
    return workbook


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--program', choices=PROGRAMS, required=True)
    result.add_argument('--fault-versions', default='all')
    result.add_argument('--testcases', type=Path)
    result.add_argument('--output-dir', type=Path, default=ROOT / 'oracle/OpenQASM')
    result.add_argument('--shots', type=int, help='Default: 100 times the output-space size (2**qubits).')
    result.add_argument('--alpha', type=float, default=0.05)
    result.add_argument('--resamples', type=int, default=9999, help='Conditional resamples for sparse Pearson tables.')
    result.add_argument('--batch-size', type=int, default=256)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--progress-interval', type=float, default=30)
    result.add_argument('--limit', type=int, help='Smoke-test limit; writes to an isolated smoke subdirectory.')
    result.add_argument('--rerun', action='store_true', help='Explicitly recompute selected sheets and replace their labels.')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
