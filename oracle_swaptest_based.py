import numpy as np
import argparse
import hashlib
import re
import os
import pandas as pd
import importlib
import matplotlib.pyplot as plt
from programs_ST import IQFT
from programs_OpenQASM import adder_n7, qft_8, pe_9, rand_cliff_8_AG, QAOA_12
from qiskit.extensions import Initialize
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.providers.aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error
import json
import ast
import testcase_io
from pathlib import Path

from qiskit import (

    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)


def dec2bin(n, dec2bin_param):
    a = 1
    list = []
    while a > 0:
        a, b = divmod(n, 2)
        list.append(str(b))
        n = a
    s = ""
    for i in range(len(list) - 1, -1, -1):
        s += str(list[i])
    s = s.zfill(dec2bin_param)
    return s


def wrong_output(i, right_output):
    set_output = set(right_output)
    if i not in set_output:
        return True
    return False


def create_correct_circuit(program_name, count_times):
    from experiment_config import SUBJECTS
    if program_name not in SUBJECTS:
        raise ValueError('Unknown subject: ' + program_name)
    module = importlib.import_module('programs_OpenQASM.' + program_name)
    return getattr(module, program_name)(count_times)


def create_error_circuit(program_name, fault_version, count_times):
    from experiment_config import get_fault_versions
    if fault_version not in get_fault_versions(program_name):
        raise ValueError('Fault version must be 1 through 5')
    module = importlib.import_module('programs_OpenQASM.' + program_name)
    return getattr(module, program_name + '_M' + str(fault_version))(count_times)

def load_test_cases(filename="test_cases.json", **kwargs):
    return testcase_io.load_test_cases(filename, **kwargs)


def parse_qubit_index(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return int(ast.literal_eval(value))
    return int(value)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def make_oracle_row(name, attributes, result, p):
    return {
        'input': name,
        'input_state': '',
        'attributes': json.dumps(attributes, ensure_ascii=False, default=json_default),
        'result': bool(result),
        'p': float(p),
    }


def checkpoint_path(program_name, group_name, suffix=None):
    path = Path('./oracle/OpenQASM/checkpoints')
    path.mkdir(parents=True, exist_ok=True)
    suffix_text = f'_{suffix}' if suffix else ''
    return path / f'{program_name}_{group_name}{suffix_text}_checkpoint.jsonl'


def error_checkpoint_path(program_name, group_name):
    path = Path('./oracle/OpenQASM/checkpoints')
    path.mkdir(parents=True, exist_ok=True)
    return path / f'{program_name}_{group_name}_last_error.json'


def load_checkpoint_rows(path):
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Ignoring corrupted checkpoint row: {path}, line {line_no}")
    return rows


def append_checkpoint_row(path, row):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + '\n')


def save_error_checkpoint(path, num, name, exc):
    payload = {
        'num': int(num),
        'input': name,
        'error_type': type(exc).__name__,
        'error': str(exc),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def write_oracle_sheet(output_path, sheet_name, rows):
    df = pd.DataFrame(rows, columns=['input', 'input_state', 'attributes', 'result', 'p'])
    if os.path.exists(output_path):
        writer = pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace')
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.close()
    else:
        df.to_excel(output_path, sheet_name=sheet_name, index=False)


def sheet_exists(output_path, sheet_name):
    if not os.path.exists(output_path):
        return False
    return sheet_name in pd.ExcelFile(output_path).sheet_names


def sheet_uses_current_mixed_states(output_path, sheet_name):
    if not sheet_exists(output_path, sheet_name):
        return False
    sheet = pd.read_excel(output_path, sheet_name=sheet_name, dtype=str)
    found_mixed = False
    for _, row in sheet.iterrows():
        try:
            attributes = json.loads(row['attributes'])
        except (KeyError, TypeError, json.JSONDecodeError):
            return False
        if attributes.get('type') == 'mixed':
            found_mixed = True
            if (attributes.get('state_representation') !=
                    testcase_io.MIXED_STATE_REPRESENTATION):
                return False
    return found_mixed


def mixed_dataset_fingerprint(test_cases):
    digest = hashlib.sha256()
    for name, state, attributes in test_cases:
        digest.update(name.encode('utf-8'))
        digest.update(json.dumps(
            attributes, sort_keys=True, ensure_ascii=False,
            default=json_default
        ).encode('utf-8'))
        factors = testcase_io.density_factors(state)
        if factors is not None:
            digest.update(np.ascontiguousarray(factors).view(np.uint8).tobytes())
    return digest.hexdigest()[:12]


def non_mixed_rows_from_sheet(output_path, sheet_name):
    sheet = pd.read_excel(output_path, sheet_name=sheet_name, dtype=str)
    rows = []
    for row in sheet.to_dict('records'):
        try:
            attributes = json.loads(row.get('attributes', '{}'))
        except (TypeError, json.JSONDecodeError):
            continue
        if attributes.get('type') != 'mixed':
            rows.append(row)
    return rows


def prepare_initial_state(test_case):
    name, state, attributes = test_case
    n_qubits = attributes["num_qubits"]
    qc = QuantumCircuit(n_qubits)


    if attributes["type"] == "classical":

        for qubit, bit in enumerate(attributes["binary_state"]):
            if bit == '1':
                qc.x(qubit)

    elif attributes["type"] == "superposition":

        if attributes["subtype"].startswith("uniform"):

            qc.h(attributes["superposed_qubits"])
        elif "ry" in attributes["subtype"]:

            for q_str, theta in attributes["rotation_angles"].items():
                q = parse_qubit_index(q_str)
                qc.ry(theta, q)

    elif attributes["type"] == "entangled":

        if attributes["subtype"] == "bell":
            qc.h(attributes["entangled_qubits"][0])
            qc.cx(*attributes["entangled_qubits"])
        elif attributes["subtype"] == "ghz":
            qc.h(0)
            for i in range(1, n_qubits):
                qc.cx(0, i)
        elif attributes["subtype"] == "layered":

            for pair in attributes["entangled_qubits"]:
                qc.h(pair[0])
                qc.cx(*pair)

    elif attributes["type"] == "mixed":

        noise_model = construct_noise_model(attributes)
        return {"qc": qc, "noise_model": noise_model}

    return {"qc": qc}


def low_rank_swap_zero_probability(state, correct_circuit, error_circuit):
    correct_factors = []
    error_factors = []
    for column in state.factors.T:
        weight_sqrt = np.linalg.norm(column)
        if weight_sqrt <= 1e-15:
            continue
        normalized = Statevector(column / weight_sqrt)
        correct_factors.append(
            normalized.evolve(correct_circuit).data * weight_sqrt
        )
        error_factors.append(
            normalized.evolve(error_circuit).data * weight_sqrt
        )

    correct_factors = np.column_stack(correct_factors)
    error_factors = np.column_stack(error_factors)
    hilbert_schmidt_overlap = float(
        np.linalg.norm(correct_factors.conj().T @ error_factors, 'fro') ** 2
    )
    hilbert_schmidt_overlap = min(1.0, max(0.0, hilbert_schmidt_overlap))
    return (1.0 + hilbert_schmidt_overlap) / 2.0


def construct_noise_model(attributes):
    noise_model = NoiseModel()
    gamma = attributes.get("amplitude_damping", 0)
    p = attributes.get("phase_damping", 0)

    noise_model.add_all_qubit_quantum_error(
        amplitude_damping_error(gamma),
        ['id', 'h', 'ry']
    )
    noise_model.add_all_qubit_quantum_error(
        phase_damping_error(p),
        ['id', 'h', 'ry']
    )
    return noise_model


def get_fault_versions(program_name):
    from experiment_config import get_fault_versions as versions
    return versions(program_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate OpenQASM program\'s oracle')
    from experiment_config import SUBJECTS
    parser.add_argument('--program', choices=list(SUBJECTS), default='QAOA_12')
    parser.add_argument('--shots', type=int, default=20,
                        help='SWAP shots, independent of the testing strategy 50 repeated runs')
    parser.add_argument(
        '--mixed_only', '--mixed-only', action='store_true',
        help='retain existing pure states oracle, Recompute only the current test suite\'s Mixed test cases',
    )
    args = parser.parse_args()

    input_qbit_count = 12
    output_qbit_count = 12
    output_qubits = np.array(range(12))

    qubit_count = 12
    clbit_count = 12
    program_name = 'QAOA_12'

    program_name = args.program
    input_qbit_count = output_qbit_count = qubit_count = clbit_count = SUBJECTS[program_name]['qubits']
    output_qubits = np.arange(output_qbit_count)
    shotTimes = args.shots
    if shotTimes < 1:
        parser.error('--shots must be positive')


    excluded_types = (
        {'classical', 'superposition', 'entangled'}
        if args.mixed_only else None
    )
    loaded_cases = load_test_cases(
        f"./TestCases/quantum_test_cases_{input_qbit_count}qubits.json",
        exclude_types=excluded_types,
    )
    if args.mixed_only and (not loaded_cases or any(
            attributes.get('state_representation') !=
            testcase_io.MIXED_STATE_REPRESENTATION
            for _, _, attributes in loaded_cases)):
        raise ValueError(
            "Testcase file does not yet contain the current GAD+PD Mixed Data. First use "
            "TestcaseGenerator.py --replace_mixed Update test cases. "
        )
    mode_text = 'Mixed' if args.mixed_only else 'all'
    print(f"\nLoaded {len(loaded_cases)} test cases ({mode_text})")
    mixed_fingerprint = (
        mixed_dataset_fingerprint(loaded_cases) if args.mixed_only else None
    )
    output_excel_path = './oracle/OpenQASM/' + program_name + '_oracle.xlsx'
    if args.mixed_only:
        missing_sheets = [
            program_name + '_' + str(version)
            for version in get_fault_versions(program_name)
            if not sheet_exists(
                output_excel_path, program_name + '_' + str(version)
            )
        ]
        if missing_sheets:
            raise FileNotFoundError(
                f"{output_excel_path} Missing sheet: {', '.join(missing_sheets)}. "
                "mixed-only mode will not create a workbook missing pure-state oracle sheet. "
            )

    for fault_version in get_fault_versions(program_name):
        group_name = program_name + '_' + str(fault_version)
        checkpoint_suffix = (
            f'mixed_{mixed_fingerprint}' if args.mixed_only else None
        )
        current_checkpoint_path = checkpoint_path(
            program_name, group_name, checkpoint_suffix
        )
        current_error_path = error_checkpoint_path(program_name, group_name)


        checkpoint_rows = load_checkpoint_rows(current_checkpoint_path)
        checkpoint_rows = [
            row for row in checkpoint_rows
            if (json.loads(row.get('attributes', '{}')).get('type') != 'mixed' or
                json.loads(row.get('attributes', '{}')).get('state_representation') ==
                testcase_io.MIXED_STATE_REPRESENTATION)
        ]
        if args.mixed_only:
            if not sheet_exists(output_excel_path, group_name):
                raise FileNotFoundError(
                    f"{output_excel_path} does not contain sheet {group_name}, "
                    "Cannot locate in mixed-only mode retains existing pure states oracle. "
                )

            pure_rows = non_mixed_rows_from_sheet(
                output_excel_path, group_name
            )
            mixed_case_names = {case[0] for case in loaded_cases}
            checkpoint_by_name = {
                row['input']: row
                for row in checkpoint_rows
                if row.get('input') in mixed_case_names
            }

            resumed_mixed_rows = [
                checkpoint_by_name[name]
                for name, _, _ in loaded_cases
                if name in checkpoint_by_name
            ]
            test_rsts = pure_rows + resumed_mixed_rows
            print(
                f"{group_name} mixed-only: retained {len(pure_rows)} non-mixed oracle rows; "
                f"restored {len(resumed_mixed_rows)}/{len(loaded_cases)} "
                f"mixed-state rows from checkpoint."
            )
        else:
            test_rsts = checkpoint_rows

        if (not args.mixed_only and sheet_exists(output_excel_path, group_name)
                and len(test_rsts) == 0):
            if sheet_uses_current_mixed_states(output_excel_path, group_name):
                print(f"{group_name} already exists in {output_excel_path}; skipping this faulty version.")
                continue


            old_sheet = pd.read_excel(
                output_excel_path, sheet_name=group_name, dtype=str
            )
            for row in old_sheet.to_dict('records'):
                try:
                    attributes = json.loads(row.get('attributes', '{}'))
                except (TypeError, json.JSONDecodeError):
                    continue
                if attributes.get('type') != 'mixed':
                    test_rsts.append(row)
            print(
                f"{group_name}: updated mixed-state data detected; reusing "
                f"{len(test_rsts)} pure-state oracle rows and recomputing only mixed-state cases."
            )

        completed_inputs = {row['input'] for row in test_rsts}

        pending_cases = [
            (num, test_case)
            for num, test_case in enumerate(loaded_cases)
            if test_case[0] not in completed_inputs
        ]

        if not pending_cases:
            write_oracle_sheet(output_excel_path, group_name, test_rsts)
            print(f"{group_name} restored from checkpoint and written to {output_excel_path}.")
            continue

        print(
            f"{group_name}: {len(pending_cases)}/{len(loaded_cases)} "
            f"test cases pending ({mode_text})."
        )

        for num, input in pending_cases:
            print('---------------------num='+str(num))
            name, target_state, attrs = input
            print(name)

            flag_wrong = False
            try:
                qc_correct = create_correct_circuit(program_name, 0 )


                qc_error = create_error_circuit(program_name, fault_version, 0)

                if isinstance(target_state, testcase_io.LowRankDensityMatrix):
                    zero_probability = low_rank_swap_zero_probability(
                        target_state, qc_correct, qc_error
                    )
                    zero_count = int(np.random.binomial(shotTimes, zero_probability))
                    p = zero_count / shotTimes
                    resultDec = [0] * zero_count + [1] * (shotTimes - zero_count)
                    print(resultDec)
                    print(p)
                    rst = make_oracle_row(name, attrs, p >= 1, p)
                    test_rsts.append(rst)
                    completed_inputs.add(name)
                    append_checkpoint_row(current_checkpoint_path, rst)
                    continue


                if qc_correct.num_clbits > 0:
                    qc_swap_test = QuantumCircuit(2 * qubit_count + 1, 2 * clbit_count + 1)
                else:
                    qc_swap_test = QuantumCircuit(2 * qubit_count + 1, 1)
                qc_swap_test.h(-1)


                prep_result  = prepare_initial_state(input)

                if qc_correct.num_clbits > 0:
                    qc_swap_test.compose(prep_result["qc"], qubits=range(input_qbit_count), inplace=True)
                    qc_swap_test.compose(prep_result["qc"], qubits=range(qubit_count, input_qbit_count + qubit_count), inplace=True)
                    qc_swap_test.compose(qc_correct, qubits=range(qubit_count), clbits=range(clbit_count), inplace=True)
                    qc_swap_test.compose(qc_error, qubits=range(qubit_count, 2 * qubit_count), clbits= range(clbit_count, 2 * clbit_count), inplace=True)
                else:
                    qc_swap_test.compose(prep_result["qc"], qubits=range(input_qbit_count), inplace=True)
                    qc_swap_test.compose(prep_result["qc"], qubits=range(qubit_count, input_qbit_count + qubit_count), inplace=True)
                    qc_swap_test.compose(qc_correct, qubits=range(qubit_count), inplace=True)
                    qc_swap_test.compose(qc_error, qubits=range(qubit_count, 2 * qubit_count), inplace=True)


                for k in output_qubits:
                    qc_swap_test.cswap(-1, k, k + qubit_count)


                qc_swap_test.h(-1)
                qc_swap_test.measure(-1, -1)


                if attrs["is_mixed"]:

                    simulator = Aer.get_backend('qasm_simulator')
                    result = execute(qc_swap_test, simulator,
                                     noise_model=prep_result["noise_model"],
                                     shots=shotTimes, memory=True).result().get_memory()
                    resultDec = [int(item[0]) for item in result]
                    p = resultDec.count(0) / shotTimes
                else:

                    simulator = Aer.get_backend('qasm_simulator')
                    result = execute(qc_swap_test, simulator, shots=shotTimes, memory=True).result().get_memory()
                    resultDec = [int(item[0]) for item in result]
                    p = resultDec.count(0) / shotTimes

                print(result)
                print(resultDec)
                print(p)
                if p < 1:
                    print('fail')
                    rst = make_oracle_row(name, attrs, False, p)
                else:
                    print('pass')
                    rst = make_oracle_row(name, attrs, True, p)
                test_rsts.append(rst)
                completed_inputs.add(name)
                append_checkpoint_row(current_checkpoint_path, rst)
            except Exception as exc:
                save_error_checkpoint(current_error_path, num, name, exc)
                print(f"Test case {num} ({name}) failed; checkpoint saved to {current_checkpoint_path}")
                print(f"Error details saved to: {current_error_path}")
                raise

        if args.mixed_only:
            rows_by_name = {row['input']: row for row in test_rsts}
            test_rsts = pure_rows + [
                rows_by_name[name]
                for name, _, _ in loaded_cases
                if name in rows_by_name
            ]
        write_oracle_sheet(output_excel_path, group_name, test_rsts)
        if current_error_path.exists():
            current_error_path.unlink()
        print(f"{group_name} completed and written to {output_excel_path}; checkpoint retained at {current_checkpoint_path}")

