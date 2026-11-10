"""Canonical subjects and version numbers for the clean 50-run experiment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPEATS = 50
SUBJECTS = {
    'adder_n7': {'qubits': 7},
    'qft_8': {'qubits': 8},
    'pe_9': {'qubits': 9},
    'rand_cliff_8_AG': {'qubits': 8},
    'IQFT': {'qubits': 10},
    'QAOA_12': {'qubits': 12},
}
SELECTED_GROUPS = [f'{program}_{version}' for program in SUBJECTS for version in range(1, 6)]


def get_fault_versions(program):
    if program not in SUBJECTS:
        raise ValueError(f'Unknown subject: {program}')
    return range(1, 6)
