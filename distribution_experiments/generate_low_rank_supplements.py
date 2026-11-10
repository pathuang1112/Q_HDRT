"""Generate/resume a compact supplement pool without changing the main testcases."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import testcase_io
from TestcaseGenerator import _apply_local_damping_as_low_rank
from distribution_experiments.workflow_utils import (
    atomic_cases, atomic_json, category, distribution_sizes, file_digest, load_pool,
)


def make_case(n_qubits, kind, index, seed, environment_ground_probability=0.8):
    """Per-case seeds make resumed generation independent of checkpoint frequency."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, n_qubits, kind == 'mixed', index]))
    qc = QuantumCircuit(n_qubits)
    attributes = {'type': kind, 'num_qubits': n_qubits, 'is_mixed': kind == 'mixed',
                  'generator_version': 'distribution_low_rank_v1', 'generation_seed': seed,
                  'generation_index': index}
    if kind == 'superposition':
        base = rng.integers(0, 2, n_qubits).tolist()
        for qubit, bit in enumerate(base):
            if bit:
                qc.x(qubit)
        qubits = sorted(rng.choice(n_qubits, size=int(rng.integers(1, n_qubits + 1)), replace=False).tolist())
        angles = {}
        for qubit in qubits:
            angle = float(rng.uniform(0.1, np.pi - 0.1))
            qc.ry(angle, qubit)
            angles[str(qubit)] = angle
        attributes.update(subtype='supplement_random', base_state=base,
                          superposed_qubits=qubits, rotation_angles=angles,
                          gate_distribution={q: 'ry' for q in angles})
        state = Statevector(qc)
    else:

        source = ('fixed_basis', 'random_base', 'superposition', 'entangled')[index % 4]
        if source == 'random_base':
            base = rng.integers(0, 2, n_qubits).tolist()
            for qubit, bit in enumerate(base):
                if bit:
                    qc.x(qubit)
            attributes['base_state'] = base
        elif source == 'superposition':
            qc.h(range(n_qubits))
            attributes['superposed_qubits'] = list(range(n_qubits))
        elif source == 'entangled':
            if n_qubits < 2:
                raise ValueError('Bell sources require at least two qubits.')
            qc.h(0)
            qc.cx(0, 1)
            attributes['entangled_qubits'] = [0, 1]
        gamma, phase = float(rng.uniform(0.05, 0.3)), float(rng.uniform(0.03, 0.3))
        target = 0 if source == 'entangled' else int(rng.integers(n_qubits))
        state = _apply_local_damping_as_low_rank(
            Statevector(qc), target, gamma, phase, environment_ground_probability)
        purity = state.purity()
        if not 0 < purity < 1 - 1e-10:
            raise ValueError('Generated state is not strictly mixed.')
        attributes.update(subtype=source, amplitude_damping=gamma, phase_damping=phase,
                          noise_qubits=[target], purity=purity,
                          environment_ground_probability=environment_ground_probability,
                          state_representation=testcase_io.MIXED_STATE_REPRESENTATION)
    return f'supplement_{kind}_lr_v1_s{seed}_{index:06d}', state, attributes


def generate(args):
    original = Path(testcase_io.resolve_testcase_file(str(args.original or
                    ROOT / f'TestCases/quantum_test_cases_{args.n_qubits}qubits.json')))
    output = (args.output or ROOT / f'TestCases/supplements/supplement_{args.n_qubits}qubits.pkl.gz').resolve()
    if output == original.resolve():
        raise ValueError('The supplement must not overwrite the original pool.')
    if not output.name.endswith('.pkl.gz'):
        raise ValueError('Use a .pkl.gz output file.')
    signature = {'generator': 'distribution_low_rank_v1', 'seed': args.seed,
                 'original_sha256': file_digest(original), 'n_qubits': args.n_qubits,
                 'environment_ground_probability': args.environment_ground_probability}
    metadata = output.with_name(output.name + '.metadata.json')
    if metadata.exists() and json.loads(metadata.read_text(encoding='utf-8'))['signature'] != signature:
        raise ValueError('Generation settings changed; choose a different --output.')
    if output.exists() and not metadata.exists():
        raise ValueError('Existing supplement has no matching metadata; choose a new --output.')
    original_cases = load_pool(original)
    counts = Counter(category(data['attributes']) for data in original_cases.values())
    sizes = distribution_sizes(args.n_qubits)
    targets = {kind: max(0, max(c[kind] for c in sizes.values()) - counts[kind])
               for kind in ('superposition', 'mixed')}
    for kind in targets:
        override = getattr(args, kind + '_count')
        if override is not None:
            targets[kind] = override
    existing = testcase_io.load_test_cases(str(output)) if output.exists() else []
    used = set(original_cases)
    for name, state, attributes in existing:
        if name in used or attributes.get('generator_version') != 'distribution_low_rank_v1':
            raise ValueError('Duplicate or incompatible supplement: ' + name)
        if attributes['type'] == 'mixed' and not isinstance(state, testcase_io.LowRankDensityMatrix):
            raise ValueError('Mixed supplements must use low-rank factors.')
        used.add(name)
    present = Counter(category(attrs) for _, _, attrs in existing)
    if any(present[kind] > targets[kind] for kind in targets):
        raise ValueError('Requested count is below saved count; use a different output.')
    if output.exists() and all(present[kind] == targets[kind] for kind in targets):
        previous = json.loads(metadata.read_text(encoding='utf-8'))
        if previous.get('status') == 'completed' and previous.get('sha256') == file_digest(output):
            print(f'Already complete: {output}; counts={targets}', flush=True)
            return output
    atomic_json(metadata, {'signature': signature, 'targets': targets, 'status': 'running'})
    start = time.perf_counter()
    unsaved = 0
    for kind, target in targets.items():
        for index in range(present[kind], target):
            case = make_case(args.n_qubits, kind, index, args.seed, args.environment_ground_probability)
            if case[0] in used:
                raise ValueError('Duplicate generated name: ' + case[0])
            existing.append(case)
            used.add(case[0])
            unsaved += 1
            if unsaved >= args.checkpoint_every:
                atomic_cases(existing, output)
                unsaved = 0
                print(f'{kind}: {index + 1}/{target}; elapsed={time.perf_counter()-start:.1f}s', flush=True)
    atomic_cases(existing, output)

    restored = testcase_io.load_test_cases(str(output))
    mixed = [state for _, state, attrs in restored if attrs['type'] == 'mixed']
    if any(not 0 < state.purity() < 1 - 1e-8 for state in mixed):
        raise ValueError('Stored mixed-state purity validation failed.')
    atomic_json(metadata, {'signature': signature, 'targets': targets, 'status': 'completed',
                          'counts': dict(Counter(category(a) for _, _, a in restored)),
                          'sha256': file_digest(output)})
    print(f'Supplement: {output}\nCounts: {targets}', flush=True)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-qubits', '--n_qubits', type=int, default=12)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixed-count', type=int)
    parser.add_argument('--superposition-count', type=int)
    parser.add_argument('--environment-ground-probability', type=float, default=0.8)
    parser.add_argument('--checkpoint-every', type=int, default=200)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    if (args.n_qubits < 2 or args.seed < 0 or args.checkpoint_every < 1
            or not 0 < args.environment_ground_probability < 1
            or any(v is not None and v < 0 for v in (args.mixed_count, args.superposition_count))):
        parser.error('Invalid qubit count, seed, counts, checkpoint interval or environment probability.')
    generate(args)
