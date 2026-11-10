"""Build fixed-size distribution workbooks from matching main and supplement oracles."""

import argparse
import json
import os
from pathlib import Path
import random
from shutil import copyfile
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import testcase_io
from QHDRT import validate_mixed_oracle
from distribution_experiments.workflow_utils import (
    COLUMNS, DEFAULT_VERSIONS, KS_SOURCE, atomic_json, category, distribution_sizes,
    file_digest, load_pool, parse_versions, validate_oracle,
)


class DistributionOracleCreator:
    def __init__(self, program_name, n_qubits, fault_versions=None, experiment_dir=None,
                 original_oracle=None, supplement_oracle=None, original_cases=None,
                 supplement_cases=None, seed=42, distributions=None):
        self.program_name, self.n_qubits = program_name, n_qubits
        self.fault_versions = fault_versions or parse_versions(DEFAULT_VERSIONS)
        self.experiment_full_dir = Path(experiment_dir or ROOT / 'distribution_experiments') / f'{program_name}_{n_qubits}qubits'
        self.original_oracle_path = Path(original_oracle or (KS_SOURCE if program_name == 'QAOA_12'
                                         else ROOT / f'oracle/OpenQASM/{program_name}_oracle.xlsx'))
        self.supplement_oracle_path = Path(supplement_oracle or ROOT / f'oracle/OpenQASM/{program_name}_oracle_supplement.xlsx')
        self.original_cases = Path(testcase_io.resolve_testcase_file(str(original_cases or
                                   ROOT / f'TestCases/quantum_test_cases_{n_qubits}qubits.json')))
        candidate = Path(supplement_cases or ROOT / f'TestCases/supplements/supplement_{n_qubits}qubits.pkl.gz')
        if supplement_cases is not None and not candidate.exists():
            raise FileNotFoundError(candidate)
        self.supplement_cases = candidate if candidate.exists() else None
        self.cases = load_pool(self.original_cases, self.supplement_cases)
        self.seed = seed
        self.sizes = distribution_sizes(n_qubits)
        if distributions:
            unknown = set(distributions) - set(self.sizes)
            if unknown:
                raise ValueError('Unknown distributions: ' + str(unknown))
            self.sizes = {name: self.sizes[name] for name in distributions}

    def _load_version(self, version):
        sheet = f'{self.program_name}_{version}'
        frames = [validate_oracle(pd.read_excel(self.original_oracle_path, sheet_name=sheet))]
        if self.supplement_cases is not None:
            if not self.supplement_oracle_path.exists():
                raise FileNotFoundError('Generate supplement oracle first: ' + str(self.supplement_oracle_path))
            frames.append(validate_oracle(pd.read_excel(self.supplement_oracle_path, sheet_name=sheet)))
        frame = validate_oracle(pd.concat(frames, ignore_index=True))
        unknown = set(frame['input']) - set(self.cases)
        if unknown:
            raise ValueError(f'{sheet}: oracle has unknown testcases: {sorted(unknown)[:5]}')
        available = {name: self.cases[name] for name in frame['input']}
        validate_mixed_oracle(frame, available)
        for name, encoded in zip(frame['input'], frame['attributes']):
            attributes = json.loads(encoded)
            expected = json.loads(json.dumps(self.cases[name]['attributes']))
            if attributes != expected:
                raise ValueError(f'{sheet}: oracle/testcase attributes differ: {name}')
        return frame.set_index('input', drop=False)

    def create_distribution_summary(self):
        self.experiment_full_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.experiment_full_dir / 'distribution_plan.json', {
            'program': self.program_name, 'n_qubits': self.n_qubits,
            'versions': self.fault_versions, 'sizes': self.sizes,
            'original_oracle': str(self.original_oracle_path),
            'supplement_oracle': str(self.supplement_oracle_path), 'seed': self.seed})

    def create_distribution_oracles(self):
        self.create_distribution_summary()

        frames = {version: self._load_version(version) for version in self.fault_versions}
        first_names = set(frames[self.fault_versions[0]].index)
        if any(set(frame.index) != first_names for frame in frames.values()):
            raise ValueError('All versions must have identical oracle testcase name sets.')
        available = {kind: sorted(name for name in first_names
                                 if category(self.cases[name]['attributes']) == kind)
                     for kind in ('classical', 'superposition', 'mixed')}
        selections = {}
        for distribution, counts in self.sizes.items():
            rng = random.Random(self.seed)
            selected = []
            for kind, required in counts.items():
                if len(available[kind]) < required:
                    raise ValueError(f'{distribution}: {kind} requires {required}, available {len(available[kind])}.')
                selected.extend(rng.sample(available[kind], required))
            selections[distribution] = selected
        sources = {'original_oracle_sha256': file_digest(self.original_oracle_path),
                   'original_cases_sha256': file_digest(self.original_cases),
                   'supplement_cases_sha256': file_digest(self.supplement_cases) if self.supplement_cases else None,
                   'supplement_oracle_sha256': file_digest(self.supplement_oracle_path) if self.supplement_cases else None}
        plans = []
        for distribution, selected in selections.items():
            directory = self.experiment_full_dir / 'oracle' / distribution
            output = directory / f'{self.program_name}_oracle.xlsx'
            manifest = directory / 'selection_manifest.json'
            versions = list(self.fault_versions)
            additions = {version: frames[version] for version in self.fault_versions}
            if output.exists():
                if not manifest.exists():
                    raise FileExistsError('Existing distribution has no matching manifest; choose a new --experiment_dir: ' + str(output))
                previous = json.loads(manifest.read_text(encoding='utf-8'))
                old = previous['signature']

                stable = dict(sources, seed=self.seed, counts=self.sizes[distribution], n_qubits=self.n_qubits)
                for key in ('original_oracle_sha256', 'supplement_oracle_sha256'):
                    stable.pop(key)
                if any(old.get(key) != value for key, value in stable.items()):
                    raise FileExistsError('Testcase pool or distribution changed; use a new --experiment_dir: ' + str(output))
                if previous.get('oracle_sha256') != file_digest(output):
                    raise ValueError('Distribution workbook changed after creation: ' + str(output))

                selected = old['selected_cases']
                if len(selected) != 2 ** self.n_qubits or len(set(selected)) != len(selected):
                    raise ValueError('Invalid saved selection: ' + str(manifest))
                actual = {kind: sum(category(self.cases[name]['attributes']) == kind for name in selected)
                          for kind in ('classical', 'superposition', 'mixed')}
                if actual != self.sizes[distribution]:
                    raise ValueError('Saved selection does not match distribution: ' + str(manifest))
                versions = list(dict.fromkeys(old['versions'] + self.fault_versions))
                with pd.ExcelFile(output) as workbook:
                    for version in old['versions']:
                        sheet = f'{self.program_name}_{version}'
                        if sheet not in workbook.sheet_names:
                            raise ValueError('Manifest worksheet missing: ' + sheet)


                        if version not in frames:
                            frames[version] = self._load_version(version)
                        stored = validate_oracle(pd.read_excel(workbook, sheet_name=sheet))
                        expected = frames[version].loc[selected, COLUMNS].reset_index(drop=True)
                        try:
                            pd.testing.assert_frame_equal(stored.fillna(''), expected.fillna(''),
                                                          check_dtype=False, rtol=1e-12, atol=1e-15)
                        except AssertionError as exc:
                            raise ValueError('Existing worksheet differs from its source; kept unchanged: ' + sheet) from exc
                        additions.pop(version, None)
                    conflicts = [f'{self.program_name}_{v}' for v in additions
                                 if f'{self.program_name}_{v}' in workbook.sheet_names]
                    if conflicts:
                        raise ValueError('Untracked worksheets already exist: ' + str(conflicts))
            for version in additions:
                missing = set(selected) - set(frames[version].index)
                if missing:
                    raise ValueError(f'Version {version} lacks selected testcases: {sorted(missing)[:5]}')
            signature = dict(sources, versions=versions, selected_cases=selected, seed=self.seed,
                             counts=self.sizes[distribution], n_qubits=self.n_qubits)
            plans.append((distribution, directory, output, manifest, signature, additions))

        for distribution, directory, output, manifest, signature, additions in plans:
            if not additions:
                print('Already complete: ' + str(output), flush=True)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix('.tmp.xlsx')
            try:
                options = {}
                if output.exists():
                    copyfile(output, temporary)
                    options['mode'] = 'a'
                with pd.ExcelWriter(temporary, engine='openpyxl', **options) as writer:
                    for version, frame in additions.items():
                        frame.loc[signature['selected_cases'], COLUMNS].to_excel(
                            writer, sheet_name=f'{self.program_name}_{version}', index=False)
                os.replace(temporary, output)
            finally:
                if temporary.exists():
                    temporary.unlink()
            atomic_json(manifest, {'signature': signature, 'oracle_sha256': file_digest(output)})
            print(f'{distribution}: added versions {list(additions)}, '
                  f'{len(signature["selected_cases"])} cases per sheet -> {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('program_name')
    parser.add_argument('n_qubits', type=int)
    parser.add_argument('--fault_versions', '--fault-versions', default=None)
    parser.add_argument('--experiment_dir', '--experiment-dir', type=Path, default=ROOT / 'distribution_experiments')
    parser.add_argument('--original-oracle', type=Path)
    parser.add_argument('--supplement-oracle', type=Path)
    parser.add_argument('--original-cases', type=Path)
    parser.add_argument('--supplement-cases', type=Path)
    parser.add_argument('--distributions', help='Comma-separated names; default all five.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    creator = DistributionOracleCreator(
        args.program_name, args.n_qubits, parse_versions(args.fault_versions) if args.fault_versions else None,
        args.experiment_dir, args.original_oracle, args.supplement_oracle,
        args.original_cases, args.supplement_cases, args.seed,
        args.distributions.split(',') if args.distributions else None)
    creator.create_distribution_oracles()


if __name__ == '__main__':
    main()
