# QHDRT Experiments

This project compares QHDRT (Q-HDRT in the paper) with RT, RPT, DRT, and ART.
It also includes ablation, dissimilarity, input-distribution, and sensitivity studies.

## Environment

Run commands from this directory using a compatible Python environment with
Qiskit, NumPy, SciPy, pandas, scikit-learn, scikit-learn-extra, matplotlib,
openpyxl, and psutil.

## Main Experiment

1. Edit `selected_groups` in `run_experiments.py` to choose subject versions.
   `experiment_config.py` defines subjects and the default 50 repetitions.

2. Generate test suites, including mixed-state inputs:

   ```powershell
   python run_experiments.py testcases
   ```

3. Generate matching oracle workbooks:

   ```powershell
   python generate_oracles.py
   ```

   This helper defaults to all six subjects; use `--programs QAOA_12`.

   Main oracle generation uses only project-root scripts:
   `oracle_distribution_based.py` for Adder, RC, and QAOA, and
   `oracle_swaptest_based.py` for QFT, PE, and IQFT. The former compares two
   independently sampled output distributions using Pearson's chi-square
   homogeneity test, with conditional Monte Carlo calibration for sparse tables.
   Defaults are `alpha=0.05` and `100 * 2**qubits` shots for the distribution
   oracle, and 50 shots for the SWAP oracle when called through this helper.
   Use `--alpha`, `--distribution-shots`, or `--swap-shots` to change them.

   ```powershell
   python oracle_distribution_based.py --program QAOA_12 --fault-versions all
   python oracle_swaptest_based.py --program qft_8 --shots 50
   ```

   Oracle workbooks are written directly to `oracle/OpenQASM/`, using sheets
   `<program>_1` through `<program>_5` and the columns
   `input, input_state, attributes, result, p`. Distribution-oracle checkpoints
   are stored in `oracle/OpenQASM/distribution_state/`; interrupted runs resume
   automatically. `--limit` writes isolated smoke-test outputs. Existing labels
   from another oracle method are not silently reused; use
   `python generate_oracles.py --programs QAOA_12 --rerun` to replace them
   explicitly. Changing the oracle may change PASS/FAIL labels, so regenerate
   affected experimental results rather than combining them with older labels.


4. Preview and run the five-strategy comparison:

   ```powershell
   python run_experiments.py main
   ```

   To run a subset without editing the script:

   ```powershell
   python run_experiments.py main --groups adder_n7_1 QAOA_12_1
   ```

Main results are stored under `result/<subject_version>/`; QHDRT uses unique
`runs/<run_id>/` subdirectories. Time and memory records accompany the results.


Separate batch entry points are `run_QHDRT.py`, `run_QHDRT_baselines.py`,
`run_QDRT_C.py`, and `run_QDRT_D.py`. Each reads its corresponding
`*_config.txt` file.

## Subexperiments

| Directory or scripts | Purpose | Unified command |
| --- | --- | --- |
| `distance_metric_experiments/` | RQ2: compare hybrid, fidelity, Euclidean, and trace-distance measures. | `python run_experiments.py distance` |
| `distribution_experiments/` | RQ3: evaluate different input-state compositions. | `python run_experiments.py distribution` |
| `QDRT_C.py`, `QDRT_D.py`, `AblationExp*.py` | RQ4: component ablation; use the main DRT results as an additional reference. | `python run_experiments.py ablation` |
| `k_sensitivity_experiments/` | RQ5: vary the within-type cluster count. | `python run_experiments.py k` |
| `profile_adjustment_experiments/` | RQ5: vary profile-adjustment parameters. | `python run_experiments.py profile` |


Unified commands use the selected versions and default to 50 repetitions.
RQ3 additionally requires supplemental test cases, matching supplemental oracle
labels, and distribution-specific workbooks. Preparation scripts include
`generate_low_rank_supplements.py` and `distribution_oracle_creator.py`.
When using the new distribution oracle, generate supplemental labels with
`oracle_distribution_based.py --testcases <supplement_file> --output-dir <separate_directory>`
(also supply `--program`), then pass that workbook to the creator's
`--supplement-oracle` option. Do not combine Pearson-based main labels with
supplemental labels produced by a different statistical test.

## Supporting Files

- `programs_OpenQASM/`: reference programs and faulty versions.
- `TestCases/`: generated test suites.
- `oracle/OpenQASM/`: main-experiment oracle workbooks.


