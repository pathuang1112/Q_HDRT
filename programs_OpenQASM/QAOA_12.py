from pathlib import Path

from qiskit import Aer, QuantumCircuit, execute


_PROGRAM_DIR = Path(__file__).resolve().parent / "QAOA_12"


def _run_qasm(filename, count_times):
    qc = QuantumCircuit.from_qasm_file(str(_PROGRAM_DIR / filename))
    backend = Aer.get_backend("qasm_simulator")

    if count_times > 0:
        qc.barrier()
        for i in range(qc.num_qubits):
            qc.measure(i, i)

        job = execute(qc, backend, shots=count_times * 100)
        result = job.result()
        counts = result.get_counts(qc)

    return qc


def QAOA_12(count_times):
    return _run_qasm("QAOA_12.qasm", count_times)


def QAOA_12_M2(count_times):
    return _run_qasm("QAOA_12_b2.qasm", count_times)


def QAOA_12_M1(count_times):
    return _run_qasm("QAOA_12_b1.qasm", count_times)


def QAOA_12_M3(count_times):
    return _run_qasm("QAOA_12_b3.qasm", count_times)


def QAOA_12_M4(count_times):
    return _run_qasm("QAOA_12_b4.qasm", count_times)


def QAOA_12_M5(count_times):
    return _run_qasm("QAOA_12_b5.qasm", count_times)


