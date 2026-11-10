import numpy as np
from qiskit import (

    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
import qiskit.extensions
from math import pi
from qiskit.visualization import plot_histogram
from qiskit.tools.visualization import circuit_drawer
from qiskit.circuit.library import CU1Gate
from qiskit.extensions import Initialize

def dec2bin(n):
    a = 1
    list = []
    while a > 0:
        a, b = divmod(n, 2)
        list.append(str(b))
        n = a
    s = ""
    for i in range(len(list) - 1, -1, -1):
        s += str(list[i])
    s = s.zfill(10)
    return s

def inverse(s):
    s_list = list(s)
    for i in range(len(s_list)):
        if s_list[i] == '0':
            s_list[i] = '1'
        else:
            s_list[i] ='0'
    s = "".join(s_list)
    return s

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft_rotations(circuit, qubit, p):
    """Performs qft on the first n qubits in circuit (without swaps)"""


    for j in range(qubit + 1, 10):
        circuit.append(CU1Gate(pi/(2**(p))), [qubit, j])
        p += 1
    circuit.barrier()
    return circuit

def IQFT(count_times):
    simulator = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(10)
    c = ClassicalRegister(10)

    qc = QuantumCircuit(q,c)


    swap_registers(qc,10)

    for qubit in range(10):
        qc.h(qubit)
        p = 1
        qft_rotations(qc,qubit,p)

    if count_times > 0:
        qc.barrier()

        qc.measure(q,c)


        job = execute(qc, simulator, shots=count_times * 100)
        result = job.result()
        counts = result.get_counts(qc)

    return qc

def IQFT_M1(count_times):
    simulator = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(10)
    c = ClassicalRegister(10)

    qc = QuantumCircuit(q,c)


    qc.ch(q[3],q[4])

    swap_registers(qc,10)

    for qubit in range(10):
        qc.h(qubit)
        qft_rotations(qc,qubit,1)

    if count_times > 0:

        qc.barrier()

        qc.measure(q,c)


        job = execute(qc, simulator, shots=count_times * 100)
        result = job.result()
        counts = result.get_counts(qc)

    return qc

def IQFT_M2(count_times):
    simulator = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(10)
    c = ClassicalRegister(10)

    qc = QuantumCircuit(q,c)


    swap_registers(qc,10)
    for qubit in range(3):
        qc.h(qubit)
        p = 1
        qft_rotations(qc,qubit,p)
    qc.ch(q[4],q[2])
    for qubit in range(3,10):
        qc.h(qubit)
        p = 1
        qft_rotations(qc,qubit,p)

    if count_times > 0:

        qc.barrier()

        qc.measure(q,c)


        job = execute(qc, simulator, shots=count_times * 100)
        result = job.result()
        counts = result.get_counts(qc)

    return qc


def IQFT_M3(count_times):
    simulator = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(10)
    c = ClassicalRegister(10)

    qc = QuantumCircuit(q, c)


    swap_registers(qc, 10)
    for qubit in range(3):
        qc.h(qubit)
        p = 1
        qft_rotations(qc, qubit, p)

    qc.ch(q[4],q[3])
    qft_rotations(qc,3,1)
    for qubit in range(4,10):
        qc.h(qubit)
        p = 1
        qft_rotations(qc,qubit,p)

    if count_times > 0:

        qc.barrier()

        qc.measure(q, c)


        job = execute(qc, simulator, shots=count_times * 100)
        result = job.result()
        counts = result.get_counts(qc)

    return qc

def IQFT_M4(count_times):
    simulator = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(10)
    c = ClassicalRegister(10)

    qc = QuantumCircuit(q, c)


    qc.ch(q[2], q[4])

    swap_registers(qc, 10)
    for qubit in range(10):
        qc.h(qubit)
        p = 1
        qft_rotations(qc, qubit, p)

    if count_times > 0:

        qc.barrier()

        qc.measure(q, c)


        job = execute(qc, simulator, shots=count_times * 100)
        result = job.result()
        counts = result.get_counts(qc)

    return qc

def IQFT_M5(count_times):
    simulator = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(10)
    c = ClassicalRegister(10)

    qc = QuantumCircuit(q, c)


    swap_registers(qc, 10)
    for qubit in range(6):
        qc.h(qubit)
        p = 1
        qft_rotations(qc, qubit, p)

    qc.ch(q[8],q[3])

    for qubit in range(6,10):
        qc.h(qubit)
        p = 1
        qft_rotations(qc, qubit, p)

    if count_times > 0:

        qc.barrier()

        qc.measure(q, c)


        job = execute(qc, simulator, shots=count_times * 100)
        result = job.result()
        counts = result.get_counts(qc)

    return qc

