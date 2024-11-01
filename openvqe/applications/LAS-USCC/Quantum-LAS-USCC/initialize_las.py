"""Directly initializes the LAS CI vectors using qiskit's initialize function.

The function `get_so_ci_vec` maps the fermionic CI vector to the qubit one.
It requires a converged LAS wave function to extract the CI vectors.
"""
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile

# Initialize using LASCI vector
## This function makes a few assumptions
## 1. The civector is arranged as a 2D matrix of coeffs
##    of size [nalphastr, nbetastr]
## 2. The civector contains all configurations within
##    the (localized) active space
# Here, we set up a lookup dictionary which is
# populated when either the number of alpha e-s
# or the number of beta electrons is correct
# It stores "bitstring" : decimal_value pairs
def get_so_ci_vec(ci_vec, nsporbs,nelec):
    lookup_a = {}
    lookup_b = {}
    cnt = 0
    norbs = nsporbs//2

    for ii in range (2**norbs):
        if f"{ii:0{norbs}b}".count('1') == nelec[0]:
            lookup_a[f"{ii:0{norbs}b}"] = cnt
            cnt +=1
    cnt = 0
    for ii in range (2**norbs):
        if f"{ii:0{norbs}b}".count('1') == nelec[1]:
            lookup_b[f"{ii:0{norbs}b}"] = cnt
            cnt +=1

    # Here the spin orbital CI vector is populated
    # the same lookup is used for alpha and beta, but for two different
    # sections of the bitstring
    so_ci_vec = np.zeros(2**nsporbs)
    for kk in range (2**nsporbs):
        if f"{kk:0{nsporbs}b}"[norbs:].count('1')==nelec[0] and f"{kk:0{nsporbs}b}"[:norbs].count('1')==nelec[1]:
            so_ci_vec[kk] = ci_vec[lookup_a[f"{kk:0{nsporbs}b}"[norbs:]],lookup_b[f"{kk:0{nsporbs}b}"[:norbs]]]

    return so_ci_vec

def initialize_las(las, gate_counts=False):
    ncas = np.sum(las.ncas_sub)
    qubits = np.arange(ncas*2).tolist()
    frag_qubits = []
    idx = 0
    for frag in las.ncas_sub:
        frag_qubits.append(qubits[idx:idx+frag]+qubits[ncas+idx:ncas+idx+frag])
        idx += frag

    qr1 = QuantumRegister(ncas*2, 'q1')
    new_circuit = QuantumCircuit(qr1)
    for frag in range(len(las.ncas_sub)):
        new_circuit.initialize(get_so_ci_vec(las.ci[frag][0],2*las.ncas_sub[frag],las.nelecas_sub[frag]) , frag_qubits[frag])

    if gate_counts is True:
        # Gate counts for initialization
        target_basis = ['rx', 'ry', 'rz', 'h', 'cx']
        circ_for_counts = transpile(new_circuit, basis_gates=target_basis, optimization_level=0)
        init_op_dict = circ_for_counts.count_ops()
        init_ops = sum(init_op_dict.values())
        print("Operations: {}".format(init_op_dict))
        print("Total operations: {}".format(init_ops))

    return new_circuit
