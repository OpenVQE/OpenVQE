import numpy as np
from qat.core import Observable, Term
import numpy as np
import scipy
from numpy import binary_repr
from qat.fermion import ElectronicStructureHamiltonian
from qat.fermion.chemistry.pyscf_tools import perform_pyscf_computation

from qat.lang.AQASM import Program, QRoutine, RY, CNOT, RX, Z, H, RZ, I, X
from qat.core import Observable, Term, Circuit
from qat.lang.AQASM.gates import Gate
import matplotlib as mpl
import numpy as np
from typing import Optional, List
import warnings
from qat.qpus import get_default_qpu

qpu = get_default_qpu()
method = "BFGS"



def ising(N):
    np.random.seed(123)  

    terms = []

    # Generate random coefficients for the transverse field term (X)
    a_coefficients = np.random.random(N)
    for i in range(N):
        term = Term(coefficient=a_coefficients[i], pauli_op="X", qbits=[i])
        terms.append(term)

    # Generate random coefficients for the interaction term (ZZ)
    J_coefficients = np.random.random((N, N))
    for i in range(N):
        for j in range(i):
            if i != j:  # avoid duplicate terms
                term = Term(coefficient=J_coefficients[i, j], pauli_op="ZZ", qbits=[i, j])
                terms.append(term)
    ising = Observable(N, pauli_terms=terms, constant_coeff=0.0)
    return ising


def Molecule_Generator(model):
    if model == "H2":
        r = 0.98
        geometry = [("H", (0, 0, 0)), ("H", (0, 0, r))]
        charge = 0
        spin = 0
        basis = "sto-3g"
    
    elif model == "H4":
        r = 0.85
        geometry = [
            ("H", (0, 0, 0)),
            ("H", (0, 0, 1 * r)),
            ("H", (0, 0, 2 * r)),
            ("H", (0, 0, 3 * r)),
        ]
        charge = 0
        spin = 0
        basis = "sto-3g"
  
    return r, geometry, charge, spin, basis

# Define the the curcuit that we want to use


def circuit_ansatz(nqbits, k, depth, theta_list):
    prog = Program()
    reg = prog.qalloc(nqbits)

    state = binary_repr(k)
    state_pad = state.zfill(nqbits)
    state_int_lst = [int(c) for c in state_pad]

    for j in range(min(nqbits, len(state_int_lst))):
        if state_int_lst[j] == 1:
            prog.apply(X, reg[j])

        if state_int_lst[j] == 0:
            prog.apply(I, reg[j])

    print("This applied state is ", state_pad)

    prog.apply(RY(theta_list[-2]), reg[0])
    prog.apply(RZ(theta_list[-1]), reg[0])

    for d in range(depth):
        for i in range(nqbits):
            prog.apply(RY(theta_list[2*i+2*nqbits*d]), reg[i])
            prog.apply(RZ(theta_list[2*i+1+2*nqbits*d]), reg[i])

        for i in range(nqbits//2):
            prog.apply(CNOT, reg[2*i+1], reg[2*i])

        for i in range(nqbits//2-1):
            prog.apply(CNOT, reg[2*i+2], reg[2*i+1])

    for i in range(nqbits):
        prog.apply(RY(theta_list[2*i+2*nqbits*depth]), reg[i])
        prog.apply(RZ(theta_list[2*i+1+2*nqbits*depth]), reg[i])

    return prog.to_circ()


from scipy.sparse.linalg import eigsh

def calculate_eigen_vectors(model, vals):
    model_matrix_sp = model.get_matrix(sparse=True)
    eigenvalues, eigenvectors = eigsh(model.get_matrix(), k=vals)
    print(eigenvalues)
    # Assuming eigenvalues and eigenvectors are obtained
    num_eigenvalues = len(eigenvalues)
    eigenvec_dict = {}

    # Loop through each eigenvalue and store its eigenvector
    for i in range(num_eigenvalues):
        eigenvector_key = f'ee{i}'
        eigenvector_value = eigenvectors[:, i]
        eigenvec_dict[eigenvector_key] = eigenvector_value

    # Now eigenvecToT will be a list containing all eigenvectors
    eigenvecToT = [eigenvec_dict[f'ee{i}'] for i in range(num_eigenvalues)]

    return eigenvecToT

# Example usage:
# Assuming 'your_model' is an instance of the model class you are using


method = "BFGS"

def get_statevector(result, nbqbits):

    statevector = np.zeros((2**nbqbits), np.complex128)
    for sample in result:
        statevector[sample.state.int] = sample.amplitude
    return statevector


def fun_fidelity(circ, eigenvectors, nbqbits):

    qpu = get_default_qpu()
    res = qpu.submit(circ.to_job())
    statevector = get_statevector(res, nbqbits)
    #print("statevector",statevector)
    fid = abs(np.vdot(eigenvectors, statevector)) ** 2
    return fid
def opt_funct(circuits, model, qpu, nqbits, energy_lists, fidelity_lists, weight, eigenvec_input):
    def input_funct(x):
        total_energy = 0
        for i, circ in enumerate(circuits):
            bound_circ = circ.bind_variables({k: v for k, v in zip(sorted(circ.get_variables()), x)})
            result = qpu.submit(bound_circ.to_job(observable=model))
            energy = result.value
            energy_lists[f"energy_circ_{i}"][method].append(energy)

            # Calculate fidelity
            fidelity = fun_fidelity(bound_circ, eigenvec_input[i], nqbits)
            fidelity_lists[f"fidelity_circ_{i}"][method].append(fidelity)
            #print(fidelity)

            total_energy += weight[i] * energy
        return total_energy

    def callback(x):
        for i, circ in enumerate(circuits):
            bound_circ = circ.bind_variables({k: v for k, v in zip(sorted(circ.get_variables()), x)})
            result = qpu.submit(bound_circ.to_job(observable=model))
            energy = result.value
            energy_lists[f"energy_circ_{i}"][method].append(energy)

            # Calculate fidelity
            fidelity = fun_fidelity(bound_circ, eigenvec_input[i], nqbits)
            fidelity_lists[f"fidelity_circ_{i}"][method].append(fidelity)

    return input_funct, callback