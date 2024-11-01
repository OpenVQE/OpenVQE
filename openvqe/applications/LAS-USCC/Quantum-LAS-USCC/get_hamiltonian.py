""" Script converting a fermionic Hamiltonian to a qubit Hamiltonian.

This file may change as qiskit-nature changes. It also creates fragment
Hamiltonians if `frag` is not None.
"""
# Qiskit imports
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

def get_hamiltonian(frag, nelecas_sub, ncas_sub, h1, h2):
    if frag is None:
        num_alpha = nelecas_sub[0]
        num_beta = nelecas_sub[1]
        n_so = ncas_sub*2
    else:
        # Get alpha and beta electrons from LAS
        num_alpha = nelecas_sub[frag][0]
        num_beta = nelecas_sub[frag][1]
        n_so = ncas_sub[frag]*2
        h1 = h1[frag]
        h2 = h2[frag]

    # Assuming an RHF reference for now, so h1_b, h2_ab, h2_bb are created using 
    # the corresponding spots from h1_frag and just the aa term from h2_frag
    electronic_energy = ElectronicEnergy.from_raw_integrals(h1, h2)
    second_q_op = electronic_energy.second_q_op()
    mapper = JordanWignerMapper()
    
    # This just outputs a qubit op corresponding to a 2nd quantized op
    hamiltonian = mapper.map(second_q_op)
    return hamiltonian

