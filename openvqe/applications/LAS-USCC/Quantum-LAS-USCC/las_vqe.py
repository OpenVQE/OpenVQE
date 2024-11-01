""" LASVQE class definition and run function.

The LASVQE class requires a mean field object and fragment information to run
a LASSCF calculation, followed by a VQE, returning the VQE energy and tracking
VQE convergence. It can be configured to use a LASCI vector initialized on the
qubits or an HF initial state.

Usage example can be found in run_las-vqe.py
"""

import numpy as np
import logging
import time
import itertools
# PySCF imports
from pyscf import mcscf, ao2mo
from pyscf.tools import fcidump
# mrh imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
#from c4h6_struct import structure
from get_geom import get_geom
from custom_UCC import custom_UCC
from initialize_las import initialize_las
from get_hamiltonian import get_hamiltonian
from custom_excitations import custom_excitations

# Qiskit imports
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit.primitives import Estimator
from qiskit import Aer, transpile
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.algorithms.minimum_eigensolvers import VQE 
from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA, SLSQP

# Prints info at every VQE iteration
#logging.basicConfig(level='DEBUG')

class LASVQE:
    def __init__(self, mf, f_orbs, f_elec, f_atom_list, spin_sub):
        self.mf = mf
        
        # Create LASSCF object
        # Keywords: (wavefunction obj, num_orb in each subspace, 
        #               (nelec in each subspace)/((num_alpha, num_beta) 
        #               in each subspace), spin multiplicity in each subspace)
        las = LASSCF(mf, f_orbs, f_elec, spin_sub=spin_sub)

        # Localize the chosen fragment active spaces
        loc_mo_coeff = las.localize_init_guess(f_atom_list, mf.mo_coeff)

        # Run LASSCF
        las.kernel(loc_mo_coeff)
        self.loc_mo_coeff = las.mo_coeff
        print("LASSCF energy: ", las.e_tot)

        self.ncore = las.ncore
        self.ncas = las.ncas
        self.ncas_sub = las.ncas_sub
        self.nelec_cas = las.nelecas
        
        self.las = las

    def run(self):
        # CASCI h1 & h2 for VQE Hamiltonian
        mc = mcscf.CASCI(self.mf, self.ncas, self.nelec_cas)
        cas_h1e, e_core = mc.h1e_for_cas(self.loc_mo_coeff)
        h2 = ao2mo.restore(1, mc.get_h2eff(self.loc_mo_coeff), mc.ncas)

        hamiltonian = get_hamiltonian(None, mc.nelecas, mc.ncas, cas_h1e, h2)

        initial_state = initialize_las(self.las)
        #inital_state = HartreeFock(num_spatial_orbitals=4, num_particles=(2,2), 
        #                            qubit_mapper=mapper)

        # Tracking the convergence of the VQE
        counts = []
        values = []
        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)

        mapper = JordanWignerMapper()
        estimator = Estimator()
        # Setting up the VQE
        ansatz = custom_UCC(num_spatial_orbitals=mc.ncas, num_particles=mc.nelecas,
                            excitations=custom_excitations, qubit_mapper=mapper, 
                            initial_state=initial_state, preserve_spin=False)
        optimizer = L_BFGS_B(maxiter=1, iprint=101)
        init_pt = np.zeros(ansatz.num_parameters)
        print(len(init_pt))
        algorithm = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer,
                        initial_point=init_pt, callback=store_intermediate_result) 
    
        # Running the VQE
        t0 = time.time()
        vqe_result = algorithm.compute_minimum_eigenvalue(hamiltonian)
        print(vqe_result)
        t1 = time.time()
        print("Time taken for VQE: ",t1-t0)
        print("VQE energies: ", values)

        # Return energy value and relevant results in a dict
        vqe_energy = vqe_result.eigenvalue
        result_dict = {'vqe_result':vqe_result, 'vqe_en_vals':values, 
                       'vqe_counts':counts, 'nuc_rep': self.las.energy_nuc()}
        return vqe_energy, result_dict
