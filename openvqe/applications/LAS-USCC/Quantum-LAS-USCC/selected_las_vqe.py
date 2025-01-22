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
# from custom_excitations import custom_excitations

# Qiskit imports
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit.primitives import Estimator
from qiskit import Aer, transpile
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.algorithms.minimum_eigensolvers import VQE 
from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA, SLSQP

from mrh.exploratory.citools import fockspace, lasci_ominus1
from scipy import linalg
from mrh.exploratory.unitary_cc import uccsd_sym1, lasuccsd
from typing import Tuple, List, Any

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
        self.f_orbs = f_orbs
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

    def run(self, epsilon=None):
        # CASCI h1 & h2 for VQE Hamiltonian
        self.epsilon = epsilon
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
            
        #Setup LASUSCC calculation
        lasci_ominus1.GLOBAL_MAX_CYCLE = 0
        ucc = mcscf.CASCI (self.mf, self.ncas, self.nelec_cas)
        ucc.mo_coeff = self.las.mo_coeff
        ucc.fcisolver = lasuccsd.FCISolver (self.mf.mol)
        ucc.fcisolver.norb_f = self.f_orbs 
        ucc.kernel()
        print ("LASSCF energy:   {:.9f}".format (self.las.e_tot))
        print ("LASUCCSD energy 0 iterations: {:.9f}\n".format (ucc.e_tot))
        psi = ucc.fcisolver.psi
        x = psi.x # Amplitude vector that solves the BFGS problem 
        h1, h0 = ucc.get_h1eff ()
        h2 = ucc.get_h2eff ()
        h = [h0, h1, h2] # 2nd-quantized CAS Hamiltonian
        energy, gradient = psi.e_de (x, h)
        def custom_excitations(num_spatial_orbitals: int,
                           num_particles: Tuple[int, int],
                           num_sub: List[int]
                           ) -> List[Tuple[Tuple[Any, ...], ...]]: #Add something like LAS; custom_UCC and ferm_exc_gen (feg) --> Find a definition --> expects 3 things make it expect 4(+ las) same for feg
            excitations = []
            epsilon = self.epsilon #This is the criteria for selection
            all_g, g, gen_indices, a_idxs_new, i_idxs_new, num_a_idxs, num_i_idxs = psi.get_grad_t1(x, h, epsilon=epsilon)
            print("len(a_idxs_new)",len(a_idxs_new))
            norb = num_spatial_orbitals
            norb = num_spatial_orbitals
            uop = lasuccsd.gen_uccsd_op (norb, num_sub)
            a_idxs = uop.a_idxs
            i_idxs = uop.i_idxs
            for a,i in zip(a_idxs_new,i_idxs_new):
                excitations.append((tuple(i),tuple(a[::-1])))            
            return excitations
        mapper = JordanWignerMapper()
        estimator = Estimator()
        # Setting up the VQE
        ansatz = custom_UCC(num_spatial_orbitals=mc.ncas, num_particles=mc.nelecas,
                            excitations=custom_excitations, qubit_mapper=mapper, 
                            initial_state=initial_state, preserve_spin=False)
        optimizer = L_BFGS_B(maxiter=10000, iprint=101)
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
