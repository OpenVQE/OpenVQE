#!/usr/bin/env python
# coding: utf-8

# This is a sample script to run LAS-USCCSD for the H4 molecule


import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import fockspace
from mrh.exploratory.unitary_cc import lasuccsd as lasuccsd
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccs_op
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op
import itertools
import pickle
from mrh.exploratory.citools import fockspace, lasci_ominus1
xyz = '''H 0.0 0.0 0.0;
            H 1.0 0.0 0.0;
            H 0.2 1.6 0.1;
            H 1.159166 1.3 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log.py',
    verbose=0)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, 4, 4).run () # = FCI
las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
las.verbose=1
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)

lasci_ominus1.GLOBAL_MAX_CYCLE = 0
mc = mcscf.CASCI (mf, 4, 4)
mc.mo_coeff = las.mo_coeff
mc.fcisolver = lasuccsd.FCISolver (mol)
mc.fcisolver.norb_f = [2,2] # Number of orbitals per fragment
mc.verbose=5
mc.kernel()

print ("FCI energy:      {:.9f}".format (ref.e_tot))
print ("LASSCF energy:   {:.9f}".format (las.e_tot))
print ("LASUCCSD energy: {:.9f}\n".format (mc.e_tot))
psi = mc.fcisolver.psi

x = psi.x # Amplitude vector that solves the BFGS problem 
h1, h0 = mc.get_h1eff ()
h2 = mc.get_h2eff ()
h = [h0, h1, h2] # 2nd-quantized CAS Hamiltonian
energy, gradient = psi.e_de (x, h)
print ("Recomputing LASUCC total energy with cached objective function")
print ("LASUCCSD energy: {:.9f}".format (energy))
print ("|gradient| = {:.3e}".format (linalg.norm (gradient)))
print ("If that seems too high to you, consider: BFGS sucks.\n")

epsilons = np.logspace(-1, -6, num=4)  # Generating epsilon values on a logarithmic scale
energies = []
lengths = []  # Store lengths of a_idxs_new for each epsilon
mc_uscc = mcscf.CASCI(mf, 4, 4)
mc_uscc.mo_coeff = las.mo_coeff
for epsilon in epsilons:
    all_g, g, gen_indices, a_idxs_new, i_idxs_new, num_a_idxs, num_i_idxs = psi.get_grad_t1(x, h, epsilon=epsilon)
    lengths.append(len(a_idxs_new))
    lasci_ominus1.GLOBAL_MAX_CYCLE = 15000
    mc_uscc.fcisolver = lasuccsd.FCISolver2(mol, a_idxs_new, i_idxs_new)
    mc_uscc.fcisolver.norb_f = [2,2]
    mc_uscc.verbose = 5
    mc_uscc.kernel()
    energies.append(mc_uscc.e_tot)
    print("Number of parameters: {:.0f} | LASUSCCSD energy: {:.9f}".format(len(a_idxs_new), mc_uscc.e_tot))



mc2 = mcscf.CASCI (mf, 4, 4)
mc2.mo_coeff = las.mo_coeff
mc2.fcisolver = lasuccsd.FCISolver (mol)
mc2.fcisolver.norb_f = [2,2] # Number of orbitals per fragment
mc2.verbose=5
mc2.kernel()

print ("FCI energy:      {:.9f}".format (ref.e_tot))
print ("LASSCF energy:   {:.9f}".format (las.e_tot))
print ("LASUCCSD reference energy: {:.9f}\n".format (mc2.e_tot))


