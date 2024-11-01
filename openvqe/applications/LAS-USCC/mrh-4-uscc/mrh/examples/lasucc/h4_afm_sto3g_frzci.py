import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import fockspace, lasci_ominus1
from mrh.exploratory.unitary_cc import lasuccsd 

# Antiferromagnetically pseudo-coupled triplets

# Different local spin (or charge) states are specified by setting
# the "nelec_f" attribute of mc.fcisolver.

# In the case of antiferromagnetic coupling, the S^2 spin symmetry
# is already broken by the reference, so it makes no sense to try
# to preserve spin in the UCC generators. Therefore, I have also 
# allowed the spin-up and spin-down T1 amplitudes to differ by
# setting the "t1_s2sym" attribute of mc.fcisolver to "False".

xyz = '''H 0.0 0.0 0.0
         H 1.0 0.0 0.0
         H 0.2 3.9 0.1
         H 1.159166 4.1 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_afm_sto3g_frzci.log',
    verbose=lib.logger.DEBUG)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, 4, (2,2)).run () # = FCI
las = LASSCF (mf, (2,2), ((2,0),(0,2)), spin_sub=(3,3))
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)

# Deoptimized implementation of LASSCF for validation
mc0 = mcscf.CASCI (mf, 4, (2,2))
mc0.mo_coeff = las.mo_coeff
mc0.fcisolver = lasci_ominus1.FCISolver (mol)
mc0.fcisolver.norb_f = [2,2] # Number of orbitals per fragment
mc0.fcisolver.nelec_f = [(2,0),(0,2)] # (neleca,nelecb) in each fragment
mc0.kernel ()

# LASUCC is implemented as a FCI solver for MC-SCF
# It's compatible with CASSCF as well as CASCI, but it's really slow
mc1 = mcscf.CASCI (mf, 4, (2,2))
mc1.mo_coeff = las.mo_coeff
mc1.fcisolver = lasuccsd.FCISolver (mol)
mc1.fcisolver.norb_f = [2,2] # Number of orbitals per fragment
mc1.fcisolver.nelec_f = [(2,0),(0,2)] # (neleca, nelecb) in each fragment
mc1.fcisolver.t1_s2sym = False # Allows t1a and t1b amplitudes to differ
mc1.kernel ()

# 1-shot version
mc2 = mcscf.CASCI (mf, 4, (2,2))
mc2.mo_coeff = las.mo_coeff
mc2.fcisolver = lasuccsd.FCISolver (mol)
mc2.fcisolver.norb_f = [2,2] # Number of orbitals per fragment
mc2.fcisolver.nelec_f = [(2,0),(0,2)] # (neleca, nelecb) in each fragment
mc2.fcisolver.frozen = 'CI' # Prevent reoptimization of the CI vector
mc2.fcisolver.t1_s2sym = False # Allows t1a and t1b amplitudes to differ
ci0_f = [np.squeeze (fockspace.hilbert2fock (ci[0], no, ne))
    for ci, no, ne in zip (las.ci, las.ncas_sub, las.nelecas_sub)]
mc2.fcisolver.get_init_guess = lambda *args: ci0_f
mc2.kernel ()

print ("FCI energy:               {:.9f}".format (ref.e_tot))
print ("LASSCF energy:            {:.9f}".format (las.e_tot))
print ("LASUCCnull sanity check:  {:.9f}".format (mc0.e_tot))
print ("LASUCCSD (full) energy:   {:.9f}".format (mc1.e_tot))
print ("LASUCCSD (1-shot) energy: {:.9f}\n".format (mc2.e_tot))

