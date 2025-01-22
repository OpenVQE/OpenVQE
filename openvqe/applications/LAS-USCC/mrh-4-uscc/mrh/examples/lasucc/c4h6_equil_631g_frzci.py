import numpy as np
from scipy import linalg
from pyscf import scf, mcscf
from pyscf.lib import logger
from c4h6_struct import structure
from mrh.my_pyscf.tools import molden as molden
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools.lasci_ominus1 import FCISolver as lasci
from mrh.exploratory.unitary_cc.lasuccsd import FCISolver as lasucc
from mrh.exploratory.citools import fockspace

norb = 8
nelec = 8
norb_f = (4,4)
nelec_f = ((2,2),(2,2))
mol = structure (0.0, 0.0, output='c4h6_equil_631g_frzci.log', verbose=logger.DEBUG)
mf = scf.RHF (mol).run ()

# CASSCF (for orbital initialization)
mc = mcscf.CASSCF (mf, norb, nelec).set (fcisolver = csf_solver (mol, smult=1))
mo_coeff = mc.sort_mo ([11,12,14,15,16,17,21,24])
mc.kernel (mo_coeff)
mo_coeff = mc.mo_coeff.copy ()

# LASSCF (for comparison)
las = LASSCF (mf, norb_f, nelec_f, spin_sub=(1,1)).set (mo_coeff=mo_coeff)
mo_loc = las.localize_init_guess ([[0,1,2,3,4],[5,6,7,8,9]])
ncore, ncas, nmo = las.ncore, las.ncas, mo_coeff.shape[1]
e_las = las.kernel (mo_loc)[0]
mo_loc = las.mo_coeff.copy ()
ci0_f = [np.squeeze (fockspace.hilbert2fock (ci[0], no, ne))
    for ci, no, ne in zip (las.ci, las.ncas_sub, las.nelecas_sub)]
molden.from_lasscf (las, 'c4h6_equil_lasscf88_631g.molden')

# CASCI (for comparison)
mc = mcscf.CASCI (mf, 8, 8).set (mo_coeff = mo_loc,
    fcisolver = csf_solver (mol, smult=1)).run ()
e_cas = mc.e_tot
molden.from_mcscf (mc, 'c4h6_equil_casci88_631g.molden', cas_natorb=True)

# LASUCCSD
ucc = mcscf.CASCI (mf, 8, 8).set (fcisolver = lasucc (mol))
ucc.fcisolver.norb_f = [4,4]
ucc.fcisolver.frozen = 'ci'
ucc.fcisolver.get_init_guess = lambda *args: ci0_f
try:
    mo_loc = np.load ('c4h6_equil_631g_frzci.mo.npy')
    # ^ In principle, this shouldn't be necessary, however this system happens
    # to have big gauge invariance in its orbital frame. If there was less
    # symmetry, you wouldn't need to cache this part.
    psi = ucc.fcisolver.load_psi ('c4h6_equil_631g_frzci.psi.npy',
        norb, nelec, norb_f)
    ucc.fcisolver.psi = psi
    print ("Found cached wave function data")
    print ("This calculation should terminate after 1 cycle")
except FileNotFoundError as e:
    print ("Did not find cached wave function data")
    print (("This calculation takes ~20 minutes wall time on an 8th-gen"
            " Intel i7 CPU"))
    print (("After this run finishes successfully, the wave function "
            "data will be cached to disk"))
ucc.kernel (mo_loc)
molden.from_mcscf (ucc, 'c4h6_equil_lasuccsd_631g_frzci.molden', cas_natorb=True)
e_luc = ucc.e_tot

# Cache result for the next time you run this
np.save ('c4h6_equil_631g_frzci.mo', ucc.mo_coeff)
ucc.fcisolver.save_psi ('c4h6_equil_631g_frzci.psi', ucc.fcisolver.psi)

# Some post-processing
print ('\nLASSCF: %.12f, CASCI: %.12f, LASUCC: %.12f' % (e_las, e_cas, e_luc))
logger.info (ucc, 'LASSCF: %.12f, CASCI: %.12f, LASUCC: %.12f', e_las, e_cas, e_luc)
w_singlet = fockspace.hilbert_sector_weight (ucc.ci, 8, (4,4), 1)
w_triplet = fockspace.hilbert_sector_weight (ucc.ci, 8, (4,4), 3)
w_quintet = fockspace.hilbert_sector_weight (ucc.ci, 8, (4,4), 5)
print ('LASUCC singlet weight: %.5f' % (w_singlet))
print ('LASUCC triplet weight: %.5f' % (w_triplet))
print ('LASUCC quintet weight: %.5f' % (w_quintet))
logger.info (ucc, 'LASUCC singlet weight: %.5f', w_singlet)
logger.info (ucc, 'LASUCC triplet weight: %.5f', w_triplet)
logger.info (ucc, 'LASUCC quintet weight: %.5f', w_quintet)

