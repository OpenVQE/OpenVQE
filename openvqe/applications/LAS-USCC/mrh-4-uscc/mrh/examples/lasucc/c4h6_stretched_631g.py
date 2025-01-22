import numpy as np
from scipy import linalg
from pyscf import scf, mcscf
from pyscf.lib import logger
from c4h6_struct import structure
from mrh.my_pyscf.tools import molden
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools.lasci_ominus1 import FCISolver as lasci
from mrh.exploratory.unitary_cc.lasuccsd import FCISolver as lasucc
from mrh.exploratory.citools import fockspace

norb = 8
nelec = 8
norb_f = (4,4)
nelec_f = ((2,2),(2,2))
mol = structure (3.0, 3.0, output='c4h6_stretched_631g.log', verbose=logger.DEBUG)
mf = scf.RHF (mol).run ()

# CASSCF (for orbital initialization)
mc = mcscf.CASSCF (mf, norb, nelec).set (fcisolver = csf_solver (mol, smult=1)).run ()
mo_coeff = mc.mo_coeff.copy ()

# LASSCF (for comparison)
las = LASSCF (mf, norb_f, nelec_f, spin_sub=(1,1)).set (mo_coeff=mo_coeff)
mo_loc = las.localize_init_guess ([[0,1,2,3,4],[5,6,7,8,9]])
ncore, ncas, nmo = las.ncore, las.ncas, mo_coeff.shape[1]
e_las = las.kernel (mo_loc)[0]
mo_loc = las.localize_init_guess ([[0,1,2,3,4],[5,6,7,8,9]],
    freeze_cas_spaces=True)
molden.from_lasscf (las, 'c4h6_stretched_lasscf88_631g.molden')

# CASCI (for comparison)
mc = mcscf.CASCI (mf, 8, 8).set (mo_coeff = mo_loc,
    fcisolver = csf_solver (mol, smult=1)).run ()
e_cas = mc.e_tot
molden.from_mcscf (mc, 'c4h6_stretched_casci88_631g.molden', cas_natorb=True)

# LASUCCSD
ucc = mcscf.CASCI (mf, 8, 8).set (fcisolver = lasucc (mol))
ucc.fcisolver.norb_f = [4,4]
try:
    mo_loc = np.load ('c4h6_stretched_631g.mo.npy')
    # ^ In principle, this shouldn't be necessary, however this system happens
    # to have big gauge invariance in its orbital frame. If there was less
    # symmetry, you wouldn't need to cache this part.
    psi = ucc.fcisolver.load_psi ('c4h6_stretched_631g.psi.npy',
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
molden.from_mcscf (ucc, 'c4h6_stretched_lasuccsd_631g.molden', cas_natorb=True)
e_luc = ucc.e_tot

# Cache result for the next time you run this
np.save ('c4h6_stretched_631g.mo', ucc.mo_coeff)
ucc.fcisolver.save_psi ('c4h6_stretched_631g.psi', ucc.fcisolver.psi)

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
print ("\nGetting U'HU as a dense matrix in a small subspace...")
psi = ucc.fcisolver.psi
h1, h0 = ucc.get_h1eff ()
h2 = ucc.get_h2eff ()
heff, determinants = psi.get_dense_heff (psi.x, [h0,h1,h2], 0, nelec=(2,2))
np.save ('c4h6_stretched_frag0_heff.npy', heff)
print (('The similarity-transformed Hamiltonian of the first fragment'
        ' in the nelec=(2,2) subspace is stored in '
        'c4h6_stretched_frag0_heff.npy'))
print ("Its determinant basis is:")
for ix, det_spinless in enumerate (determinants):
    deta, detb = divmod (det_spinless, 2**4)
    det_onv = fockspace.onv_str (deta, detb, 4)
    print ("{} {}".format (ix, det_onv))

