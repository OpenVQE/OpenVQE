import numpy as np
from pyscf import gto, scf, lib
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF as LASSCFci
from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox

mol = struct (3.0, 3.0, '6-31g', symmetry=False)
mol.verbose = lib.logger.INFO
mol.output = 'c2h4n4_spin.log'
mol.build ()
mf = scf.RHF (mol).run ()

# Reference object
las0 = LASSCFci (mf, (4,4), (4,4), spin_sub=(1,1))
frag_atom_list = (list (range (3)), list (range (7, 10)))
mo_coeff = las0.localize_init_guess (frag_atom_list, mf.mo_coeff)
las0.kernel (mo_coeff)
print ("E(CI algorithm) =", las0.e_tot)

# I/O for RDM algorithm
''' frs -> fragment, root, spin
    fr -> fragment, root
    Currently only one root is supported.
    casdm1frs is overall a list (len nfrag) of ndarrays of shape
        (1, 2, norb[ifrag], norb[ifrag])
    casdm2fr is a list (len nfrag) of ndarrays of shape
        (1, norb[ifrag], norb[ifrag], norb[ifrag], norb[ifrag])
    where, in this example, norb = [4,4].
    This corresponds to 8 spinorbitals in each fragment.
    The casdm2's are in Mulliken order and are summed over spin as
        aaaa + aabb + bbaa + bbbb'''
casdm1frs = las0.states_make_casdm1s_sub ()
casdm2fr = las0.states_make_casdm2_sub ()
def get_kernel_fn (ifrag):
    dm1s = casdm1frs[ifrag][0]
    dm2 = casdm2fr[ifrag][0]
    def kernel (norb, nelec, h0, h1s, h2):
        ''' This shows the call and return signature that you have to spoof in
            order to package arbitrary functions into this LASSCF algorithm. '''
        e1 = np.dot (h1s.ravel (), dm1s.ravel ())
        e2 = 0.5 * np.dot (h2.ravel (), dm2.ravel ())
        etot = h0 + e1 + e2
        return etot, dm1s, dm2
    return kernel

# Plugging in to the RDM algorithm
las1 = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
las1.fciboxes = [make_fcibox (las1.mol, kernel=get_kernel_fn (ifrag))
    for ifrag in range (2)]
''' Note kernel is optional. If omitted, it defaults to the original CSF
    solver, but it still uses the alternating orbitals->rdm->orbitals->
    algorithm, as opposed to the original coupled algorithm that las0
    is using. '''
las1.kernel (las0.mo_coeff)
''' Warning: currently, this doesn't converge correctly with the highly fake
    kernel I wrote above unless I initialize from the converged orbitals of the
    CI algorithm. However, I'm able to complete a full calculation from scratch
    using the original CI solver in this framework, so this should work with a
    real kernel function. Also see tests/lasscf/test_lasscf_rdm.py '''
print ("E(RDM algorithm) =", las1.e_tot)


