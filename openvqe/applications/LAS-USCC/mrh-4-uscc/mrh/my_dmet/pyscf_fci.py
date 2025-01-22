'''
pyscf-FCI SOLVER for DMET

author: Hung Pham (email: phamx494@umn.edu)
'''

import numpy as np
from mrh.my_dmet import localintegrals
import os, time
import sys
#import qcdmet_paths
from pyscf import gto, ao2mo, fci
from pyscf.fci.addons import fix_spin_
#np.set_printoptions(threshold=np.nan)
from mrh.util.basis import represent_operator_in_basis
from mrh.util.rdm import get_2CDM_from_2RDM
from mrh.util.tensors import symmetrize_tensor

#def solve( CONST, OEI, FOCK, TEI, frag.norbs_imp, frag.nelec_imp, frag.norbs_frag, impCAS, frag.active_orb_list, guess_1RDM, energytype='CASCI', chempot_frag=0.0, printoutput=True ):
def solve (frag, guess_1RDM, chempot_imp):

    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI_C - chempot_imp
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.spin = int (round (2 * frag.target_MS))
    mol.verbose = 0 if frag.mol_output is None else 4
    mol.output = frag.mol_output
    mol.build ()
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    #mol.incore_anyway = True
    #mf.get_hcore = lambda *args: OEI
    #mf.get_ovlp = lambda *args: np.eye(frag.norbs_imp)
    #mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    h1e = OEI
    eri = ao2mo.restore (8, frag.impham_TEI, frag.norbs_imp)

    ed = fci.FCI (mol, singlet=(frag.target_S == 0))
    if frag.target_S != 0:
        s2_eval = frag.target_S * (frag.target_S + 1)
        fix_spin_(ed, ss=s2_eval)

    # Guess vector
    ci = None
    if len (frag.imp_cache) == 1:
        ci = frag.imp_cache[0]
        print ("Taking initial ci vector from cache")

    t_start = time.time()
    ed.conv_tol = 1e-12
    E_FCI, ci = ed.kernel (h1e, eri, frag.norbs_imp, frag.nelec_imp, ci0=ci)
    assert (ed.converged)
    frag.imp_cache = [ci]
    t_end = time.time()
    print('Impurity FCI energy (incl chempot): {}; spin multiplicity: {}; time to solve: {}'.format (
        frag.impham_CONST + E_FCI,
        ed.spin_square (ci, frag.norbs_imp, frag.nelec_imp)[1],
        t_end - t_start))
    
    # oneRDM and twoCDM
    oneRDM_imp, twoRDM_imp = ed.make_rdm12 (ci, frag.norbs_imp, frag.nelec_imp)
    oneRDMs_imp = ed.make_rdm1s (ci, frag.norbs_imp, frag.nelec_imp)
    twoCDM_imp = get_2CDM_from_2RDM (twoRDM_imp, oneRDMs_imp)

    # General impurity data
    frag.oneRDM_loc = symmetrize_tensor (frag.oneRDMfroz_loc + represent_operator_in_basis (oneRDM_imp, frag.imp2loc))
    frag.twoCDM_imp = symmetrize_tensor (twoCDM_imp)
    frag.E_imp      = frag.impham_CONST + E_FCI + np.einsum ('ab,ab->', chempot_imp, oneRDM_imp)

    return None


