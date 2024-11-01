'''
    QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import numpy as np
from pyscf import gto, scf, ao2mo, mp
from mrh.util.basis import represent_operator_in_basis
from mrh.util.rdm import get_2CDM_from_2RDM
from mrh.util.tensors import symmetrize_tensor

#def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF, chempot_imp=0.0, printoutput=True ):
def solve (frag, guess_1RDM, chempot_imp):

    # Augment OEI with the chemical potential
    OEI = frag.impham_OEI_C - chempot_imp

    # Get the RHF solution
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = frag.nelec_imp
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( frag.norbs_imp )
    mf._eri = ao2mo.restore(8, frag.impham_TEI, frag.norbs_imp)
    mf.scf( guess_1RDM )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    if ( mf.converged == False ):
        mf = mf.newton ()
        mf.kernel ()
    
    # Get the MP2 solution
    mp2 = mp.MP2( mf )
    mp2.kernel()
    imp2mo         = mf.mo_coeff
    mo2imp         = imp2mo.conjugate ().T
    oneRDMimp_imp  = mf.make_rdm1()
    twoRDMimp_mo   = mp2.make_rdm2()
    twoRDMimp_imp  = represent_operator_in_basis (twoRDMimp_mo, mo2imp)
    twoCDM_imp = get_2CDM_from_2RDM (twoRDMimp_imp, oneRDMimp_imp)

    # General impurity data
    frag.oneRDM_loc     = symmetrize_tensor (frag.oneRDMfroz_loc + represent_operator_in_basis (oneRDMimp_imp, frag.imp2loc))
    frag.twoCDM_imp = symmetrize_tensor (twoCDM_imp)
    frag.E_imp          = frag.impham_CONST + mp2.e_tot + np.einsum ('ab,ab->', oneRDMimp_imp, chempot_imp)

    return None

