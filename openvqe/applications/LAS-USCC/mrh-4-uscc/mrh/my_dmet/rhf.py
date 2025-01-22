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
import scipy.sparse.linalg
#import qcdmet_paths
from pyscf import gto, scf, ao2mo
from pyscf.scf.hf import dot_eri_dm
from pyscf.lib import logger
from mrh.util.basis import get_complementary_states
from mrh.util.la import matrix_eigen_control_options

# The TEI is tagged and I must wrap_my_veff and wrap_my_jk in solve_ERI as well

def wrap_my_jk_ERI (TEI):

    def my_jk (mol, dm, hermi=1, vhfopt=None):
        vj, vk = dot_eri_dm (TEI, TEI.loc2eri_op (dm), hermi=1)
        vj = TEI.eri2loc_op (vj)
        vk = TEI.eri2loc_op (vk)
        return vj, vk
    return my_jk

def wrap_my_veff_ERI (TEI):

    def my_veff (mol, dm, dm_last=0, vhf_last=0, hermi=1, vhfopt=None):
        
        ddm_basis    = np.array(dm, copy=False) - np.array (dm_last, copy=False)
        ddm_ao       = TEI.loc2eri_op (ddm_basis)
        vj_ao, vk_ao = dot_eri_dm (TEI, ddm_ao, hermi=hermi)
        veff_ao      = vj_ao - 0.5 * vk_ao
        veff_basis   = TEI.eri2loc_op (veff_ao) + np.array (vhf_last, copy=False)
        return veff_basis
        
    return my_veff

def solve_ERI( OEI, TEI, oneRDMguess_loc, numPairs, num_mf_stab_checks):

    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = 2 * numPairs

    L = OEI.shape[0]
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( L )
    mf._eri = None # ao2mo.restore(8, TEI, L)
    mf.get_jk   = wrap_my_jk_ERI   (TEI)
    mf.get_veff = wrap_my_veff_ERI (TEI)
    mf.verbose = 4
    mf.scf( oneRDMguess_loc )
    if ( mf.converged == False ):
        mf = mf.newton ()
        mf.kernel ()
    oneRDM_loc = mf.make_rdm1 ()
    assert (mf.converged)

    # Instability check and repeat
    for i in range (num_mf_stab_checks):
        mf.mo_coeff = mf.stability ()[0]
        oneRDMguess_loc = mf.make_rdm1 ()
        mf = scf.RHF( mol )
        mf.get_hcore = lambda *args: OEI
        mf.get_ovlp = lambda *args: np.eye( L )
        mf._eri = None # ao2mo.restore(8, TEI, L)
        mf.get_jk   = wrap_my_jk_ERI   (TEI)
        mf.get_veff = wrap_my_veff_ERI (TEI)
        mf.verbose=0
        mf.scf( oneRDMguess_loc )
        oneRDM_loc = mf.make_rdm1 ()
        if ( mf.converged == False ):
            mf.newton ().kernel ( oneRDM_loc )
            oneRDM_loc = mf.make_rdm1 () #np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    
    return oneRDM_loc
    
def wrap_my_jk (get_jk_ao, ao2basis ): # mol_orig works in ao

    #get_jk(mol, dm, hermi=1, vhfopt=None)
    def my_jk (mol, dm, hermi=1): # mol works in basis, dm is in basis
    
        dm_ao        = np.dot( np.dot( ao2basis, dm ), ao2basis.T )
        vj_ao, vk_ao = get_jk_ao (dm_ao, hermi)
        vj_basis     = np.dot( np.dot( ao2basis.T, vj_ao ), ao2basis )
        vk_basis     = np.dot( np.dot( ao2basis.T, vk_ao ), ao2basis )
        return vj_basis, vk_basis
    
    return my_jk

def wrap_my_veff (get_veff_ao, ao2basis ): 

    #get_veff(mol, dm, dm_last=0, vhf_last=0, hermi=1, vhfopt=None)
    def my_veff (mol, dm, dm_last=0, vhf_last=0, hermi=1): # mol works in basis, dm is in basis
        
        ddm_basis   = np.array(dm, copy=False) - np.array(dm_last, copy=False)
        ddm_ao      = np.dot( np.dot( ao2basis, ddm_basis ), ao2basis.T )
        if isinstance (dm_last, np.ndarray):
            dm_last_ao  = np.dot( np.dot( ao2basis, dm_last ), ao2basis.T )
        else:
            dm_last_ao = dm_last
        if isinstance (vhf_last, np.ndarray):
            vhf_last_ao = np.dot( np.dot( ao2basis, vhf_last ), ao2basis.T )
        else:
            vhf_last_ao = vhf_last
        veff_ao     = get_veff_ao (ddm_ao, dm_last_ao, vhf_last_ao, hermi)
        veff_basis  = np.dot( np.dot( ao2basis.T, veff_ao ), ao2basis ) + np.array( vhf_last, copy=False )
        return veff_basis
        
    return my_veff

def solve_JK(CONST, OEI, ao2basis, oneRDMguess_loc, numPairs, num_mf_stab_checks, get_veff_ao, get_jk_ao,
    groupname=None, symm_orb=None, irrep_name=None, irrep_id=None, enforce_symmetry=False,
    verbose=logger.INFO, output=None):

    mol = gto.Mole()
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = 2 * numPairs
    mol.verbose = 0 if output is None else verbose
    if enforce_symmetry:
        mol.groupname = groupname
        mol.symm_orb = symm_orb
        mol.irrep_name = irrep_name
        mol.irrep_id = irrep_id
    if output is not None: mol.output = output
    mol.build ()
    if enforce_symmetry: mol.symmetry = True

    L = OEI.shape[0]
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: OEI
    mf.get_ovlp = lambda *args: np.eye( L )
    mf.energy_nuc = lambda *args: CONST
    mf._eri = None
    mf.get_jk   = wrap_my_jk   (get_jk_ao, ao2basis)
    #mf.get_veff = wrap_my_veff (get_veff_ao, ao2basis)
    mf.max_cycle = 500
    mf.damp = 0.33
    
    mf.scf( oneRDMguess_loc )
    oneRDM_loc = mf.make_rdm1 ()
    if not mf.converged:
        mf = mf.newton ()
        def my_intor (intor, comp=None, hermi=0, aosym='s1', out=None, shls_slice=None):
            if intor == 'int1e_ovlp':
                return np.eye (L)
            return gto.Mole.intor(mol, intor, comp=comp, hermi=hermi, aosym=aosym, out=out, shls_slice=shls_slice)
        mf.mol.intor = my_intor
        mf.kernel ( oneRDM_loc )
        oneRDM_loc = mf.make_rdm1 ()
    assert (mf.converged)

    # Instability check and repeat
    for i in range (num_mf_stab_checks):
        mf.mo_coeff = mf.stability ()[0]
        oneRDMguess_loc = mf.make_rdm1 ()
        mf = scf.RHF( mol )
        mf.get_hcore = lambda *args: OEI
        mf.get_ovlp = lambda *args: np.eye( L )
        mf.energy_nuc = lambda *args: CONST
        mf._eri = None # ao2mo.restore(8, TEI, L)
        mf._eri = None
        mf.get_jk   = wrap_my_jk   (get_jk_ao, ao2basis)
        #mf.get_veff = wrap_my_veff (get_veff_ao, ao2basis)
        mf.verbose=0
        mf.scf( oneRDMguess_loc )
        oneRDM_loc = mf.make_rdm1 ()
        if ( mf.converged == False ):
            mf.newton ().kernel ( oneRDM_loc )
            oneRDM_loc = mf.make_rdm1 () #np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )

    return oneRDM_loc
    
def get_unfrozen_states (oneRDMfroz_loc):
    _, loc2froz = matrix_eigen_control_options (oneRDMfroz_loc, only_nonzero_vals=True)
    if loc2froz.shape[1] == loc2froz.shape[0]:
        raise RuntimeError ("No unfrozen states: eigenbasis of oneRDMfroz_loc is complete!")
    return get_complementary_states (loc2froz)



