### Adapted from github.com/hczhai/fci-siso/blob/master/fcisiso.py ###

import numpy as np
from scipy import linalg
import copy
from pyscf.data import nist
from pyscf.lib import logger, param
from pyscf.data import elements

def get_jk(mol, dm0):
    log = logger.new_logger (mol, mol.verbose)
    t0 = (logger.process_clock (), logger.perf_counter ())
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(3, mol.nao, mol.nao, mol.nao, mol.nao)
    t0 = log.timer ('SSO eri for {} AOs'.format (mol.nao), *t0)
    vj = np.einsum('yijkl,lk->yij', hso2e, dm0)
    vk = np.einsum('yijkl,jk->yil', hso2e, dm0)
    vk += np.einsum('yijkl,li->ykj', hso2e, dm0)
    t0 = log.timer ('SSO vj & vk for {} AOs'.format (mol.nao), *t0)
    return vj, vk

def get_jk_amfi(mol, dm0):
    nao = mol.nao_nr()
    aoslice = mol.aoslice_by_atom()
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    atom = copy.copy(mol)

    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = get_jk(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk

def compute_hso_amfi(mol, dm0): 
    alpha2 = param.LIGHT_SPEED**(-2)
    #alpha2 = nist.ALPHA ** 2
    aoslice = mol.aoslice_by_atom()
    nao = mol.nao_nr()
    hso_1e = np.zeros((3,nao,nao))
    for i in range(mol.natm):
        si, sf, ai, af = aoslice[i]
        slices = (si, sf, si, sf)
        #mol.set_rinv_origin(mol.atom_coord(i))
        with mol.with_rinv_as_nucleus (i):
            atom_1e = mol.intor('int1e_prinvxp', comp=3, shls_slice=slices)
        hso_1e[:,ai:af,ai:af] = - atom_1e * (mol.atom_charge(i))

    vj, vk = get_jk_amfi(mol, dm0)
    hso_2e = vj - vk * 1.5
    
    hso = (alpha2 / 2) * (hso_1e + hso_2e)
    return hso

def compute_hso(mol, dm0, amfi=True):  
    alpha2 = param.LIGHT_SPEED**(-2)
    #alpha2 = nist.ALPHA ** 2
    
    if amfi:
        hso = compute_hso_amfi(mol, dm0)
    
    else:
        hso_1e = mol.intor('int1e_prinvxp', comp=3)
        vj, vk = get_jk(mol, dm0)
        hso_2e = vj - vk * 1.5
        hso = (alpha2 / 2) * (hso_1e + hso_2e)
    return hso * 1j

def amfi_dm (mol, atomic_configuration=elements.CONFIGURATION):
    '''Generate AMFI density matrix, which is exactly like the
    "init_guess_by_atom" density matrix except that the orbitals
    of the atom hf's aren't optimized.

    Returns:
        dm : ndarray of shape (nao, nao)
            AMFI density matrix
    '''
    # TODO: refactor so that the discarded AO optimization doesn't happen
    # and waste cycles
    from pyscf.scf import atom_hf
    atm_scf = atom_hf.get_atm_nrhf(mol, atomic_configuration=atomic_configuration)
    aoslice = mol.aoslice_by_atom()
    atm_dms = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb not in atm_scf:
            symb = mol.atom_pure_symbol(ia)

        if symb in atm_scf:
            e_hf, e, c, occ = atm_scf[symb]
            #dm = np.dot(c*occ, c.conj().T)
            dm = np.diag (occ)
        else:  # symb's basis is not specified in the input
            nao_atm = aoslice[ia,3] - aoslice[ia,2]
            dm = np.zeros((nao_atm, nao_atm))

        atm_dms.append(dm)

    dm = linalg.block_diag(*atm_dms)

    if mol.cart:
        cart2sph = mol.cart2sph_coeff(normalized='sp')
        dm = reduce(np.dot, (cart2sph, dm, cart2sph.T))

    for k, v in atm_scf.items():
        logger.debug1(mol, 'Atom %s, E = %.12g', k, v[0])
    return dm
