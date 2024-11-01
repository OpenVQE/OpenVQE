from pyscf.scf.rohf import get_roothaan_fock
from pyscf.fci import cistring
from pyscf.mcscf import casci, casci_symm, df
from pyscf import symm, gto, scf, ao2mo, lib
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.my_pyscf.mcscf.addons import state_average_n_mix, get_h1e_zipped_fcisolver, las2cas_civec
from mrh.my_pyscf.mcscf import lasci_sync, _DFLASCI, lasscf_guess
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.df.sparse_df import sparsedf_array
from mrh.my_pyscf.mcscf import chkfile
from mrh.my_pyscf.mcscf.productstate import ImpureProductStateFCISolver
from mrh.util.la import matrix_svd_control_options
from itertools import combinations
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg
import numpy as np
import copy

def LASCI (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASCISymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASCINoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = density_fit (las, with_df = mf.with_df) 
    return las

def get_grad (las, mo_coeff=None, ci=None, ugg=None, h1eff_sub=None, h2eff_sub=None,
              veff=None, dm1s=None):
    '''Return energy gradient for orbital rotation and CI relaxation.

    Args:
        las : instance of :class:`LASCINoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        ugg : instance of :class:`LASCI_UnitaryGroupGenerators`
        h1eff_sub : list (length=nfrags) of list (length=nroots) of ndarray
            Contains effective one-electron Hamiltonians experienced by each fragment
            in each state
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        dm1s : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-RDM in the AO basis

    Returns:
        gorb : ndarray of shape (ugg.nvar_orb,)
            Orbital rotation gradients as a flat array
        gci : ndarray of shape (sum(ugg.ncsf_sub),)
            CI relaxation gradients as a flat array
        gx : ndarray
            Orbital rotation gradients for temporarily frozen orbitals in the "LASCI" problem
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if ugg is None: ugg = las.get_ugg (mo_coeff, ci)
    if dm1s is None: dm1s = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci)
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if veff is None:
        veff = las.get_veff (dm1s = dm1s.sum (0))
        veff = las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, ci=ci)
    if h1eff_sub is None: h1eff_sub = las.get_h1eff (mo_coeff, ci=ci, veff=veff,
                                                     h2eff_sub=h2eff_sub)

    gorb = get_grad_orb (las, mo_coeff=mo_coeff, ci=ci, h2eff_sub=h2eff_sub, veff=veff, dm1s=dm1s)
    gci = get_grad_ci (las, mo_coeff=mo_coeff, ci=ci, h1eff_sub=h1eff_sub, h2eff_sub=h2eff_sub,
                       veff=veff)

    idx = ugg.get_gx_idx ()
    gx = gorb[idx]
    gint = ugg.pack (gorb, gci)
    gorb = gint[:ugg.nvar_orb]
    gci = gint[ugg.nvar_orb:]
    return gorb, gci, gx.ravel ()

def get_grad_orb (las, mo_coeff=None, ci=None, h2eff_sub=None, veff=None, dm1s=None, hermi=-1):
    '''Return energy gradient for orbital rotation.

    Args:
        las : instance of :class:`LASCINoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        dm1s : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-RDM in the AO basis
        hermi : integer
            Control (anti-)symmetrization. 0 means to return the effective Fock matrix,
            F1 = h.D + g.d. -1 means to return the true orbital-rotation gradient, which is skew-
            symmetric: gorb = F1 - F1.T. +1 means to return the symmetrized effective Fock matrix,
            (F1 + F1.T) / 2. The factor of 2 difference between hermi=-1 and the other two options
            is intentional and necessary.

    Returns:
        gorb : ndarray of shape (nmo,nmo)
            Orbital rotation gradients as a square antihermitian array
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if dm1s is None: dm1s = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci)
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if veff is None:
        veff = las.get_veff (dm1s = dm1s.sum (0))
        veff = las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, ci=ci)
    nao, nmo = mo_coeff.shape
    ncore = las.ncore
    ncas = las.ncas
    nocc = las.ncore + las.ncas
    smo_cas = las._scf.get_ovlp () @ mo_coeff[:,ncore:nocc]
    smoH_cas = smo_cas.conj ().T

    # The orbrot part
    h1s = las.get_hcore ()[None,:,:] + veff
    f1 = h1s[0] @ dm1s[0] + h1s[1] @ dm1s[1]
    f1 = mo_coeff.conjugate ().T @ f1 @ las._scf.get_ovlp () @ mo_coeff
    # ^ I need the ovlp there to get dm1s back into its correct basis
    casdm2 = las.make_casdm2 (ci=ci)
    casdm1s = np.stack ([smoH_cas @ d @ smo_cas for d in dm1s], axis=0)
    casdm1 = casdm1s.sum (0)
    casdm2 -= np.multiply.outer (casdm1, casdm1)
    casdm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
    casdm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
    eri = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
    eri = lib.numpy_helper.unpack_tril (eri).reshape (nmo, ncas, ncas, ncas)
    f1[:,ncore:nocc] += np.tensordot (eri, casdm2, axes=((1,2,3),(1,2,3)))

    if hermi == -1:
        return f1 - f1.T
    elif hermi == 1:
        return .5*(f1+f1.T)
    elif hermi == 0:
        return f1
    else:
        raise ValueError ("kwarg 'hermi' must = -1, 0, or +1")

def get_grad_ci (las, mo_coeff=None, ci=None, h1eff_sub=None, h2eff_sub=None, veff=None):
    '''Return energy gradient for CI relaxation.

    Args:
        las : instance of :class:`LASCINoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        h1eff_sub : list (length=nfrags) of list (length=nroots) of ndarray
            Contains effective one-electron Hamiltonians experienced by each fragment
            in each state
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis

    Returns:
        gci : list (length=nfrags) of list (length=nroots) of ndarray
            CI relaxation gradients in the shape of CI vectors
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if h1eff_sub is None: h1eff_sub = las.get_h1eff (mo_coeff, ci=ci, veff=veff,
                                                     h2eff_sub=h2eff_sub)
    gci = []
    for isub, (fcibox, h1e, ci0, ncas, nelecas) in enumerate (zip (
            las.fciboxes, h1eff_sub, ci, las.ncas_sub, las.nelecas_sub)):
        eri_cas = las.get_h2eff_slice (h2eff_sub, isub, compact=8)
        max_memory = max(400, las.max_memory-lib.current_memory()[0])
        linkstrl = fcibox.states_gen_linkstr (ncas, nelecas, True)
        linkstr  = fcibox.states_gen_linkstr (ncas, nelecas, False)
        h2eff = fcibox.states_absorb_h1e(h1e, eri_cas, ncas, nelecas, .5)
        hc0 = fcibox.states_contract_2e(h2eff, ci0, ncas, nelecas, link_index=linkstrl)
        hc0 = [hc.ravel () for hc in hc0]
        ci0 = [c.ravel () for c in ci0]
        gci.append ([2.0 * (hc - c * (c.dot (hc))) for c, hc in zip (ci0, hc0)])
    return gci

def density_fit (las, auxbasis=None, with_df=None):
    ''' Here I ONLY need to attach the tag and the df object because I put conditionals in
        LASCINoSymm to make my life easier '''
    las_class = las.__class__
    if with_df is None:
        if (getattr(las._scf, 'with_df', None) and
            (auxbasis is None or auxbasis == las._scf.with_df.auxbasis)):
            with_df = las._scf.with_df
        else:
            with_df = df.DF(las.mol)
            with_df.max_memory = las.max_memory
            with_df.stdout = las.stdout
            with_df.verbose = las.verbose
            with_df.auxbasis = auxbasis
    class DFLASCI (las_class, _DFLASCI):
        def __init__(self, scf, ncas_sub, nelecas_sub):
            self.with_df = with_df
            self._keys = self._keys.union(['with_df'])
            las_class.__init__(self, scf, ncas_sub, nelecas_sub)
    new_las = DFLASCI (las._scf, las.ncas_sub, las.nelecas_sub)
    new_las.__dict__.update (las.__dict__)
    return new_las

def h1e_for_las (las, mo_coeff=None, ncas=None, ncore=None, nelecas=None, ci=None, ncas_sub=None,
                 nelecas_sub=None, veff=None, h2eff_sub=None, casdm1s_sub=None, casdm1frs=None):
    ''' Effective one-body Hamiltonians (plural) for a LASCI problem

    Args:
        las: a LASCI object

    Kwargs:
        mo_coeff: ndarray of shape (nao,nmo)
            Orbital coefficients ordered on the columns as: 
            core orbitals, subspace 1, subspace 2, ..., external orbitals
        ncas: integer
            As in PySCF's existing CASCI/CASSCF implementation
        nelecas: sequence of 2 integers
            As in PySCF's existing CASCI/CASSCF implementation
        ci: list (length=nfrags) of list (length=nroots) of ndarrays
            Contains CI vectors
        ncas_sub: ndarray of shape (nsub)
            Number of active orbitals in each subspace
        nelecas_sub: ndarray of shape (nsub,2)
            na, nb in each subspace
        veff: ndarray of shape (2, nao, nao)
            Contains spin-separated, state-averaged effective potential
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        casdm1s_sub : list (length=nfrags) of ndarrays
            Contains state-averaged, spin-separated 1-RDMs in the localized active subspaces
        casdm1frs : list (length=nfrags) of list (length=nroots) of ndarrays
            Contains spin-separated 1-RDMs for each state in the localized active subspaces

    Returns:
        h1e_fr: list (length=nfrags) of list (length=nroots) of ndarrays
            Spin-separated 1-body Hamiltonian operator for each fragment and state
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ncas is None: ncas = las.ncas
    if ncore is None: ncore = las.ncore
    if ncas_sub is None: ncas_sub = las.ncas_sub
    if nelecas_sub is None: nelecas_sub = las.nelecas_sub
    if ncore is None: ncore = las.ncore
    if ci is None: ci = las.ci
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if casdm1frs is None: casdm1frs = las.states_make_casdm1s_sub (ci=ci)
    if casdm1s_sub is None: casdm1s_sub = [np.einsum ('rsij,r->sij',dm,las.weights)
                                           for dm in casdm1frs]
    if veff is None:
        veff = las.get_veff (dm1s = las.make_rdm1 (mo_coeff=mo_coeff, ci=ci))
        veff = las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, ci=ci, casdm1s_sub=casdm1s_sub)

    # First pass: split by root  
    nocc = ncore + ncas
    nmo = mo_coeff.shape[-1]
    mo_cas = mo_coeff[:,ncore:nocc]
    moH_cas = mo_cas.conj ().T 
    h1e = moH_cas @ (las.get_hcore ()[None,:,:] + veff) @ mo_cas
    h1e_r = np.empty ((las.nroots, 2, ncas, ncas), dtype=h1e.dtype)
    h2e = lib.numpy_helper.unpack_tril (h2eff_sub.reshape (nmo*ncas,
        ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
    avgdm1s = np.stack ([linalg.block_diag (*[dm[spin] for dm in casdm1s_sub])
                         for spin in range (2)], axis=0)
    for state in range (las.nroots):
        statedm1s = np.stack ([linalg.block_diag (*[dm[state][spin] for dm in casdm1frs])
                               for spin in range (2)], axis=0)
        dm1s = statedm1s - avgdm1s 
        j = np.tensordot (dm1s, h2e, axes=((1,2),(2,3)))
        k = np.tensordot (dm1s, h2e, axes=((1,2),(2,1)))
        h1e_r[state] = h1e + j + j[::-1] - k


    # Second pass: split by fragment and subtract double-counting
    h1e_fr = []
    for ix, casdm1s_r in enumerate (casdm1frs):
        p = sum (las.ncas_sub[:ix])
        q = p + las.ncas_sub[ix]
        h1e = h1e_r[:,:,p:q,p:q]
        h2e = las.get_h2eff_slice (h2eff_sub, ix)
        j = np.tensordot (casdm1s_r, h2e, axes=((2,3),(2,3)))
        k = np.tensordot (casdm1s_r, h2e, axes=((2,3),(2,1)))
        h1e_fr.append (h1e - j - j[:,::-1] + k)

    return h1e_fr

def get_fock (las, mo_coeff=None, ci=None, eris=None, casdm1s=None, verbose=None, veff=None,
              dm1s=None):
    ''' f_pq = h_pq + (g_pqrs - g_psrq/2) D_rs, AO basis
    Note the difference between this and h1e_for_las: h1e_for_las only has
    JK terms from electrons outside the "current" active subspace; get_fock
    includes JK from all electrons. This is also NOT the "generalized Fock matrix"
    of orbital gradients (but it can be used in calculating those if you do a
    semi-cumulant decomposition).
    The "eris" kwarg does not do anything and is retained only for backwards
    compatibility (also why I don't just call las.make_rdm1) '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if dm1s is None:
        if casdm1s is None: casdm1s = las.make_casdm1s (ci=ci)
        mo_cas = mo_coeff[:,las.ncore:][:,:las.ncas]
        moH_cas = mo_cas.conjugate ().T
        mo_core = mo_coeff[:,:las.ncore]
        moH_core = mo_core.conjugate ().T
        dm1s = [(mo_core @ moH_core) + (mo_cas @ d @ moH_cas) for d in list(casdm1s)]
    if veff is not None:
        fock = las.get_hcore()[None,:,:] + veff
        return get_roothaan_fock (fock, dm1s, las._scf.get_ovlp ())
    dm1 = dm1s[0] + dm1s[1]
    if isinstance (las, _DFLASCI):
        vj, vk = las.with_df.get_jk(dm1, hermi=1)
    else:
        vj, vk = las._scf.get_jk(las.mol, dm1, hermi=1)
    fock = las.get_hcore () + vj - (vk/2)
    return fock

def _eig_inactive_virtual (las, fock, orbsym=None):
    '''Generate the unitary matrix canonicalizing the inactive and virtual orbitals only.

    Args:
        las : object of :class:`LASCINoSymm`
        fock : ndarray of shape (nmo,nmo)
            Contains Fock matrix in MO basis

    Kwargs:
        orbsym : list of length nmo
        umat : ndarray of shape (nmo, nmo)

    Returns:
        ene : ndarray of shape (nmo,)
        umat : ndarray of shape (nmo, nmo)'''
    nmo = fock.shape[0]
    ncore = las.ncore
    nocc = ncore + las.ncas
    ene = np.zeros (nmo)
    umat = np.eye (nmo)
    if ncore:
        orbsym_i = None if orbsym is None else orbsym[:ncore]
        fock_i = fock[:ncore,:ncore]
        ene[:ncore], umat[:ncore,:ncore] = las._eig (fock_i, 0, 0, orbsym_i)
    if nmo-nocc:
        orbsym_i = None if orbsym is None else orbsym[nocc:]
        fock_i = fock[nocc:,nocc:]
        ene[nocc:], umat[nocc:,nocc:] = las._eig (fock_i, 0, 0, orbsym_i)
    return ene, umat

def canonicalize (las, mo_coeff=None, ci=None, casdm1fs=None, natorb_casdm1=None, veff=None,
                  h2eff_sub=None, orbsym=None):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci

    # In-place safety
    mo_coeff = mo_coeff.copy ()
    ci = copy.deepcopy (ci)

    # Temporary lroots safety
    # The desired behavior is that the inactive and external orbitals should
    # be canonicalized according to the density matrix used for orbital optimization
    # TODO: once orbital optimization with lroots is enabled, change this behavior
    # TODO: possibly move this logic to the make_casdm* functions
    if casdm1fs is None:
        ci_dm = []
        for i in range (len (ci)):
            ci_i = []
            for j in range (len (ci[i])):
                if ci[i][j].ndim>2:
                    ci_i.append (ci[i][j][0])
                else:
                    ci_i.append (ci[i][j])
            ci_dm.append (ci_i)
        casdm1fs = las.make_casdm1s_sub (ci=ci_dm)

    nao, nmo = mo_coeff.shape
    ncore = las.ncore
    nocc = ncore + las.ncas
    ncas_sub = las.ncas_sub
    nelecas_sub = las.nelecas_sub

    # Passing casdm1 or lasdm1 only affects the canonicalization of the active orbitals
    umat = np.zeros_like (mo_coeff)
    casdm1s = np.stack ([linalg.block_diag (*[dm[0] for dm in casdm1fs]),
                         linalg.block_diag (*[dm[1] for dm in casdm1fs])], axis=0)
    fock = mo_coeff.conjugate ().T @ las.get_fock (mo_coeff=mo_coeff, casdm1s=casdm1s, veff=veff)
    fock = fock @ mo_coeff
    if natorb_casdm1 is None: # State-average natural orbitals by default
        natorb_casdm1 = casdm1s.sum (0)

    # Inactive-inactive and virtual-virtual
    ene, umat = _eig_inactive_virtual (las, fock, orbsym=orbsym)
    idx = np.arange (nmo, dtype=int)
    if ncore: idx[:ncore] = idx[:ncore][np.argsort (ene[:ncore])]
    if nmo-nocc: idx[nocc:] = idx[nocc:][np.argsort (ene[nocc:])]
    umat = umat[:,idx]
    if orbsym is not None: orbsym = orbsym[idx]
    # Active-active
    check_diag = natorb_casdm1.copy ()
    for ix, ncas in enumerate (ncas_sub):
        i = sum (ncas_sub[:ix])
        j = i + ncas
        check_diag[i:j,i:j] = 0.0
    if np.amax (np.abs (check_diag)) < 1e-8:
        # No off-diagonal RDM elements -> extra effort to prevent diagonalizer from breaking frags
        for isub, (ncas, nelecas) in enumerate (zip (ncas_sub, nelecas_sub)):
            i = sum (ncas_sub[:isub])
            j = i + ncas
            dm1 = natorb_casdm1[i:j,i:j]
            i += ncore
            j += ncore
            orbsym_i = None if orbsym is None else orbsym[i:j]
            occ, umat[i:j,i:j] = las._eig (dm1, 0, 0, orbsym_i)
            idx = np.argsort (occ)[::-1]
            umat[i:j,i:j] = umat[i:j,i:j][:,idx]
            if orbsym_i is not None: orbsym[i:j] = orbsym[i:j][idx]
            if ci is not None:
                fcibox = las.fciboxes[isub]
                ci[isub] = fcibox.states_transform_ci_for_orbital_rotation (
                    ci[isub], ncas, nelecas, umat[i:j,i:j])
    else: # You can't get proper LAS-type CI vectors w/out active space fragmentation
        ci = None 
        orbsym_cas = None if orbsym is None else orbsym[ncore:nocc]
        occ, umat[ncore:nocc,ncore:nocc] = las._eig (natorb_casdm1, 0, 0, orbsym_cas)
        idx = np.argsort (occ)[::-1]
        umat[ncore:nocc,ncore:nocc] = umat[ncore:nocc,ncore:nocc][:,idx]
        if orbsym_cas is not None: orbsym[ncore:nocc] = orbsym[ncore:nocc][idx]

    # Final
    mo_occ = np.zeros (nmo, dtype=natorb_casdm1.dtype)
    if ncore: mo_occ[:ncore] = 2
    ucas = umat[ncore:nocc,ncore:nocc]
    mo_occ[ncore:nocc] = ((natorb_casdm1 @ ucas) * ucas).sum (0)
    mo_ene = ((fock @ umat) * umat.conjugate ()).sum (0)
    mo_ene[ncore:][:sum (ncas_sub)] = 0.0
    mo_coeff = mo_coeff @ umat
    if orbsym is not None:
        '''
        print ("This is the second call to label_orb_symm inside of canonicalize") 
        orbsym = symm.label_orb_symm (las.mol, las.mol.irrep_id,
                                      las.mol.symm_orb, mo_coeff,
                                      s=las._scf.get_ovlp ())
        #mo_coeff = las.label_symmetry_(mo_coeff)
        '''
        mo_coeff = lib.tag_array (mo_coeff, orbsym=orbsym)
    if h2eff_sub is not None:
        h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub.reshape (nmo*las.ncas, -1))
        h2eff_sub = h2eff_sub.reshape (nmo, las.ncas, las.ncas, las.ncas)
        h2eff_sub = np.tensordot (umat, h2eff_sub, axes=((0),(0)))
        h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(1))).transpose (1,0,2,3)
        h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(2))).transpose (1,2,0,3)
        h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(3))).transpose (1,2,3,0)
        h2eff_sub = h2eff_sub.reshape (nmo*las.ncas, las.ncas, las.ncas)
        h2eff_sub = lib.numpy_helper.pack_tril (h2eff_sub).reshape (nmo, -1)
    return mo_coeff, mo_ene, mo_occ, ci, h2eff_sub

def get_init_guess_ci (las, mo_coeff=None, h2eff_sub=None, ci0=None):
    # TODO: come up with a better algorithm? This might be working better than what I had before
    # but it omits inter-active Coulomb and exchange interactions altogether. Is there a
    # non-outer-product algorithm for finding the lowest-energy single product of CSFs?
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci0 is None: ci0 = [[None for i in range (las.nroots)] for j in range (las.nfrags)]
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    nmo = mo_coeff.shape[-1]
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    dm1_core= 2 * mo_coeff[:,:ncore] @ mo_coeff[:,:ncore].conj ().T
    h1e_ao = las._scf.get_fock (dm=dm1_core)
    eri_cas = lib.numpy_helper.unpack_tril (h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2))
    eri_cas = eri_cas.reshape (nmo, ncas, ncas, ncas)
    eri_cas = eri_cas[ncore:nocc]
    for ix, (fcibox, norb, nelecas) in enumerate (zip (las.fciboxes,las.ncas_sub,las.nelecas_sub)):
        i = sum (las.ncas_sub[:ix])
        j = i + norb
        mo = mo_coeff[:,ncore+i:ncore+j]
        moH = mo.conj ().T
        h1e = moH @ h1e_ao @ mo
        h1e = [h1e, h1e]
        eri = eri_cas[i:j,i:j,i:j,i:j]
        for iy, solver in enumerate (fcibox.fcisolvers):
            nelec = fcibox._get_nelec (solver, nelecas)
            ndet = tuple ([cistring.num_strings (norb, n) for n in nelec])
            if isinstance (ci0[ix][iy], np.ndarray) and ci0[ix][iy].size==ndet[0]*ndet[1]: continue
            if hasattr (mo_coeff, 'orbsym'):
                solver.orbsym = mo_coeff.orbsym[ncore+i:ncore+j]
            hdiag_csf = solver.make_hdiag_csf (h1e, eri, norb, nelec)
            ci0[ix][iy] = solver.get_init_guess (norb, nelec, solver.nroots, hdiag_csf)[0]
    return ci0

def get_space_info (las):
    ''' Retrieve the quantum numbers defining the states of a LASSCF calculation '''
    nfrags, nroots = las.nfrags, las.nroots
    charges = np.zeros ((nroots, nfrags), dtype=np.int32)
    wfnsyms, spins, smults = charges.copy (), charges.copy (), charges.copy ()
    for ifrag, fcibox in enumerate (las.fciboxes):
     for iroot, solver in enumerate (fcibox.fcisolvers):
        nelec = fcibox._get_nelec (solver, las.nelecas_sub[ifrag])
        charges[iroot,ifrag] = np.sum (las.nelecas_sub[ifrag]) - np.sum (nelec)
        spins[iroot,ifrag] = nelec[0]-nelec[1]
        smults[iroot,ifrag] = solver.smult
        try:
            wfnsyms[iroot,ifrag] = solver.wfnsym or 0
        except ValueError as e:
            wfnsyms[iroot,ifrag] = symm.irrep_name2id (las.mol.groupname, solver.wfnsym)
    return charges, spins, smults, wfnsyms
   
def assert_no_duplicates (las, tab=None):
    log = lib.logger.new_logger (las, las.verbose)
    if tab is None: tab = np.stack (get_space_info (las), axis=-1)
    tab_uniq, uniq_idx, uniq_inv, uniq_cnts = np.unique (tab, return_index=True,
        return_inverse=True, return_counts=True, axis=0)
    idx_dupe = uniq_cnts>1
    try:
        err_str = ('LAS state basis has duplicates; details in logfile for '
                   'verbose >= INFO (4) [more details for verbose > INFO].\n'
                   '(Disable this assertion by passing assert_no_dupes=False '
                   'to the kernel, lasci, and state_average(_) functions.)')
        assert (~np.any (idx_dupe)), err_str
    except AssertionError as e:
        dupe_idx = uniq_idx[idx_dupe]
        dupe_cnts = uniq_cnts[idx_dupe]
        for i, (ix, cnt, col) in enumerate (zip (uniq_idx, uniq_cnts, tab_uniq)):
            if cnt==1: continue
            log.info ('State %d appears %d times', ix, cnt)
            idx_thisdupe = np.where (uniq_inv==i)[0]
            row = col.T
            log.debug ('As states {}'.format (idx_thisdupe))
            log.debug ('Charges = {}'.format (row[0]))
            log.debug ('2M_S = {}'.format (row[1]))
            log.debug ('2S+1 = {}'.format (row[2]))
            log.debug ('Wfnsyms = {}'.format (row[3]))
        raise e from None

def state_average_(las, weights=[0.5,0.5], charges=None, spins=None,
        smults=None, wfnsyms=None, assert_no_dupes=True):
    ''' Transform LASCI/LASSCF object into state-average LASCI/LASSCF 

    Args:
        las: LASCI/LASSCF instance

    Kwargs:
        weights: list of float; required
            E_SA = sum_i weights[i] E[i] is used to optimize the orbitals
        charges: 2d ndarray or nested list of integers
        spins: 2d ndarray or nested list of integers
            For the jth fragment in the ith state,
            neleca = (sum(las.nelecas_sub[j]) - charges[i][j] + spins[i][j]) // 2
            nelecb = (sum(las.nelecas_sub[j]) - charges[i][j] - spins[i][j]) // 2
            Defaults to
            charges[i][j] = 0
            spins[i][j] = las.nelecas_sub[j][0] - las.nelecas_sub[j][1]
        smults: 2d ndarray or nested list of integers
            For the jth fragment in the ith state,
            smults[i][j] = (2*s)+1
            where "s" is the total spin quantum number,
            S^2|j,i> = s*(s+1)|j,i>
            Defaults to
            smults[i][j] = abs (spins[i][j]) + 1
        wfnsyms: 2d ndarray or nested list of integers or strings
            For the jth fragment of the ith state,
            wfnsyms[i][j]
            identifies the point-group irreducible representation
            Defaults to all zeros (i.e., the totally-symmetric irrep)

    Returns:
        las: LASCI/LASSCF instance
            The first positional argument, modified in-place into a
            state-averaged LASCI/LASSCF instance.

    '''
    old_states = np.stack (get_space_info (las), axis=-1)
    nroots = len (weights)
    nfrags = las.nfrags
    if charges is None: charges = np.zeros ((nroots, nfrags), dtype=np.int32)
    if wfnsyms is None: wfnsyms = np.zeros ((nroots, nfrags), dtype=np.int32)
    if spins is None: spins = np.asarray ([[n[0]-n[1] for n in las.nelecas_sub] for i in weights]) 
    if smults is None: smults = np.abs (spins)+1 

    charges = np.asarray (charges)
    wfnsyms = np.asarray (wfnsyms)
    spins = np.asarray (spins)
    smults = np.asarray (smults)
    if np.issubsctype (wfnsyms.dtype, np.str_):
        wfnsyms_str = wfnsyms
        wfnsyms = np.zeros (wfnsyms_str.shape, dtype=np.int32)
        for ix, wfnsym in enumerate (wfnsyms_str.flat):
            try:
                wfnsyms.flat[ix] = symm.irrep_name2id (las.mol.groupname, wfnsym)
            except (TypeError, KeyError) as e:
                wfnsyms.flat[ix] = int (wfnsym)
    if nfrags == 1:
        charges = np.atleast_2d (np.squeeze (charges)).T
        wfnsyms = np.atleast_2d (np.squeeze (wfnsyms)).T
        spins = np.atleast_2d (np.squeeze (spins)).T
        smults = np.atleast_2d (np.squeeze (smults)).T
    new_states = np.stack ([charges, spins, smults, wfnsyms], axis=-1)
    if assert_no_dupes: assert_no_duplicates (las, tab=new_states)

    las.fciboxes = [get_h1e_zipped_fcisolver (state_average_n_mix (
        las, [csf_solver (las.mol, smult=s2p1).set (charge=c, spin=m2, wfnsym=ir)
              for c, m2, s2p1, ir in zip (c_r, m2_r, s2p1_r, ir_r)], weights).fcisolver)
        for c_r, m2_r, s2p1_r, ir_r in zip (charges.T, spins.T, smults.T, wfnsyms.T)]
    las.e_states = np.zeros (nroots)
    las.nroots = nroots
    las.weights = weights

    if las.ci is not None:
        log = lib.logger.new_logger(las, las.verbose)
        log.debug (("lasci.state_average: Cached CI vectors may be present.\n"
                    "Looking for matches between old and new LAS states..."))
        ci0 = [[None for i in range (nroots)] for j in range (nfrags)]
        new_states = np.stack ([charges, spins, smults, wfnsyms],
            axis=-1).reshape (nroots, nfrags*4)
        old_states = old_states.reshape (-1, nfrags*4)
        for iroot, row in enumerate (old_states):
            idx = np.all (new_states == row[None,:], axis=1)
            if np.count_nonzero (idx) == 1:
                jroot = np.where (idx)[0][0] 
                log.debug ("Old state {} -> New state {}".format (iroot, jroot))
                for ifrag in range (nfrags):
                    ci0[ifrag][jroot] = las.ci[ifrag][iroot]
            elif np.count_nonzero (idx) > 1:
                raise RuntimeError ("Duplicate states specified?\n{}".format (idx))
        las.ci = ci0 
    las.converged = False
    return las

@lib.with_doc(''' A version of lasci.state_average_ that creates a copy instead of modifying the 
    LASCI/LASSCF method instance in place.

    See lasci.state_average_ docstring below:\n\n''' + state_average_.__doc__)
def state_average (las, weights=[0.5,0.5], charges=None, spins=None,
        smults=None, wfnsyms=None, assert_no_dupes=True):
    new_las = las.__class__(las._scf, las.ncas_sub, las.nelecas_sub)
    new_las.__dict__.update (las.__dict__)
    new_las.mo_coeff = las.mo_coeff.copy ()
    if getattr (las.mo_coeff, 'orbsym', None) is not None:
        new_las.mo_coeff = lib.tag_array (new_las.mo_coeff,
            orbsym=las.mo_coeff.orbsym)
    new_las.ci = None
    if las.ci is not None:
        new_las.ci = [[c2.copy () if isinstance (c2, np.ndarray) else None
            for c2 in c1] for c1 in las.ci]
    return state_average_(new_las, weights=weights, charges=charges, spins=spins,
        smults=smults, wfnsyms=wfnsyms, assert_no_dupes=assert_no_dupes)

def run_lasci (las, mo_coeff=None, ci0=None, lroots=None, lweights=None, verbose=0, assert_no_dupes=False):
    '''Self-consistently optimize the CI vectors of a LAS state with 
    frozen orbitals using a fixed-point algorithm. "lasci_" (with the
    trailing underscore) sets self.mo_coeff from the kwarg if it is passed;
    "lasci" (without the trailing underscore) leaves self.mo_coeff unchanged.

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            MO coefficients; defaults to self.mo_coeff
        ci0 : list (length nfrags) of list (length nroots) of ndarrays
            Contains CI vectors for initial guess
        lroots : ndarray of shape (nfrags,nroots)
            Number of local roots in each fragment for each global state. 
            The corresponding local weights are set to [1,0,0,0,...].
        lweights : list of length nfrags of list of length nroots of sequence
            Weights of local roots in each fragment for each global state.
            Passing lweights is incompatible with passing lroots. Defaults
            to, i.e., np.ones (las.nfrags, las.nroots, 1).tolist ()
        verbose : integer
            See pyscf.lib.logger.
        assert_no_dupes : logical
            If True, checks state list for duplicate states

    Returns:
        converged : list of length nroots of logical
            Stores whether the calculation for each state successfully converged
        e_tot : float
            (State-averaged) total energy
        e_states : list of length nroots
            List of each state energy
        e_cas : list of length nroots
            List of the CAS space energy of each state
        ci : list (length nfrags) of list (length nroots) of ndarrays
            Contains optimized CI vectors
    '''
    if assert_no_dupes: assert_no_duplicates (las)
    if lroots is not None and lweights is not None:
        raise RuntimeError ("lroots sets lweights: pass either or none but not both")
    elif lweights is None:
        if lroots is None: lroots = np.ones ((las.nfrags, las.nroots), dtype=int)
        lweights = []
        for i in range (las.nfrags):
            lwi = []
            for j in range (las.nroots):
                lwij = np.zeros (lroots[i,j])
                lwij[0] = 1
                lwi.append (lwij)
            lweights.append (lwi)
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    ncas_sub = las.ncas_sub
    nelecas_sub = las.nelecas_sub
    orbsym = getattr (mo_coeff, 'orbsym', None)
    if orbsym is not None: orbsym=orbsym[ncore:nocc]
    elif isinstance (las, LASCISymm):
        mo_coeff = las.label_symmetry_(mo_coeff)
        orbsym = mo_coeff.orbsym[ncore:nocc]
    log = lib.logger.new_logger (las, verbose)

    h1eff, energy_core = las.h1e_for_cas (mo_coeff=mo_coeff,
        ncas=las.ncas, ncore=las.ncore)
    h2eff = las.get_h2eff (mo_coeff) 
    if (ci0 is None or any ([c is None for c in ci0]) or
            any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
        ci0 = las.get_init_guess_ci (mo_coeff, h2eff, ci0)
    eri_cas = lib.numpy_helper.unpack_tril (
            h2eff.reshape (nmo*ncas, ncas*(ncas+1)//2)).reshape (nmo, ncas,
            ncas, ncas)[ncore:nocc]

    e_cas = np.empty (las.nroots)
    e_states = np.empty (las.nroots)
    ci1 = [[None for c2 in c1] for c1 in ci0]
    converged = []
    t = (lib.logger.process_clock(), lib.logger.perf_counter())
    e_lexc = [[None for i in range (las.nroots)] for j in range (las.nfrags)]
    for state in range (las.nroots):
        fcisolvers = [b.fcisolvers[state] for b in las.fciboxes]
        ci0_i = [c[state] for c in ci0]
        solver = ImpureProductStateFCISolver (fcisolvers, stdout=las.stdout,
            lweights=[l[state] for l in lweights], verbose=verbose)
        # TODO: better handling of CSF symmetry quantum numbers in general
        for ix, s in enumerate (solver.fcisolvers):
            i = sum (ncas_sub[:ix])
            j = i + ncas_sub[ix]
            if orbsym is not None: s.orbsym = orbsym[i:j]
            s.norb = ncas_sub[ix]
            s.nelec = solver._get_nelec (s, nelecas_sub[ix])
            s.check_transformer_cache ()
        conv, e_i, ci_i = solver.kernel (h1eff, eri_cas, ncas_sub, nelecas_sub,
            ecore=0, ci0=ci0_i, orbsym=orbsym, conv_tol_grad=las.conv_tol_grad,
            conv_tol_self=las.conv_tol_self, max_cycle_macro=las.max_cycle_macro)
        e_cas[state] = e_i
        e_states[state] = e_i + energy_core
        for frag, s in enumerate (solver.fcisolvers):
            e_loc = np.atleast_1d (getattr (s, 'e_states', e_i))
            e_lexc[frag][state] = e_loc - e_i
        for c1, c2, s, no, ne in zip (ci1, ci_i, solver.fcisolvers, ncas_sub, nelecas_sub):
            ne = solver._get_nelec (s, ne)
            ndeta, ndetb = [cistring.num_strings (no, n) for n in ne]
            shape = [s.nroots, ndeta, ndetb] if s.nroots>1 else [ndeta, ndetb]
            c1[state] = np.asarray (c2).reshape (*shape)
        if not conv: log.warn ('State %d LASCI not converged!', state)
        converged.append (conv)
        t = log.timer ('State {} LASCI'.format (state), *t)

    e_tot = np.dot (las.weights, e_states)
    return converged, e_tot, e_states, e_cas, e_lexc, ci1

class LASCINoSymm (casci.CASCI):

    def __init__(self, mf, ncas, nelecas, ncore=None, spin_sub=None, frozen=None, **kwargs):
        if isinstance(ncas,int):
            ncas = [ncas]
        ncas_tot = sum (ncas)
        nel_tot = [0, 0]
        new_nelecas = []
        for ix, nel in enumerate (nelecas):
            if isinstance (nel, (int, np.integer)):
                nb = nel // 2
                na = nb + (nel % 2)
            else:
                na, nb = nel
            new_nelecas.append ((na, nb))
            nel_tot[0] += na
            nel_tot[1] += nb
        nelecas = new_nelecas
        self.nroots = 1
        super().__init__(mf, ncas=ncas_tot, nelecas=nel_tot, ncore=ncore)
        self.chkfile = self._scf.chkfile
        if spin_sub is None: spin_sub = [1 + abs(ne[0]-ne[1]) for ne in nelecas]
        self.ncas_sub = np.asarray (ncas)
        self.nelecas_sub = np.asarray (nelecas)
        assert (len (self.nelecas_sub) == self.nfrags)
        self.frozen = frozen
        self.conv_tol_grad = 1e-4
        self.conv_tol_self = 1e-10
        self.ah_level_shift = 1e-8
        self.max_cycle_macro = 50
        self.max_cycle_micro = 5
        keys = set(('e_states', 'fciboxes', 'nroots', 'weights', 'ncas_sub', 'nelecas_sub',
                    'conv_tol_grad', 'conv_tol_self', 'max_cycle_macro', 'max_cycle_micro',
                    'ah_level_shift', 'states_converged', 'chkfile', 'e_lexc'))
        self._keys = set(self.__dict__.keys()).union(keys)
        self.fciboxes = []
        if isinstance(spin_sub,int):
            self.fciboxes.append(self._init_fcibox(spin_sub,self.nelecas_sub[0]))
        else:
            assert (len (spin_sub) == self.nfrags)
            for smult, nel in zip (spin_sub, self.nelecas_sub):
                self.fciboxes.append (self._init_fcibox (smult, nel)) 
        self.weights = [1.0]
        self.e_states = [0.0]
        self.e_lexc = [[np.array ([0]),],]

    def _init_fcibox (self, smult, nel): 
        s = csf_solver (self.mol, smult=smult)
        s.spin = nel[0] - nel[1] 
        return get_h1e_zipped_fcisolver (state_average_n_mix (self, [s], [1.0]).fcisolver)

    @property
    def nfrags (self): return len (self.ncas_sub)

    def get_mo_slice (self, idx, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        mo = mo_coeff[:,self.ncore:]
        for offs in self.ncas_sub[:idx]:
            mo = mo[:,offs:]
        mo = mo[:,:self.ncas_sub[idx]]
        return mo

    def ao2mo (self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nao, nmo = mo_coeff.shape
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas
        mo_cas = mo_coeff[:,ncore:nocc]
        mo = [mo_coeff, mo_cas, mo_cas, mo_cas]
        if getattr (self, 'with_df', None) is not None:
            # Store intermediate with one contracted ao index for faster calculation of exchange!
            bPmn = sparsedf_array (self.with_df._cderi)
            bmuP = bPmn.contract1 (mo_cas)
            buvP = np.tensordot (mo_cas.conjugate (), bmuP, axes=((0),(0)))
            eri_muxy = np.tensordot (bmuP, buvP, axes=((2),(2)))
            eri = np.tensordot (mo_coeff.conjugate (), eri_muxy, axes=((0),(0)))
            eri = lib.pack_tril (eri.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            eri = lib.tag_array (eri, bmPu=bmuP.transpose (0,2,1))
            if self.verbose > lib.logger.DEBUG:
                eri_comp = self.with_df.ao2mo (mo, compact=True)
                lib.logger.debug(self,"CDERI two-step error: {}".format(linalg.norm(eri-eri_comp)))
        elif getattr (self._scf, '_eri', None) is not None:
            eri = ao2mo.incore.general (self._scf._eri, mo, compact=True)
        else:
            eri = ao2mo.outcore.general_iofree (self.mol, mo, compact=True)
        if eri.shape != (nmo,ncas*ncas*(ncas+1)//2):
            try:
                eri = eri.reshape (nmo, ncas*ncas*(ncas+1)//2)
            except ValueError as e:
                assert (nmo == ncas), str (e)
                eri = ao2mo.restore ('2kl', eri, nmo).reshape (nmo, ncas*ncas*(ncas+1)//2)
        return eri

    def get_h2eff_slice (self, h2eff, idx, compact=None):
        ncas_cum = np.cumsum ([0] + self.ncas_sub.tolist ())
        i = ncas_cum[idx] 
        j = ncas_cum[idx+1]
        ncore = self.ncore
        nocc = ncore + self.ncas
        eri = h2eff[ncore:nocc,:].reshape (self.ncas*self.ncas, -1)
        ix_i, ix_j = np.tril_indices (self.ncas)
        eri = eri[(ix_i*self.ncas)+ix_j,:]
        eri = ao2mo.restore (1, eri, self.ncas)[i:j,i:j,i:j,i:j]
        if compact: eri = ao2mo.restore (compact, eri, j-i)
        return eri

    get_h1eff = get_h1las = h1e_for_las = h1e_for_las
    get_h2eff = ao2mo
    '''
    def get_h2eff (self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if isinstance (self, _DFLASCI):
            mo_cas = mo_coeff[:,self.ncore:][:,:self.ncas]
            return self.with_df.ao2mo (mo_cas)
        return self.ao2mo (mo_coeff)
    '''

    get_fock = get_fock
    get_grad = get_grad
    get_grad_orb = get_grad_orb
    get_grad_ci = get_grad_ci
    _hop = lasci_sync.LASCI_HessianOperator
    _kern = lasci_sync.kernel
    def get_hop (self, mo_coeff=None, ci=None, ugg=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ugg is None: ugg = self.get_ugg ()
        return self._hop (self, ugg, mo_coeff=mo_coeff, ci=ci, **kwargs)
    canonicalize = canonicalize

    def _finalize(self):
        log = lib.logger.new_logger (self, self.verbose)
        nroots_prt = len (self.e_states)
        if self.verbose <= lib.logger.INFO:
            nroots_prt = min (nroots_prt, 100)
        if nroots_prt < len (self.e_states):
            log.info (("Printing a maximum of 100 state energies;"
                       " increase self.verbose to see them all"))
        if nroots_prt > 1:
            log.info ("LASCI state-average energy = %.15g", self.e_tot)
            for i, e in enumerate (self.e_states[:nroots_prt]):
                log.info ("LASCI state %d energy = %.15g", i, e)
        else:
            log.info ("LASCI energy = %.15g", self.e_tot)
        return


    def kernel(self, mo_coeff=None, ci0=None, casdm0_fr=None, conv_tol_grad=None,
            assert_no_dupes=False, verbose=None, _kern=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None: ci0 = self.ci
        if verbose is None: verbose = self.verbose
        if conv_tol_grad is None: conv_tol_grad = self.conv_tol_grad
        if _kern is None: _kern = self._kern
        log = lib.logger.new_logger(self, verbose)

        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        self.dump_flags(log)

        for fcibox in self.fciboxes:
            fcibox.verbose = self.verbose
            fcibox.stdout = self.stdout
            fcibox.nroots = self.nroots
            fcibox.weights = self.weights
        # TODO: local excitations and locally-impure states in LASSCF kernel
        do_warn=False
        if ci0 is not None:
            for i, ci0_i in enumerate (ci0):
                if ci0_i is None: continue
                for j, ci0_ij in enumerate (ci0_i):
                    if ci0_ij is None: continue
                    if np.asarray (ci0_ij).ndim>2:
                        do_warn=True
                        ci0_i[j] = ci0_ij[0]
        if do_warn: log.warn ("Discarding all but the first root of guess CI vectors!")

        self.converged, self.e_tot, self.e_states, self.mo_energy, self.mo_coeff, self.e_cas, \
                self.ci, h2eff_sub, veff = _kern(mo_coeff=mo_coeff, ci0=ci0, verbose=verbose, \
                casdm0_fr=casdm0_fr, conv_tol_grad=conv_tol_grad, assert_no_dupes=assert_no_dupes)

        self._finalize ()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy, h2eff_sub, veff

    def states_make_casdm1s_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Spin-separated 1-RDMs in the MO basis for each subspace in sequence '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None:
            return [np.zeros ((self.nroots,2,ncas,ncas)) for ncas in ncas_sub] 
        casdm1s = []
        for fcibox, ci_i, ncas, nelecas in zip (self.fciboxes, ci, ncas_sub, nelecas_sub):
            if ci_i is None:
                dm1a = dm1b = np.zeros ((ncas, ncas))
            else: 
                dm1a, dm1b = fcibox.states_make_rdm1s (ci_i, ncas, nelecas)
            casdm1s.append (np.stack ([dm1a, dm1b], axis=1))
        return casdm1s

    def make_casdm1s_sub (self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm1frs=None, w=None, **kwargs):
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        if w is None: w = self.weights
        return [np.einsum ('rspq,r->spq', dm1, w) for dm1 in casdm1frs]

    def states_make_casdm1s (self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm1frs=None, **kwargs):
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        return np.stack ([np.stack ([linalg.block_diag (*[dm1rs[iroot][ispin] 
                                                          for dm1rs in casdm1frs])
                                     for ispin in (0, 1)], axis=0)
                          for iroot in range (self.nroots)], axis=0)

    def states_make_casdm2_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Spin-separated 1-RDMs in the MO basis for each subspace in sequence '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        casdm2 = []
        for fcibox, ci_i, ncas, nel in zip (self.fciboxes, ci, ncas_sub, nelecas_sub):
            casdm2.append (fcibox.states_make_rdm12 (ci_i, ncas, nel)[-1])
        return casdm2

    def make_casdm2_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, casdm2fr=None, **kwargs):
        if casdm2fr is None: casdm2fr = self.states_make_casdm2_sub (ci=ci, ncas_sub=ncas_sub,
            nelecas_sub=nelecas_sub, **kwargs)
        return [np.einsum ('rijkl,r->ijkl', dm2, box.weights)
                for dm2, box in zip (casdm2fr, self.fciboxes)]

    def states_make_rdm1s (self, mo_coeff=None, ci=None, ncas_sub=None,
            nelecas_sub=None, casdm1rs=None, casdm1frs=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1rs is None: casdm1rs = self.states_make_casdm1s (ci=ci, 
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, casdm1frs=casdm1frs, 
            **kwargs)
        mo_core = mo_coeff[:,:self.ncore]
        mo_cas = mo_coeff[:,self.ncore:][:,:self.ncas]
        dm1rs = np.tensordot (mo_cas.conj (), np.dot (casdm1rs, mo_cas.conj ().T), axes=((1),(2)))
        dm1rs = dm1rs.transpose (1,2,0,3)
        dm1rs += (mo_core @ mo_core.conj ().T)[None,None,:,:]
        return dm1rs

    def make_rdm1s_sub (self, mo_coeff=None, ci=None, ncas_sub=None,
            nelecas_sub=None, include_core=False, casdm1s_sub=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1s_sub is None: casdm1s_sub = self.make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        ''' Same as make_casdm1s_sub, but in the ao basis '''
        rdm1s = []
        for idx, casdm1s in enumerate (casdm1s_sub):
            mo = self.get_mo_slice (idx, mo_coeff=mo_coeff)
            moH = mo.conjugate ().T
            rdm1s.append (np.tensordot (mo, np.dot (casdm1s,moH), axes=((1),(1))).transpose(1,0,2))
        if include_core and self.ncore:
            mo_core = mo_coeff[:,:self.ncore]
            moH_core = mo_core.conjugate ().T
            dm_core = mo_core @ moH_core
            rdm1s = [np.stack ([dm_core, dm_core], axis=0)] + rdm1s
        rdm1s = np.stack (rdm1s, axis=0)
        return rdm1s

    def make_rdm1_sub (self, **kwargs):
        return self.make_rdm1s_sub (**kwargs).sum (1)

    def make_rdm1s (self, mo_coeff=None, ncore=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        mo = mo_coeff[:,:ncore]
        moH = mo.conjugate ().T
        dm_core = mo @ moH
        dm_cas = self.make_rdm1s_sub (mo_coeff=mo_coeff, **kwargs).sum (0)
        return dm_core[None,:,:] + dm_cas

    def make_rdm1 (self, mo_coeff=None, ci=None, **kwargs):
        return self.make_rdm1s (mo_coeff=mo_coeff, ci=ci, **kwargs).sum (0)

    def make_casdm1s (self, ci=None, **kwargs):
        ''' Make the full-dimensional casdm1s spanning the collective active space '''
        casdm1s_sub = self.make_casdm1s_sub (ci=ci, **kwargs)
        casdm1a = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
        casdm1b = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
        return np.stack ([casdm1a, casdm1b], axis=0)

    def make_casdm1 (self, ci=None, **kwargs):
        ''' Spin-sum make_casdm1s '''
        return self.make_casdm1s (ci=ci, **kwargs).sum (0)

    def states_make_casdm2 (self, ci=None, ncas_sub=None, nelecas_sub=None, 
            casdm1frs=None, casdm2fr=None, **kwargs):
        ''' Make the full-dimensional casdm2 spanning the collective active space '''
        log = lib.logger.new_logger (self, self.verbose)
        log.warn (("You have found yourself in states_make_casdm2, which is "
                   "a very bad piece of code that Matt should be avoiding. "
                   "Please yell at him about this at earliest convenience."))
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci)
        if casdm2fr is None: casdm2fr = self.states_make_casdm2_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        ncas = sum (ncas_sub)
        ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
        casdm2r = np.zeros ((self.nroots,ncas,ncas,ncas,ncas))
        # Diagonal 
        for isub, dm2 in enumerate (casdm2fr):
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            casdm2r[:, i:j, i:j, i:j, i:j] = dm2
        # Off-diagonal
        for (isub1, dm1s1_r), (isub2, dm1s2_r) in combinations (enumerate (casdm1frs), 2):
            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            for dm1s1, dm1s2, casdm2 in zip (dm1s1_r, dm1s2_r, casdm2r):
                dma1, dmb1 = dm1s1[0], dm1s1[1]
                dma2, dmb2 = dm1s2[0], dm1s2[1]
                # Coulomb slice
                casdm2[i:j, i:j, k:l, k:l] = np.multiply.outer (dma1+dmb1, dma2+dmb2)
                casdm2[k:l, k:l, i:j, i:j] = casdm2[i:j, i:j, k:l, k:l].transpose (2,3,0,1)
                # Exchange slice
                casdm2[i:j, k:l, k:l, i:j] = -(np.multiply.outer (dma1, dma2)
                                               +np.multiply.outer (dmb1, dmb2)).transpose (0,3,2,1)
                casdm2[k:l, i:j, i:j, k:l] = casdm2[i:j, k:l, k:l, i:j].transpose (1,0,3,2)
        return casdm2r 

    def state_make_casdm1s(self, ci=None, state=0,  ncas_sub=None, nelecas_sub=None,
        casdm1frs=None, **kwargs):
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        casdm1s = np.stack([np.stack ([linalg.block_diag (*[dm1rs[iroot][ispin] 
                                                            for dm1rs in casdm1frs])
                                        for ispin in (0, 1)], axis=0)
                            for iroot in range (self.nroots)], axis=0)
        return casdm1s[state]
    
    def state_make_casdm2(self, ci=None, state=0, ncas_sub=None, nelecas_sub=None, 
            casdm1frs=None, casdm2fr=None, **kwargs):
        ''' State wise casdm2 spanning the collective active space. '''
        # This is producing the casdm2 for all states, but need to generate only for one state
        casdm2r = self.states_make_casdm2(ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, 
            casdm1frs=casdm1frs, casdm2fr=casdm2fr, **kwargs)
        return casdm2r[state] 
    
    def make_casdm2 (self, ci=None, ncas_sub=None, nelecas_sub=None, 
            casdm2r=None, casdm2f=None, casdm1frs=None, casdm2fr=None,
            **kwargs):
        ''' Make the full-dimensional casdm2 spanning the collective active space '''
        if casdm2r is not None: 
            return np.einsum ('rijkl,r->ijkl', casdm2r, self.weights)
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm2f is None: casdm2f = self.make_casdm2_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, casdm2fr=casdm2fr)
        ncas = sum (ncas_sub)
        ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
        weights = self.weights
        casdm2 = np.zeros ((ncas,ncas,ncas,ncas))
        # Diagonal 
        for isub, dm2 in enumerate (casdm2f):
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            casdm2[i:j, i:j, i:j, i:j] = dm2
        # Off-diagonal
        for (isub1, dm1rs1), (isub2, dm1rs2) in combinations (enumerate (casdm1frs), 2):
            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            dma1r, dmb1r = dm1rs1[:,0], dm1rs1[:,1]
            dma2r, dmb2r = dm1rs2[:,0], dm1rs2[:,1]
            dm1r = dma1r + dmb1r
            dm2r = dma2r + dmb2r
            # Coulomb slice
            casdm2[i:j, i:j, k:l, k:l] = lib.einsum ('r,rij,rkl->ijkl', weights, dm1r, dm2r)
            casdm2[k:l, k:l, i:j, i:j] = casdm2[i:j, i:j, k:l, k:l].transpose (2,3,0,1)
            # Exchange slice
            d2exc = (lib.einsum ('rij,rkl->rilkj', dma1r, dma2r)
                   + lib.einsum ('rij,rkl->rilkj', dmb1r, dmb2r))
            casdm2[i:j, k:l, k:l, i:j] -= np.tensordot (weights, d2exc, axes=1)
            casdm2[k:l, i:j, i:j, k:l] = casdm2[i:j, k:l, k:l, i:j].transpose (1,0,3,2)
        return casdm2 

    def get_veff (self, mol=None, dm1s=None, hermi=1, spin_sep=False, **kwargs):
        ''' Returns a spin-summed veff! If dm1s isn't provided, builds from self.mo_coeff, self.ci
            etc. '''
        if mol is None: mol = self.mol
        nao = mol.nao_nr ()
        if dm1s is None: dm1s = self.make_rdm1 (include_core=True, **kwargs).reshape (nao, nao)
        dm1s = np.asarray (dm1s)
        if dm1s.ndim == 2: dm1s = dm1s[None,:,:]
        if isinstance (self, _DFLASCI):
            vj, vk = self.with_df.get_jk(dm1s, hermi=hermi)
        else:
            vj, vk = self._scf.get_jk(mol, dm1s, hermi=hermi)
        if spin_sep:
            assert (dm1s.shape[0] == 2)
            return vj.sum (0)[None,:,:] - vk
        else:
            veff = np.stack ([j - k/2 for j, k in zip (vj, vk)], axis=0)
            return np.squeeze (veff)

    def split_veff (self, veff, h2eff_sub, mo_coeff=None, ci=None, casdm1s_sub=None):
        ''' Split a spin-summed veff into alpha and beta terms using the h2eff eri array.
        Note that this will omit v(up_active - down_active)^virtual_inactive by necessity; 
        this won't affect anything because the inactive density matrix has no spin component.
        On the other hand, it ~is~ necessary to correctly do 

        v(up_active - down_active)^unactive_active

        in order to calculate the external orbital gradient at the end of the calculation.
        This means that I need h2eff_sub spanning both at least two active subspaces
        ~and~ the full orbital range. '''
        veff_c = veff.copy ()
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if casdm1s_sub is None: casdm1s_sub = self.make_casdm1s_sub (ci = ci)
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        nao, nmo = mo_coeff.shape
        moH_coeff = mo_coeff.conjugate ().T
        smo_coeff = self._scf.get_ovlp () @ mo_coeff
        smoH_coeff = smo_coeff.conjugate ().T
        veff_s = np.zeros_like (veff_c)
        for ix, (ncas_i, casdm1s) in enumerate (zip (self.ncas_sub, casdm1s_sub)):
            i = sum (self.ncas_sub[:ix])
            j = i + ncas_i
            eri_k = h2eff_sub.reshape (nmo, ncas, -1)[:,i:j,...].reshape (nmo*ncas_i, -1)
            eri_k = lib.numpy_helper.unpack_tril (eri_k)[:,i:j,:]
            eri_k = eri_k.reshape (nmo, ncas_i, ncas_i, ncas)
            sdm = casdm1s[0] - casdm1s[1]
            vk_pa = -np.tensordot (eri_k, sdm, axes=((1,2),(0,1))) / 2
            veff_s[:,ncore:nocc] += vk_pa
            veff_s[ncore:nocc,:] += vk_pa.T
            veff_s[ncore:nocc,ncore:nocc] -= vk_pa[ncore:nocc,:] / 2
            veff_s[ncore:nocc,ncore:nocc] -= vk_pa[ncore:nocc,:].T / 2
        veff_s = smo_coeff @ veff_s @ smoH_coeff
        veffa = veff_c + veff_s
        veffb = veff_c - veff_s
        return np.stack ([veffa, veffb], axis=0)
         

    def states_energy_elec (self, mo_coeff=None, ncore=None, ncas=None,
            ncas_sub=None, nelecas_sub=None, ci=None, h2eff=None, veff=None, 
            casdm1frs=None, casdm2fr=None, veff_core=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        if ncas is None: ncas = self.ncas
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None: ci = self.ci
        if h2eff is None: h2eff = self.get_h2eff (mo_coeff)
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci, ncas_sub=ncas_sub,
                                                                        nelecas_sub=nelecas_sub)
        if casdm2fr is None: casdm2fr = self.states_make_casdm2_sub (ci=ci, ncas_sub=ncas_sub,
                                                                     nelecas_sub=nelecas_sub)
        nao, nmo = mo_coeff.shape
        nocc = ncore + ncas
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:nocc]
        dm_core = 2*mo_core @ mo_core.conj ().T
        if veff_core is None: veff_core = getattr (veff, 'c', None)
        if veff_core is None: veff_core = self.get_veff (dm1s=dm_core)
        h1eff = self.get_hcore () + veff_core
        e0 = 2*np.dot (((h1eff-(veff_core/2)) @ mo_core).ravel (), mo_core.conj().ravel ())
        h1eff = mo_cas.conj ().T @ h1eff @ mo_cas
        eri_cas = lib.numpy_helper.unpack_tril (h2eff.reshape (nmo*ncas, ncas*(ncas+1)//2))
        eri_cas = eri_cas.reshape (nmo, ncas, ncas, ncas)
        eri_cas = eri_cas[ncore:nocc]
        casdm1rs = self.states_make_casdm1s (ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub,
                                             casdm1frs=casdm1frs)
        vj_r = np.tensordot (casdm1rs.sum (1), eri_cas, axes=2)
        vk_rs = np.tensordot (casdm1rs, eri_cas, axes=((2,3),(2,1)))
        veff_rs = vj_r[:,None,:,:] - vk_rs

        energy_elec = []
        for idx, (dm1s, v) in enumerate (zip (casdm1rs, veff_rs)):
            casdm1fs = [dm[idx] for dm in casdm1frs]
            casdm2f = [dm[idx] for dm in casdm2fr]
            
            # 1-body veff terms
            h1e = h1eff[None,:,:] + v/2
            e1 = np.dot (h1e.ravel (), dm1s.ravel ())

            # 2-body cumulant terms
            e2 = 0
            for isub, (dm1s, dm2) in enumerate (zip (casdm1fs, casdm2f)):
                dm1a, dm1b = dm1s[0], dm1s[1]
                dm1 = dm1a + dm1b
                cdm2 = dm2 - np.multiply.outer (dm1, dm1)
                cdm2 += np.multiply.outer (dm1a, dm1a).transpose (0,3,2,1)
                cdm2 += np.multiply.outer (dm1b, dm1b).transpose (0,3,2,1)
                eri = self.get_h2eff_slice (h2eff, isub)
                te2 = np.tensordot (eri, cdm2, axes=4) / 2
                e2 += te2
            energy_elec.append (e0 + e1 + e2)
            self._e1_ref = e0 + e1
            self._e2_ref = e2

        return energy_elec

    def energy_elec (self, mo_coeff=None, ncore=None, ncas=None,
            ncas_sub=None, nelecas_sub=None, ci=None, h2eff=None, veff=None,
            casdm1frs=None, casdm2fr=None, **kwargs):
        ''' Since the LASCI energy cannot be calculated as simply as ecas + ecore, I need this '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        if ncas is None: ncas = self.ncas
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None: ci = self.ci
        if h2eff is None: h2eff = self.get_h2eff (mo_coeff)
        casdm1s_sub = self.make_casdm1s_sub (ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub,
                                             casdm1frs=casdm1frs)
        if veff is None:
            veff = self.get_veff (dm1s = self.make_rdm1(mo_coeff=mo_coeff,casdm1s_sub=casdm1s_sub))
            veff = self.split_veff (veff, h2eff, mo_coeff=mo_coeff, casdm1s_sub=casdm1s_sub)

        # 1-body veff terms
        h1e = self.get_hcore ()[None,:,:] + veff/2
        dm1s = self.make_rdm1s (mo_coeff=mo_coeff, ncore=ncore, ncas_sub=ncas_sub,
            nelecas_sub=nelecas_sub, casdm1s_sub=casdm1s_sub)
        e1 = np.dot (h1e.ravel (), dm1s.ravel ())

        # 2-body cumulant terms
        casdm1s = self.make_casdm1s (ci=ci, ncas_sub=ncas_sub, 
            nelecas_sub=nelecas_sub, casdm1frs=casdm1frs)
        casdm1 = casdm1s.sum (0)
        casdm2 = self.make_casdm2 (ci=ci, ncas_sub=ncas_sub,
            nelecas_sub=nelecas_sub, casdm1frs=casdm1frs, casdm2fr=casdm2fr)
        casdm2 -= np.multiply.outer (casdm1, casdm1)
        casdm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
        casdm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
        ncore, ncas, nocc = self.ncore, self.ncas, self.ncore + self.ncas
        eri = lib.numpy_helper.unpack_tril (h2eff[ncore:nocc].reshape (ncas*ncas, -1))
        eri = eri.reshape ([ncas,]*4)
        e2 = np.tensordot (eri, casdm2, axes=4)/2

        e0 = self.energy_nuc ()
        self._e1_test = e1
        self._e2_test = e2
        return e1 + e2

    _ugg = lasci_sync.LASCI_UnitaryGroupGenerators
    def get_ugg (self, mo_coeff=None, ci=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        return self._ugg (self, mo_coeff, ci)

    def cderi_ao2mo (self, mo_i, mo_j, compact=False):
        assert (isinstance (self, _DFLASCI))
        nmo_i, nmo_j = mo_i.shape[-1], mo_j.shape[-1]
        if compact:
            assert (nmo_i == nmo_j)
            bPij = np.empty ((self.with_df.get_naoaux (), nmo_i*(nmo_i+1)//2), dtype=mo_i.dtype)
        else:
            bPij = np.empty ((self.with_df.get_naoaux (), nmo_i, nmo_j), dtype=mo_i.dtype)
        ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos (mo_i, mo_j, compact=compact)
        b0 = 0
        for eri1 in self.with_df.loop ():
            b1 = b0 + eri1.shape[0]
            eri2 = bPij[b0:b1]
            eri2 = ao2mo._ao2mo.nr_e2 (eri1, moij, ijslice, aosym='s2', mosym=ijmosym, out=eri2)
            b0 = b1
        return bPij

    def fast_veffa (self, casdm1s_sub, h2eff_sub, mo_coeff=None, ci=None, _full=False):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        assert (isinstance (self, _DFLASCI) or _full)
        ncore = self.ncore
        ncas_sub = self.ncas_sub
        ncas = sum (ncas_sub)
        nocc = ncore + ncas
        nao, nmo = mo_coeff.shape

        mo_cas = mo_coeff[:,ncore:nocc]
        moH_cas = mo_cas.conjugate ().T
        moH_coeff = mo_coeff.conjugate ().T
        dma = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
        dmb = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
        casdm1s = np.stack ([dma, dmb], axis=0)
        if not (isinstance (self, _DFLASCI)):
            dm1s = np.dot (mo_cas, np.dot (casdm1s, moH_cas)).transpose (1,0,2)
            return self.get_veff (dm1s = dm1s, spin_sep=True)
        casdm1 = casdm1s.sum (0)
        dm1 = np.dot (mo_cas, np.dot (casdm1, moH_cas))
        bPmn = sparsedf_array (self.with_df._cderi)

        # vj
        dm_tril = dm1 + dm1.T - np.diag (np.diag (dm1.T))
        rho = np.dot (bPmn, lib.pack_tril (dm_tril))
        vj = lib.unpack_tril (np.dot (rho, bPmn))

        # vk
        bmPu = h2eff_sub.bmPu
        if _full:
            vmPsu = np.dot (bmPu, casdm1s)
            vk = np.tensordot (vmPsu, bmPu, axes=((1,3),(1,2))).transpose (1,0,2)
            return vj[None,:,:] - vk
        else:
            vmPu = np.dot (bmPu, casdm1)
            vk = np.tensordot (vmPu, bmPu, axes=((1,2),(1,2)))
            return vj - vk/2

    @lib.with_doc(run_lasci.__doc__)
    def lasci (self, mo_coeff=None, ci0=None, lroots=None, lweights=None, verbose=None,
               assert_no_dupes=False):
        if mo_coeff is None: mo_coeff=self.mo_coeff
        if ci0 is None: ci0 = self.ci
        if verbose is None: verbose = self.verbose
        converged, e_tot, e_states, e_cas, e_lexc, ci = run_lasci (
            self, mo_coeff=mo_coeff, ci0=ci0, lroots=lroots, lweights=lweights,
            verbose=verbose, assert_no_dupes=assert_no_dupes)
        self.converged, self.ci = converged, ci
        self.e_tot, self.e_states, self.e_cas, self.e_lexc = e_tot, e_states, e_cas, e_lexc
        if mo_coeff is self.mo_coeff:
            self.dump_chk ()
        elif getattr (self, 'chkfile', None) is not None:
            lib.logger.warn (self, 'orbitals changed; chkfile not dumped!')
        self._finalize ()
        return self.converged, self.e_tot, self.e_states, self.e_cas, e_lexc, self.ci

    @lib.with_doc(run_lasci.__doc__)
    def lasci_(self, mo_coeff=None, ci0=None, verbose=None,
            assert_no_dupes=False):
        if mo_coeff is not None:
            self.mo_coeff = mo_coeff
        return self.lasci (mo_coeff=mo_coeff, ci0=ci0, verbose=verbose,
                           assert_no_dupes=assert_no_dupes)

    state_average = state_average
    state_average_ = state_average_

    def lassi(self, **kwargs):
        #import warnings
        #lassi_kernel_warn = "Now LASSI have kernel, which takes las instance as input. This [las.lassi()] function " \
        #                    "will be removed soon."
        #warnings.warn(lassi_kernel_warn, stacklevel=3)
        from mrh.my_pyscf.lassi import lassi
        mylassi = lassi.LASSI(self, **kwargs)
        return mylassi.kernel(**kwargs)

    las2cas_civec = las2cas_civec
    assert_no_duplicates = assert_no_duplicates
    get_init_guess_ci = get_init_guess_ci
    localize_init_guess=lasscf_guess.localize_init_guess
    def _svd (self, mo_lspace, mo_rspace, s=None, **kwargs):
        if s is None: s = self._scf.get_ovlp ()
        return matrix_svd_control_options (s, lspace=mo_lspace, rspace=mo_rspace, full_matrices=True)[:3]

    def dump_flags (self, verbose=None, _method_name='LASCI'):
        log = lib.logger.new_logger (self, verbose)
        log.info ('')
        log.info ('******** %s flags ********', _method_name)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        nfrags = len (self.nelecas_sub)
        log.info ('CAS (%de+%de, %do), ncore = %d, nvir = %d',
                  self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        log.info ('Divided into %d LAS spaces', nfrags)
        for i, (no, ne) in enumerate (zip (self.ncas_sub, self.nelecas_sub)):
            log.info ('LAS %d : (%de+%de, %do)', i, ne[0], ne[1], no)
        log.info ('nroots = %d', self.nroots)
        log.info ('max_memory %d (MB)', self.max_memory)
        for i, fcibox in enumerate (self.fciboxes):
            if getattr (fcibox, 'dump_flags', None):
                log.info ('fragment %d FCI solver flags:', i)
                fcibox.dump_flags (log.verbose)

    @property
    def converged (self):
        return all (self.states_converged)
    @converged.setter
    def converged (self, x):
        if hasattr (x, '__len__'):
            self.states_converged = list (x)
        else:
            self.states_converged = [x,]*self.nroots

    def dump_spaces (self, nroots=None, sort_energy=False):
        log = lib.logger.new_logger (self, self.verbose)
        log.info ("******** LAS space tables ********")
        ci = self.ci
        if nroots is None and self.verbose <= lib.logger.INFO:
            nroots = min (self.nroots, 100)
        elif nroots is None:
            nroots = self.nroots
        if nroots < self.nroots:
            log.warn ("Dumping only 100 of %d spaces", self.nroots)
            log.warn ("To see more, explicitly pass nroots to dump_spaces or increase verbosity")
        if sort_energy:
            idx = np.argsort (self.e_states)
        else:
            idx = range (nroots)
        for state in idx:
            neleca_f = []
            nelecb_f = []
            wfnsym_f = []
            wfnsym = 0
            m_f = []
            s_f = []
            lroots = []
            s2_tot = 0
            for ifrag, (fcibox, nelecas) in enumerate (zip (self.fciboxes, self.nelecas_sub)):
                solver = fcibox.fcisolvers[state]
                na, nb = _unpack_nelec (fcibox._get_nelec (solver, nelecas))
                neleca_f.append (na)
                nelecb_f.append (nb)
                m_f.append ((na-nb)/2)
                s_f.append ((solver.smult-1)/2)
                s2_tot += s_f[-1] * (s_f[-1] + 1)
                fragsym = getattr (solver, 'wfnsym', 0) or 0
                if isinstance (fragsym, str):
                    fragsym_str = fragsym
                    fragsym_id = symm.irrep_name2id (solver.mol.groupname, fragsym)
                else:
                    fragsym_id = fragsym
                    fragsym_str = symm.irrep_id2name (solver.mol.groupname, fragsym)
                wfnsym ^= fragsym_id
                wfnsym_f.append (fragsym_str)
                lroots_i = 0
                if ci is not None:
                    if ci[ifrag] is not None:
                        ci_i = ci[ifrag]
                        if ci_i[state] is not None:
                            ci_ij = ci_i[state]
                            lroots_i = 1 if ci_ij.ndim<3 else ci_ij.shape[0]
                lroots.append (lroots_i)
            s2_tot += sum ([2*m1*m2 for m1, m2 in combinations (m_f, 2)])
            s_f, m_f = np.asarray (s_f), np.asarray (m_f)
            if np.all (m_f<0): m_f *= -1
            s_pure = bool (np.all (s_f==m_f))
            wfnsym = symm.irrep_id2name (self.mol.groupname, wfnsym)
            neleca = sum (neleca_f)
            nelecb = sum (nelecb_f)
            log.info ("LAS space %d: (%de+%de,%do) wfynsm=%s", state, neleca, nelecb, self.ncas, wfnsym)
            log.info ("Converged? %s", self.states_converged[state])
            log.info ("E(LAS) = %.15g", self.e_states[state])
            log.info ("S^2 = %.7f (%s)", s2_tot, ('Impure','Pure')[s_pure])
            log.info ("Space table")
            log.info (" frag    (ae+be,no)  2S+1   ir   lroots")
            for i in range (self.nfrags):
                smult_f = int (round (2*s_f[i] + 1))
                tupstr = '({}e+{}e,{}o)'.format (neleca_f[i], nelecb_f[i], self.ncas_sub[i])
                log.info (" %4d %13s  %4d  %3s   %6d", i, tupstr, smult_f, wfnsym_f[i], lroots[i])

    def check_sanity (self):
        casci.CASCI.check_sanity (self)
        self.get_ugg () # constructor encounters impossible states and raises error

    dump_chk = chkfile.dump_las
    load_chk = load_chk_ = chkfile.load_las_

class LASCISymm (casci_symm.CASCI, LASCINoSymm):

    def __init__(self, mf, ncas, nelecas, ncore=None, spin_sub=None, wfnsym_sub=None, frozen=None,
                 **kwargs):
        LASCINoSymm.__init__(self, mf, ncas, nelecas, ncore=ncore, spin_sub=spin_sub,
                             frozen=frozen, **kwargs)
        if getattr (self.mol, 'groupname', None) in ('Dooh', 'Coov'):
            raise NotImplementedError ("LASSCF support for cylindrical point group {}".format (
                self.mol.groupname))
        if wfnsym_sub is None: wfnsym_sub = [0 for icas in self.ncas_sub]
        # TODO: guess wfnsym_sub intelligently (0 can be impossible for some multiplicities)
        for wfnsym, frag in zip (wfnsym_sub, self.fciboxes):
            if isinstance (wfnsym, (str, np.str_)):
                wfnsym = symm.irrep_name2id (self.mol.groupname, wfnsym)
            frag.fcisolvers[0].wfnsym = wfnsym

    make_rdm1s = LASCINoSymm.make_rdm1s
    make_rdm1 = LASCINoSymm.make_rdm1
    get_veff = LASCINoSymm.get_veff
    get_h1eff = get_h1las = h1e_for_las
    dump_flags = LASCINoSymm.dump_flags
    dump_spaces = LASCINoSymm.dump_spaces
    check_sanity = LASCINoSymm.check_sanity
    _ugg = lasci_sync.LASCISymm_UnitaryGroupGenerators

    @property
    def wfnsym (self):
        ''' This now returns the product of the irreps of the subspaces '''
        wfnsym = [0,]*self.nroots
        for frag in self.fciboxes:
            for state, solver in enumerate (frag.fcisolvers):
                wfnsym[state] ^= solver.wfnsym
        if self.nroots == 1: wfnsym = wfnsym[0]
        return wfnsym
    @wfnsym.setter
    def wfnsym (self, ir):
        raise RuntimeError (("Cannot assign the whole-system symmetry of a LASCI wave function. "
                             "Address fciboxes[ifrag].fcisolvers[istate].wfnsym instead."))

    def kernel(self, mo_coeff=None, ci0=None, casdm0_fr=None, verbose=None, assert_no_dupes=False):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci0 is None:
            ci0 = self.ci

        # Initialize/overwrite mo_coeff.orbsym. Don't pass ci0 because it's not the right shape
        lib.logger.info (self, ("LASCI lazy hack note: lines below reflect the point-group "
                                "symmetry of the whole molecule but not of the individual "
                                "subspaces"))
        mo_coeff = self.mo_coeff = self.label_symmetry_(mo_coeff)
        return LASCINoSymm.kernel(self, mo_coeff=mo_coeff, ci0=ci0,
            casdm0_fr=casdm0_fr, verbose=verbose, assert_no_dupes=assert_no_dupes)

    def canonicalize (self, mo_coeff=None, ci=None, natorb_casdm1=None, veff=None, h2eff_sub=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        mo_coeff = self.label_symmetry_(mo_coeff)
        return canonicalize (self, mo_coeff=mo_coeff, ci=ci, natorb_casdm1=natorb_casdm1,
                             h2eff_sub=h2eff_sub, orbsym=mo_coeff.orbsym)

    def label_symmetry_(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff=self.mo_coeff
        ncore = self.ncore
        ncas_sub = self.ncas_sub
        nocc = ncore + sum (ncas_sub)
        mo_coeff[:,:ncore] = symm.symmetrize_space (self.mol, mo_coeff[:,:ncore])
        for isub, ncas in enumerate (ncas_sub):
            i = ncore + sum (ncas_sub[:isub])
            j = i + ncas
            mo_coeff[:,i:j] = symm.symmetrize_space (self.mol, mo_coeff[:,i:j])
        mo_coeff[:,nocc:] = symm.symmetrize_space (self.mol, mo_coeff[:,nocc:])
        orbsym = symm.label_orb_symm (self.mol, self.mol.irrep_id,
                                      self.mol.symm_orb, mo_coeff,
                                      s=self._scf.get_ovlp ())
        mo_coeff = lib.tag_array (mo_coeff, orbsym=orbsym)
        return mo_coeff
        
    @lib.with_doc(LASCINoSymm.localize_init_guess.__doc__)
    def localize_init_guess (self, frags_atoms, mo_coeff=None, spin=None, lo_coeff=None, fock=None,
                             freeze_cas_spaces=False):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_coeff = casci_symm.label_symmetry_(self, mo_coeff)
        return LASCINoSymm.localize_init_guess (self, frags_atoms, mo_coeff=mo_coeff, spin=spin,
            lo_coeff=lo_coeff, fock=fock, freeze_cas_spaces=freeze_cas_spaces)

    def _svd (self, mo_lspace, mo_rspace, s=None, **kwargs):
        if s is None: s = self._scf.get_ovlp ()
        lsymm = getattr (mo_lspace, 'orbsym', None)
        if lsymm is None:
            mo_lspace = symm.symmetrize_space (self.mol, mo_lspace)
            lsymm = symm.label_orb_symm(self.mol, self.mol.irrep_id,
                self.mol.symm_orb, mo_lspace, s=s)
        rsymm = getattr (mo_rspace, 'orbsym', None)
        if rsymm is None:
            mo_rspace = symm.symmetrize_space (self.mol, mo_rspace)
            rsymm = symm.label_orb_symm(self.mol, self.mol.irrep_id,
                self.mol.symm_orb, mo_rspace, s=s)
        decomp = matrix_svd_control_options (s,
            lspace=mo_lspace, rspace=mo_rspace,
            lspace_symmetry=lsymm, rspace_symmetry=rsymm,
            full_matrices=True, strong_symm=True)
        mo_lvecs, svals, mo_rvecs, lsymm, rsymm = decomp
        mo_lvecs = lib.tag_array (mo_lvecs, orbsym=lsymm)
        mo_rvecs = lib.tag_array (mo_rvecs, orbsym=rsymm)
        return mo_lvecs, svals, mo_rvecs
     
