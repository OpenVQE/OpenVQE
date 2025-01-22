''' 
MRH: In which I attempt to modify the pyscf MC-SCF class(es) to allow arbitrary constraints on the active orbitals by messing with the gradients
'''

import numpy as np
import scipy as sp
from pyscf.lib import logger
from pyscf.lo.orth import orth_ao
from pyscf.scf import hf
from pyscf.mcscf import mc1step, addons
from pyscf.mcscf.mc1step import expmat
from pyscf.tools import molden
from functools import reduce
from mrh.my_pyscf.scf import hf_as
from mrh.util.basis import is_basis_orthonormal, are_bases_orthogonal, basis_olap, orthonormalize_a_basis
from mrh.util.la import is_matrix_eye

def orth_orb (orb, ovlp):
    ovlp_orb = reduce (np.dot, [orb.conjugate ().T, ovlp, orb])
    evals, evecs = sp.linalg.eigh (ovlp_orb)
    idx = evals>1e-10
    return np.dot (orb, evecs[:,idx]) / np.sqrt (evals[idx])

def _get_u_casrot (mo, casrot, ovlp, ncore, ncas, ncasrot):
    ''' Obtain a unitary matrix for transforming from the mo basis to the casrot basis.
    The first ncas columns (block 1) span the active mos , the next ncasrot - ncas columns (block 2)
    are the orbitals the amos are allowed to rotate in to in descending order of their entanglement
    with the final nmo - ncasrot columns (block 3). An SVD diagonalizes the density matrices inside
    blocks 2 and 3, but does not sort them by occupancy. Also returns the occupancy by core electrons.'''

    mo2casrot = reduce (np.dot, [mo.conjugate ().T, ovlp, casrot])
    nocc = ncore + ncas
    nmo = mo.shape[1]

    # block 1
    proj = np.dot (mo2casrot[ncore:nocc,:ncasrot].conjugate ().T, mo2casrot[ncore:nocc,:ncasrot])
    evals, evecs = sp.linalg.eigh (proj)
    idx = evals.argsort ()[::-1]
    assert (np.all (np.logical_or (np.isclose (evals, 1), np.isclose (evals, 0))))
    mo2casrot[:,:ncasrot] = np.dot (mo2casrot[:,:ncasrot], evecs[:,idx])

    # block 2
    coredm1 = 2 * np.dot (mo2casrot[:ncore,:].conjugate ().T, mo2casrot[:ncore,:])
    block2_mat = np.dot (coredm1[ncas:ncasrot,ncasrot:], coredm1[ncasrot:,ncas:ncasrot])
    evals, evecs = sp.linalg.eigh (block2_mat)
    idx = evals.argsort ()[::-1]
    test_evals = np.append (evals[idx], np.zeros (nmo - ncasrot - evals.size))
    mo2casrot[:,ncas:ncasrot] = np.dot (mo2casrot[:,ncas:ncasrot], evecs[:,idx])

    # Make block 3
    block3_mat = np.dot (coredm1[ncasrot:,ncas:ncasrot], coredm1[ncas:ncasrot,ncasrot:])
    evals, evecs = sp.linalg.eigh (block3_mat)
    idx = evals.argsort ()[::-1]
    evals = evals[idx]
    assert (np.all (np.isclose (evals, test_evals)))
    mo2casrot[:,ncasrot:] = np.dot (mo2casrot[:,ncasrot:], evecs[:,idx])

    # Get occupancy  
    coreoccs = 2 * np.einsum ('ip,ip->p', mo2casrot[:ncore,:].conjugate (), mo2casrot[:ncore,:])
    assert (abs (np.sum (coreoccs) - 2*ncore) < 1e-10), "{} {}".format (2*ncore, np.sum (coreoccs))
    return mo2casrot, coreoccs


def rotate_orb_cc_wrapper (casscf, mo, fcivec, fcasdm1, fcasdm2, eris, x0_guess=None,
                  conv_tol_grad=1e-4, max_stepsize=None, verbose=None):
    ncore = casscf.ncore
    ncas = casscf.ncas
    ncasrot = casscf.ncasrot
    nelectron = casscf.mol.nelectron
    nocc = ncore + ncas

    # Test to make sure the orbitals never leave the proper space
    cas_ao = casscf.cas_ao
    err = np.linalg.norm (mo[~cas_ao,ncore:nocc])
    assert (abs (err) < 1e-10), err

    # Make _u_casrot
    casscf._u_casrot, casscf._crocc = casscf.get_u_casrot (mo)

    rota = mc1step.rotate_orb_cc (casscf, mo, fcivec, fcasdm1, fcasdm2, eris, 
            x0_guess=x0_guess, conv_tol_grad=conv_tol_grad, max_stepsize=max_stepsize, 
            verbose=verbose)
    fock_mo = reduce (np.dot, [mo.conjugate ().T,
        casscf.get_fock (mo_coeff=mo, ci=fcivec, eris=eris, casdm1=fcasdm1(), verbose=verbose),
        mo])
    for u_mo, g_orb, njk, r0 in rota:
        ''' This is not very efficient, because it constitutes a sort of rectangular taxicab descent on the orbital surface, but at least it's stable '''
        idx = np.zeros(mo.shape[0], dtype=np.bool_)
        idx[:ncore] = True
        idx[nocc:] = True
        idx2 = np.ix_(idx,idx)
        fock_mo1 = reduce (np.dot, [u_mo.conjugate ().T, fock_mo, u_mo])[idx2]
        evals, evecs = sp.linalg.eigh (fock_mo1)
        evecs = evecs[:,evals.argsort ()]
        evecs[:,np.diag(evecs)<0] *= -1
        u_fock = np.eye (u_mo.shape[0], dtype=u_mo.dtype)
        u_fock[idx2] = evecs
        u_mo = np.dot (u_mo, u_fock)
        yield u_mo, g_orb, njk, r0

def casci_scf_relaxation (envs):
    mc = envs['casscf']
    oldverbose = mc._scf.verbose
    mc._scf.verbose = 0
    mc._scf.build_frozen_from_mo (envs['mo'], mc.ncore, mc.ncas, envs['casdm1'], envs['casdm2'])
    mc._scf.diis = None
    dm0 = mc.make_rdm1 (mo_coeff=envs['mo'], ci=envs['fcivec'], ncas=mc.ncas, nelecas=mc.nelecas, ncore=mc.ncore)
    mc._scf.kernel (dm0)
    mo_change = reduce (np.dot, [envs['mo'].conjugate ().T, mc._scf.get_ovlp (), mc._scf.mo_coeff])
    mo_change = np.dot (mo_change, mo_change.conjugate ().T) - np.eye (mo_change.shape[0])
    print (np.linalg.norm (mo_change))
    envs['mo'] = mc._scf.mo_coeff
    mc._scf.verbose = oldverbose

class CASSCF(mc1step.CASSCF):
    '''MRH: In principle, to render certain orbitals orthogonal to certain other orbitals, all I should need to do is to project
    the initial guess and then mess with the gradients.  In this way I should be able to apply arbitrary constraints to the
    active orbitals, without affecting the cores and virtuals and without writing a whole Lagrangian optimizer.  Fingers crossed.
    I'm going to restrict myself to excluding entire atomic orbitals for now

    Extra attributes:

    cas_ao : ndarray, shape=(nao)
        boolean mask array for ao's allowed to contribute to cas

    casrot_coeff : ndarray, shape=(nao,nmo)
        orbital coefficients describing rotation space, the first ncasrot of which contain the active orbitals

    ncasrot : int
        number of orthonormal orbitals spanning cas_ao

    Parent class documentation follows:

    ''' + mc1step.CASSCF.__doc__

    def __init__(self, mf, ncas, nelecas, ncore=None, frozen=None, cas_ao=None):
        self.cas_ao = cas_ao
        self.casrot_coeff = None
        self.ncasrot = None
        assert (isinstance (mf, hf_as.RHF))
        if frozen is not None:
            raise NotImplementedError ("frozen mos in a constrained CASSCF context")
        mc1step.CASSCF.__init__(self, mf, ncas, nelecas, ncore, frozen)

    def kernel (self, mo_coeff=None, ci0=None, cas_ao=None, callback=None, _kern=mc1step.kernel):
        '''MRH: The only thing I need to do is to project the mo_coeffs, then pass along to the parent class member.
        The active orbitals need to be directly projected away from the inactive, and a reasonable
        selection of cores needs to be made.

        Extra kwargs:

        cas_ao : list of ints or boolean mask array of shape=(nao)
            aos allowed to contribute to cas

        see mc1step.CASSCF.kernel.__doc__ for parent method
        '''

        if self.frozen is not None:
            raise NotImplementedError ("frozen mos in a constrained CASSCF context")

        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        if cas_ao is None:
            cas_ao = self.cas_ao

        self.cas_ao, self.casrot_coeff, self.ncasrot = self.build_casrot (cas_ao)

        mo_coeff = self.project_init_guess (mo_coeff)

        #molden.from_mo (self.mol, 'init.molden', mo_coeff, occ=self._scf.mo_occ)
        self.mo_coeff = mo_coeff

        return mc1step.CASSCF.kernel (self, mo_coeff=mo_coeff, ci0=ci0, callback=callback, _kern=_kern)

    def build_casrot (self, cas_ao=None):
        if cas_ao is None:
            cas_ao = self.cas_ao
        nao = self._scf.get_ovlp ().shape[1]
        x = np.zeros (nao, np.bool_)
        x[cas_ao] = True
        cas_ao = x
        nocas_ao = np.logical_not (x)
        ovlp_ao = self._scf.get_ovlp ()

        idx_mat = np.ix_(cas_ao,cas_ao)
        ovlp_cas_ao = ovlp_ao[idx_mat] 
        evals, evecs = sp.linalg.eigh (ovlp_cas_ao)
        idx_lindep = evals > 1e-12
        ncasrot = np.count_nonzero (idx_lindep)
        p_coeff = np.zeros ((nao, ncasrot))
        p_coeff[cas_ao,:] = evecs[:,idx_lindep] / np.sqrt (evals[idx_lindep])

        projector = reduce (np.dot, [p_coeff, p_coeff.conjugate ().T, ovlp_ao])
        projector = np.eye (nao) - projector
        projector = np.dot (ovlp_ao, projector)
        evals, evecs = sp.linalg.eigh (projector, ovlp_ao)
        assert (np.all (np.logical_or (np.isclose (evals, 1), np.isclose (evals, 0)))), "{0}".format (evals)
        idx = np.isclose (evals, 1)
        q_coeff = evecs[:,idx]

        casrot_coeff = np.append (p_coeff, q_coeff, axis=1)

        # Check orthonormality
        err_mat = np.eye (casrot_coeff.shape[1]) - reduce (np.dot, [casrot_coeff.conjugate ().T, ovlp_ao, casrot_coeff])
        err_norm = np.asarray ([np.linalg.norm (e) for e in (err_mat[:ncasrot,:ncasrot], 
                                                             err_mat[ncasrot:,:ncasrot],
                                                             err_mat[:ncasrot,ncasrot:],
                                                             err_mat[ncasrot:,ncasrot:])])

        assert (np.linalg.norm (err_norm) < 1e-10), "norm sectors = {0}".format (err_norm)

        return cas_ao, casrot_coeff, ncasrot

    def project_init_guess (self, mo_coeff=None, cas_ao=None, prev_mol=None):

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if prev_mol is not None:
            mo_coeff = addons.project_init_guess (self, mo_coeff, prev_mol)
        if cas_ao is None:
            cas_ao = self.cas_ao
        else:
            self.cas_ao, self.casrot_coeff, self.ncasrot = self.build_casrot (cas_ao)
        ncas = self.ncas
        ncore = self.ncore
        nocc = self.mol.nelectron // 2
        dm0 = 2 * np.dot (mo_coeff[:,:nocc], mo_coeff[:,:nocc].conjugate ().T)
        self._scf.kernel (dm0)
        nocc = ncas + ncore

        ovlp_ao = self._scf.get_ovlp ()
        fock_ao = self._scf.get_fock ()
        ncasrot = self.ncasrot
        u_casrot = self.casrot_coeff[:,:ncasrot]
        assert (np.allclose (u_casrot[~cas_ao,:], 0))
        projector = np.dot (u_casrot, u_casrot.conjugate ().T)

        # Project active orbitals
        mo_coeff[:,ncore:nocc] = reduce (np.dot, [projector, ovlp_ao, mo_coeff[:,ncore:nocc]])
        assert (np.allclose (mo_coeff[~cas_ao,ncore:nocc], 0))
        mo_coeff[:,ncore:nocc] = orth_orb (mo_coeff[:,ncore:nocc], ovlp_ao)
        # Remove active component of core orbitals
        mo_coeff[:,:ncore] -= reduce (np.dot, [mo_coeff[:,ncore:nocc], mo_coeff[:,ncore:nocc].conjugate ().T, ovlp_ao, mo_coeff[:,:ncore]])
        mo_coeff[:,:ncore] = orth_orb (mo_coeff[:,:ncore], ovlp_ao)
        # Remove core-active component of virtual orbitals
        mo_coeff[:,nocc:] -= reduce (np.dot, [mo_coeff[:,:nocc], mo_coeff[:,:nocc].conjugate ().T, ovlp_ao, mo_coeff[:,nocc:]])
        mo_coeff[:,nocc:] = orth_orb (mo_coeff[:,nocc:], ovlp_ao)
        assert (np.allclose (mo_coeff[~cas_ao,ncore:nocc], 0))
        assert (is_basis_orthonormal (mo_coeff, ovlp_ao))

        # sort active orbitals by energy
        mo_energy = np.einsum ('ip,ij,jp->p', mo_coeff.conjugate (), fock_ao, mo_coeff.T)
        amo_energy = mo_energy[ncore:nocc]
        amo_coeff = mo_coeff[:,ncore:nocc]
        idx = amo_energy.argsort ()
        amo_energy = amo_energy[idx]
        amo_coeff = amo_coeff[:,idx]
        mo_energy[ncore:nocc] = amo_energy
        mo_coeff[:,ncore:nocc] = amo_coeff

        # fc-scf to get the correct core
        nelecb = self.mol.nelectron // 2
        neleca = nelecb + (self.mol.nelectron % 2)
        mo_occ = np.zeros (mo_coeff.shape[1])
        mo_occ[:neleca] += 1
        mo_occ[:nelecb] += 1
        casdm1 = np.diag (mo_occ[ncore:nocc])
        self._scf.build_frozen_from_mo (mo_coeff, ncore, ncas)
        self._scf.diis = None
        dm0 = hf.make_rdm1 (mo_coeff, mo_occ)
        self._scf.kernel (dm0)
        amo_ovlp = reduce (np.dot, [mo_coeff[:,ncore:nocc].conjugate ().T, ovlp_ao, self._scf.mo_coeff[:,ncore:nocc]])
        amo_ovlp = np.dot (amo_ovlp, amo_ovlp.conjugate ().T)
        err = np.trace (amo_ovlp) - ncas
        assert (abs (err) < 1e-10), "{0}".format (amo_ovlp)
        assert (np.allclose (self._scf.mo_coeff[~cas_ao,ncore:nocc], 0))

        return self._scf.mo_coeff

    rotate_orb_cc = rotate_orb_cc_wrapper

    def get_u_casrot (self, mo=None, casrot=None, ovlp=None):
        if mo is None:
            mo = self.mo_coeffs
        if ovlp is None:
            ovlp = self._scf.get_ovlp ()
        if casrot is None:
            casrot = self.casrot_coeff

        return _get_u_casrot (mo, casrot, ovlp, self.ncore, self.ncas, self.ncasrot)

    def uniq_var_indices (self):
        # Most essential: active to casrot
        nmo = self.casrot_coeff.shape[1]
        ncas = self.ncas
        ncasrot = self.ncasrot
        mask = np.zeros ((nmo,nmo), dtype=np.bool_)
        mask[ncas:ncasrot,:ncas] = True
        # Can I add others to get occ-vir rotation without spoiling it?
        # Yes I can, but this ~slows it down~ somehow! (At least it doesn't break it, but I'm not sure it's doing anything?)
        #idx_unocc1 = np.isclose (self._crocc, 0)
        #idx_occ1 = ~idx_unocc1
        #idx_occ2, idx_unocc2 = np.copy (idx_occ1), np.copy (idx_unocc1)
        #idx_occ1[ncasrot:] = idx_unocc1[ncasrot:] = False
        #idx_occ2[:ncasrot] = idx_unocc2[:ncasrot] = False
        #mask[np.ix_(idx_occ1,idx_unocc1)] = True
        #mask[np.ix_(idx_occ2,idx_unocc2)] = True
        return mask

    def pack_uniq_var (self, rot):
        u = self._u_casrot
        uH = u.conjugate ().T
        rot = reduce (np.dot, [uH, rot, u])
        idx = self.uniq_var_indices ()
        #rot = np.dot (rot[ncore:nocc,:], u[:,ncas:ncasrot]).ravel ()
        return rot[idx]

    def unpack_uniq_var (self, rot):
        u = self._u_casrot
        uH = u.conjugate ().T
        nmo = self.casrot_coeff.shape[1]
        mat = np.zeros ((nmo,nmo), dtype=u.dtype)
        idx = self.uniq_var_indices ()
        mat[idx] = rot
        mat = mat - mat.T
        mat = reduce (np.dot, [u, mat, uH])
        return mat

