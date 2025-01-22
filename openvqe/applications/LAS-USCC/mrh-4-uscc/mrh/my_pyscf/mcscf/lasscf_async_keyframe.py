import numpy as np
from scipy import linalg

class LASKeyframe (object):
    '''Shallow struct for various intermediates. DON'T put complicated code in here Matt!!!'''

    def __init__(self, las, mo_coeff, ci):
        self.las = las
        self.mo_coeff = mo_coeff
        self.ci = ci
        self._dm1s = self._veff = self._fock1 = self._h1eff_sub = self._h2eff_sub = None

    @property
    def dm1s (self):
        if self._dm1s is None:
            self._dm1s = self.las.make_rdm1s (mo_coeff=self.mo_coeff, ci=self.ci)
        return self._dm1s

    @property
    def veff (self):
        if self._veff is None:
            self._veff = self.las.get_veff (dm1s=self.dm1s, spin_sep=True)
        return self._veff

    @property
    def fock1 (self):
        if self._fock1 is None:
            self._fock1 = self.las.get_grad_orb (
                mo_coeff=self.mo_coeff, ci=self.ci, h2eff_sub=self.h2eff_sub, veff=self.veff,
                dm1s=self.dm1s, hermi=0)
        return self._fock1

    @property
    def h2eff_sub (self):
        if self._h2eff_sub is None:
            self._h2eff_sub = self.las.get_h2eff (self.mo_coeff)
        return self._h2eff_sub

    @property
    def h1eff_sub (self):
        if self._h1eff_sub is None:
            self._h1eff_sub = self.las.get_h1eff (self.mo_coeff, ci=self.ci, veff=self.veff,
                h2eff_sub=self.h2eff_sub)
        return self._h1eff_sub

    def copy (self):
        ''' MO coefficients deepcopy; CI vectors shallow copy. Everything else, drop. '''
        mo1 = self.mo_coeff.copy ()
        ci1_fr = []
        ci0_fr = self.ci
        for ci0_r in ci0_fr:
            ci1_r = []
            for ci0 in ci0_r:
                ci1 = ci0.view ()
                ci1_r.append (ci1)
            ci1_fr.append (ci1_r)
        return LASKeyframe (self.las, mo1, ci1_fr)


def approx_keyframe_ovlp (las, kf1, kf2):
    '''Evaluate the similarity of two keyframes in terms of orbital and CI vector overlaps.

    Args:
        las : object of :class:`LASCINoSymm`
        kf1 : object of :class:`LASKeyframe`
        kf2 : object of :class:`LASKeyframe`

    Returns:
        mo_ovlp : float
            Products of the overlaps of the rotationally-invariant subspaces across the two
            keyframes; i.e.: prod (svals (inactive orbitals)) * prod (svals (virtual orbitals))
            * prod (svals (active 1)) * prod (svals (active 2)) * ...
        ci_ovlp : list of length nfrags of list of length nroots of floats
            Overlaps of the CI vectors, assuming that prod (svals (active n)) = 1. Meaningless
            if mo_ovlp deviates significantly from 1.
    '''

    nao, nmo = kf1.mo_coeff.shape    
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nvirt = nmo - nocc

    s0 = las._scf.get_ovlp ()
    mo1 = kf1.mo_coeff[:,:ncore]
    mo2 = kf2.mo_coeff[:,:ncore]
    s1 = mo1.conj ().T @ s0 @ mo2
    u, svals, vh = linalg.svd (s1)
    mo_ovlp = np.prod (svals) # inactive orbitals
    mo1 = kf1.mo_coeff[:,nocc:]
    mo2 = kf2.mo_coeff[:,nocc:]
    s1 = mo1.conj ().T @ s0 @ mo2
    u, svals, vh = linalg.svd (s1)
    mo_ovlp *= np.prod (svals) # virtual orbitals

    ci_ovlp = []
    for ifrag, (fcibox, c1_r, c2_r) in enumerate (zip (las.fciboxes, kf1.ci, kf2.ci)):
        nlas, nelelas = las.ncas_sub[ifrag], las.nelecas_sub[ifrag]
        i = ncore + sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        mo1 = kf1.mo_coeff[:,i:j]
        mo2 = kf2.mo_coeff[:,i:j]
        s1 = mo1.conj ().T @ s0 @ mo2
        u, svals, vh = linalg.svd (s1)
        mo_ovlp *= np.prod (svals) # ifrag active orbitals
        c1_r = fcibox.states_transform_ci_for_orbital_rotation (c1_r, nlas, nelelas, u @ vh)
        ci_ovlp.append ([abs (c1.conj ().ravel ().dot (c2.ravel ()))
                         for c1, c2 in zip (c1_r, c2_r)])

    return mo_ovlp, ci_ovlp
    

