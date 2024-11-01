import os
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.lo import orth
from pyscf.scf.rohf import get_roothaan_fock
from mrh.my_pyscf.mcscf import lasci, _DFLASCI

# TODO: symmetry
def orth_orb (las, kf2_list):
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nao, nmo = las.mo_coeff.shape
    nfrags = len (kf2_list)
    log = lib.logger.new_logger (las, las.verbose)

    # orthonormalize active orbitals
    mo_cas = np.empty ((nao, ncas), dtype=las.mo_coeff.dtype)
    ci = []
    for ifrag, kf2 in enumerate (kf2_list):
        i = sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        k, l = i + ncore, j + ncore
        mo_cas[:,i:j] = kf2.mo_coeff[:,k:l]
        ci.append (kf2.ci[ifrag])
    mo_cas_preorth = mo_cas.copy ()
    s0 = las._scf.get_ovlp ()
    mo_cas = orth.vec_lowdin (mo_cas_preorth, s=s0)
    
    # reassign orthonormalized active orbitals
    proj = []
    for ifrag in range (nfrags):
        i = sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        proj.append (mo_cas_preorth[:,i:j] @ mo_cas_preorth[:,i:j].conj ().T)
    smo1 = s0 @ mo_cas
    frag_weights = np.stack ([((p @ smo1) * smo1.conjugate ()).sum (0)
                              for p in proj], axis=-1)
    idx = np.argsort (frag_weights, axis=1)[:,-1]
    mo_las = []
    for ifrag in range (nfrags):
        mo = mo_cas[:,(idx == ifrag)]
        i = sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        s1 = mo.conj ().T @ s0 @ mo_cas_preorth[:,i:j]
        u, svals, vh = linalg.svd (s1)
        mo_las.append (mo @ u @ vh)
    mo_cas = np.concatenate (mo_las, axis=1)
    
    # non-active orbitals
    ucas = las.mo_coeff.conj ().T @ s0 @ mo_cas
    u, R = linalg.qr (ucas)
    # Isn't it weird that you do Gram-Schmidt by doing QR?
    errmax = np.amax (np.abs (np.abs (R[:ncas,:ncas]) - np.eye (ncas)))
    if errmax>1e-8:
        log.warn ('Active orbital orthogonalization may have failed: %e', errmax)
    mo1 = las.mo_coeff @ u
    errmax = np.amax (np.abs (np.abs (mo_cas.conj ().T @ s0 @ mo1[:,:ncas]) - np.eye (ncas)))
    if errmax>1e-8:
        log.warn ('Active orbitals leaking into non-active space: %e', errmax)
    errmax = np.amax (np.abs ((mo1.conj ().T @ s0 @ mo1) - np.eye (mo1.shape[1])))
    if errmax>1e-8:
        log.warn ('Non-orthogonal AOs in lasscf_async_combine.orth_orb: %e', errmax)
    mo1 = mo1[:,ncas:]
    veff = sum ([kf2.veff for kf2 in kf2_list]) / nfrags
    dm1s = sum ([kf2.dm1s for dm1s in kf2_list]) / nfrags
    fock = las.get_hcore ()[None,:,:] + veff
    fock = get_roothaan_fock (fock, dm1s, s0)
    orbsym = None # TODO: symmetry
    fock = mo1.conj ().T @ fock @ mo1
    ene, umat = las._eig (fock, 0, 0, orbsym)
    mo_core = mo1 @ umat[:,:ncore]
    mo_virt = mo1 @ umat[:,ncore:]
    mo_coeff = np.concatenate ([mo_core, mo_cas, mo_virt], axis=1)

    return las.get_keyframe (mo_coeff, ci)

class flas_stdout_env (object):
    def __init__(self, las, flas_stdout):
        self.las = las
        self.flas_stdout = flas_stdout
        self.las_stdout = las.stdout
    def __enter__(self):
        self.las.stdout = self.flas_stdout
        self.las._scf.stdout = self.flas_stdout
        self.las.fcisolver.stdout = self.flas_stdout
        for fcibox in self.las.fciboxes:
            fcibox.stdout = self.flas_stdout
            for fcisolver in fcibox.fcisolvers:
                fcisolver.stdout = self.flas_stdout
        if getattr (self.las, 'with_df', None):
            self.las.with_df.stdout = self.flas_stdout
    def __exit__(self, type, value, traceback):
        self.las.stdout = self.las_stdout
        self.las._scf.stdout = self.las_stdout
        self.las.fcisolver.stdout = self.las_stdout
        for fcibox in self.las.fciboxes:
            fcibox.stdout = self.las_stdout
            for fcisolver in fcibox.fcisolvers:
                fcisolver.stdout = self.las_stdout
        if getattr (self.las, 'with_df', None):
            self.las.with_df.stdout = self.las_stdout

def relax (las, kf):
    log = lib.logger.new_logger (las, las.verbose)
    flas_stdout = getattr (las, '_flas_stdout', None)
    if flas_stdout is None:
        output = getattr (las.mol, 'output', None)
        if not ((output is None) or (output=='/dev/null')):
            flas_output = output + '.flas'
            if las.verbose > lib.logger.QUIET:
                if os.path.isfile (flas_output):
                    print('overwrite output file: %s' % flas_output)
                else:
                    print('output file: %s' % flas_output)
            flas_stdout = open (flas_output, 'w')
            las._flas_stdout = flas_stdout
        else:
            flas_stdout = las.stdout
    with flas_stdout_env (las, flas_stdout):
        flas = lasci.LASCI (las._scf, las.ncas_sub, las.nelecas_sub)
        flas.__dict__.update (las.__dict__)
        e_tot, e_cas, ci, mo_coeff, mo_energy, h2eff_sub, veff = \
            flas.kernel (kf.mo_coeff, ci0=kf.ci)
    ovlp = mo_coeff.conj ().T @ las._scf.get_ovlp () @ mo_coeff
    errmat = ovlp - np.eye (ovlp.shape[0])
    errmax = np.amax (np.abs (errmat))
    if errmax>1e-8:
        log.warn ('Non-orthogonal AOs in lasscf_async_combine.relax: max ovlp error = %e', errmax)
    return las.get_keyframe (mo_coeff, ci)

def combine_o0 (las, kf2_list):
    kf1 = orth_orb (las, kf2_list)
    kf1 = relax (las, kf1)
    return kf1


