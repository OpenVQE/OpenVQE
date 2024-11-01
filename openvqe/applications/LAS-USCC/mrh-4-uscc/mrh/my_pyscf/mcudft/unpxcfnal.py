import time
import numpy as np
from scipy import linalg
from pyscf.lib import logger, tag_array
from pyscf.dft.rks import _dft_common_init_
from mrh.my_pyscf.mcpdft import otfnal

def kernel (fnal, dm, max_memory=None, hermi=1):
    if max_memory is None: max_memory = fnal.max_memory
    s0 = fnal.get_ovlp ()
    dmc = dm.copy ()
    dms = (2 * dm) - (dm @ s0 @ dm)
    dm1 = np.stack ((dmc,dms), axis=0)
    if getattr (dm, 'mo_coeff', None) is not None and getattr (dm, 'mo_occ', None) is not None:
        mo_coeff, mo_occ = dm.mo_coeff, dm.mo_occ
        mo_coeff = np.stack ((mo_coeff, mo_coeff), axis=0)
        mo_occ = np.stack ((mo_occ, mo_occ * (2 - mo_occ)), axis=0)
        dm1 = tag_array (dm1, mo_coeff=mo_coeff, mo_occ=mo_occ)
        if fnal.verbose >= logger.DEBUG:
            smo = s0 @ mo_coeff[1]
            mo_occ_test = ((dms @ smo) * smo.conjugate ()).sum (0)
            assert (linalg.norm (mo_occ[1] - mo_occ_test) < 1e-10), '{} disagrees with {}'.format (mo_occ[1], mo_occ_test)
            logger.debug (fnal, 'tr[dm_unpaired] = %s', mo_occ[1].sum ())
    ni, xctype, dens_deriv = fnal._numint, fnal.xctype, fnal.dens_deriv

    Exc = 0.0
    make_rho, ndms, nao = ni._gen_rho_evaluator (fnal.mol, dm1, hermi)
    t0 = (logger.process_clock (), logger.perf_counter ())
    for ao, mask, weight, coords in ni.block_loop (fnal.mol, fnal.grids, nao, dens_deriv, max_memory):
        rho_eff = np.stack ([make_rho (spin, ao, mask, xctype) for spin in range (ndms)], axis=0)
        rho_eff = 0.5 * np.stack ((rho_eff.sum (0), rho_eff[0] - rho_eff[1]), axis=0)
        # I do it this way, rather than just passing (dma_eff,dmb_eff) to make_rho, in order to exploit
        # NO-based calculation of densities, which is faster than AO-based calculation but requires
        # positive-definite matrices (dmb_eff is non-positive-definite).
        t0 = logger.timer (fnal, 'effective densities', *t0)
        Exc += fnal.get_E_xc (rho_eff, weight)
        t0 = logger.timer (fnal, 'exchange-correlation energy', *t0)
    return Exc

def _get_E_xc (fnal, rho_eff, weight):
    dexc_ddens  = fnal._numint.eval_xc (fnal.xc, (rho_eff[0,:,:], rho_eff[1,:,:]), spin=1, relativity=0, deriv=0, verbose=fnal.verbose)[0]
    rho = rho_eff[:,0,:].sum (0)
    rho *= weight
    dexc_ddens *= rho

    if fnal.verbose >= logger.DEBUG:
        nelec = rho.sum ()
        logger.debug (fnal, 'MC-UDFT: Total number of electrons in (this chunk of) the total density = %s', nelec)
        ms = np.dot (rho_eff[0,0,:] - rho_eff[1,0,:], weight) / 2.0
        logger.debug (fnal, 'MC-UDFT: Total ms = (neleca - nelecb) / 2 in (this chunk of) the unpaired density = %s', ms)
    return dexc_ddens.sum ()

class unpxcfnal (otfnal.otfnal):

    def __init__(self, mol, xc='LDA,WVN', grids_level=None):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mol.max_memory
        self._keys = set (())
        _dft_common_init_(self, xc=xc)
        if grids_level is not None: self.grids.level = grids_level

    # Fix some names
    @property
    def otxc (self):
        return self.xc
    @property
    def get_E_ot (self, *args, **kwargs):
        return self.get_E_xc (*args, **kwargs)
    @property
    def get_dEot_drho (self, *args, **kwargs):
        return self.get_dExc_drho (*args, **kwargs)

    def get_ovlp (self, mol=None):
        if mol is None: mol = self.mol
        return mol.intor_symmetric('int1e_ovlp')

    get_E_xc = _get_E_xc
    kernel = kernel

