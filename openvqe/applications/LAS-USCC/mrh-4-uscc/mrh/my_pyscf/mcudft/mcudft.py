import numpy as np
from scipy import linalg
from pyscf import fci
from pyscf.lib import logger, tag_array
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from mrh.my_pyscf.mcudft import unpxcfnal

def kernel (mc, fnal, e_mcscf, ci, ncas, nelecas):
    mo_coeff = mc.mo_coeff[:,:mc.ncore+mc.ncas].copy ()
    mo_occ = np.zeros (mo_coeff.shape[-1], dtype=mo_coeff.dtype)
    mo_occ[:mc.ncore] = 2.0
    dm_cas = fci.solver (mc._scf.mol, singlet=False, symm=False).make_rdm12 (ci, ncas, nelecas)[0]
    no_occ, no_umat = linalg.eigh (dm_cas)
    mo_coeff[:,mc.ncore:] = mo_coeff[:,mc.ncore:] @ no_umat
    mo_occ[mc.ncore:] = no_occ
    dm1 = (mo_coeff * mo_occ[None,:]) @ mo_coeff.T
    dm1 = tag_array (dm1, mo_coeff=mo_coeff, mo_occ=mo_occ)
    
    hcore = mc._scf.get_hcore ()
    vj = mc._scf.get_j (dm=dm1)

    logger.debug (mc, 'CAS energy decomposition:')
    Vnn = mc._scf.energy_nuc ()
    logger.debug (mc, 'Vnn = %s', Vnn)
    Te_Vne = np.tensordot (hcore, dm1)
    logger.debug (mc, 'Te + Vne = %s', Te_Vne)
    Ej = 0.5 * np.tensordot (vj, dm1)
    logger.debug (mc, 'Ej = %s', Ej)
    Exc_wfn = e_mcscf - (Vnn + Te_Vne + Ej)
    logger.debug (mc, 'Exc (wfn) = %s', Exc_wfn)
    Exc = fnal.kernel (dm1, max_memory=mc.max_memory)
    logger.debug (mc, 'Exc (UDFT) = %s', Exc)
    return Vnn + Te_Vne + Ej + Exc, Exc

def get_mcudft_child_class (mc, xc, **kwargs):

    class UDFT (mc.__class__):

        def __init__(self, scf, ncas, nelecas, my_xc='LDA,WVN', grids_level=None, **kwargs):
            # Keep the same initialization pattern for backwards-compatibility. Use a separate intializer for the ot functional
            try:
                super().__init__(scf, ncas, nelecas)
            except TypeError as e:
                # I think this is the same DFCASSCF problem as with the DF-SACASSCF gradients earlier
                super().__init__()
            keys = set (('e_xc', 'xc', 'grids_level', 'fnal', 'e_mcscf' 'e_states'))
            self._keys = set ((self.__dict__.keys ())).union (keys)
            self.fnal = unpxcfnal.unpxcfnal (self.mol, xc=xc, grids_level=grids_level)

        @property
        def xc (self):
            return self.fnal.xc
        @xc.setter
        def xc (self, x):
            self.fnal = unpxcfnal.unpxcfnal (self.mol, xc=x, grids_level=self.grids_level)

        @property
        def grids_level (self):
            return self.fnal.grids.level
        @grids_level.setter
        def grids_level (self, x):
            self.fnal = unpxcfnal.unpxcfnal (self.mol, xc=self.xc, grids_level=x)
          
        def kernel (self, mo=None, ci=None, **kwargs):
            # Hafta reset the grids so that geometry optimization works!
            self.fnal = unpxcfnal.unpxcfnal (self.mol, xc=self.xc, grids_level=self.grids_level)
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = super().kernel (mo, ci, **kwargs)
            if isinstance (self, StateAverageMCSCFSolver):
                raise NotImplementedError ('State average extension of MC-UDFT')
            else:
                self.e_tot, self.e_xc = kernel (self, self.fnal, self.e_mcscf, self.ci, self.ncas, self.nelecas)
            return self.e_tot, self.e_xc, self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

        def nuc_grad_method (self):
            return NotImplementedError ('MC-UDFT analytic gradients')

        def get_energy_decomposition (self, mo_coeff=None, ci=None):
            if mo_coeff is None: mo_coeff = self.mo_coeff
            if ci is None: ci = self.ci
            return get_energy_decomposition (self, self.otfnal, mo_coeff=mo_coeff, ci=ci)

        def state_average (self, weights=(0.5,0.5)):
            raise NotImplementedError ('State average extension of MC-UDFT')

        def state_average_(self, weights=(0.5,0.5)):
            raise NotImplementedError ('State average extension of MC-UDFT')


    udft = UDFT (mc._scf, mc.ncas, mc.nelecas, my_xc=xc, **kwargs)
    udft.__dict__.update (mc.__dict__)
    return udft

  
