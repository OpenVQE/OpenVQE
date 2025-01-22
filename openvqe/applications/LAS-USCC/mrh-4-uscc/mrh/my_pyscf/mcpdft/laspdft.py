from pyscf import ao2mo, lib
import numpy as np
import copy
from scipy import linalg
from types import MethodType
from copy import deepcopy
from mrh.my_pyscf.df.sparse_df import sparsedf_array
from mrh.my_pyscf.lassi import lassi

try:
    from pyscf.mcpdft.mcpdft import _PDFT, _mcscf_env
except ImportError:
        msg = "For performing LASPDFT, you will require pyscf-forge.\n" +\
        "pyscf-forge can be found at : https://github.com/pyscf/pyscf-forge"
        raise ImportError(msg)

class _LASPDFT(_PDFT):
    'MC-PDFT energy for a LASSCF wavefunction'
        
    def get_h2eff(self, mo_coeff=None):
        'Compute the active space two-particle Hamiltonian.'
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        if mo_coeff is None: mo_coeff = self.mo_coeff[:,ncore:nocc]
        elif mo_coeff.shape[1] != ncas: mo_coeff = mo_coeff[:,ncore:nocc]

        if getattr (self._scf, '_eri', None) is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                                max_memory=self.max_memory)
        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                                max_memory=self.max_memory)
        return eri
        
def get_mcpdft_child_class(mc, ot, DoLASSI=False,  **kwargs):
    mc_doc = (mc.__class__.__doc__ or 'No docstring for MC-SCF parent method')
   
    class PDFT(_LASPDFT, mc.__class__):
        __doc__= mc_doc + '\n\n' + _LASPDFT.__doc__
        _mc_class = mc.__class__
        setattr(_mc_class, 'DoLASSI', None)

        def get_h2eff(self, mo_coeff=None):
            if self._in_mcscf_env: return mc.__class__.get_h2eff(self, mo_coeff=mo_coeff)
            else: return _LASPDFT.get_h2eff(self, mo_coeff=mo_coeff)
        
        if DoLASSI:  _mc_class.DoLASSI = True
        else: _mc_class.DoLASSI = False

        if _mc_class.DoLASSI:
            # This code doesn't seem efficent, have to calculate the casdm1 and casdm2 in different functions.
            def make_one_casdm1s(self, ci=None, state=0, **kwargs):
                with lib.temporary_env (self, verbose=2):
                    casdm1s = lassi.root_make_rdm12s (self, ci=ci, si=self.si, state=state)[0]
                return casdm1s
            def make_one_casdm2(self, ci=None, state=0, **kwargs):
                with lib.temporary_env (self, verbose=2):
                    casdm2s = lassi.root_make_rdm12s (self, ci=ci, si=self.si, state=state)[1]
                return casdm2s.sum ((0,3))
        else:
            make_one_casdm1s=mc.__class__.state_make_casdm1s
            make_one_casdm2=mc.__class__.state_make_casdm2

        # TODO: in pyscf-forge/pyscf/mcpdft/mcpdft.py::optimize_mcscf_, generalize the number
        # of return arguments. Then the redefinition below will be unnecessary. 
        def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
            '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
            Has the same calling signature as the parent kernel method. '''
            with _mcscf_env(self):
                self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
                    self._mc_class.kernel(self, mo_coeff, ci0=ci0, **kwargs)[:-2]
                self.fcisolver.nroots = self.nroots
                if self.DoLASSI:
                    self.e_states, self.si = self.lassi()
                return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    pdft = PDFT(mc._scf, mc.ncas_sub, mc.nelecas_sub, my_ot=ot, **kwargs)

    _keys = pdft._keys.copy()
    pdft.__dict__.update (mc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    return pdft

