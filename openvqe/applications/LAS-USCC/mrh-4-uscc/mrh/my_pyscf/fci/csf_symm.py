import numpy as np
import scipy
from pyscf import symm, __config__
from pyscf.lib import logger, davidson1
from pyscf.fci import direct_spin1_symm, cistring, direct_uhf
from pyscf.lib.numpy_helper import tag_array
from pyscf.fci.direct_spin1 import _unpack_nelec, _get_init_guess, kernel_ms1
from pyscf.fci.direct_spin1_symm import _gen_strs_irrep, _id_wfnsym
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from mrh.my_pyscf.fci.csf import kernel, pspace, get_init_guess, make_hdiag_csf, make_hdiag_det, unpack_h1e_cs, CSFFCISolver
'''
    MRH 03/24/2019
    IMPORTANT: this solver will interpret a two-component one-body Hamiltonian as [h1e_charge, h1e_spin] where
    h1e_charge = h^p_q (a'_p,up a_q,up + a'_p,down a_q,down)
    h1e_spin   = h^p_q (a'_p,up a_q,up - a'_p,down a_q,down)
    This is to preserve interoperability with the members of direct_spin1_symm, since there is no direct_uhf_symm in pyscf yet.
    Only with an explicitly CSF-based solver can such potentials be included in a calculation that retains S^2 symmetry.
    Multicomponent two-body integrals are currently not available (so this feature is only for use with, e.g., ROHF-CASSCF with 
    with some SOMOs outside of the active space or LASSCF with multiple nonsinglet fragments, not UHF-CASSCF).
'''


class FCISolver (direct_spin1_symm.FCISolver, CSFFCISolver):
    r''' get_init_guess uses csfstring.py and csdstring.py to construct a spin-symmetry-adapted initial guess, and the Davidson algorithm is carried
    out in the CSF basis. However, the ci attribute is put in the determinant basis at the end of it all, and "ci0" is also assumed
    to be in the determinant basis.

    ...However, I want to also do point-group symmetry better than direct_spin1_symm...
    '''

    pspace_size = getattr(__config__, 'fci_csf_FCI_pspace_size', 200)

    def __init__(self, mol=None, smult=None):
        self.smult = smult
        self.transformer = None
        super().__init__(mol)

    make_hdiag = make_hdiag_det

    def absorb_h1e (self, h1e, eri, norb, nelec, fac=1):
        h1e_c, h1e_s = unpack_h1e_cs (h1e)
        h2eff = super().absorb_h1e (h1e_c, eri, norb, nelec, fac)
        if h1e_s is not None:
            h2eff = tag_array (h2eff, h1e_s=h1e_s)
        return h2eff

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        hc = super().contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)
        if hasattr (eri, 'h1e_s'):
           hc += direct_uhf.contract_1e ([eri.h1e_s, -eri.h1e_s], fcivec, norb, nelec, link_index)  
        return hc

    def make_hdiag_csf (self, h1e, eri, norb, nelec, hdiag_det=None):
        self.norb, self.nelec = norb, nelec
        self.check_transformer_cache ()
        return make_hdiag_csf (h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det)

    def pspace (self, h1e, eri, norb, nelec, hdiag_det=None, hdiag_csf=None, npsp=200, **kwargs):
        self.norb, self.nelec = norb, nelec
        self.check_transformer_cache ()
        return pspace (self, h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det,
            hdiag_csf=hdiag_csf, npsp=npsp)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        ''' Over the top of the existing kernel, I just need to set the parameters and cache values related to spin.

        ...and electron configuration point group '''
        if 'nroots' not in kwargs:
            nroots = self.nroots
            kwargs['nroots'] = nroots
        orbsym_back = self.orbsym
        if 'orbsym' not in kwargs:
            kwargs['orbsym'] = self.orbsym
        orbsym = kwargs['orbsym']
        wfnsym_back = self.wfnsym
        if 'wfnsym' not in kwargs:
            wfnsym = self.wfnsym
            kwargs['wfnsym'] = wfnsym
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec
        if 'smult' in kwargs:
            self.smult = kwargs['smult']
            kwargs.pop ('smult')

        # The order of the four things below is super sensitive
        self.orbsym = orbsym
        wfnsym = self.guess_wfnsym(norb, nelec, ci0, **kwargs)
        self.wfnsym = wfnsym
        kwargs['wfnsym'] = wfnsym
        self.check_transformer_cache ()

        idx_sym = self.transformer.confsym[self.transformer.econf_csf_mask] == wfnsym
        e, c = kernel (self, h1e, eri, norb, nelec, smult=self.smult, idx_sym=idx_sym, ci0=ci0, transformer=self.transformer, **kwargs)
        self.eci, self.ci = e, c

        self.orbsym = orbsym_back
        self.wfnsym = wfnsym_back
        return e, c

    def get_init_guess (self, norb, nelec, nroots, hdiag_csf, **kwargs):
        orbsym = kwargs['orbsym'] if 'orbsym' in kwargs else self.orbsym
        wfnsym = kwargs['wfnsym'] if 'wfnsym' in kwargs else self.wfnsym
        self.orbsym = orbsym
        self.wfnsym = wfnsym
        assert ((self.orbsym is not None) and (self.wfnsym is not None))
        self.norb = norb
        self.nelec = nelec
        self.check_transformer_cache ()
        return get_init_guess (norb, nelec, nroots, hdiag_csf, self.transformer)
           

    def check_transformer_cache (self):
        assert (isinstance (self.smult, (int, np.number)))
        neleca, nelecb = _unpack_nelec (self.nelec)
        if isinstance (self.wfnsym, str):
            wfnsym = symm.irrep_name2id (self.mol.groupname, self.wfnsym)
        else:
            wfnsym = self.wfnsym
        if self.transformer is None:
            self.transformer = CSFTransformer (self.norb, neleca, nelecb, self.smult, orbsym=self.orbsym, wfnsym=wfnsym)
        else:
            self.transformer._update_spin_cache (self.norb, neleca, nelecb, self.smult)
            self.transformer._update_symm_cache (self.orbsym)
            self.transformer.wfnsym = wfnsym






