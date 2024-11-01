# Lahh dee dah
from mrh.my_pyscf.mcudft.mcudft import get_mcudft_child_class
from pyscf import mcscf

def CASSCFPDFT (mf_or_mol, xc, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    mc = mcscf.CASSCF (mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    return get_mcudft_child_class (mc, xc, **kwargs)

def CASCIPDFT (mf_or_mol, xc, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    mc = mcscf.CASCI (mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    return get_mcudft_child_class (mc, xc, **kwargs)

CASSCF=CASSCFPDFT
CASCI=CASCIPDFT

