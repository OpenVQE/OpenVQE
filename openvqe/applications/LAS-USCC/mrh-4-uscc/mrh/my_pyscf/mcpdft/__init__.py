# Lahh dee dah
import copy
from pyscf.mcpdft.mcpdft import get_mcpdft_child_class
from pyscf import mcscf, gto
from pyscf.lib import logger
from pyscf.mcscf import mc1step, casci
import mrh
from mrh.util.io import mcpdft_removal_warn
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCFNoSymm, LASSCFSymm 


mcpdft_removal_warn ()

# NOTE: As of 02/06/2022, initializing PySCF mcscf classes with a symmetry-enabled molecule
# doesn't work.

def _MCPDFT (mc_class, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None,
             **kwargs):
    if isinstance (mc_or_mf_or_mol, (mc1step.CASSCF, casci.CASCI)):
        mc0 = mc_or_mf_or_mol
        mf_or_mol = mc_or_mf_or_mol._scf
    else:
        mc0 = None
        mf_or_mol = mc_or_mf_or_mol
    if isinstance (mf_or_mol, gto.Mole) and mf_or_mol.symmetry:
        logger.warn (mf_or_mol,
                     'Initializing MC-SCF with a symmetry-adapted Mole object may not work!')
    if frozen is not None: mc1 = mc_class (mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    else: mc1 = mc_class (mf_or_mol, ncas, nelecas, ncore=ncore)
    
    mc2 = get_mcpdft_child_class (mc1, ot, **kwargs)

    if mc0 is not None:
        mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy ()    
        mc2.ci = copy.deepcopy (mc_or_mf_or_mol.ci)
        mc2.converged = mc0.converged
    return mc2

def CASSCFPDFT (mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None,
                **kwargs):
    return _MCPDFT (mcscf.CASSCF, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                    **kwargs)

def CASCIPDFT (mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, **kwargs):
    return _MCPDFT (mcscf.CASCI, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore,
                    **kwargs)

CASSCF=CASSCFPDFT
CASCI=CASCIPDFT


# LAS-PDFT
def _laspdftEnergy(mc_class, mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, DoLASSI=False, ncore=None, spin_sub=None,
                   frozen=None, **kwargs):
    if isinstance (mc_or_mf_or_mol, (LASSCFNoSymm, LASSCFSymm)):
        mc0 = mc_or_mf_or_mol
        mf_or_mol = mc_or_mf_or_mol._scf
    else:
        mc0 = None
        mf_or_mol = mc_or_mf_or_mol
    if isinstance (mf_or_mol, gto.Mole) and mf_or_mol.symmetry:
        logger.warn (mf_or_mol,
                    'Initializing MC-SCF with a symmetry-adapted Mole object may not work!')

    if frozen is not None: mc1 = mc_class(mf_or_mol, ncas_sub, nelecas_sub, ncore=ncore, spin_sub = spin_sub, frozen=frozen)
    else:  mc1 = mc_class(mf_or_mol, ncas_sub, nelecas_sub, ncore=ncore, spin_sub = spin_sub)
    
    from mrh.my_pyscf.mcpdft.laspdft import get_mcpdft_child_class
    mc2 = get_mcpdft_child_class(mc1, ot, DoLASSI=DoLASSI, **kwargs)

    if mc0 is not None:
        mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy ()    
        mc2.ci = copy.deepcopy (mc_or_mf_or_mol.ci)
        mc2.converged = mc0.converged
    return mc2
 
def LASSCFPDFT(mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub,  ncore=None, spin_sub=None, frozen=None,
               **kwargs):
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    return _laspdftEnergy(LASSCF,  mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, ncore=ncore,
                          spin_sub=spin_sub, frozen=frozen, **kwargs)

def LASSIPDFT(mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, ncore=None, spin_sub=None, frozen=None,
               **kwargs):
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    return _laspdftEnergy(LASSCF,  mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, DoLASSI=True, ncore=ncore,
                          spin_sub=spin_sub, frozen=frozen, **kwargs)

LASSCF = LASSCFPDFT
LASSI = LASSIPDFT

def CIMCPDFT (*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn (mcscf.CASCI, *args, **kwargs)
def CIMCPDFT_SCF (*args, **kwargs):
    from mrh.my_pyscf.mcpdft.var_mcpdft import CIMCPDFT as fn
    return fn (mcscf.CASSCF, *args, **kwargs)


