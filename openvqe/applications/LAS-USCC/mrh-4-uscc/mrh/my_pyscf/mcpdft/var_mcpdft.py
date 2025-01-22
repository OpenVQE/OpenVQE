from pyscf import gto
from pyscf.mcscf import mc1step, casci
from pyscf.mcpdft import mcpdft
from mrh.my_pyscf.mcpdft import ci_scf


def CIMCPDFT (mc_class, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None,
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

    class PDFT (mcpdft._PDFT, mc1.__class__):
        if isinstance (mc1, mc1step.CASSCF):
            casci=ci_scf.mc1step_casci # CASSCF CI step
            update_casdm=ci_scf.mc1step_update_casdm # innercycle CI update
        else:
            kernel=ci_scf.casci_kernel # CASCI
            _finalize=ci_scf.casci_finalize # I/O clarity

    mc2 = PDFT (mc1._scf, mc1.ncas, mc1.nelecas, my_ot=ot, **kwargs)
    _keys = mc1._keys.copy ()
    mc2.__dict__.update (mc1.__dict__)
    mc2._keys = mc2._keys.union (_keys)

    if mc0 is not None:
        mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy ()
        mc2.ci = copy.deepcopy (mc_or_mf_or_mol.ci)
        mc2.converged = mc0.converged
    return mc2



