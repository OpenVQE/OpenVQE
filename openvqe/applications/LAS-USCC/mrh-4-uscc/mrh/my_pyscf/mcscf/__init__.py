# Ohh boy
from mrh.my_pyscf.mcscf.mc1step_csf import fix_ci_response_csf

class _DFLASCI: # Tag
    pass

def constrCASSCF(mf, ncas, nelecas, **kwargs):
    from pyscf import scf
    from pyscf.mcscf import addons
    from mrh.my_pyscf.mcscf import mc1step_constrained
    mf = scf.addons.convert_to_rhf(mf)
    return mc1step_constrained.CASSCF (mf, ncas, nelecas, **kwargs)

constrRCASSCF = constrCASSCF
