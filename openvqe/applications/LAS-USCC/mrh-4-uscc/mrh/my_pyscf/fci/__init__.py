from mrh.my_pyscf.fci import csf, csf_symm
from mrh.my_pyscf.fci.csfstring import CSFTransformer

def csf_solver(mol=None, smult=None, symm=None):
    if mol and symm is None:
        symm = mol.symmetry
    if symm:
        return csf_symm.FCISolver (mol=mol, smult=smult)
    else:
        return csf.FCISolver (mol=mol, smult=smult)

