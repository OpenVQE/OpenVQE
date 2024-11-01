import numpy as np

def cas_mo_energy_shift_4_jmol (mo_ene, norb, nelec, ncas, nelecas):
    ''' Return shifted mo energy to preserve orbital order when viewing molden file in jmol '''
    nelec = np.sum (np.asarray (nelec))
    nelecas = np.sum (np.asarray (nelecas))
    ncore = (nelec - nelecas) // 2
    assert ((nelec - nelecas) % 2 == 0)
    nocc = ncore+ncas
    mo_ene[ncore:nocc] = 0
    if np.any (np.diff (mo_ene[:ncore]) < 0):
        mo_ene[:ncore] = 0
    elif ncore > 0 and mo_ene[ncore-1] > 0:
        mo_ene[:ncore] -= mo_ene[ncore-1]
    if np.any (np.diff (mo_ene[nocc:]) < 0):
        mo_ene[nocc:] = 0
    elif nocc < norb and mo_ene[nocc] < 0:
        mo_ene[nocc:] += mo_ene[nocc]
    return mo_ene




