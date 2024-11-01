import numpy as np
from pyscf.lo import orth
from pyscf.lib import tag_array

def _localize (las, frags_orbs, mo_coeff, spin, lo_coeff, fock, ao_ovlp, freeze_cas_spaces=False):
    ''' Project active orbitals into sets of orthonormal "fragments" defined by lo_coeff
    and frags_orbs, and orthonormalize inactive and virtual orbitals in the orthogonal complement
    space. Beware that unless freeze_cas_spaces=True, frozen orbitals will not be preserved.

    Args:
        las: LASSCF or LASCI object
        frags_orbs: list of length nfrags
            Contains list of AO indices formally defining the fragments
            into which the active orbitals are to be localized

    Kwargs: (some of these are args here but kwargs in the actual caller)
        mo_coeff: ndarray of shape (nao, nmo)
            Molecular orbital coefficients containing active orbitals
            on columns ncore:ncore+ncas
        spin: integer
            Unused; retained for backwards compatibility I guess
        lo_coeff: ndarray of shape (nao, nao)
            Linear combinations of AOs that are localized and orthonormal
        fock: ndarray of shape (nmo, nmo)
            Effective 1-electron Hamiltonian matrix for recanonicalizing
            the inactive and external sectors after the latter are
            possibly distorted by the projection of the active orbitals
        ao_ovlp: ndarray of shape (nao, nao)
            Overlap matrix of the underlying AO basis
        freeze_cas_spaces: logical
            If true, then active orbitals are mixed only among themselves
            when localizing, which leaves the inactive and external sectors
            unchanged (to within numerical precision). Otherwise, active
            orbitals are projected into the localized-orbital space and
            the inactive and external orbitals are reconstructed as closely
            as possible using SVD.

    Returns:
        mo_coeff: ndarray of shape (nao,nmo)
            Orbital coefficients after localization of the active space;
            columns in the order (inactive,las1,las2,...,lasn,external)
    '''
    # For reasons that pass my understanding, mo_coeff sometimes can't be assigned symmetry
    # by PySCF's own code. Therefore, I'm going to keep the symmetry tags on mo_coeff
    # and make sure the SVD engine sees them and doesn't try to figure it out itself.
    # Hopefully this never becomes a problem with the lo_coeff.
    ncore, ncas, ncas_sub = las.ncore, las.ncas, las.ncas_sub
    nocc = ncore + ncas
    nfrags = len (frags_orbs)
    nao, nmo = mo_coeff.shape
    unused_aos = np.ones (nao, dtype=np.bool_)
    for frag_orbs in frags_orbs: unused_aos[frag_orbs] = False
    has_orbsym = hasattr (mo_coeff, 'orbsym')
    mo_orbsym = getattr (mo_coeff, 'orbsym', np.zeros (nmo))
    mo_coeff = mo_coeff.copy () # Safety

    # SVD to pick active orbitals
    mo_cas = tag_array (mo_coeff[:,ncore:nocc], orbsym=mo_orbsym[ncore:nocc])
    if freeze_cas_spaces:
        null_coeff = np.hstack ([mo_coeff[:,:ncore], mo_coeff[:,nocc:]])
    else:
        null_coeff = lo_coeff[:,unused_aos]
    for ix, (nlas, frag_orbs) in enumerate (zip (las.ncas_sub, frags_orbs)):
        try:
            mo_proj, sval, mo_cas = las._svd (lo_coeff[:,frag_orbs], mo_cas, s=ao_ovlp)
        except ValueError as e:
            print (ix, lo_coeff[:,frag_orbs].shape, ao_ovlp.shape, mo_cas.shape)
            print (mo_cas.orbsym)
            raise (e)
        i, j = ncore + sum (las.ncas_sub[:ix]), ncore + sum (las.ncas_sub[:ix]) + nlas
        mo_las = mo_cas if freeze_cas_spaces else mo_proj
        mo_coeff[:,i:j] = mo_las[:,:nlas]
        if has_orbsym: mo_orbsym[i:j] = mo_las.orbsym[:nlas]
        if freeze_cas_spaces:
            if has_orbsym: orbsym = mo_cas.orbsym[nlas:]
            mo_cas = mo_cas[:,nlas:]
            if has_orbsym: mo_cas = tag_array (mo_cas, orbsym=orbsym)
        else:
            null_coeff = np.hstack ([null_coeff, mo_proj[:,nlas:]])

    # SVD of null space to pick inactive orbitals
    assert (null_coeff.shape[-1] + ncas == nmo)
    mo_core = tag_array (mo_coeff[:,:ncore], orbsym=mo_orbsym[:ncore])
    mo_proj, sval, mo_core = las._svd (null_coeff, mo_core, s=ao_ovlp)
    mo_coeff[:,:ncore], mo_coeff[:,nocc:] = mo_proj[:,:ncore], mo_proj[:,ncore:]
    if has_orbsym:
        mo_orbsym[:ncore] = mo_proj.orbsym[:ncore]
        mo_orbsym[nocc:] = mo_proj.orbsym[ncore:]
    mo_coeff = tag_array (mo_coeff, orbsym=mo_orbsym)

    # Canonicalize for good init CI guess and visualization
    ranges = [(0,ncore),(nocc,nmo)]
    for ix, di in enumerate (ncas_sub):
        i = ncore + sum (ncas_sub[:ix])
        ranges.append ((i,i+di))
    fock = mo_coeff.conj ().T @ fock @ mo_coeff
    for i, j in ranges:
        if (j == i): continue
        e, c = las._eig (fock[i:j,i:j], i, j)
        idx = np.argsort (e)
        mo_coeff[:,i:j] = mo_coeff[:,i:j] @ c[:,idx]
        mo_orbsym[i:j] = mo_orbsym[i:j][idx]
    if has_orbsym: mo_coeff = tag_array (mo_coeff, orbsym=mo_orbsym)
    else: mo_coeff = np.array (mo_coeff) # remove spurious tag
    return mo_coeff

def localize_init_guess (las, frags_atoms, mo_coeff=None, spin=None, lo_coeff=None, fock=None,
                         freeze_cas_spaces=False):
    ''' Project active orbitals into sets of orthonormal "fragments" defined by lo_coeff
    and frags_orbs, and orthonormalize inactive and virtual orbitals in the orthogonal complement
    space. Beware that unless freeze_cas_spaces=True, frozen orbitals will not be preserved.

    Args:
        frags_atoms: list of length nfrags
            Contains either lists of integer atom indices, or lists of
            strings which are passed to mol.search_ao_label, which define
            fragments into which the active orbitals are to be localized

    Kwargs:
        mo_coeff: ndarray of shape (nao, nmo)
            Molecular orbital coefficients containing active orbitals
            on columns ncore:ncore+ncas
        spin: integer
            Unused; retained for backwards compatibility I guess
        lo_coeff: ndarray of shape (nao, nao)
            Linear combinations of AOs that are localized and orthonormal
        fock: ndarray of shape (nmo, nmo)
            Effective 1-electron Hamiltonian matrix for recanonicalizing
            the inactive and external sectors after the latter are
            possibly distorted by the projection of the active orbitals
        ao_ovlp: ndarray of shape (nao, nao)
            Overlap matrix of the underlying AO basis
        freeze_cas_spaces: logical
            If true, then active orbitals are mixed only among themselves
            when localizing, which leaves the inactive and external sectors
            unchanged (to within numerical precision). Otherwise, active
            orbitals are projected into the localized-orbital space and
            the inactive and external orbitals are reconstructed as closely
            as possible using SVD.

    Returns:
        mo_coeff: ndarray of shape (nao,nmo)
            Orbital coefficients after localization of the active space;
            columns in the order (inactive,las1,las2,...,lasn,external)
    '''
    if mo_coeff is None:
        mo_coeff = las.mo_coeff
    if lo_coeff is None:
        lo_coeff = orth.orth_ao (las.mol, 'meta_lowdin')
    if spin is None:
        spin = las.nelecas[0] - las.nelecas[1]
    assert (spin % 2 == sum (las.nelecas) % 2)
    assert (len (frags_atoms) == len (las.ncas_sub))
    frags_atoms_int = all ([all ([isinstance (i, int) for i in j]) for j in frags_atoms])
    frags_atoms_str = all ([all ([isinstance (i, str) for i in j]) for j in frags_atoms])
    if frags_atoms_int:
        ao_offset = las.mol.offset_ao_by_atom ()
        frags_orbs = [[orb for atom in frags_atoms 
                       for orb in list (range (ao_offset[atom,2], ao_offset[atom,3]))]
                      for frags_atoms in frags_atoms]
    elif frags_atoms_str:
        frags_orbs = [las.mol.search_ao_label (i) for i in frags_atoms]
    else:
        raise RuntimeError ('localize_init_guess requires either all integers or all strings to identify fragments')
    if fock is None: fock = las._scf.get_fock ()
    ao_ovlp = las._scf.get_ovlp ()
    return _localize (las, frags_orbs, mo_coeff, spin, lo_coeff, fock, ao_ovlp, freeze_cas_spaces=freeze_cas_spaces)



