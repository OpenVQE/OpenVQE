import itertools

import numpy as np
from qat.core import Term
from qat.fermion import FermionHamiltonian as Hamiltonian
from qat.fermion.chemistry.ucc_deprecated import get_cluster_ops_and_init_guess
from qat.fermion.transforms import (
    transform_to_bk_basis,
    transform_to_jw_basis,
    transform_to_parity_basis,
)

from .fermion_util import order_fermionic_term



def uccsd(hamiltonian, n_elec, noons_full, orb_energies_full, transform):
    """
    This function is responsible for constructing the cluster operators, the MP2 initial guesses variational parameters of the UCCSD ansatz.
    It computes the cluster operators in the spin representation and the size of the pool.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The electronic structure Hamiltonian
    
    n_elec : int
        The number of electrons
    
    noons_full : List<float>
        The list of noons
    
    orb_energies_full : List<float>
        The list of orbital energies with double degeneracy
        
    transform : string
        type of transformation. Either 'JW', 'Bravyi-Kitaev', or 'parity_basis'

    Returns
    -------
    pool_size: int
        The number of the cluster operators
    cluster_ops: list[Hamiltonian]
        list of fermionic cluster operators
    cluster_ops_sp: list[Hamiltonian]
        list of spin cluster operators
    theta_MP2: list[float]
        list of parameters in the MP2 pre-screening process
    hf_init: int
        the integer corresponding to the occupation of the Hartree-Fock solution
    """

    cluster_ops, theta_MP2, hf_init = get_cluster_ops_and_init_guess(
        n_elec, noons_full, orb_energies_full, hamiltonian.hpqrs
    )
    if transform == "JW":
        pool_size = len(cluster_ops)
        cluster_ops_sp = []
        for y in cluster_ops:
            hamilt_sp = transform_to_jw_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops_sp.append(hamilt_sp)
        return pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init
    if transform == "Bravyi-Kitaev":
        pool_size = len(cluster_ops)
        cluster_ops_sp = []
        for y in cluster_ops:
            hamilt_sp = transform_to_bk_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops_sp.append(hamilt_sp)
        return pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init
    if transform == "parity_basis":
        pool_size = len(cluster_ops)
        cluster_ops_sp = []
        for y in cluster_ops:
            hamilt_sp = transform_to_parity_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops_sp.append(hamilt_sp)
        return pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init


def spin_complement_gsd(n_elec, orbital_number, transform):
    """
    This function is responsible for generating the spin complement generalized single and double excitations.

    Parameters
    ----------
    n_elec: int
        The number of electrons

    orbital_number: int
        The number of orbitals
    
    transform: string
        type of transformation. Either 'JW', 'Bravyi-Kitaev', or 'parity_basis'
    
    Returns
    -------

    pool_size: int
        The number of the cluster operators

    cluster_ops: List<Hamiltonian>
        List of fermionic cluster operators
        
    cluster_ops_sp: List<Hamiltonian>
        List of spin cluster operators

    """
    spin_complement_gsd = []
    for p in range(0, orbital_number):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, orbital_number):
            qa = 2 * q
            qb = 2 * q + 1

            term_a = Hamiltonian(
                2 * orbital_number,
                [
                    Term(1, "Cc", [pa, qa]),
                    Term(-1, "Cc", [qa, pa]),
                    Term(1, "Cc", [pb, qb]),
                    Term(-1, "Cc", [qb, pb]),
                ],
            )
            spin_complement_gsd.append(term_a)

    pq = -1
    for p in range(0, orbital_number):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, orbital_number):
            qa = 2 * q
            qb = 2 * q + 1

            pq += 1

            rs = -1
            for r in range(0, orbital_number):
                ra = 2 * r
                rb = 2 * r + 1

                for s in range(r, orbital_number):
                    sa = 2 * s
                    sb = 2 * s + 1

                    rs += 1

                    if pq > rs:
                        continue
                    term_a = [
                        Term(1, "CcCc", [ra, pa, sa, qa]),
                        Term(-1, "CcCc", [qa, sa, pa, ra]),
                        Term(1, "CcCc", [rb, pb, sb, qb]),
                        Term(-1, "CcCc", [qb, sb, pb, rb]),
                    ]

                    term_b = [
                        Term(1, "CcCc", [ra, pa, sb, qb]),
                        Term(-1, "CcCc", [qb, sb, pa, ra]),
                        Term(1, "CcCc", [rb, pb, sa, qa]),
                        Term(-1, "CcCc", [qa, sa, pb, rb]),
                    ]

                    term_c = [
                        Term(1, "CcCc", [ra, pb, sb, qa]),
                        Term(-1, "CcCc", [qa, sb, pb, ra]),
                        Term(1, "CcCc", [rb, pa, sa, qb]),
                        Term(-1, "CcCc", [qb, sa, pa, rb]),
                    ]
                    #
                    ordered_term_a = 0
                    for t1 in term_a:
                        t_list = order_fermionic_term(t1)
                        ordered_term_a = ordered_term_a + Hamiltonian(
                            2 * orbital_number, terms=[t for t in t_list]
                        )
                    ordered_term_b = 0
                    for t2 in term_b:
                        t_list = order_fermionic_term(t2)
                        ordered_term_b = ordered_term_b + Hamiltonian(
                            2 * orbital_number, terms=[t for t in t_list]
                        )
                    ordered_term_c = 0
                    for t2 in term_c:
                        t_list = order_fermionic_term(t2)
                        ordered_term_c = ordered_term_c + Hamiltonian(
                            2 * orbital_number, terms=[t for t in t_list]
                        )
                    spin_complement_gsd.append(ordered_term_a)
                    spin_complement_gsd.append(ordered_term_b)
                    spin_complement_gsd.append(ordered_term_c)

    if transform == "JW":
        cluster_ops = []
        cluster_ops_sp = []
        for y in spin_complement_gsd:
            hamilt_sp = transform_to_jw_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "Bravyi-Kitaev":
        cluster_ops = []
        cluster_ops_sp = []
        for y in spin_complement_gsd:
            hamilt_sp = transform_to_bk_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "parity_basis":
        cluster_ops = []
        cluster_ops_sp = []
        for y in spin_complement_gsd:
            hamilt_sp = transform_to_parity_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp

# ----------------------------------------twin QLM---------------------------------------
def spin_complement_gsd_twin(n_elec, orbital_number, transform):
    """
    This function is responsible for generating the spin complement generalized single and double excitations. This is the twin
    version of spin_complement_gsd

    Parameters
    ----------
    n_elec: int
        The number of electrons

    orbital_number: int
        The number of orbitals
    
    transform: string
        type of transformation. Either 'JW', 'Bravyi-Kitaev', or 'parity_basis'
    
    Returns
    -------

    pool_size: int
        The number of the cluster operators

    cluster_ops: List<Hamiltonian>
        List of fermionic cluster operators
        
    cluster_ops_sp: List<Hamiltonian>
        List of spin cluster operators

    """

    alpha_orbs = [2 * i for i in range(orbital_number)]
    beta_orbs = [2 * i + 1 for i in range(orbital_number)]

    spin_complement_gsd_twin = []
    term_a = []

    # aa
    for p in alpha_orbs:
        for q in alpha_orbs:
            if p >= q:
                continue

            term_a = Hamiltonian(
                2 * orbital_number,
                [
                    Term(1, "Cc", [q, p]),
                    Term(-1, "Cc", [p, q]),
                    Term(1, "Cc", [q + 1, p + 1]),
                    Term(-1, "Cc", [p + 1, q + 1]),
                ],
            )

            spin_complement_gsd_twin.append(term_a)
    pq = 0
    term_b = []
    for p in alpha_orbs:
        for q in alpha_orbs:
            if p > q:
                continue
            rs = 0
            for r in alpha_orbs:
                for s in alpha_orbs:
                    if r > s:
                        continue
                    if pq < rs:
                        continue

                    term_b = [
                        Term(1, "CcCc", [r, p, s, q]),
                        Term(-1, "CcCc", [q, s, p, r]),
                        Term(1, "CcCc", [r + 1, p + 1, s + 1, q + 1]),
                        Term(-1, "CcCc", [q + 1, s + 1, p + 1, r + 1]),
                    ]
                    ordered_term_b = 0
                    for t1 in term_b:
                        t_list = order_fermionic_term(t1)
                        ordered_term_b = ordered_term_b + Hamiltonian(
                            2 * orbital_number, terms=[t for t in t_list]
                        )

                    spin_complement_gsd_twin.append(ordered_term_b)

                    rs += 1
            pq += 1

    pq = 0
    for p in alpha_orbs:
        for q in beta_orbs:
            rs = 0
            for r in alpha_orbs:
                for s in beta_orbs:
                    if pq < rs:
                        continue
                    ordered_term_c = 0
                    term_c = [Term(1, "CcCc", [r, p, s, q])]
                    if p > q:
                        continue
                    term_c += [
                        Term(1, "CcCc", [s - 1, q - 1, r + 1, p + 1]),
                        Term(-1, "CcCc", [q, s, p, r]),
                        Term(-1, "CcCc", [p + 1, r + 1, q - 1, s - 1]),
                    ]

                    for t1 in term_c:
                        t_list = order_fermionic_term(t1)
                        ordered_term_c = ordered_term_c + Hamiltonian(
                            2 * orbital_number, terms=[t for t in t_list]
                        )
                    spin_complement_gsd_twin.append(ordered_term_c)
                    rs += 1
            pq += 1

    if transform == "JW":
        cluster_ops = []
        cluster_ops_sp = []
        for y in spin_complement_gsd_twin:
            hamilt_sp = transform_to_jw_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "Bravyi-Kitaev":
        cluster_ops = []
        cluster_ops_sp = []
        for y in spin_complement_gsd_twin:
            hamilt_sp = transform_to_bk_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "parity_basis":
        cluster_ops = []
        cluster_ops_sp = []
        for y in spin_complement_gsd_twin:
            hamilt_sp = transform_to_parity_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp


def singlet_sd(n_elec, orbital_number, transform):
    """
    This function is responsible for generating the single and double excitations occured
    from occupied to unoccupied orbitals within singlet spin symmetry.

    Parameters
    ----------
    n_elec: int
        The number of electrons

    orbital_number: int
        The number of orbitals
    
    transform: string
        type of transformation. Either 'JW', 'Bravyi-Kitaev', or 'parity_basis'
    
    Returns
    -------

    pool_size: int
        The number of the cluster operators

    cluster_ops: List<Hamiltonian>
        List of fermionic cluster operators
        
    cluster_ops_sp: List<Hamiltonian>
        List of spin cluster operators

    """

    singlet_sd = []
    n_occ = int(np.ceil(n_elec / 2))
    n_vir = orbital_number - n_occ

    for i in range(0, n_occ):
        ia = 2 * i
        ib = 2 * i + 1

        for a in range(0, n_vir):
            aa = 2 * n_occ + 2 * a
            ab = 2 * n_occ + 2 * a + 1

            term_a = Hamiltonian(
                2 * orbital_number,
                [
                    Term(1 / np.sqrt(2), "Cc", [aa, ia]),
                    Term(+1 / np.sqrt(2), "Cc", [ab, ib]),
                    Term(-1 / np.sqrt(2), "Cc", [ia, aa]),
                    Term(-1 / np.sqrt(2), "Cc", [ib, ab]),
                ],
            )
            # Normalize
            coeff_a = 0
            for t in term_a.terms:
                coeff_t = t.coeff
                coeff_a += coeff_t * coeff_t
            term_a = term_a / np.sqrt(coeff_a)
            singlet_sd.append(term_a)

    term_a = 0
    term_b = 0
    for i in range(0, n_occ):
        ia = 2 * i
        ib = 2 * i + 1
        for j in range(i, n_occ):
            ja = 2 * j
            jb = 2 * j + 1
            for a in range(0, n_vir):
                aa = 2 * n_occ + 2 * a
                ab = 2 * n_occ + 2 * a + 1
                for b in range(a, n_vir):
                    ba = 2 * n_occ + 2 * b
                    bb = 2 * n_occ + 2 * b + 1
                    term_a = [
                        Term(2 / np.sqrt(12), "CCcc", [aa, ba, ia, ja]),
                        Term(-2 / np.sqrt(12), "CCcc", [ja, ia, ba, aa]),
                        Term(2 / np.sqrt(12), "CCcc", [ab, bb, ib, jb]),
                        Term(-2 / np.sqrt(12), "CCcc", [jb, ib, bb, ab]),
                        Term(1 / np.sqrt(12), "CCcc", [aa, bb, ia, jb]),
                        Term(-1 / np.sqrt(12), "CCcc", [jb, ia, bb, aa]),
                        Term(1 / np.sqrt(12), "CCcc", [ab, ba, ib, ja]),
                        Term(-1 / np.sqrt(12), "CCcc", [ja, ib, ba, ab]),
                        Term(1 / np.sqrt(12), "CCcc", [aa, bb, ib, ja]),
                        Term(-1 / np.sqrt(12), "CCcc", [ja, ib, bb, aa]),
                        Term(1 / np.sqrt(12), "CCcc", [ab, ba, ia, jb]),
                        Term(-1 / np.sqrt(12), "CCcc", [jb, ia, ba, ab]),
                    ]
                    term_b = [
                        Term(1 / 2, "CCcc", [aa, bb, ia, jb]),
                        Term(-1 / 2, "CCcc", [jb, ia, bb, aa]),
                        Term(1 / 2, "CCcc", [ab, ba, ib, ja]),
                        Term(-1 / 2, "CCcc", [ja, ib, ba, ab]),
                        Term(-1 / 2, "CCcc", [aa, bb, ib, ja]),
                        Term(1 / 2, "CCcc", [ja, ib, bb, aa]),
                        Term(-1 / 2, "CCcc", [ab, ba, ia, jb]),
                        Term(1 / 2, "CCcc", [jb, ia, ba, ab]),
                    ]

                    ordered_term_a = 0
                    for t1 in term_a:
                        t_list = order_fermionic_term(t1)
                        ordered_term_a = ordered_term_a + Hamiltonian(
                            2 * orbital_number, terms=[t for t in t_list]
                        )
                    ordered_term_b = 0
                    for t2 in term_b:
                        t_list = order_fermionic_term(t2)
                        ordered_term_b = ordered_term_b + Hamiltonian(
                            2 * orbital_number, terms=[t for t in t_list]
                        )
                    coeff_a = 0
                    coeff_b = 0
                    for t in ordered_term_a.terms:
                        coeff_t = t.coeff
                        coeff_a += abs(coeff_t * coeff_t)
                    for t1 in ordered_term_b.terms:
                        coeff_t = t1.coeff
                        coeff_b += abs(coeff_t * coeff_t)
                    if coeff_a > 0:
                        term_a = ordered_term_a / np.sqrt(coeff_a)
                        singlet_sd.append(term_a)
                    if coeff_b > 0:
                        term_b = ordered_term_b / np.sqrt(coeff_b)
                        singlet_sd.append(term_b)

    if transform == "JW":
        cluster_ops = []
        cluster_ops_sp = []
        for y in singlet_sd:
            hamilt_sp = transform_to_jw_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "Bravyi-Kitaev":
        cluster_ops = []
        cluster_ops_sp = []
        for y in singlet_sd:
            hamilt_sp = transform_to_bk_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "parity_basis":
        cluster_ops = []
        cluster_ops_sp = []
        for y in singlet_sd:
            hamilt_sp = transform_to_parity_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp

# From fermion_util.py file
def merge_duplicate_terms(hamiltonian):

    """
    Take a fermionic Hamiltonian and merge terms with same operator content

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Of type fermionic cluster operator

    Returns
    ----------
    merged_hamiltonian: Hamiltonian
        The listed merged operators
    
    """
    

    terms = {}

    for term in hamiltonian.terms:

        key = tuple([term.op, tuple(term.qbits)])

        if key in terms.keys():

            terms[key] += term.coeff

        else:

            terms[key] = term.coeff

    terms = [Term(v, k[0], list(k[1])) for k, v in terms.items()]

    merged_hamiltonian = Hamiltonian(
        hamiltonian.nbqbits, terms=terms, constant_coeff=hamiltonian.constant_coeff
    )

    return merged_hamiltonian


def singlet_upccgsd(n_orb, transform, perm):
    """
    This function is responsible for generating the paired generalized single and double excitations within singlet spin symmetry.
    Note that 'k' is always equal to 1 but in molecule_factory.py file, the user can type k>1.
    'k' denotes the products of unitary paired generalized double excitations, along with the full set of generalized single excitations.

    Parameters
    ----------
    n_orb: int
        The number of orbitals
    
    transform: string
        type of transformation. Either 'JW', 'Bravyi-Kitaev', or 'parity_basis'
    
    perm: int
        the prefactor number (which is 'k')
    
    Returns
    -------

    pool_size: int
        The number of the cluster operators

    cluster_ops: List<Hamiltonian>
        List of fermionic cluster operators
        
    cluster_ops_sp: List<Hamiltonian>
        List of spin cluster operators

    """

    # perm: is k here
    print("Form spin-adapted UpCCGSD operators pool: ")
    fermi_ops = []
    # Construct general singles
    for p in range(n_orb):
        for q in range(n_orb):
            pa = 2 * p
            qa = 2 * q
            pb = 2 * p + 1
            qb = 2 * q + 1
            if p > q:
                term_a = [
                    Term(1, "Cc", [qa, pa]),
                    Term(-1, "Cc", [pa, qa]),
                    Term(1, "Cc", [qb, pb]),
                    Term(-1, "Cc", [pb, qb]),
                ]
                #                 print(term_a)
                ordered_term_a = 0
                for t1 in term_a:
                    t_list = order_fermionic_term(t1)
                    ordered_term_a = ordered_term_a + Hamiltonian(
                        2 * n_orb, terms=[t for t in t_list]
                    )
                    ordered_term_a = merge_duplicate_terms(ordered_term_a)
                fermi_ops.append(ordered_term_a)
    # Construct general paired doubles
    spatial_orb = list(range(n_orb))
    double_excitations = []
    for i, (p, q) in enumerate(itertools.combinations(spatial_orb, 2)):
        pa = 2 * p
        qa = 2 * q
        pb = 2 * p + 1
        qb = 2 * q + 1
        double_excitations.append([qa, pa, qb, pb])
    #         print("double_excitations",double_excitations)
    for (i, j, k, L) in double_excitations:
        i, j, k, L = int(i), int(j), int(k), int(L)
        term_b = [Term(1.0, "CcCc", [i, j, k, L]), Term(-1.0, "CcCc", [L, k, j, i])]
        ordered_term_b = 0
        for t1 in term_b:
            t_list = order_fermionic_term(t1)
            ordered_term_b = ordered_term_b + Hamiltonian(
                2 * n_orb, terms=[t for t in t_list]
            )
            ordered_term_b = merge_duplicate_terms(ordered_term_b)

        fermi_ops.append(ordered_term_b)
    if transform == "JW":
        cluster_ops = []
        cluster_ops_sp = []
        for y in fermi_ops:
            hamilt_sp = transform_to_jw_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        for i in range(1,perm+1):
            cluster_ops+=cluster_ops
            cluster_ops_sp+=cluster_ops_sp
            pool_size = len(cluster_ops)
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "Bravyi-Kitaev":
        cluster_ops = []
        cluster_ops_sp = []
        for y in fermi_ops:
            hamilt_sp = transform_to_bk_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        for i in range(1,perm+1):
            cluster_ops+=cluster_ops
            cluster_ops_sp+=cluster_ops_sp
            pool_size = len(cluster_ops)                    
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "parity_basis":
        pool_size = len(fermi_ops)
        cluster_ops = []
        cluster_ops_sp = []
        for y in fermi_ops:
            hamilt_sp = transform_to_parity_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        for i in range(1,perm+1):
            cluster_ops+=cluster_ops
            cluster_ops_sp+=cluster_ops_sp
            pool_size = len(cluster_ops)                    
        return pool_size, cluster_ops, cluster_ops_sp

def singlet_gsd(n_elec, orbital_number, transform):
    """
    This function is responsible for generating the 'generalized' single and double excitations occured
    from occupied-occupied, occupied-unoccupied, and unoccupied-unoccupied orbitals within singlet spin symmetry.

    Parameters
    ----------
    n_elec: int
        The number of electrons

    orbital_number: int
        The number of orbitals
    
    transform: string
        type of transformation. Either 'JW', 'Bravyi-Kitaev', or 'parity_basis'
    
    Returns
    -------

    pool_size: int
        The number of the cluster operators

    cluster_ops: List<Hamiltonian>
        List of fermionic cluster operators
        
    cluster_ops_sp: List<Hamiltonian>
        List of spin cluster operators

    """

    singlet_gsd = []
    for p in range(0, orbital_number):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, orbital_number):
            qa = 2 * q
            qb = 2 * q + 1

            term_a = Hamiltonian(
                2 * orbital_number,
                [
                    Term(1, "Cc", [pa, qa]),
                    Term(-1, "Cc", [qa, pa]),
                    Term(1, "Cc", [pb, qb]),
                    Term(-1, "Cc", [qb, pb]),
                ],
            )
            # Normalize
            coeff_a = 0
            for t in term_a.terms:
                coeff_t = t.coeff
                coeff_a += coeff_t * coeff_t
            if coeff_a > 0:
                term_a = term_a / np.sqrt(coeff_a)
                singlet_gsd.append(term_a)

    term_a = 0
    term_b = 0
    pq = -1
    for p in range(0, orbital_number):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, orbital_number):
            qa = 2 * q
            qb = 2 * q + 1

            pq += 1

            rs = -1
            for r in range(0, orbital_number):
                ra = 2 * r
                rb = 2 * r + 1
                for s in range(r, orbital_number):
                    sa = 2 * s
                    sb = 2 * s + 1
                    rs += 1
                    if pq > rs:
                        continue
                    term_a = [
                        Term(2 / np.sqrt(12), "CcCc", [ra, pa, sa, qa]),
                        Term(-2 / np.sqrt(12), "CcCc", [qa, sa, pa, ra]),
                        Term(2 / np.sqrt(12), "CcCc", [rb, pb, sb, qb]),
                        Term(-2 / np.sqrt(12), "CcCc", [qb, sb, pb, rb]),
                        Term(1 / np.sqrt(12), "CcCc", [ra, pa, sb, qb]),
                        Term(-1 / np.sqrt(12), "CcCc", [qb, sb, pa, ra]),
                        Term(1 / np.sqrt(12), "CcCc", [rb, pb, sa, qa]),
                        Term(-1 / np.sqrt(12), "CcCc", [qa, sa, pb, rb]),
                        Term(1 / np.sqrt(12), "CcCc", [ra, pb, sb, qa]),
                        Term(-1 / np.sqrt(12), "CcCc", [qa, sb, pb, ra]),
                        Term(1 / np.sqrt(12), "CcCc", [rb, pa, sa, qb]),
                        Term(-1 / np.sqrt(12), "CcCc", [qb, sa, pa, rb]),
                    ]
                    term_b = [
                        Term(1 / 2.0, "CcCc", [ra, pa, sb, qb]),
                        Term(-1 / 2.0, "CcCc", [qb, sb, pa, ra]),
                        Term(1 / 2.0, "CcCc", [rb, pb, sa, qa]),
                        Term(-1 / 2.0, "CcCc", [qa, sa, pb, rb]),
                        Term(-1 / 2.0, "CcCc", [ra, pb, sb, qa]),
                        Term(1 / 2.0, "CcCc", [qa, sb, pb, ra]),
                        Term(-1 / 2.0, "CcCc", [rb, pa, sa, qb]),
                        Term(1 / 2.0, "CcCc", [qb, sa, pa, rb]),
                    ]
                    ordered_term_a = 0
                    for t1 in term_a:
                        t_list = order_fermionic_term(t1)
                        ordered_term_a = ordered_term_a + Hamiltonian(
                            2 * orbital_number, terms=[t for t in t_list]
                        )
                    ordered_term_b = 0
                    for t2 in term_b:
                        t_list = order_fermionic_term(t2)
                        ordered_term_b = ordered_term_b + Hamiltonian(
                            2 * orbital_number, terms=[t for t in t_list]
                        )
                    coeff_a = 0
                    coeff_b = 0
                    for t in ordered_term_a.terms:
                        coeff_t = t.coeff
                        coeff_a += abs(coeff_t * coeff_t)
                    for t1 in ordered_term_b.terms:
                        coeff_t = t1.coeff
                        coeff_b += abs(coeff_t * coeff_t)
                    if coeff_a > 0:
                        term_a = ordered_term_a / np.sqrt(coeff_a)
                        singlet_gsd.append(term_a)
                    if coeff_b > 0:
                        term_b = ordered_term_b / np.sqrt(coeff_b)
                        singlet_gsd.append(term_b)
    if transform == "JW":
        cluster_ops = []
        cluster_ops_sp = []
        for y in singlet_gsd:
            hamilt_sp = transform_to_jw_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "Bravyi-Kitaev":
        cluster_ops = []
        cluster_ops_sp = []
        for y in singlet_gsd:
            hamilt_sp = transform_to_bk_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp
    if transform == "parity_basis":
        cluster_ops = []
        cluster_ops_sp = []
        for y in singlet_gsd:
            hamilt_sp = transform_to_parity_basis(y)
            if hamilt_sp.terms != []:
                cluster_ops.append(y)
                cluster_ops_sp.append(hamilt_sp)
        pool_size = len(cluster_ops_sp)
        return pool_size, cluster_ops, cluster_ops_sp
