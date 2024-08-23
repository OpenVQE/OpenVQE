import itertools
from itertools import combinations

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

def _apply_transforms(cluster_ops_fr, transform, perm=0):
    if transform == "JW":
        transform_func = transform_to_jw_basis
    elif transform == "Bravyi-Kitaev":
        transform_func = transform_to_bk_basis
    elif transform == "parity_basis":
        transform_func = transform_to_parity_basis
    else:
        return
    
    cluster_ops = []
    cluster_ops_sp = []
    for y in cluster_ops_fr:
        hamilt_sp = transform_func(y)
        if hamilt_sp.terms != []:
            cluster_ops.append(y)
            cluster_ops_sp.append(hamilt_sp)
    cluster_ops += cluster_ops * perm
    cluster_ops_sp += cluster_ops_sp * perm
    pool_size = len(cluster_ops_sp)
    return pool_size, cluster_ops, cluster_ops_sp



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
    pool_size, cluster_ops, cluster_ops_sp = _apply_transforms(cluster_ops, transform)
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
    spin_complement_single = []
    spin_complement_double = []

    for p in range(0, 2*orbital_number, 2):
        for q in range(p, 2*orbital_number, 2):
            term_a = [
                    Term(1, "Cc", [p, q]),
                    Term(-1, "Cc", [q, p]),
                    Term(1, "Cc", [p+1, q+1]),
                    Term(-1, "Cc", [q+1, p+1]),
                ]
            hamiltonian = Hamiltonian(2*orbital_number, term_a)
            spin_complement_single.append(hamiltonian)

            for r in range(p, 2*orbital_number, 2):
                for s in range(q if r==p else r, 2*orbital_number, 2):

                    term_a = [
                        Term(1, "CcCc", [r, p, s, q]),
                        Term(-1, "CcCc", [q, s, p, r]),
                        Term(1, "CcCc", [r+1, p+1, s+1, q+1]),
                        Term(-1, "CcCc", [q+1, s+1, p+1, r+1]),
                    ]

                    term_b = [
                        Term(1, "CcCc", [r, p, s+1, q+1]),
                        Term(-1, "CcCc", [q+1, s+1, p, r]),
                        Term(1, "CcCc", [r+1, p+1, s, q]),
                        Term(-1, "CcCc", [q, s, p+1, r+1]),
                    ]

                    term_c = [
                        Term(1, "CcCc", [r, p+1, s+1, q]),
                        Term(-1, "CcCc", [q, s+1, p+1, r]),
                        Term(1, "CcCc", [r+1, p, s, q+1]),
                        Term(-1, "CcCc", [q+1, s, p, r+1]),
                    ]
                    
                    for term_x in [term_a, term_b, term_c]:
                        term_x_ordered_terms = map(order_fermionic_term, term_x)
                        term_x = sum(term_x_ordered_terms, [])
                        hamiltonian = Hamiltonian(2*orbital_number, terms=term_x)
                        spin_complement_double.append(hamiltonian)

    spin_complements = spin_complement_single + spin_complement_double
    return _apply_transforms(spin_complements, transform)

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

    return _apply_transforms(spin_complement_gsd_twin, transform)


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

    singlet_s = []
    singlet_d = []

    n_occ = int(np.ceil(n_elec / 2))
    for i in range(0, 2*n_occ, 2):
        for j in range(i, 2*n_occ, 2):
            for a in range(2*n_occ, 2*orbital_number, 2):
                if j == i: # only once
                    term_a = [
                        Term(1/2, "Cc", [a, i]),
                        Term(1/2, "Cc", [a+1, i+1]),
                        Term(-1/2, "Cc", [i, a]),
                        Term(-1/2, "Cc", [i+1, a+1]),
                    ]
                    hamiltonian = Hamiltonian(2*orbital_number, term_a)
                    singlet_s.append(hamiltonian)
                for b in range(a, 2*orbital_number, 2):
                    term_a = [
                        Term(2 / np.sqrt(12), "CCcc", [a, b, i, j]),
                        Term(-2 / np.sqrt(12), "CCcc", [j, i, b, a]),
                        Term(2 / np.sqrt(12), "CCcc", [a+1, b+1, i+1, j+1]),
                        Term(-2 / np.sqrt(12), "CCcc", [j+1, i+1, b+1, a+1]),
                        Term(1 / np.sqrt(12), "CCcc", [a, b+1, i, j+1]),
                        Term(-1 / np.sqrt(12), "CCcc", [j+1, i, b+1, a]),
                        Term(1 / np.sqrt(12), "CCcc", [a+1, b, i+1, j]),
                        Term(-1 / np.sqrt(12), "CCcc", [j, i+1, b, a+1]),
                        Term(1 / np.sqrt(12), "CCcc", [a, b+1, i+1, j]),
                        Term(-1 / np.sqrt(12), "CCcc", [j, i+1, b+1, a]),
                        Term(1 / np.sqrt(12), "CCcc", [a+1, b, i, j+1]),
                        Term(-1 / np.sqrt(12), "CCcc", [j+1, i, b, a+1]),
                    ]
                    term_b = [
                        Term(1 / 2, "CCcc", [a, b+1, i, j+1]),
                        Term(-1 / 2, "CCcc", [j+1, i, b+1, a]),
                        Term(1 / 2, "CCcc", [a+1, b, i+1, j]),
                        Term(-1 / 2, "CCcc", [j, i+1, b, a+1]),
                        Term(-1 / 2, "CCcc", [a, b+1, i+1, j]),
                        Term(1 / 2, "CCcc", [j, i+1, b+1, a]),
                        Term(-1 / 2, "CCcc", [a+1, b, i, j+1]),
                        Term(1 / 2, "CCcc", [j+1, i, b, a+1]),
                    ]

                    for term_x in [term_a, term_b]:
                        # order
                        term_x_ordered_terms = map(order_fermionic_term, term_x)
                        term_x = sum(term_x_ordered_terms, [])
                        hamiltonian = Hamiltonian(2*orbital_number, terms=term_x)
                        # normalize
                        norm = sum(map(lambda term: abs(term.coeff**2), hamiltonian.terms)) ** 0.5
                        if norm > 0:
                            hamiltonian /= norm
                            singlet_d.append(hamiltonian)

    singlets = singlet_s + singlet_d

    return _apply_transforms(singlets, transform)

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
    single_excitations = []
    double_excitations = []
    # Construct general singles
    for p in range(0, 2*n_orb, 2):
        for q in range(0, p, 2):
            term_a = [
                Term(1, "Cc", [q, p]),
                Term(-1, "Cc", [p, q]),
                Term(1, "Cc", [q+1, p+1]),
                Term(-1, "Cc", [p+1, q+1]),
            ]
            hamiltonian = Hamiltonian(2*n_orb, terms=term_a)
            hamiltonian = merge_duplicate_terms(hamiltonian)
            single_excitations.append(hamiltonian)
    # Construct general paired doubles
    even_spatial_orb = list(range(0, 2*n_orb, 2))

    for p, q in itertools.combinations(even_spatial_orb, 2):
        term_b = [
                    Term(1.0, "CcCc", [q, p, q+1, p+1]),
                    Term(-1.0, "CcCc", [p+1, q+1, p, q])
                ]
        
        term_b_ordered_terms = map(order_fermionic_term, term_b)
        term_B = sum(term_b_ordered_terms, [])
        hamiltonian = Hamiltonian(2*n_orb, terms=term_B)
        hamiltonian = merge_duplicate_terms(hamiltonian)
        double_excitations.append(hamiltonian)

    fermi_ops = single_excitations + double_excitations
    return _apply_transforms(fermi_ops, transform, perm=perm)

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

    singlet_single = []
    singlet_double = []

    for p in range(0, 2*orbital_number, 2):
        for q in range(p, 2*orbital_number, 2):
            term_a = [
                    Term(1/2, "Cc", [p, q]),
                    Term(-1/2, "Cc", [q, p]),
                    Term(1/2, "Cc", [p+1, q+1]),
                    Term(-1/2, "Cc", [q+1, p+1]),
                ]
            hamiltonian = Hamiltonian(2*orbital_number, term_a)
            singlet_single.append(hamiltonian)

            for r in range(p, 2*orbital_number, 2):
                for s in range(q if r==p else r, 2*orbital_number, 2):
                    term_a = [
                        Term(2 / np.sqrt(12), "CcCc", [r, p, s, q]),
                        Term(-2 / np.sqrt(12), "CcCc", [q, s, p, r]),
                        Term(2 / np.sqrt(12), "CcCc", [r+1, p+1, s+1, q+1]),
                        Term(-2 / np.sqrt(12), "CcCc", [q+1, s+1, p+1, r+1]),
                        Term(1 / np.sqrt(12), "CcCc", [r, p, s+1, q+1]),
                        Term(-1 / np.sqrt(12), "CcCc", [q+1, s+1, p, r]),
                        Term(1 / np.sqrt(12), "CcCc", [r+1, p+1, s, q]),
                        Term(-1 / np.sqrt(12), "CcCc", [q, s, p+1, r+1]),
                        Term(1 / np.sqrt(12), "CcCc", [r, p+1, s+1, q]),
                        Term(-1 / np.sqrt(12), "CcCc", [q, s+1, p+1, r]),
                        Term(1 / np.sqrt(12), "CcCc", [r+1, p, s, q+1]),
                        Term(-1 / np.sqrt(12), "CcCc", [q+1, s, p, r+1]),
                    ]
                    term_b = [
                        Term(1 / 2.0, "CcCc", [r, p, s+1, q+1]),
                        Term(-1 / 2.0, "CcCc", [q+1, s+1, p, r]),
                        Term(1 / 2.0, "CcCc", [r+1, p+1, s, q]),
                        Term(-1 / 2.0, "CcCc", [q, s, p+1, r+1]),
                        Term(-1 / 2.0, "CcCc", [r, p+1, s+1, q]),
                        Term(1 / 2.0, "CcCc", [q, s+1, p+1, r]),
                        Term(-1 / 2.0, "CcCc", [r+1, p, s, q+1]),
                        Term(1 / 2.0, "CcCc", [q+1, s, p, r+1]),
                    ]

                    for term_x in [term_a, term_b]:
                        # order
                        term_x_ordered_terms = map(order_fermionic_term, term_x)
                        term_x = sum(term_x_ordered_terms, [])
                        hamiltonian = Hamiltonian(2*orbital_number, terms=term_x)
                        # normalize
                        norm = sum(map(lambda term: abs(term.coeff**2), hamiltonian.terms)) ** 0.5
                        if norm > 0:
                            hamiltonian /= norm
                            singlet_double.append(hamiltonian)
    
    singlets = singlet_single + singlet_double

    return _apply_transforms(singlets, transform)


def uccgsd(n_elec, orbital_number, transform):
    """
    TBD

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
    spin_complement_single = []
    spin_complement_double = []

    for p in range(0, 2*orbital_number):
        for q in range(p, 2*orbital_number):
            term_a = [
                    Term(1, "Cc", [p, q]),
                    Term(-1, "Cc", [q, p])
                ]
            hamiltonian = Hamiltonian(2*orbital_number, term_a)
            spin_complement_single.append(hamiltonian)

            for r in range(p, 2*orbital_number):
                for s in range(q if r==p else r, 2*orbital_number):

                    term_a = [
                        Term(1, "CCcc", [p, q, r, s]),
                        Term(-1, "CCcc", [s, r, q, p])
                    ]
                    
                    term_a_ordered_terms = map(order_fermionic_term, term_a)
                    term_a = sum(term_a_ordered_terms, [])
                    hamiltonian = Hamiltonian(2*orbital_number, terms=term_a)
                    spin_complement_double.append(hamiltonian)

    spin_complements = spin_complement_single + spin_complement_double
    return _apply_transforms(spin_complements, transform)


# It's available for the user
def generalized_singlet_and_triplet_excitations(n_orb, transform):
    """
    from paper:
    Qubit-ADAPT-VQE: An Adaptive Algorithm for Constructing
    Hardware-Efficient Ansätze on a Quantum Processor
    """
    groups = {
        "1": {"singlet": [], "triplet": []},
        "2": {"singlet": [], "triplet": []},
        "3": [],
        "4": [],
        "5": [],
    }

    # group 1, triplet
    for p in range(0, n_orb):
        for q in range(p + 1, n_orb):
            for r in range(q + 1, n_orb):
                for s in range(r + 1, n_orb):
                    def evodd(s1, s2, s3, s4):
                        return [2 * p + s1, 2 * q + s2, 2 * r + s3, 2 * s + s4]

                    term = [
                        Term(1.0, "CCcc", evodd(0, 0, 0, 0)),
                        Term(0.5, "CCcc", evodd(0, 1, 0, 1)),
                        Term(0.5, "CCcc", evodd(0, 1, 1, 0)),
                        Term(0.5, "CCcc", evodd(1, 0, 0, 1)),
                        Term(0.5, "CCcc", evodd(1, 0, 1, 0)),
                        Term(1.0, "CCcc", evodd(1, 1, 1, 1)),
                        Term(-1.0, "CCcc", evodd(0, 0, 0, 0)[::-1]),
                        Term(-0.5, "CCcc", evodd(0, 1, 0, 1)[::-1]),
                        Term(-0.5, "CCcc", evodd(0, 1, 1, 0)[::-1]),
                        Term(-0.5, "CCcc", evodd(1, 0, 0, 1)[::-1]),
                        Term(-0.5, "CCcc", evodd(1, 0, 1, 0)[::-1]),
                        Term(-1.0, "CCcc", evodd(1, 1, 1, 1)[::-1]),
                    ]
                    hamiltonian = Hamiltonian(2 * n_orb, term)
                    groups["1"]["triplet"].append(hamiltonian)

    # group 1, singlet
    for p in range(0, n_orb):
        for q in range(p + 1, n_orb):
            for r in range(q + 1, n_orb):
                for s in range(r + 1, n_orb):
                    def evodd(s1, s2, s3, s4):
                        return [2 * p + s1, 2 * q + s2, 2 * r + s3, 2 * s + s4]

                    term = [
                        Term(0.5, "CCcc", evodd(0, 1, 0, 1)),
                        Term(-0.5, "CCcc", evodd(0, 1, 1, 0)),
                        Term(-0.5, "CCcc", evodd(1, 0, 0, 1)),
                        Term(0.5, "CCcc", evodd(1, 0, 1, 0)),
                        Term(-0.5, "CCcc", evodd(0, 1, 0, 1)[::-1]),
                        Term(0.5, "CCcc", evodd(0, 1, 1, 0)[::-1]),
                        Term(0.5, "CCcc", evodd(1, 0, 0, 1)[::-1]),
                        Term(-0.5, "CCcc", evodd(1, 0, 1, 0)[::-1]),
                    ]
                    hamiltonian = Hamiltonian(2 * n_orb, term)
                    groups["1"]["singlet"].append(hamiltonian)

    # group 2, triplet
    for a1 in range(0, n_orb):
        for a2 in range(a1 + 1, n_orb):
            for a3 in range(a2 + 1, n_orb):
                # above are 3 unique numbers.
                # we implement 4 cases:
                #   p = r
                #   q = r
                #   p = s
                #   q = s

                evodd_case = {}
                evodd_case["p=r"] = lambda s1, s2, s3, s4: [
                    2 * a1 + s1,
                    2 * a2 + s2,
                    2 * a1 + s3,
                    2 * a3 + s4,
                ]
                evodd_case["q=r"] = lambda s1, s2, s3, s4: [
                    2 * a1 + s1,
                    2 * a2 + s2,
                    2 * a2 + s3,
                    2 * a3 + s4,
                ]
                evodd_case["p=s"] = lambda s1, s2, s3, s4: [
                    2 * a1 + s1,
                    2 * a2 + s2,
                    2 * a3 + s3,
                    2 * a1 + s4,
                ]
                evodd_case["q=s"] = lambda s1, s2, s3, s4: [
                    2 * a1 + s1,
                    2 * a2 + s2,
                    2 * a3 + s3,
                    2 * a2 + s4,
                ]

                for key in evodd_case.keys():
                    evodd = evodd_case[key]
                    term = [
                        Term(1.0, "CCcc", evodd(0, 0, 0, 0)),
                        Term(0.5, "CCcc", evodd(0, 1, 0, 1)),
                        Term(0.5, "CCcc", evodd(0, 1, 1, 0)),
                        Term(0.5, "CCcc", evodd(1, 0, 0, 1)),
                        Term(0.5, "CCcc", evodd(1, 0, 1, 0)),
                        Term(1.0, "CCcc", evodd(1, 1, 1, 1)),
                        Term(-1.0, "CCcc", evodd(0, 0, 0, 0)[::-1]),
                        Term(-0.5, "CCcc", evodd(0, 1, 0, 1)[::-1]),
                        Term(-0.5, "CCcc", evodd(0, 1, 1, 0)[::-1]),
                        Term(-0.5, "CCcc", evodd(1, 0, 0, 1)[::-1]),
                        Term(-0.5, "CCcc", evodd(1, 0, 1, 0)[::-1]),
                        Term(-1.0, "CCcc", evodd(1, 1, 1, 1)[::-1]),
                    ]
                    hamiltonian = Hamiltonian(2 * n_orb, term)
                    groups["2"]["triplet"].append(hamiltonian)

    # group 2, singlet
    for a1 in range(0, n_orb):
        for a2 in range(a1 + 1, n_orb):
            for a3 in range(a2 + 1, n_orb):
                # above are 3 unique numbers.
                # we implement 4 cases:
                #   p = r
                #   q = r
                #   p = s
                #   q = s

                evodd_case = {}
                evodd_case["p=r"] = lambda s1, s2, s3, s4: [
                    2 * a1 + s1,
                    2 * a2 + s2,
                    2 * a1 + s3,
                    2 * a3 + s4,
                ]
                evodd_case["q=r"] = lambda s1, s2, s3, s4: [
                    2 * a1 + s1,
                    2 * a2 + s2,
                    2 * a2 + s3,
                    2 * a3 + s4,
                ]
                evodd_case["p=s"] = lambda s1, s2, s3, s4: [
                    2 * a1 + s1,
                    2 * a2 + s2,
                    2 * a3 + s3,
                    2 * a1 + s4,
                ]
                evodd_case["q=s"] = lambda s1, s2, s3, s4: [
                    2 * a1 + s1,
                    2 * a2 + s2,
                    2 * a3 + s3,
                    2 * a2 + s4,
                ]

                for key in evodd_case.keys():
                    evodd = evodd_case[key]

                    term = [
                        Term(0.5, "CCcc", evodd(0, 1, 0, 1)),
                        Term(-0.5, "CCcc", evodd(0, 1, 1, 0)),
                        Term(-0.5, "CCcc", evodd(1, 0, 0, 1)),
                        Term(0.5, "CCcc", evodd(1, 0, 1, 0)),
                        Term(-0.5, "CCcc", evodd(0, 1, 0, 1)[::-1]),
                        Term(0.5, "CCcc", evodd(0, 1, 1, 0)[::-1]),
                        Term(0.5, "CCcc", evodd(1, 0, 0, 1)[::-1]),
                        Term(-0.5, "CCcc", evodd(1, 0, 1, 0)[::-1]),
                    ]
                    hamiltonian = Hamiltonian(2 * n_orb, term)
                    groups["2"]["singlet"].append(hamiltonian)

    # group 3
    for pq in range(0, n_orb):
        for r in range(pq + 1, n_orb):
            for s in range(r + 1, n_orb):
                def evodd(s1, s2, s3, s4):
                    return [2 * pq + s1, 2 * pq + s2, 2 * r + s3, 2 * s + s4]

                term = [
                    Term(1.0, "CCcc", evodd(0, 1, 0, 1)),
                    Term(1.0, "CCcc", evodd(0, 1, 1, 0)),
                    Term(-1.0, "CCcc", evodd(0, 1, 0, 1)[::-1]),
                    Term(-1.0, "CCcc", evodd(0, 1, 1, 0)[::-1]),
                ]
                hamiltonian = Hamiltonian(2 * n_orb, term)
                groups["3"].append(hamiltonian)

    # group 4
    for pqr in range(0, n_orb):
        for s in range(pqr + 1, n_orb):
            def evodd(s1, s2, s3, s4):
                return [2 * pqr + s1, 2 * pqr + s2, 2 * pqr + s3, 2 * s + s4]

            term = [
                Term(1.0, "CCcc", evodd(1, 0, 0, 1)),
                Term(1.0, "CCcc", evodd(0, 1, 1, 0)),
                Term(-1.0, "CCcc", evodd(1, 0, 0, 1)[::-1]),
                Term(-1.0, "CCcc", evodd(0, 1, 1, 0)[::-1]),
            ]
            hamiltonian = Hamiltonian(2 * n_orb, term)
            groups["4"].append(hamiltonian)

    # group 5
    for pq in range(0, n_orb):
        for rs in range(pq + 1, n_orb):
            def evodd(s1, s2, s3, s4):
                return [2 * pq + s1, 2 * pq + s2, 2 * rs + s3, 2 * rs + s4]

            term = [
                Term(2.0, "CCcc", evodd(0, 1, 0, 1)),
                Term(-2.0, "CCcc", evodd(0, 1, 0, 1)[::-1]),
            ]
            hamiltonian = Hamiltonian(2 * n_orb, term)
            groups["5"].append(hamiltonian)

    spin_complements = (
        groups["1"]["singlet"]
        + groups["1"]["triplet"]
        + groups["2"]["singlet"]
        + groups["2"]["triplet"]
        + groups["3"]
        + groups["4"]
        + groups["5"]
    )
    return _apply_transforms(spin_complements, transform)


# Triple excitations
def get_possibilities(num, parity, minimum, maximum):
    assert parity in ["odd", "even"], "parity must be either 'even' or 'odd'"
    shift = int(parity == "odd")
    lista = list(range(minimum, maximum))
    lista = lista[(shift + minimum) % 2 :: 2]
    return combinations(lista, num)


def get_hamiltonian(i, j, k, a, b, c, n_orb):
    term = [
        Term(1, "CCCccc", [a, b, c, k, j, i]),
        Term(-1, "CCCccc", [i, j, k, c, b, a]),
    ]
    hamiltonian = Hamiltonian(n_orb, term)
    return hamiltonian


def triple_excitation(n_spatial_orb, n_occ):
    n_orb = 2 * n_spatial_orb
    triple = []

    # eee -> eee
    for i, j, k in get_possibilities(3, "even", 0, n_occ):
        for a, b, c in get_possibilities(3, "even", n_occ, n_orb):
            triple.append(get_hamiltonian(i, j, k, a, b, c, n_orb))

    # ooo -> ooo
    for i, j, k in get_possibilities(3, "odd", 0, n_occ):
        for a, b, c in get_possibilities(3, "odd", n_occ, n_orb):
            triple.append(get_hamiltonian(i, j, k, a, b, c, n_orb))

    # ooe -> ooe
    for i, j in get_possibilities(2, "odd", 0, n_occ):
        for (k,) in get_possibilities(1, "even", 0, n_occ):

            for a, b in get_possibilities(2, "odd", n_occ, n_orb):
                for (c,) in get_possibilities(1, "even", n_occ, n_orb):
                    triple.append(get_hamiltonian(i, j, k, a, b, c, n_orb))

    # eeo -> eeo
    for i, j in get_possibilities(2, "even", 0, n_occ):
        for (k,) in get_possibilities(1, "odd", 0, n_occ):

            for a, b in get_possibilities(2, "even", n_occ, n_orb):
                for (c,) in get_possibilities(1, "odd", n_occ, n_orb):
                    triple.append(get_hamiltonian(i, j, k, a, b, c, n_orb))

    triple = [g for g in triple if g is not None]
    return triple

# Single and Double excitations without mp2 guess
def generate_cluster_ops_without_mp2(n_orb, n_occ):
    n_orb = 2 * n_orb
    single = []
    for a in range(0, n_occ):
        for i in range(n_occ, n_orb):
            if i % 2 + a % 2 == 1:
                continue
            term = [Term(1, "Cc", [a, i]), Term(-1, "Cc", [i, a])]
            hamiltonian = Hamiltonian(n_orb, term)
            single.append(hamiltonian)

    double = []
    for i in range(n_occ, n_orb):
        for j in range(i + 2, n_orb):
            for a in range(0, n_occ):
                for b in range(a + 2, n_occ):
                    if i % 2 + a % 2 + j % 2 + b % 2 in [1, 2, 3]:
                        continue
                    term = [
                        Term(1, "CCcc", [a, b, j, i]),
                        Term(-1, "CCcc", [i, j, b, a]),
                    ]
                    hamiltonian = Hamiltonian(n_orb, term)
                    double.append(hamiltonian)

    for i in range(n_occ, n_orb):
        if i % 2 == 1:
            continue
        for j in range(n_occ, n_orb):
            if j % 2 == 0:
                continue
            for a in range(0, n_occ):
                if a % 2 == 1:
                    continue
                for b in range(0, n_occ):
                    if b % 2 == 0:
                        continue
                    term = [
                        Term(1, "CCcc", [a, b, j, i]),
                        Term(-1, "CCcc", [i, j, b, a]),
                    ]
                    hamiltonian = Hamiltonian(n_orb, term)
                    double.append(hamiltonian)

    return single + double
