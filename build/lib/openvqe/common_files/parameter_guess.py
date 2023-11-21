from openvqe.common_files.orbital_symmetry import reverse_according_to_n_occ, OrbSym
import numpy as np
from qat.fermion.chemistry.pyscf_tools import perform_pyscf_computation
from qat.fermion.chemistry.ucc import convert_to_h_integrals
from qat.fermion import ElectronicStructureHamiltonian
from qat.fermion.chemistry.ucc_deprecated import get_cluster_ops_and_init_guess


def get_parameters(molecule_symbol):
    if molecule_symbol == "LiH":
        r = 1.45
        geometry = [("Li", (0, 0, 0)), ("H", (0, 0, r))]
        charge = 0
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "BeH2":
        r = 1.3264
        geometry = [("Be", (0, 0, 0 * r)), ("H", (0, 0, r)), ("H", (0, 0, -r))]
        charge = 0
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "CH4":
        r = 1.0
        geometry = [
            ("C", (0.0, 0.0, 0.0)),
            ("H", (0.6276, 0.6276, 0.6276)),
            ("H", (0.6276, -0.6276, -0.6276)),
            ("H", (-0.6276, 0.6276, -0.6276)),
            ("H", (-0.6276, -0.6276, 0.6276)),
        ]
        basis = "sto-3g"
        spin = 0
        charge = 0
    else:
        raise Exception("Only LiH, BeH2, and CH4 are supported")

    return r, geometry, charge, spin, basis


def generate_cluster_ops_with_mp2(molecule, mp2_thresh=1e-8):
    """from qlm code"""
    r, geometry, charge, spin, basis = get_parameters(molecule)

    (
        rdm1,
        orbital_energies,
        nuclear_repulsion,
        nels,
        one_body_integrals,
        two_body_integrals,
        info,
    ) = perform_pyscf_computation(
        geometry=geometry, basis=basis, spin=spin, charge=charge, run_fci=True
    )

    hpq, hpqrs = convert_to_h_integrals(one_body_integrals, two_body_integrals)
    H_s = ElectronicStructureHamiltonian(hpq, hpqrs, constant_coeff=nuclear_repulsion)
    noons, basis_change = np.linalg.eigh(rdm1)
    noons = list(reversed(noons))

    noons_full, orb_energies_full = [], []
    for ind in range(len(noons)):
        noons_full.extend([noons[ind], noons[ind]])
        orb_energies_full.extend([orbital_energies[ind], orbital_energies[ind]])

    cluster_ops, theta_0, hf_init = get_cluster_ops_and_init_guess(
        nels, noons_full, orb_energies_full, H_s.hpqrs
    )

    # remove coeffs
    new_cluster_ops = []

    for cluster_op, theta in zip(cluster_ops, theta_0):
        if theta < mp2_thresh:
            continue
        new_cluster_ops.append(cluster_op)
    return new_cluster_ops


def ccsd_check(molecule, n_occ, ops, CCSD_THRESH):
    new_ops = []
    sym_class = OrbSym(molecule, n_occ, CCSD_THRESH)

    for op in ops:
        qbits = op.terms[0].qbits
        qbits = reverse_according_to_n_occ(n_occ, qbits)
        if len(qbits) == 2:
            # single excitation
            if sym_class.ccsd_check1(*qbits):
                new_ops.append(op)
        elif len(qbits) == 4:
            # double excitation
            if sym_class.ccsd_check2(*qbits):
                new_ops.append(op)
        else:
            raise Exception("Only single or double excitations are supported")

    return new_ops
