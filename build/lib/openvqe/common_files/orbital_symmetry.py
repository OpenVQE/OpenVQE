import pyscf
from pyscf import scf, symm, cc

import numpy as np

MULTI_IRREPS = -1


class OrbSym:
    def __init__(self, molecule, n_occ, CCSD_THRESH=1e-8):
        if molecule == "LiH":
            mol = pyscf.M(atom="Li 0 0 0; H 0 0 1.5949", symmetry=True)
            mol.symmetry_subgroup = "C2v"
            self.groupname = "C2v"
        elif molecule == "CH4":
            geometry = [
                ("C", (0.0, 0.0, 0.0)),
                ("H", (0.6276, 0.6276, 0.6276)),
                ("H", (0.6276, -0.6276, -0.6276)),
                ("H", (-0.6276, 0.6276, -0.6276)),
                ("H", (-0.6276, -0.6276, 0.6276)),
            ]
            atom = "; ".join(
                [atom + " " + " ".join(map(str, geo)) for atom, geo in geometry]
            )
            mol = pyscf.M(atom=atom, symmetry=True)
            mol.symmetry_subgroup = "D2"
            self.groupname = "D2"
        elif molecule == "BeH2":
            mol = pyscf.M(atom="Be 0 0 0; H 0 0 1.3264; H 0 0 -1.3264", symmetry=True)
            mol.symmetry_subgroup = "D2h"
            self.groupname = "D2h"
        else:
            raise ValueError("Supported molecules are LiH, CH4, and BeH2")
        self.n_occ = n_occ
        mol.basis = "sto-3g"
        mol.build()
        mf = scf.HF(mol).run()  # this is UHF
        self.mycc = cc.CCSD(mf).run()  # this is UCCSD
        mf = scf.RHF(mol)
        mf.kernel()
        self.label_orb_symm = symm.label_orb_symm(
            mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff
        )
        self.label_orb_symm_id = symm.label_orb_symm(
            mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff
        )
        self.n_spin_orb = len(self.label_orb_symm) * 2
        self.HF_diagram = list(range(n_occ)) + [None] * (self.n_spin_orb - self.n_occ)
        self.HF_irrep = self.rep_after_excitations([], [])
        self.thresh = CCSD_THRESH

    def ccsd_check1(self, ann, crea):
        return abs(self.mycc.t1[ann // 2, (crea - self.n_occ) // 2]) > self.thresh

    def ccsd_check2(self, ann1, ann2, crea1, crea2):
        return (
            abs(
                self.mycc.t2[
                    ann1 // 2,
                    ann2 // 2,
                    (crea1 - self.n_occ) // 2,
                    (crea2 - self.n_occ) // 2,
                ]
            )
            > self.thresh
        )

    def se1(self, orb1):
        index1 = orb1 // 2

        orb_id1 = np.array([self.label_orb_symm_id[index1]])

        return orb_id1

    def se2(self, orb1, orb2):
        index1 = orb1 // 2
        index2 = orb2 // 2

        orb_id1 = np.array([self.label_orb_symm_id[index1]])
        orb_id2 = np.array([self.label_orb_symm_id[index2]])

        return self.direct_prod(orb_id1, orb_id2, self.groupname)

    def se3(self, orb1, orb2, orb3):
        index1 = orb1 // 2
        index2 = orb2 // 2
        index3 = orb3 // 2

        orb_id1 = np.array([self.label_orb_symm_id[index1]])
        orb_id2 = np.array([self.label_orb_symm_id[index2]])
        orb_id3 = np.array([self.label_orb_symm_id[index3]])

        prod_temp = self.direct_prod(orb_id1, orb_id2, self.groupname)

        return self.direct_prod(prod_temp, orb_id3, self.groupname)

    def rep_after_excitations(self, list_annihilated, list_created):
        default_diagram = self.HF_diagram.copy()
        for annihilated in list_annihilated:
            default_diagram[annihilated] = None
        for created in list_created:
            default_diagram[created] = created
        even_indices = default_diagram[::2]
        odd_indices = default_diagram[1::2]
        reprs = []
        for even, odd in zip(even_indices, odd_indices):
            if even is None and odd is None:
                pass
            elif even is not None and odd is not None:
                reprs.append(self.se2(even, odd))
            elif even is not None:
                reprs.append(self.se1(even))
            elif odd is not None:
                reprs.append(self.se1(odd))

        assert reprs
        if len(reprs) == 1:
            return reprs[0]
        final = self.direct_prod(reprs[0], reprs[1], self.groupname)
        for rep in reprs[2:]:
            final = self.direct_prod(final, rep, self.groupname)

        return final

    def direct_prod(self, orbsym1, orbsym2, groupname="D2h"):
        """implementation taken from Pyscf"""
        if groupname == "SO3":
            prod = orbsym1[:, None] ^ orbsym2
            orbsym1_not_s = orbsym1 != 0
            orbsym2_not_s = orbsym2 != 0
            prod[orbsym1_not_s[:, None] & orbsym2_not_s != 0] = MULTI_IRREPS
            prod[orbsym1[:, None] == orbsym2] = 0
        elif groupname == "Dooh":
            orbsym1_octa = (orbsym1 // 10) * 8 + orbsym1 % 10
            orbsym2_octa = (orbsym2 // 10) * 8 + orbsym2 % 10
            prod = orbsym1_octa[:, None] ^ orbsym2_octa
            prod = (prod % 8) + (prod // 8) * 10
            orbsym1_irrepE = (orbsym1 >= 2) & (orbsym1 != 4) & (orbsym1 != 5)
            orbsym2_irrepE = (orbsym2 >= 2) & (orbsym2 != 4) & (orbsym2 != 5)
            prod[orbsym1_irrepE[:, None] & orbsym2_irrepE] = MULTI_IRREPS
            prod[orbsym1[:, None] == orbsym2] = 0
        elif groupname == "Coov":
            prod = orbsym1[:, None] ^ orbsym2
            orbsym1_irrepE = orbsym1 >= 2
            orbsym2_irrepE = orbsym2 >= 2
            prod[orbsym1_irrepE[:, None] & orbsym2_irrepE] = MULTI_IRREPS
            prod[orbsym1[:, None] == orbsym2] = 0
        else:  # D2h and subgroup
            prod = orbsym1[:, None] ^ orbsym2
        return prod


def reverse_according_to_n_occ(n_occ, qbits):
    if qbits[0] >= n_occ:
        return qbits[::-1]
    return qbits


def HF_sym(molecule, n_occ, ops):
    new_ops = []
    sym_class = OrbSym(molecule, n_occ)

    for op in ops:
        qbits = op.terms[0].qbits
        qbits = reverse_according_to_n_occ(n_occ, qbits)
        if len(qbits) == 2:
            # single excitation
            if (
                sym_class.rep_after_excitations([qbits[0]], [qbits[1]])
                == sym_class.HF_irrep
            ):
                new_ops.append(op)

        elif len(qbits) == 4:
            # double excitation
            if (
                sym_class.rep_after_excitations(qbits[:2], qbits[2:])
                == sym_class.HF_irrep
            ):
                new_ops.append(op)

        elif len(qbits) == 6:
            # triple excitation
            if (
                sym_class.rep_after_excitations(qbits[:3], qbits[3:])
                == sym_class.HF_irrep
            ):
                new_ops.append(op)
        else:
            raise Exception("Only single, double, or triple excitations are supported")

    return new_ops
