"""
script to ...
"""

from openvqe.common_files.generator_excitations import (
    triple_excitation,
    _apply_transforms,
)
from openvqe.common_files.orbital_symmetry import HF_sym


def generate(
    molecule,
    n_occ,
    n_spatial_orb,
    apply_HF_sym,
    apply_transform,
):
    generated_ops = triple_excitation(n_spatial_orb, n_occ)

    if apply_HF_sym:
        generated_ops = HF_sym(molecule, n_occ, generated_ops)

    if apply_transform:
        generated_ops = _apply_transforms(generated_ops, "JW")[-1]
    return generated_ops


ops = generate(
    molecule="LiH", n_occ=4, n_spatial_orb=6, apply_HF_sym=True, apply_transform=True
)

print(len(ops))
