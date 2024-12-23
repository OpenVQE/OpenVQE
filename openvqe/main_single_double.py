from openvqe.common_files.generator_excitations import (
    _apply_transforms,
    generate_cluster_ops_without_mp2,
)
from openvqe.common_files.orbital_symmetry import HF_sym
from openvqe.common_files.parameter_guess import (
    ccsd_check,
    generate_cluster_ops_with_mp2,
)


def generate(
    molecule,
    n_occ,
    n_spatial_orb,
    apply_HF_sym,
    apply_ccsd_check,
    use_mp2,
    CCSD_THRESH,
    MP2_THRESH,
    apply_transform,
):
    if use_mp2:
        ops = generate_cluster_ops_with_mp2(molecule, MP2_THRESH)
    else:
        ops = generate_cluster_ops_without_mp2(n_spatial_orb, n_occ)

    if apply_HF_sym:
        ops = HF_sym(molecule, n_occ, ops)

    if apply_ccsd_check:
        ops = ccsd_check(molecule, n_occ, ops, CCSD_THRESH)

    if apply_transform:
        ops = _apply_transforms(ops, "JW")[-1]
    return ops

def main():
    ops = generate(
        molecule="LiH",
        n_occ=4,
        n_spatial_orb=6,
        apply_HF_sym=True,
        apply_ccsd_check=True,
        use_mp2=True,
        CCSD_THRESH=1e-8,
        MP2_THRESH=1e-8,
        apply_transform=True,
    )

    print(len(ops))

if __name__ == "__main__":
    main()