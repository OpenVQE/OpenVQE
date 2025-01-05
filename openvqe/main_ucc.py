
from openvqe.vqe import VQE

def main():
    # user can type the name of molecule (H2, LIH, CO, CO2 such that their geormetries and properties are defined in MoleculeQlm())
    # the type of generators: UCCSD, singlet_sd, singlet_gsd, spin_complement_gsd, spin_complement_gsd_twin, sUPCCGSD
    # suppose user type sUPCCGSD
    # user can type any of the following three transformations: JW,  Bravyi-Kitaev and Parity-basis
    # the non_active space selection
    VQE.algorithm('ucc', 'H2', 'sUPCCGSD', 'JW', False).execute()

if __name__ == "__main__":
    main()