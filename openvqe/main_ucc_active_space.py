from openvqe.vqe import VQE

def main():
    # Here user is obliged to check the thresholds epsilon_1 and epsilon_2  inserted in "MoleculeFactory" to select: the active electorns and active orbitals
    VQE('ucc', 'H4', 'sUPCCGSD', 'JW', True).execute()

if __name__ == "__main__":
    main()