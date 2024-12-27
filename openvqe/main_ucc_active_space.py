
from openvqe.main_ucc import execute

def main():
    # Here user is obliged to check the thresholds epsilon_1 and epsilon_2  inserted in "MoleculeFactory" to select: the active electorns and active orbitals
    execute('H4', 'sUPCCGSD', 'JW', True)

if __name__ == "__main__":
    main()