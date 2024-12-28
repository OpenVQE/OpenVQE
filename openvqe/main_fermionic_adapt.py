from openvqe.vqe import VQE

def main():
    ## non active case
    VQE('fermionic_adapt', 'H4', 'spin_complement_gsd', 'JW', False).execute()
    ## active case
    VQE('fermionic_adapt', 'H4', 'spin_complement_gsd', 'JW', True).execute()

if __name__ == "__main__":
    main()