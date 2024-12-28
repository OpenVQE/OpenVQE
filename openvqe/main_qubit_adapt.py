from openvqe.vqe import VQE

def main():
    VQE('qubit_adapt', 'H2', 'singlet_gsd', 'JW', False).execute()
    
if __name__ == "__main__":
    main()