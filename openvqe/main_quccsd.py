from openvqe.vqe import VQE

def main():
    VQE.algorithm('quccsd', 'H4', 'QUCCSD', 'JW', False).execute()

if __name__ == "__main__":
    main()