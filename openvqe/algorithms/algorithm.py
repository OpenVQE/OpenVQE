class Algorithm:

    def __init__(self, molecule_symbol, type_of_generator, transform, active, opts={}):
        self.molecule_symbol = molecule_symbol
        self.type_of_generator = type_of_generator
        self.transform = transform
        self.active = active
        self.opts = opts

    def presentation(self, molecule_factory):

        r, geometry, charge, spin, basis = molecule_factory.get_parameters(self.molecule_symbol)
        print(" --------------------------------------------------------------------------")
        if self.active:
            print("Running in the active case: ")
        else:
            print("Running in the non active case: ")
        print("molecule symbol: %s " %(self.molecule_symbol))
        print("molecule basis: %s " %(basis))
        print("type of generator: %s " %(self.type_of_generator))
        print("transform: %s " %(self.transform))
        print("options: %s " %(self.opts))
        print(" --------------------------------------------------------------------------")

    def generate_hamiltonian(self, molecule_factory):
        print(" --------------------------------------------------------------------------")
        print("                                                          ")
        print("                      Generate Hamiltonians and Properties from :")
        print("                                                          ")
        print(" --------------------------------------------------------------------------")
        print("                                                          ")

        res = molecule_factory.generate_hamiltonian(self.molecule_symbol, active=self.active, transform=self.transform)
        
        print(f'Hamiltonian info {res[-1]}')
        
        return res

    def generate_cluster_ops(self, molecule_factory):
        print(" --------------------------------------------------------------------------")
        print("                                                          ")
        print("                      Generate Cluster OPS:")
        print("                                                          ")
        print(" --------------------------------------------------------------------------")
        print("                                                           ")

        args = molecule_factory.generate_cluster_ops(self.molecule_symbol, type_of_generator=self.type_of_generator, transform=self.transform, active=self.active)

        print('Pool size: ', args[0])
        print('length of the cluster OP: ', len(args[1]))
        print('length of the cluster OPS: ', len(args[2]))
        if molecule_factory.sparse():
            print('length of the cluster _sparse: ', len(args[3]))

        return args