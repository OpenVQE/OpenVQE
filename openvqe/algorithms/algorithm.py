import matplotlib.pyplot as plt
import numpy as np

class Algorithm:

    def __init__(self, molecule_symbol, type_of_generator, transform, active, opts={}):
        """Initialize the VQE calculation.

        Args:
            molecule_symbol (string): Symbol of the molecule whose geometries and properties 
                are defined (e.g., "H2", "LiH").
            type_of_generator (string): The type of generator used to construct the ansatz. 
                Examples include:
                - 'UCCSD': Unitary Coupled Cluster with Singles and Doubles
                - 'singlet_sd': Singlet Singles and Doubles
                - 'singlet_gsd': Generalized Singles and Doubles with Singlet Spin
                - 'spin_complement_gsd': Spin Complemented Generalized Singles and Doubles
                - 'spin_complement_gsd_twin': Twin Spin Complemented GSD
                - 'sUPCCGSD': Simplified Unitary Pair Coupled Cluster GSD
            transform (string): Specifies the qubit mapping transformation to use. Options include:
                - 'JW': Jordan-Wigner
                - 'Bravyi-Kitaev'
                - 'Parity-basis'
            active (string): Active space definition or selection strategy for the molecular orbitals.
                It can specify the subset of orbitals to include in the calculation, such as 
                active occupied and virtual orbitals, or the method to select them.
        """
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
    
    def energy_list(self):
        return self.result['energies_1'], self.result['energies_2']
    
    def plot_energy_result(self):
        energies_1, energies_2 = self.energy_list()
        # Plot results with custom styles
        plt.figure(figsize=(14, 8))  # Larger plot size
        plt.plot(
            energies_1,
            "-o",  # Line style with circle markers
            color="orange",  # Use custom color
            label=f"Energies Cluster operators"
        )
        plt.plot(
            energies_2,
            "-o",  # Line style with circle markers
            color="red",  # Use custom color
            label=f"Pool generators"
        )
        plt.plot(
            [self.info['FCI']] * max([len(energies_1), len(energies_2)]), 
            "k--", 
            label="True ground state energy(FCI)"
        )
        plt.xlabel("Optimization step", fontsize=20)
        plt.ylabel("Energy (Ha)", fontsize=20)

        plt.xticks(fontsize=16)  # Set font size for x-axis tick labels
        plt.yticks(fontsize=16) 

        # Move the legend box outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
        plt.grid()
        plt.title(f"Energy evolution of {type(self).__name__} on {self.molecule_symbol} molecule", fontsize=20)
        plt.tight_layout()  # Adjust layout to prevent clipping

        plt.show()
        
    def plot_error_result(self):
        energies_1, energies_2 = self.energy_list()
        err1 = np.maximum(energies_1 - self.info['FCI'], 1e-16)
        err2 = np.maximum(energies_2 - self.info['FCI'], 1e-16)
        # Plot results with custom styles
        plt.figure(figsize=(14, 8))  # Larger plot size
        plt.plot(
            err1,
            "-o",  # Line style with circle markers
            color="orange",  # Use custom color
            label=f"Energies Cluster operators"
        )
        plt.plot(
            err2,
            "-o",  # Line style with circle markers
            color="red",  # Use custom color
            label=f"Pool generators"
        )
        plt.fill_between(
            np.arange(0, max([len(energies_1), len(energies_2)])), 
            min(min(err1), min(err2)), 
            1e-3, 
            color="cadetblue", 
            alpha=0.2, 
            interpolate=True, 
            label="Chemical Accuracy"
        )
        plt.yscale('log')
        plt.xlabel("Optimization step", fontsize=20)
        plt.ylabel("Energy (Ha)", fontsize=20)
        plt.xticks(fontsize=16)  # Set font size for x-axis tick labels
        plt.yticks(fontsize=16) 

        # Move the legend box outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
        plt.grid()
        plt.title(f"Error on log scale for {type(self).__name__} on {self.molecule_symbol} molecule", fontsize=20)
        plt.tight_layout()  # Adjust layout to prevent clipping

        plt.show()