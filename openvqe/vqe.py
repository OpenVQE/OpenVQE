from openvqe.algorithms.ucc import UCC
from openvqe.algorithms.fermionic_adapt import FermionicAdapt
from openvqe.algorithms.qubit_adapt import QubitAdapt
from openvqe.algorithms.quccsd import QUCCSD
from openvqe.algorithms import algorithm
from openvqe.common_files.molecule_factory import MoleculeFactory
from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory as SparseMoleculeFactory

import matplotlib.pyplot as plt
import numpy as np

class VQE:
    
    algorithms = {
        'ucc': UCC,
        'fermionic_adapt': FermionicAdapt,
        'qubit_adapt': QubitAdapt,
        'quccsd': QUCCSD
    }
    
    def __init__(self, algo_name, molecule_symbol, type_of_generator, transform, active, opts={}):
        """Initialize the VQE calculation with the chosen algorithm and molecular configuration.

        Args:
            algo_name (string): Name of the VQE algorithm to use. Options include:
                - 'ucc': Unitary Coupled Cluster
                - 'fermionic_adapt': Fermionic Adaptive
                - 'qubit_adapt': Qubit Adaptive
                - 'quccsd': Quasi-Unrestricted Coupled Cluster with Singles and Doubles
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

        Raises:
            Exception: If the specified algorithm name (`algo_name`) is not found in the `algorithms` dictionary.
        """
        self.molecule_symbol = molecule_symbol
        self.active = active
        self.type_of_generator = type_of_generator
        self.transform = transform
        self.opts = opts
        self.algorithm = self.algorithms[algo_name]
        if self.algorithm is None:
            raise Exception(f'Algorithm not found. Please choose from the following: {self.algorithms.keys()}')
        self.algorithm = self.algorithm(self.molecule_symbol, self.type_of_generator, self.transform, self.active, self.opts)

    def execute(self):
        self.iterations, self.results = self.algorithm.execute()
        self.info = self.algorithm.info
        
    def energy_list(self):
        return self.results['energies_1'], self.results['energies_2']

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
        plt.title(f"Energy evolution of {type(self.algorithm).__name__} on {self.molecule_symbol} molecule", fontsize=20)
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
        plt.title(f"Error on log scale for {type(self.algorithm).__name__} on {self.molecule_symbol} molecule", fontsize=20)
        plt.tight_layout()  # Adjust layout to prevent clipping

        plt.show()