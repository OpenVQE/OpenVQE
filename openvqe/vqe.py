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
    
    @classmethod
    def algorithm(cls, algo_name, molecule_symbol, type_of_generator, transform, active, opts={}):
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
        algorithm = cls.algorithms[algo_name]
        if algorithm is None:
            raise Exception(f'Algorithm not found. Please choose from the following: {cls.algorithms.keys()}')
        return algorithm(molecule_symbol, type_of_generator, transform, active, opts)