from openvqe.common_files.qubit_pool import QubitPool
from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.qubit_adapt_vqe import qubit_adapt_vqe
from openvqe.algorithms.algorithm import Algorithm

class QubitAdapt(Algorithm):

    def generate_pool_without_cluster(self, cluster_ops, nbqbits, molecule_symbol):
        print(" --------------------------------------------------------------------------")
        print("                                                          ")
        print("                      Generate Pool without Cluster:")
        print("                                                          ")
        print(" --------------------------------------------------------------------------")
        print("                                                           ")
        
        qubitpool = QubitPool()
        pool_type = 'random'
        qubit_pool = qubitpool.generate_pool(cluster_ops)
        len_returned_pool, returned_pool = qubitpool.generate_pool_without_cluster(pool_type=pool_type, 
                                                                                nbqbits=nbqbits, 
                                                                                qubit_pool=qubit_pool,
                                                                                molecule_symbol=molecule_symbol)
        return len_returned_pool, returned_pool

    def execute(self):
        """
        Executes the ADAPT-VQE algorithm for a given molecule.
        
        The ADAPT-VQE (Adaptive Derivative-Assembled Pseudo-Trotter Variational Quantum Eigensolver) algorithm adaptively 
        builds the quantum circuit by iteratively adding operators that most significantly reduce the energy, leading to 
        efficient and accurate ground state energy estimation.
        
        This implementation uses qubit operators directly to construct the ansatz, potentially simplifying the implementation 
        and making it more efficient for certain quantum hardware.

        Args:
            molecule_symbol (str): The chemical symbol of the molecule.
            type_of_generator (str): The type of generator to use for the algorithm.
            transform (str): The type of transformation to apply.
            active (list): List of active orbitals.
        Returns:
            None: The function prints the number of iterations and the resulting energies.
        """
        
        # Parameters for the ADAPT-VQE algorithm
        self.opts = {
            'n_max_grads': 1,
            'optimizer': 'BFGS',
            'tolerance': 1e-9,
            'type_conver': 'norm',
            'threshold_needed': 1e-7,
            'max_external_iterations': 29
        } | self.opts
        
        molecule_factory = MoleculeFactory()

        self.presentation(molecule_factory)
        hamiltonian, hamiltonian_sparse, hamiltonian_sp, hamiltonian_sp_sparse, n_elec, noons_full, orb_energies_full, info = self.generate_hamiltonian(molecule_factory)
        pool_size, cluster_ops, cluster_ops_sp, cluster_ops_sparse = self.generate_cluster_ops(molecule_factory)
        nbqbits = hamiltonian_sp.nbqbits
        len_returned_pool, returned_pool = self.generate_pool_without_cluster(cluster_ops, hamiltonian_sp.nbqbits, self.molecule_symbol)
        hf_init = molecule_factory.find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
        reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, len(orb_energies_full), self.transform)       
        
        pool_mix = returned_pool
        print("length of the pool",len(pool_mix))
        
        self.info = info
        
        iterations_sim, iterations_ana, result_sim, result_ana = qubit_adapt_vqe(
            hamiltonian_sp, 
            hamiltonian_sp_sparse,
            reference_ket, 
            nbqbits, 
            pool_mix, 
            hf_init_sp, 
            info['FCI'],
            n_max_grads     = self.opts['n_max_grads'],
            adapt_conver    = self.opts['type_conver'],
            adapt_thresh    = self.opts['threshold_needed'],
            adapt_maxiter   = self.opts['max_external_iterations'],
            tolerance_sim   = self.opts['tolerance'],
            method_sim      = self.opts['optimizer']
        )
        print("iterations are:",iterations_sim)    
        print("results are:",result_sim)
        self.iterations = iterations_sim
        self.result = result_sim
