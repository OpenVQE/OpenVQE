from openvqe.common_files.qubit_pool import QubitPool
from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.qubit_adapt_vqe import qubit_adapt_vqe
from openvqe.algorithms import tools

def generate_pool_without_cluster(cluster_ops, nbqbits, molecule_symbol):
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

def execute(molecule_symbol, type_of_generator, transform, active):
    
    # Parameters for the ADAPT-VQE algorithm
    n_max_grads = 1
    optimizer = 'BFGS'                
    tolerance = 10**(-9)            
    type_conver = 'norm'
    threshold_needed = 1e-7
    max_external_iterations = 29
    
    molecule_factory = MoleculeFactory()

    tools.presentation(molecule_symbol, type_of_generator, transform, active)
    hamiltonian, hamiltonian_sparse, hamiltonian_sp, hamiltonian_sp_sparse, n_elec, noons_full, orb_energies_full, info = tools.generate_hamiltonian(molecule_factory, molecule_symbol, type_of_generator, transform, active)
    pool_size, cluster_ops, cluster_ops_sp, cluster_ops_sparse = tools.generate_cluster_ops(molecule_factory, molecule_symbol, type_of_generator, transform, active)
    nbqbits = hamiltonian_sp.nbqbits
    len_returned_pool, returned_pool = generate_pool_without_cluster(cluster_ops, hamiltonian_sp.nbqbits, molecule_symbol)
    hf_init = molecule_factory.find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
    reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, len(orb_energies_full), transform)
    
    pool_mix = returned_pool
    print("length of the pool",len(pool_mix))
    iterations_sim, iterations_ana, result_sim, result_ana = qubit_adapt_vqe(
        hamiltonian_sp, 
        hamiltonian_sp_sparse,
        reference_ket, 
        nbqbits, 
        pool_mix, 
        hf_init_sp, 
        info['FCI'],
        n_max_grads     = n_max_grads,
        adapt_conver    = type_conver,
        adapt_thresh    = threshold_needed,
        adapt_maxiter   = max_external_iterations,
        tolerance_sim   = tolerance,
        method_sim      = optimizer
    )
    print("iterations are:",iterations_sim)    
    print("results are:",result_sim)
