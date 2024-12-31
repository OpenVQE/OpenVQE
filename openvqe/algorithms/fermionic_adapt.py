from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.fermionic_adapt_vqe import fermionic_adapt_vqe
from openvqe.algorithms import tools

def execute(molecule_symbol, type_of_generator, transform, active):
    
    # Parameters for the ADAPT-VQE algorithm
    n_max_grads = 1
    optimizer = 'COBYLA'                
    tolerance = 10**(-6)            
    type_conver = 'norm'
    threshold_needed = 1e-2
    max_external_iterations = 35
    
    molecule_factory = MoleculeFactory()

    tools.presentation(molecule_symbol, type_of_generator, transform, active)
    hamiltonian, hamiltonian_sparse, hamiltonian_sp, hamiltonian_sp_sparse, n_elec, noons_full, orb_energies_full, info = tools.generate_hamiltonian(molecule_factory, molecule_symbol, type_of_generator, transform, active)
    pool_size, cluster_ops, cluster_ops_sp, cluster_ops_sparse = tools.generate_cluster_ops(molecule_factory, molecule_symbol, type_of_generator, transform, active)
    nbqbits = len(orb_energies_full)
    hf_init = molecule_factory.find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
    reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, nbqbits, transform)
    print(n_elec)
    print(hf_init_sp)
    print(reference_ket)
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Start ADAPT-VQE algorithm:")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")

    iterations, result = fermionic_adapt_vqe(
        hamiltonian_sparse, 
        cluster_ops_sparse, 
        reference_ket, 
        hamiltonian_sp,
        cluster_ops_sp, 
        hf_init_sp, 
        n_max_grads, 
        info['FCI'], 
        optimizer,                
        tolerance,                
        type_conver = type_conver,
        threshold_needed = threshold_needed,
        max_external_iterations = max_external_iterations
    )
    print("iterations are:",iterations)    
    print("results are:",result)