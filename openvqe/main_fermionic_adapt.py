from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.fermionic_adapt_vqe import fermionic_adapt_vqe

def main():
    molecule_factory = MoleculeFactory()

    ## non active case
    molecule_symbol = 'H4'
    type_of_generator = 'spin_complement_gsd'
    transform = 'JW'
    active = False
    r, geometry, charge, spin, basis = molecule_factory.get_parameters(molecule_symbol)
    print(" --------------------------------------------------------------------------")
    print("Running in the non active case: ")
    print("                     molecule symbol: %s " %(molecule_symbol))
    print("                     molecule basis: %s " %(basis))
    print("                     type of generator: %s " %(type_of_generator))
    print("                     transform: %s " %(transform))
    print(" --------------------------------------------------------------------------")

    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Hamiltonians and Properties from :")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    hamiltonian, hamiltonian_sparse, hamiltonian_sp, hamiltonian_sp_sparse, n_elec, noons_full, orb_energies_full, info = molecule_factory.generate_hamiltonian(molecule_symbol,active=active, transform=transform)
    nbqbits = len(orb_energies_full)
    print(n_elec)
    hf_init = molecule_factory.find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
    reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, nbqbits, transform)
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Cluster OPS from :")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    pool_size,cluster_ops, cluster_ops_sp, cluster_ops_sparse = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)
    # for case of UCCSD from  library
    # pool_size,cluster_ops, cluster_ops_sp, cluster_ops_sparse,theta_MP2, hf_init = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator,transform=transform, active=active)

    print('Pool size: ', pool_size)
    print('length of the cluster OP: ', len(cluster_ops))
    print('length of the cluster OPS: ', len(cluster_ops_sp))
    print(hf_init_sp)
    print(reference_ket)
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Start ADAPT-VQE algorithm:")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")


    n_max_grads = 1
    optimizer = 'COBYLA'                
    tolerance = 10**(-6)            
    type_conver = 'norm'
    threshold_needed = 1e-2
    max_external_iterations = 35
    fci = info['FCI']
    fermionic_adapt_vqe(hamiltonian_sparse, cluster_ops_sparse, reference_ket, hamiltonian_sp,
            cluster_ops_sp, hf_init_sp, n_max_grads, fci, 
            optimizer,                
            tolerance,                
            type_conver = type_conver,
            threshold_needed = threshold_needed,
            max_external_iterations = max_external_iterations)



    ## active case

    # initializing the variables in the case of active 
    molecule_symbol = 'H4'
    type_of_generator = 'spin_complement_gsd' #'spin_complement_gsd_twin'
    transform = 'JW'
    active = True

    r, geometry, charge, spin, basis = molecule_factory.get_parameters(molecule_symbol)
    print(" --------------------------------------------------------------------------")
    print("Running in the active case: ")
    print("                     molecule symbol: %s " %(molecule_symbol))
    print("                     molecule basis: %s " %(basis))
    print("                     type of generator: %s " %(type_of_generator))
    print("                     transform: %s " %(transform))
    print(" --------------------------------------------------------------------------")


    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Hamiltonians and Properties from :")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    hamiltonian_active, hamiltonian_active_sparse, hamiltonian_sp,hamiltonian_sp_sparse,nb_active_els,active_noons,active_orb_energies,info=molecule_factory.generate_hamiltonian(molecule_symbol,active=active,transform=transform)
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Cluster OPS from :")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    nbqbits = hamiltonian_sp.nbqbits
    hf_init = molecule_factory.find_hf_init(hamiltonian_active, nb_active_els,active_noons, active_orb_energies)
    reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, nbqbits, transform)
    pool_size,cluster_ops, cluster_ops_sp, cluster_ops_sparse = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)
    print("Clusters were generated...")
    print('Pool size: ', pool_size)
    print('length of the cluster OP: ', len(cluster_ops))
    print('length of the cluster OPS: ', len(cluster_ops_sp))
    print('length of the cluster OPS_sparse: ', len(cluster_ops_sp))

    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Start ADAPT-VQE algorithm:")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")

    n_max_grads = 1
    optimizer = 'COBYLA'                
    tolerance = 10**(-7)            
    type_conver = 'norm'
    threshold_needed = 1e-3
    max_external_iterations = 30
    fci = info['FCI']
    fermionic_adapt_vqe(hamiltonian_active_sparse, cluster_ops_sparse, reference_ket, hamiltonian_sp,
            cluster_ops_sp, hf_init_sp, n_max_grads, fci, 
            optimizer,                
            tolerance,                
            type_conver = type_conver,
            threshold_needed = threshold_needed,
            max_external_iterations = max_external_iterations)

if __name__ == "__main__":
    main()