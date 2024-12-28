from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.fermionic_adapt_vqe import fermionic_adapt_vqe

def presentation(molecule_symbol, type_of_generator, transform, active):
    molecule_factory = MoleculeFactory()

    r, geometry, charge, spin, basis = molecule_factory.get_parameters(molecule_symbol)
    print(" --------------------------------------------------------------------------")
    if active:
        print("Running in the active case: ")
    else:
        print("Running in the non active case: ")
    print("                     molecule symbol: %s " %(molecule_symbol))
    print("                     molecule basis: %s " %(basis))
    print("                     type of generator: %s " %(type_of_generator))
    print("                     transform: %s " %(transform))
    print(" --------------------------------------------------------------------------")

def generate_sparse_hamiltonian(molecule_symbol, type_of_generator, transform, active):
    molecule_factory = MoleculeFactory()
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Hamiltonians and Properties from :")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")

    return molecule_factory.generate_hamiltonian(molecule_symbol, active=active, transform=transform)   

def generate_cluster_ops(molecule_symbol, type_of_generator, transform, active):
    molecule_factory = MoleculeFactory()
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Cluster OPS:")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                           ")

    pool_size,cluster_ops, cluster_ops_sp, cluster_ops_sparse = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)

    print('Pool size: ', pool_size)
    print('length of the cluster OP: ', len(cluster_ops))
    print('length of the cluster OPS: ', len(cluster_ops_sp))
    print('length of the cluster _sparse: ', len(cluster_ops_sp))

    return pool_size,cluster_ops,cluster_ops_sp, cluster_ops_sparse

def execute(molecule_symbol, type_of_generator, transform, active):
    molecule_factory = MoleculeFactory()

    presentation(molecule_symbol, type_of_generator, transform, active)
    hamiltonian, hamiltonian_sparse, hamiltonian_sp, hamiltonian_sp_sparse, n_elec, noons_full, orb_energies_full, info = generate_sparse_hamiltonian(molecule_symbol, type_of_generator, transform, active)
    pool_size, cluster_ops, cluster_ops_sp, cluster_ops_sparse = generate_cluster_ops(molecule_symbol, type_of_generator, transform, active)
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


    n_max_grads = 1
    optimizer = 'COBYLA'                
    tolerance = 10**(-6)            
    type_conver = 'norm'
    threshold_needed = 1e-2
    max_external_iterations = 35
    fermionic_adapt_vqe(hamiltonian_sparse, cluster_ops_sparse, reference_ket, hamiltonian_sp,
            cluster_ops_sp, hf_init_sp, n_max_grads, info['FCI'], 
            optimizer,                
            tolerance,                
            type_conver = type_conver,
            threshold_needed = threshold_needed,
            max_external_iterations = max_external_iterations)
