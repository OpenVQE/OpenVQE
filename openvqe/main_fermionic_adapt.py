from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.fermionic_adapt_vqe import fermionic_adapt_vqe

def main():
    ## non active case
    execute('H4', 'spin_complement_gsd', 'JW', False)
    ## active case
    execute('H4', 'spin_complement_gsd', 'JW', True)

def execute(molecule_symbol, type_of_generator, transform, active):
    molecule_factory = MoleculeFactory()

    r, geometry, charge, spin, basis = molecule_factory.get_parameters(molecule_symbol)
    print(" --------------------------------------------------------------------------")
    if active:
        print("Running in the active case:")
    else:
        print("Running in the non active case:")
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

if __name__ == "__main__":
    main()