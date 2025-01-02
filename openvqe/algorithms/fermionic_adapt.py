from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.fermionic_adapt_vqe import fermionic_adapt_vqe
from openvqe.algorithms import tools

def execute(molecule_symbol, type_of_generator, transform, active, opts=None):
    """
    Executes the ADAPT-VQE algorithm for a given molecule.
    
    The ADAPT-VQE (Adaptive Derivative-Assembled Pseudo-Trotter Variational Quantum Eigensolver) algorithm adaptively 
    builds the quantum circuit by iteratively adding operators that most significantly reduce the energy, leading to 
    efficient and accurate ground state energy estimation.
    
    This implementation uses fermionic operators to construct the ansatz. The fermionic operators are transformed into 
    qubit operators (e.g., using Jordan-Wigner or Bravyi-Kitaev transformation) to be used in the quantum circuit.

    Args:
        molecule_symbol (str): The chemical symbol of the molecule.
        type_of_generator (str): The type of generator to use for the algorithm.
        transform (str): The type of transformation to apply.
        active (list): List of active orbitals.
    Returns:
        None: The function prints the number of iterations and the resulting energies.
    """
    
    # Parameters for the ADAPT-VQE algorithm
    opts = {
        'n_max_grads': 1,
        'optimizer': 'COBYLA',
        'tolerance': 10**(-6),
        'type_conver': 'norm',
        'threshold_needed': 1e-2,
        'max_external_iterations': 35
    } | opts
    
    molecule_factory = MoleculeFactory()

    tools.presentation(molecule_symbol, type_of_generator, transform, active, opts)
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
        opts['n_max_grads'], 
        info['FCI'], 
        opts['optimizer'],                
        opts['tolerance'],                
        type_conver = opts['type_conver'],
        threshold_needed = opts['threshold_needed'],
        max_external_iterations = opts['max_external_iterations']
    )
    print("iterations are:",iterations)    
    print("results are:",result)
    
    return iterations, result