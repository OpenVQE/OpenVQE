
from openvqe.common_files.qubit_pool import QubitPool
from openvqe.common_files.molecule_factory import MoleculeFactory
from openvqe.ucc_family.get_energy_ucc import EnergyUCC
from qat.fermion.transforms import (get_jw_code, recode_integer)
from openvqe.algorithms import tools

def generate_pool_from_cluster(cluster_ops, nbqbits):
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Pool from Cluster:")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                           ")

    qubit_pool = QubitPool()
    pool_condition='reduced_without_Z'
    len_returned_pool, returned_pool = qubit_pool.generate_pool_from_cluster(pool_condition, cluster_ops, nbqbits)
    return len_returned_pool, returned_pool

def get_ansatz(cluster_ops_sp, returned_pool):
    # define two lists: 1. ansatz_ops to append operators direct coming from fermionic evolutions (after spin-transformation) 
    # and 2. ansatz_q_ops to append operators coming from "Qubit Pool generators"
    # important note: when UCCSD  is called at the top, then we must skip 1j in ansatz_ops (why? due to the internal functions defined by "mqlm-fermion" package)
    ansatz_ops = []
    ansatz_q_ops = []

    for i in cluster_ops_sp:
        ansatz_ops.append(i*1j)
    for i in returned_pool:
        ansatz_q_ops.append(i)
        
    return ansatz_ops, ansatz_q_ops

def execute(molecule_symbol, type_of_generator, transform, active, opts={}):
    """
    Executes the UCC (Unitary Coupled Cluster) algorithm for a given molecule.
        molecule_symbol (str): The chemical symbol of the molecule.
        type_of_generator (str): The type of generator to use for the UCC algorithm.
        transform (str): The type of transformation to apply (e.g., Jordan-Wigner, Bravyi-Kitaev).
        active (list): List of active orbitals to consider in the calculation.
    Returns:
        None: The function prints the number of iterations and the resulting energies.
    """
    # Parameters for the UCC algorithm
    opts = {
        'step': 0.01
    } | opts
    
    molecule_factory = MoleculeFactory()
    energy_ucc = EnergyUCC()

    tools.presentation(molecule_symbol, type_of_generator, transform, active, opts)
    hamiltonian, hamiltonian_sp, n_elec, noons_full, orb_energies_full, info = tools.generate_hamiltonian(molecule_factory, molecule_symbol, type_of_generator, transform, active)
    _, cluster_ops, cluster_ops_sp = tools.generate_cluster_ops(molecule_factory, molecule_symbol, type_of_generator, transform, active)
    _, returned_pool = generate_pool_from_cluster(cluster_ops, hamiltonian_sp.nbqbits)
    hf_init = molecule_factory.find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
    _, hf_init_sp = molecule_factory.get_reference_ket(hf_init, hamiltonian_sp.nbqbits, transform)
    ansatz_ops, ansatz_q_ops = get_ansatz(cluster_ops_sp, returned_pool)
    # when UCCSD is not used then one can make theta_current1 same as theta_current2     
    theta_current = []
    for i in range(len(returned_pool)):
        theta_current.append(opts['step'])
    
    # now we can run EnergyUCC.get_energies to get to get the energies and properties (CNOT counts, )
    # 1.  from  UCC-family (with ansatz_ops and theta-current1) 
    # 2. from  UCC-family but with qubit evolutions i.e from "Qubit Pool" (with ansatz_q_ops and theta-current2)

    iterations, result = energy_ucc.get_energies(
        hamiltonian_sp,
        ansatz_ops,
        ansatz_q_ops,
        hf_init_sp,
        theta_current,
        theta_current,
        info['FCI']
    )
    print("iterations are:", iterations)
    print("results are:", result)
    print(iterations, result)
    
    return iterations, result