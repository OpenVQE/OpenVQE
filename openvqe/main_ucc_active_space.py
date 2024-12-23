
from qat.fermion.transforms import (get_jw_code, recode_integer)
from openvqe.common_files.qubit_pool import QubitPool
from openvqe.common_files.molecule_factory import MoleculeFactory
from openvqe.ucc_family.get_energy_ucc import EnergyUCC

def main():
    qubitpool = QubitPool()
    molecule_factory = MoleculeFactory()
    energy_ucc = EnergyUCC()


    molecule_symbol = 'H4'
    type_of_generator = 'sUPCCGSD'
    transform = 'JW'
    # the user type "active = True" the active space selection case
    # Here user is obliged to check the thresholds epsilon_1 and epsilon_2  inserted in "MoleculeFactory" to select: the active electorns and active orbitals
    active = True

    # 


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

    hamiltonian, hamiltonian_sp, n_elec, noons_full, orb_energies_full, info = molecule_factory.generate_hamiltonian(molecule_symbol, active=active, transform=transform)

    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Cluster OPS from :")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")

    # for UCCSD
    # pool_size,cluster_ops,cluster_ops_sp, theta_MP2, hf_init = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, 
    #                                transform=transform, active=active)
    # hf_init_sp = recode_integer(hf_init, get_jw_code(hamiltonian_sp.nbqbits))


    # for other types of generators
    pool_size,cluster_ops,cluster_ops_sp =molecule_factory.generate_cluster_ops(molecule_symbol, 
                        type_of_generator=type_of_generator, transform=transform, active=active)
    hf_init = molecule_factory.find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
    nbqbits = hamiltonian_sp.nbqbits
    reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, nbqbits, transform)



    print('Pool size: ', pool_size)
    print('length of the cluster OP: ', len(cluster_ops))
    print('length of the cluster OPS: ', len(cluster_ops_sp))
    FCI = info['FCI']
    # print(hf_init_sp)


    nbqbits = hamiltonian_sp.nbqbits
    pool_type = 'without_Z_from_generator'
    qubit_pool = qubitpool.generate_pool(cluster_ops)
    len_returned_pool, returned_pool = qubitpool.generate_pool_without_cluster(pool_type=pool_type, 
                                                                            nbqbits=nbqbits, 
                                                                            qubit_pool=qubit_pool,
                                                                            molecule_symbol=
                                                                            molecule_symbol)

    # pool_condition='full_without_Z'
    # len_returned_pool, returned_pool = qubitpool.generate_hamiltonian_from_cluster(pool_condition, cluster_ops, nbqbits)
        


    # theta_current1 = theta_MP2
    # theta_current2 = theta_MP2
    # print(theta_current1)
    print(len(cluster_ops_sp))
    pool_generator = returned_pool
    theta_current1 = []
    theta_current2 = []
    ansatz_ops = []
    ansatz_q_ops = []
    for i in range(len(cluster_ops_sp)):
        theta_current1.append(0.01)

    for i in range(len(returned_pool)):
        theta_current2.append(0.01)
    # for UCCS skip 1j
    for i in cluster_ops_sp:
        ansatz_ops.append(i*1j)
    for i in pool_generator:
        ansatz_q_ops.append(i)
    iterations, result = energy_ucc.get_energies(hamiltonian_sp,ansatz_ops,ansatz_q_ops,hf_init_sp,theta_current1,theta_current2,FCI)
    print("iterations are:", iterations)
    print("results are:", result)


    print(iterations, result)

if __name__ == "__main__":
    main()