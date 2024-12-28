from openvqe.ucc_family.get_energy_qucc import EnergyUCC
from openvqe.common_files.molecule_factory import MoleculeFactory
from qat.fermion.transforms import (get_jw_code, recode_integer)

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


def generate_hamiltonian(molecule_symbol, type_of_generator, transform, active):
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

    pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)
    
    print('Pool size: ', pool_size)
    print('length of the cluster OP: ', len(cluster_ops))
    print('length of the cluster OPS: ', len(cluster_ops_sp))
    
    return pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init

def execute(molecule_symbol, type_of_generator, transform, active):
    molecule_factory = MoleculeFactory()
    energy_ucc = EnergyUCC()

    presentation(molecule_symbol, type_of_generator, transform, active)
    hamiltonian, hamiltonian_sp, n_elec, noons_full, orb_energies_full, info = generate_hamiltonian(molecule_symbol, type_of_generator, transform, active)
    pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init = generate_cluster_ops(molecule_symbol, type_of_generator, transform, active)
    hf_init_sp = recode_integer(hf_init, get_jw_code(hamiltonian_sp.nbqbits))
    theta_current1 = theta_MP2
    theta_current2 = []
    for i in range(len(cluster_ops)):
        theta_current2.append(0.01)

    iterations, result = energy_ucc.get_energies(hamiltonian_sp,cluster_ops,hf_init_sp,theta_current1,theta_current2,info['FCI'])
    print("iterations are:", iterations)
    print("results are:", result)
