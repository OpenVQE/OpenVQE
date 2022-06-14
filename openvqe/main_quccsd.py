

from openvqe.ucc_family.get_energy_qucc import EnergyUCC
from openvqe.common_files.molecule_factory import MoleculeFactory

from qat.fermion.transforms import (get_jw_code, recode_integer)


molecule_factory = MoleculeFactory()
energy_ucc = EnergyUCC()

molecule_symbol = 'H4'
type_of_generator = 'QUCCSD'
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

hamiltonian, hamiltonian_sp, n_elec, noons_full, orb_energies_full, info = molecule_factory.generate_hamiltonian(molecule_symbol, active=active, transform=transform)

print(" --------------------------------------------------------------------------")
print("                                                          ")
print("                      Generate Cluster OPS from :")
print("                                                          ")
print(" --------------------------------------------------------------------------")
print("                                                          ")

pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)
hf_init_sp = recode_integer(hf_init, get_jw_code(hamiltonian_sp.nbqbits))


# for other type of generators
# pool_size,cluster_ops,cluster_ops_sp =molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)
# nbqbits = hamiltonian_sp.nbqbits
# hf_init = molecule_factory.find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
# reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, nbqbits, transform)
print('Pool size: ', pool_size)
print('length of the cluster OP: ', len(cluster_ops))
print('length of the cluster OPS: ', len(cluster_ops_sp))

FCI = info['FCI']
nbqbits = hamiltonian_sp.nbqbits


theta_current1 = theta_MP2
# theta_current1 = []
theta_current2 = []
ansatz_ops = []
# for UCCS skip 1j
# for i in range(len(cluster_ops_sp)):
#     theta_current1.append(0.01)
for i in range(len(cluster_ops)):
    theta_current2.append(0.01)
for i in cluster_ops_sp:
    ansatz_ops.append(i)
iterations, result = energy_ucc.get_energies(hamiltonian_sp,cluster_ops,ansatz_ops,hf_init_sp,theta_current1,theta_current2,FCI)
print("iterations are:", iterations)
print("results are:", result)








