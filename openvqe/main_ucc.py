
from openvqe.common_files.qubit_pool import QubitPool
from openvqe.common_files.molecule_factory import MoleculeFactory
from openvqe.ucc_family.get_energy_ucc import EnergyUCC

qubit_pool = QubitPool()
molecule_factory = MoleculeFactory()
energy_ucc = EnergyUCC()


# user can type the name of molecule (H2, LIH, CO, CO2 such that their geormetries and properties are defined in MoleculeQlm())
molecule_symbol = 'H2'
# the type of generators: UCCSD, singlet_sd, singlet_gsd, spin_complement_gsd, spin_complement_gsd_twin, sUPCCGSD
# suppose user type sUPCCGSD
type_of_generator = 'sUPCCGSD'
# user can type any of the following three transformations: JW,  Bravyi-Kitaev and Parity-basis
transform = 'JW'
# the non_active space selection
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
print("                      Generate Hamiltonians and Properties:")
print("                                                          ")
print(" --------------------------------------------------------------------------")
print("                                                          ")

hamiltonian, hamiltonian_sp, n_elec, noons_full, orb_energies_full, info = molecule_factory.generate_hamiltonian(molecule_symbol, active=active, transform=transform)

print(" --------------------------------------------------------------------------")
print("                                                          ")
print("                      Generate Cluster OPS:")
print("                                                          ")
print(" --------------------------------------------------------------------------")
print("                                                           ")


# properties 
nbqbits = hamiltonian_sp.nbqbits

# for uccsd_case 
# pool_size,cluster_ops,cluster_ops_sp, theta_MP2, hf_init = molecule_factory.generate_cluster_ops(molecule_symbol,
#                                         type_of_generator=type_of_generator, transform=transform, active=active)
# hf_init_sp = recode_integer(hf_init, get_jw_code(hamiltonian_sp.nbqbits))

# for other type of excitations like UCCGSD, k-UpCCGSD, spin-complement pairs etc  
pool_size,cluster_ops,cluster_ops_sp = molecule_factory.generate_cluster_ops(molecule_symbol, 
                    type_of_generator=type_of_generator, transform=transform, active=active)
hf_init = molecule_factory .find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, nbqbits, transform)

print('Pool size: ', pool_size)
print('length of the cluster OP: ', len(cluster_ops))
print('length of the cluster OPS: ', len(cluster_ops_sp))
fci = info['FCI']



# suppose user wants to generate cluster operators using the Qubit Pool generators: then we can have two functions, 
# 1. generate_pool_without_cluster 2.generate_pool_from_cluster

#1. using generate_pool_without_cluster:
# pool_type = 'full_without_Z'
# qubit_pool = QubitPool.generate_pool(cluster_ops)
# len_returned_pool, returned_pool = qubit_pool.generate_pool_without_cluster(pool_type=pool_type, 
#                                                                         nbqbits=nbqbits, 
#                                                                         qubit_pool=qubit_pool,
#                                                                         molecule_symbol=
#                                                                         molecule_symbol)

# 2. generate_pool_from_cluster
pool_condition='reduced_without_Z'
len_returned_pool, returned_pool = qubit_pool.generate_pool_from_cluster(pool_condition, cluster_ops, nbqbits)
    


# HERE USER can apply UCC-family VQE using MP2 guess defined by theta_current1; and using either a constant value starting guess 
#(0.0,0.001,1.0, etc) or random numbers defined by theta_current2


pool_generator = returned_pool
theta_current2 = []
for i in range(len(returned_pool)):
    theta_current2.append(0.01)
# when UCCSD is not used then one can make theta_current1 same as theta_current2     
theta_current1  = theta_current2
# define two lists: 1. ansatz_ops to append operators direct coming from fermionic evolutions (after spin-transformation) 
# and 2. ansatz_q_ops to append operators coming from "Qubit Pool generators"
ansatz_ops = []
ansatz_q_ops = []

# important note: when UCCSD  is called at the top, then we must skip 1j in ansatz_ops (why? due to the internal functions defined by "mqlm-fermion" package)


for i in cluster_ops_sp:
    ansatz_ops.append(i*1j)
for i in pool_generator:
    ansatz_q_ops.append(i)
    
# now we can run EnergyUCC.get_energies to get to get the energies and properties (CNOT counts, )
# 1.  from  UCC-family (with ansatz_ops and theta-current1) 
# 2. from  UCC-family but with qubit evolutions i.e from "Qubit Pool" (with ansatz_q_ops and theta-current2)

iterations, result = energy_ucc.get_energies(hamiltonian_sp,ansatz_ops,ansatz_q_ops,hf_init_sp,theta_current1,theta_current2,fci)
print("iterations are:", iterations)
print("results are:", result)


print(iterations, result)





