
from openvqe.adapt.qubit_adapt_vqe import qubit_adapt_vqe
from openvqe.common_files.qubit_pool import QubitPool
from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory

qubitpool = QubitPool()
molecule_factory = MoleculeFactory()



molecule_symbol = 'H2'
# In qubit ADAPT-VQE normally we choose the generalized single and double excitations
type_of_generator = 'singlet_gsd'
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
hf_init = molecule_factory.find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, nbqbits, transform)
print(" --------------------------------------------------------------------------")
print("                                                          ")
print("                      Generate Cluster OPS from :")
print("                                                          ")
print(" --------------------------------------------------------------------------")
print("                                                          ")

pool_size,cluster_ops, cluster_ops_sp, cluster_ops_sparse = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)

print('Pool size: ', pool_size)
print('length of the cluster OP: ', len(cluster_ops))
print('length of the cluster OPS: ', len(cluster_ops_sp))
print('length of the cluster _sparse: ', len(cluster_ops_sp))


nbqbits = hamiltonian_sp.nbqbits
# user can just type the name of pool wanted: full, full_without_Z, reduced_without_Z, YXXX, XYXX,XXYX,XXXY,random, two, four, eight
# pure_with_symmetry

# for example here user can put radom (YXXX, XYXX, XXYX, XXXY):
pool_type = 'random'
qubit_pool =qubitpool.generate_pool(cluster_ops)
len_returned_pool, returned_pool = qubitpool.generate_pool_without_cluster(pool_type=pool_type, 
                                                                        nbqbits=nbqbits, 
                                                                        qubit_pool=qubit_pool,
                                                                        molecule_symbol=molecule_symbol)
# or user can type:
# pool_condition='full_without_Z'
# len_returned_pool, returned_pool = qubitpool.generate_hamiltonian_from_cluster(pool_condition, cluster_ops, nbqbits)
pool_mix = returned_pool
pool_pure = returned_pool
print("length of the pool",len(pool_mix))
iterations_sim, iterations_ana, result_sim, result_ana = qubit_adapt_vqe(hamiltonian_sp, hamiltonian_sp_sparse,
       reference_ket, nbqbits, pool_mix, hf_init_sp, info['FCI'],
        chosen_grad = 1,
        adapt_conver    = 'norm',
        adapt_thresh    = 1e-07,
        adapt_maxiter   = 29,
        tolerance_sim = 1e-09,
        method_sim = 'BFGS')
print("iterations",iterations_sim)    
print("results",result_sim)





