from openvqe.common_files.molecule_factory import MoleculeFactory
from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory as SparseMoleculeFactory

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

def generate_hamiltonian(molecule_factory, molecule_symbol, type_of_generator, transform, active):
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Hamiltonians and Properties from :")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")

    return molecule_factory.generate_hamiltonian(molecule_symbol, active=active, transform=transform)

def generate_cluster_ops(molecule_factory, molecule_symbol, type_of_generator, transform, active):
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Generate Cluster OPS:")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                           ")

    args = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)

    print('Pool size: ', args[0])
    print('length of the cluster OP: ', len(args[1]))
    print('length of the cluster OPS: ', len(args[2]))
    if molecule_factory.sparse():
        print('length of the cluster _sparse: ', len(args[3]))

    return args