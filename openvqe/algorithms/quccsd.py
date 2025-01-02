from openvqe.ucc_family.get_energy_qucc import EnergyUCC
from openvqe.common_files.molecule_factory import MoleculeFactory
from qat.fermion.transforms import (get_jw_code, recode_integer)
from openvqe.algorithms import tools

def execute(molecule_symbol, type_of_generator, transform, active, opts={}):
    # Parameters for the QUCCSD algorithm
    opts = {
        'step': 0.01
    } | opts
    
    molecule_factory = MoleculeFactory()
    energy_ucc = EnergyUCC()

    tools.presentation(molecule_symbol, type_of_generator, transform, active, opts)
    _, hamiltonian_sp, _, _, _, info = tools.generate_hamiltonian(molecule_factory, molecule_symbol, type_of_generator, transform, active)
    _, cluster_ops, _, theta_MP2, hf_init = tools.generate_cluster_ops(molecule_factory, molecule_symbol, type_of_generator, transform, active)
    hf_init_sp = recode_integer(hf_init, get_jw_code(hamiltonian_sp.nbqbits))
    theta_current1 = theta_MP2
    theta_current2 = []
    for i in range(len(cluster_ops)):
        theta_current2.append(opts['step'])

    iterations, result = energy_ucc.get_energies(
        hamiltonian_sp,
        cluster_ops,
        hf_init_sp,
        theta_current1,
        theta_current2,
        info['FCI']
    )
    print("iterations are:", iterations)
    print("results are:", result)
    
    return iterations, result
