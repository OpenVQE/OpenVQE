from openvqe.ucc_family.get_energy_qucc import EnergyUCC
from openvqe.common_files.molecule_factory import MoleculeFactory
from qat.fermion.transforms import (get_jw_code, recode_integer)
from openvqe.algorithms.algorithm import Algorithm

class QUCCSD(Algorithm):

    def execute(self):
        # Parameters for the QUCCSD algorithm
        self.opts = {
            'step': 0.01
        } | self.opts
        
        molecule_factory = MoleculeFactory()
        energy_ucc = EnergyUCC()

        self.presentation(molecule_factory)
        _, hamiltonian_sp, _, _, _, info = self.generate_hamiltonian(molecule_factory)
        _, cluster_ops, _, theta_MP2, hf_init = self.generate_cluster_ops(molecule_factory)
        hf_init_sp = recode_integer(hf_init, get_jw_code(hamiltonian_sp.nbqbits))
        theta_current1 = theta_MP2
        theta_current2 = []
        for i in range(len(cluster_ops)):
            theta_current2.append(self.opts['step'])

        self.info = info
        
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
        self.iterations = iterations
        self.result = result
