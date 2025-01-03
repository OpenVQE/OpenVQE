import scipy.optimize
from qat.fermion.chemistry.ucc_deprecated import build_ucc_ansatz
from qat.lang.AQASM import Program
from qat.qpus import get_default_qpu
from ..common_files.circuit import count

class EnergyUCC:
    def ucc_action(self, theta_current, hamiltonian_sp, cluster_ops_sp, hf_init_sp, energies=[]):
        """
        It maps the exponential of cluster operators ("cluster_ops_sp") associated by their parameters ("theta_current")
        using the CNOTS-staircase method, which is done by "build_ucc_ansatz" which creates the circuit on the top of
        the HF-state ("hf_init_sp"). Then, this function also calculates the expected value of the hamiltonian ("hamiltonian_sp").

        Parameters
        ----------
        theta_current: List<float>
            the Parameters of the cluster operators
        
        hamiltonian_sp: Hamiltonian
                Hamiltonian in the spin representation
            
        cluster_ops_sp: list[Hamiltonian]
            list of spin cluster operators
        
        hf_init_sp: int
            the integer corresponds to the hf_init (The Hartree-Fock state in integer representation) obtained by using
            "qat.fermion.transforms.record_integer".
        
        Returns
        --------
            res.value: float
                the resulted energy

        """
        qpu = 0
        prog = 0
        reg = 0
        qpu = get_default_qpu()
        prog = Program()
        reg = prog.qalloc(hamiltonian_sp.nbqbits)
        qrout = 0
        for n_term, (term, theta_term) in enumerate(zip(cluster_ops_sp, theta_current)):
            init = hf_init_sp if n_term == 0 else 0
            qprog = build_ucc_ansatz([term], init, n_steps=1)
            prog.apply(qprog([theta_term]), reg)
        circ = prog.to_circ()
        job = circ.to_job(job_type="OBS", observable=hamiltonian_sp)
        res = qpu.submit(job)
        energies.append(res.value)
        return res.value

    def prepare_state_ansatz(
        self, hamiltonian_sp, cluster_ops_sp, hf_init_sp, parameters
    ):
        """
        It constructs the trial wave function (ansatz) 

        Parameters
        ----------
        hamiltonian_sp: Hamiltonian
                Hamiltonian in the spin representation
            
        cluster_ops_sp: list[Hamiltonian]
            list of spin cluster operators
        
        hf_init_sp: int
            the integer corresponds to the hf_init (The Hartree-Fock state in integer representation) obtained by using
            "qat.fermion.transforms.record_integer".
        
        parameters: List<float>
            the Parameters for the trial wave function to be constructed
        


        Returns
        --------
            curr_state: qat.core.Circuit
                the circuit that represent the trial wave function
        
        """
        qpu = get_default_qpu()
        prog = Program()
        reg = prog.qalloc(hamiltonian_sp.nbqbits)
        for n_term, (term, theta_term) in enumerate(zip(cluster_ops_sp, parameters)):
            init = hf_init_sp if n_term == 0 else 0
            qprog = build_ucc_ansatz([term], init, n_steps=1)
            prog.apply(qprog([theta_term]), reg)
        circ = prog.to_circ()
        curr_state = circ
        return curr_state

    def get_energies(
        self,
        hamiltonian_sp,
        cluster_ops_sp,
        pool_generator,
        hf_init_sp,
        theta_current1,
        theta_current2,
        fci,
    ):
        """
        It calls internally the functions "ucc_action" and "prepare_state_ansatz", and uses scipy.optimize to
        return the properties of the ucc energy and wave function.

        Parameters
        ----------
        hamiltonian_sp: Hamiltonian
                Hamiltonian in the spin representation
            
        cluster_ops_sp: list[Hamiltonian]
            list of spin cluster operators

        pool_generator: 
            the pool containing the operators made of Pauli strings that doesn't contain Z-Pauli term.
        
        hf_init_sp: int
            the integer corresponds to the hf_init (The Hartree-Fock state in integer representation) obtained by using
            "qat.fermion.transforms.record_integer".
        
        theta_current1: List<float>
            the Parameters of the cluster operators of "cluster_ops_sp"
        
        theta_current2: List<float>
            the Parameters of the cluster operators of "pool_generator"
        
        fci: float
            the full configuration interaction energy (for any basis set)
    
        
        Returns
        --------
            iterations: Dict
                the minimum energy and the optimized parameters
            
            result: Dict
                the number of CNOT gates, the number of operators/parameters, and the substraction of the optimized energy from fci.
        
        """

        iterations = {
            "minimum_energy_result1_guess": [],
            "minimum_energy_result2_guess": [],
            "theta_optimized_result1": [],
            "theta_optimized_result2": [],
        }
        result = {}
        tolerance = 10 ** (-4)
        method = "BFGS"
        print("tolerance= ", tolerance)
        print("method= ", method)

        theta_optimized_result1 = []
        theta_optimized_result2 = []
        energies_1 = []
        energies_2 = []

        opt_result1 = scipy.optimize.minimize(
            lambda theta: self.ucc_action(
                theta, hamiltonian_sp, cluster_ops_sp, hf_init_sp, energies_1
            ),
            x0=theta_current1,
            method=method,
            tol=tolerance,
            options={"maxiter": 50000, "disp": True},
        )
        opt_result2 = scipy.optimize.minimize(
            lambda theta: self.ucc_action(
                theta, hamiltonian_sp, pool_generator, hf_init_sp, energies_2
            ),
            x0=theta_current2,
            method=method,
            tol=tolerance,
            options={"maxiter": 50000, "disp": True},
        )

        xlist1 = opt_result1.x
        xlist2 = opt_result2.x

        for si in range(len(theta_current1)):
            theta_optimized_result1.append(xlist1[si])
        for si in range(len(theta_current2)):
            theta_optimized_result2.append(xlist2[si])
        curr_state_result1 = self.prepare_state_ansatz(
            hamiltonian_sp, cluster_ops_sp, hf_init_sp, theta_optimized_result1
        )
        curr_state_result2 = self.prepare_state_ansatz(
            hamiltonian_sp, cluster_ops_sp, hf_init_sp, theta_optimized_result2
        )
        gates1 = curr_state_result1.ops
        gates2 = curr_state_result2.ops
        cnot1 = count("CNOT", gates1)
        cnot2 = count("CNOT", gates2)
        iterations["minimum_energy_result1_guess"].append(opt_result1.fun)
        iterations["minimum_energy_result2_guess"].append(opt_result2.fun)
        iterations["theta_optimized_result1"].append(theta_optimized_result1)
        iterations["theta_optimized_result2"].append(theta_optimized_result2)
        result["CNOT1"] = cnot1
        result["CNOT2"] = cnot2
        result["len_op1"] = len(theta_optimized_result1)
        result["len_op2"] = len(theta_optimized_result2)
        result["energies1_substracted_from_FCI"] = abs(opt_result1.fun - fci)
        result["energies2_substracted_from_FCI"] = abs(opt_result2.fun - fci)
        result['energies_1'] = energies_1
        result['energies_2'] = energies_2
        return iterations, result
