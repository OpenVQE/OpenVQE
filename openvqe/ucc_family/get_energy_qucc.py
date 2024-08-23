import scipy.optimize
from numpy import binary_repr

from qat.lang.AQASM import Program, X
from qat.qpus import get_default_qpu

from ..common_files.circuit import efficient_fermionic_ansatz, count


class EnergyUCC:
    def action_quccsd(self, theta_0, hamiltonian_sp, cluster_ops, hf_init_sp):
        """
        It returns the energy from the qubit coupled cluster ansatz which are obtained from common_files.circuit

        Parameters
        ----------
        theta_0: List<float>
            the Parameters of the cluster operators
        
        hamiltonian_sp: Hamiltonian
                Hamiltonian in the spin representation
            
        cluster_ops: list[Hamiltonian]
            list of fermionic cluster operators
        
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
        prog = Program()
        q = prog.qalloc(hamiltonian_sp.nbqbits)
        ket_hf = binary_repr(hf_init_sp)
        list_ket_hf = [int(c) for c in ket_hf]
        # print(list_ket_hf)
        for j in range(hamiltonian_sp.nbqbits):
            if int(list_ket_hf[j] == 1):
                prog.apply(X, q[j])
        list_exci = []
        for j in cluster_ops:
            s = j.terms[0].qbits
            list_exci.append(s)
        qpu = get_default_qpu()
        qprog = efficient_fermionic_ansatz(q, prog, list_exci, theta_0)
        circ = qprog.to_circ()
        job = circ.to_job(job_type="OBS", observable=hamiltonian_sp)
        res = qpu.submit(job)
        return res.value

    def prepare_hf_state(self, hf_init_sp, cluster_ops_sp):
        """
        It constructs the Hartree-Fock state (ansatz)

        Parameters
        ----------

        hf_init_sp: int
            the integer corresponds to the hf_init (The Hartree-Fock state in integer representation) obtained by using
            "qat.fermion.transforms.record_integer".

        cluster_ops_sp: list[Hamiltonian]
            list of spin cluster operators
        

        Returns
        --------
            circuit: qat.core.Circuit
                the circuit representing the HF-state
        
        """
        prog = Program()
        nbqbits = cluster_ops_sp[0].nbqbits
        ket_hf = binary_repr(hf_init_sp)
        list_ket_hf = [int(c) for c in ket_hf]
        qb = prog.qalloc(nbqbits)
        # print(list_ket_hf)
        for j in range(nbqbits):
            if int(list_ket_hf[j] == 1):
                prog.apply(X, qb[j])
        circuit = prog.to_circ()
        return circuit

    def prepare_state_ansatz(self, hamiltonian_sp, hf_init_sp, cluster_ops, theta):
        """
        It constructs the "qubit coupled cluster" trial wave function (ansatz) 

        Parameters
        ----------
        hamiltonian_sp: Hamiltonian
                Hamiltonian in the spin representation
            
        cluster_ops: list[Hamiltonian]
            list of fermionic cluster operators
        
        hf_init_sp: int
            the integer corresponds to the hf_init (The Hartree-Fock state in integer representation) obtained by using
            "qat.fermion.transforms.record_integer".
        
        theta: List<float>
            the Parameters for the trial wave function to be constructed
        


        Returns
        --------
            curr_state: qat.core.Circuit
                the circuit that represent the trial wave function
        
        """
        prog = Program()
        q = prog.qalloc(hamiltonian_sp.nbqbits)
        ket_hf = binary_repr(hf_init_sp)
        list_ket_hf = [int(c) for c in ket_hf]
        # print(list_ket_hf)
        for j in range(hamiltonian_sp.nbqbits):
            if int(list_ket_hf[j] == 1):
                prog.apply(X, q[j])
        list_exci = []
        for j in cluster_ops:
            s = j.terms[0].qbits
            list_exci.append(s)
        qprog = efficient_fermionic_ansatz(q, prog, list_exci, theta)
        circ = qprog.to_circ()
        curr_state = circ
        return curr_state

    def get_energies(
        self,
        hamiltonian_sp,
        cluster_ops,
        hf_init_sp,
        theta_current1,
        theta_current2,
        FCI,
    ):
        """
        It calls internally the functions "action_quccsd" and "prepare_state_ansatz", and uses scipy.optimize to
        return the properties of 

        Parameters
        ----------
        hamiltonian_sp: Hamiltonian
                Hamiltonian in the spin representation
            
        cluster_ops: list[Hamiltonian]
            list of fermionic cluster operators

        hf_init_sp: int
            the integer corresponds to the hf_init (The Hartree-Fock state in integer representation) obtained by using
            "qat.fermion.transforms.record_integer".
        
        theta_current1: List<float>
            MP2 initial guess obtained from "qat.fermion.chemistry.ucc_deprecated.get_cluster_ops_and_init_guess"
        
        theta_current2: List<float>
            fixed values (e.g. 0.0, 0.001, ...) or random values (random.uniform(0,1))
        
        FCI: float
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
        tolerance = 10 ** (-5)
        method = "BFGS"
        print("tolerance= ", tolerance)
        print("method= ", method)
        theta_optimized_result1 = []
        theta_optimized_result2 = []
        opt_result1 = scipy.optimize.minimize(
            lambda theta: self.action_quccsd(
                theta, hamiltonian_sp, cluster_ops, hf_init_sp
            ),
            x0=theta_current1,
            method=method,
            tol=tolerance,
            options={"maxiter": 50000, "disp": True},
        )
        opt_result2 = scipy.optimize.minimize(
            lambda theta: self.action_quccsd(
                theta, hamiltonian_sp, cluster_ops, hf_init_sp
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
            hamiltonian_sp, hf_init_sp, cluster_ops, theta_optimized_result1
        )
        curr_state_result2 = self.prepare_state_ansatz(
            hamiltonian_sp, hf_init_sp, cluster_ops, theta_optimized_result2
        )
        gates1 = curr_state_result1.ops
        gates2 = curr_state_result2.ops
        CNOT1 = count("CNOT", gates1)
        CNOT2 = count("CNOT", gates2)
        iterations["minimum_energy_result1_guess"].append(opt_result1.fun)
        iterations["minimum_energy_result2_guess"].append(opt_result2.fun)
        iterations["theta_optimized_result1"].append(theta_optimized_result1)
        iterations["theta_optimized_result2"].append(theta_optimized_result2)
        result["CNOT1"] = CNOT1
        result["CNOT2"] = CNOT2
        result["len_op1"] = len(theta_optimized_result1)
        result["len_op2"] = len(theta_optimized_result2)
        result["energies1_substracted_from_FCI"] = abs(opt_result1.fun - FCI)
        result["energies2_substracted_from_FCI"] = abs(opt_result2.fun - FCI)
        return iterations, result
