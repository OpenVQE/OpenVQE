import scipy.optimize

from qat.fermion.chemistry.ucc_deprecated import build_ucc_ansatz

from qat.lang.AQASM import Program
from qat.qpus import get_default_qpu


class EnergyUCC:
    def ucc_action(self, theta_current, hamiltonian_sp, cluster_ops_sp, hf_init_sp):
        qpu = 0
        prog = 0
        reg = 0
        # qpu = LinAlg()
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
        return res.value

    def count(self, gate, mylist):
        if type(gate) == type(str):
            gate = str(gate)
        if gate == gate.lower():
            gate = gate.upper()
        mylist = [str(i) for i in mylist]
        count = 0
        for i in mylist:
            if i.find("gate='{}'".format(gate)) == -1:
                pass
            else:
                count += 1
        return count

    def prepare_state_ansatz(
        self, hamiltonian_sp, cluster_ops_sp, hf_init_sp, parameters
    ):
        # qpu = LinAlg()
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

        opt_result1 = scipy.optimize.minimize(
            lambda theta: self.ucc_action(
                theta, hamiltonian_sp, cluster_ops_sp, hf_init_sp
            ),
            x0=theta_current1,
            method=method,
            tol=tolerance,
            options={"maxiter": 50000, "disp": True},
        )
        opt_result2 = scipy.optimize.minimize(
            lambda theta: self.ucc_action(
                theta, hamiltonian_sp, pool_generator, hf_init_sp
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
        cnot1 = self.count("CNOT", gates1)
        cnot2 = self.count("CNOT", gates2)
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
        return iterations, result
