import numpy as np
import scipy
import scipy.optimize
from numpy import binary_repr
from qat.fermion.chemistry.ucc_deprecated import build_ucc_ansatz
from qat.lang.AQASM import Program, X
from qat.qpus import get_default_qpu

from ..common_files.sorted_gradient import value_without_0, index_without_0, abs_sort_desc, corresponding_index
from ..common_files.circuit import count

def prepare_state(reference_ket, spmat_ops, parameters):
    new_state = reference_ket * 1.0
    g = spmat_ops
    for k in range(len(parameters)):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k] * g[k]), new_state)
    return new_state


def compute_gradient_i(i, cluster_ops_sparse, v, sig):
    # """
    # For a previously optimized state |n>, compute the gradient g(k) of exp(c(k) A(k))|n>
    # g(k) = 2Real<HA(k)>
    # Note - this assumes A(k) is an antihermitian operator. If this is not the case,
    # the derived class should
    # reimplement this function. Of course, also assumes H is hermitian
    # v = current_state
    # sig = H*v
    # """
    op_a = cluster_ops_sparse[i]
    gi = 2 * (sig.transpose().conj().dot(op_a.dot(v)))
    assert gi.shape == (1, 1)
    gi = gi[0, 0]
    # print('gi values are: ', gi)
    assert np.isclose(gi.imag, 0)
    gi = gi.real
    return gi


def return_gradient_list(
    cluster_ops, cluster_ops_sparse, hamiltonian_sparse, curr_state
):
    list_grad = []
    curr_norm = 0
    next_deriv = 0
    next_index = 0
    sig = hamiltonian_sparse.dot(curr_state)
    for oi in range(len(cluster_ops)):
        gi = compute_gradient_i(oi, cluster_ops_sparse, curr_state, sig)
        list_grad.append(abs(gi))
        curr_norm += gi * gi
        if abs(gi) > abs(next_deriv):
            next_deriv = gi
            next_index = oi
    return list_grad, curr_norm, next_deriv, next_index


# create the function related to myADAPT-VQE:
def ucc_action(hamiltonian_sp, cluster_ops_sp, hf_init_sp, theta):
    qpu = get_default_qpu()
    prog = Program()
    reg = prog.qalloc(hamiltonian_sp.nbqbits)
    for n_term, (term, theta_term) in enumerate(zip(cluster_ops_sp, theta)):
        init = hf_init_sp if n_term == 0 else 0
        qprog = build_ucc_ansatz([term], init, n_steps=1)
        prog.apply(qprog([theta_term]), reg)
    circ = prog.to_circ()
    res = qpu.submit(circ.to_job(job_type="OBS", observable=hamiltonian_sp))
    circ.empty(hamiltonian_sp.nbqbits)
    return res.value


def print_gradient_lists_and_indices(list_grad):
    #         print("list_gradorginal",list_grad)
    mylist_value_without_0 = value_without_0(list_grad)
    #         print('mylist_value_without_0:',mylist_value_without_0)

    mylist_index_without_0 = index_without_0(list_grad)
    #         print('mylist_index_without_0:',mylist_index_without_0)

    sorted_mylist_value_without_0 = abs_sort_desc(value_without_0(list_grad))
    #         print('sorted_mylist_value_without_0', sorted_mylist_value_without_0)

    sorted_index = corresponding_index(
        mylist_value_without_0, mylist_index_without_0, sorted_mylist_value_without_0
    )
    #         print('sorted Index: ',sorted_index)
    return sorted_mylist_value_without_0, sorted_index


def prepare_hf_state(hf_init_sp, cluster_ops_sp):
    prog = Program()
    nbqbits = cluster_ops_sp[0].nbqbits
    print(nbqbits)
    ket_hf = binary_repr(hf_init_sp)
    list_ket_hf = [int(c) for c in ket_hf]
    print(ket_hf)
    print(list_ket_hf)
    qb = prog.qalloc(nbqbits)
    # print(list_ket_hf)
    for j in range(nbqbits):
        if int(list_ket_hf[j] == 1):
            prog.apply(X, qb[j])
    circuit = prog.to_circ()
    qpu = get_default_qpu()
    res = qpu.submit(circuit.to_job())
    statevector = get_statevector(res, nbqbits)
    vec = statevector
    print("vec", vec)
    return circuit


def hf_energy(hf_state, hamiltonian_sp):
    qpu = get_default_qpu()
    circ = hf_state
    res = qpu.submit(circ.to_job(job_type="OBS", observable=hamiltonian_sp))
    return res.value


def commutators_calculations(cluster_ops_sp, hamiltonian_sp):
    print("Compute the commutator for the first time from ")
    list_commutators = []
    for oi in cluster_ops_sp:
        X = -(
            hamiltonian_sp * oi * (complex(0, 1))
            - oi * (complex(0, 1)) * hamiltonian_sp
        )
        list_commutators.append(X)
    return list_commutators


def prepare_state_ansatz(hamiltonian_sp, cluster_ops_sp, hf_init_sp, parameters):
    prog = Program()
    reg = prog.qalloc(hamiltonian_sp.nbqbits)
    for n_term, (term, theta_term) in enumerate(zip(cluster_ops_sp, parameters)):
        init = hf_init_sp if n_term == 0 else 0
        qprog = build_ucc_ansatz([term], init, n_steps=1)
        prog.apply(qprog([theta_term]), reg)
    circ = prog.to_circ()
    curr_state = circ
    return curr_state


def get_statevector(result, nbqbits):
    # "Get the statevector from the Result class:"
    statevector = np.zeros((2**nbqbits), np.complex128)
    for sample in result:
        statevector[sample.state.int] = sample.amplitude
    return statevector


## check fidelity of the circuit:
def fun_fidelity(circ, eigenvalues, eigenvectors, nbqbits):
    ee = eigenvectors[:, np.argmin(eigenvalues)]
    qpu = get_default_qpu()
    res = qpu.submit(circ.to_job())
    statevector = get_statevector(res, nbqbits)
    fid = abs(np.vdot(ee, statevector)) ** 2
    return fid


# cluster_ops, h_active_sparse, cluster_ops_sparse, reference_ket, hamiltonian_sp,
#         cluster_ops_sp, hf_init_sp, n_max_grads, FCI,
#         optimizer,
#         tolerance,
#         type_conver = type_conver,
#         threshold_needed = threshold_needed,
#         max_external_iterations = max_external_iterations
def fermionic_adapt_vqe(
    cluster_ops,
    hamiltonian_sparse,
    cluster_ops_sparse,
    reference_ket,
    hamiltonian_sp,
    cluster_ops_sp,
    hf_init_sp,
    n_max_grads,
    fci,
    optimizer,
    tolerance,
    type_conver,
    threshold_needed,
    max_external_iterations=30,
):
    iterations = {
        "energies": [],
        "energies_substracted_from_FCI": [],
        "norms": [],
        "Max_gradients": [],
        "fidelity": [],
        "CNOTs": [],
        "Hadamard": [],
        "RY": [],
        "RX": [],
    }
    result = {}
    print("threshold needed for convergence", threshold_needed)
    print("Max_external_iterations:", max_external_iterations)
    print("how many maximum gradient are selected", n_max_grads)
    print("The optimizer method used:", optimizer)
    print("Tolerance for reaching convergence", tolerance)
    # define list to store the operators needed to evolve the ansatz
    ansatz_ops = []
    # define a list to store the sparse matrices from open_fermion
    ansatz_mat = []
    # define a list which store the indices of the operators
    op_indices = []
    # use this list in case we need reversing the indices
    listreverse = []
    # define a list parameters use open_fermion
    parameters_open_f = []
    # define a list to store parameters for the ansatz
    parameters_ansatz = []
    # to see fidelity
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_sp.get_matrix())
    # use hf_state to compute the HF energy from simulator
    hf_state = prepare_hf_state(hf_init_sp, cluster_ops_sp)

    ref_energy = hf_energy(hf_state, hamiltonian_sp)
    print(ref_energy)
    print(" The reference energy of the molecular system is: %12.8f" % ref_energy)
    curr_state = hf_state
    # curr_state = prepare_state_ansatz(hamiltonian_sp,cluster_ops_sp, hf_init_sp, parameters_ansatz)
    curr_state_open_f = 1.0 * reference_ket
    prev_norm = 0.0
    for n_iter in range(0, max_external_iterations):
        print("\n\n\n")
        print(
            " --------------------------------------------------------------------------"
        )
        print("                     Fermionic_ADAPT-VQE iteration: ", n_iter)
        print(
            " --------------------------------------------------------------------------"
        )
        # few commands to initialize at each new iteration
        print(" Check gradient list chronological order")
        list_grad, curr_norm, next_deriv, next_index = return_gradient_list(
            cluster_ops, cluster_ops_sparse, hamiltonian_sparse, curr_state_open_f
        )
        sorted_mylist_value_without_0, sorted_index = print_gradient_lists_and_indices(
            list_grad
        )
        # the norm from the current iteration:
        curr_norm = np.sqrt(curr_norm)
        print(" Norm of the gradients in current iteration = %12.8f" % curr_norm)
        print(" Max gradient in current iteration= %12.8f" % next_deriv)
        print(" Index of the Max gradient in current iteration= ", next_index)
        # fidelity fucntion
        nbqbits = hamiltonian_sp.nbqbits
        fid = fun_fidelity(curr_state, eigenvalues, eigenvectors, nbqbits)
        # check now convergence:
        converged = False
        if type_conver == "norm":
            if curr_norm < threshold_needed:
                converged = True
        else:
            print(" type convergence is not defined")
            exit()
        if converged or (abs(curr_norm - prev_norm) < 10 ** (-8)):
            print("Convergence is done")
            result["indices"] = op_indices
            result["Number_operators"] = len(ansatz_ops)
            result["final_norm"] = curr_norm
            result["parameters"] = parameters_ansatz
            gates = curr_state.ops
            n_cnot = count("CNOT", gates)
            n_had = count("H", gates)
            n_rx = count("_2", gates)
            result["Number_CNOT_gates"] = n_cnot
            result["Number_Hadamard_gates"] = n_had
            result["Number_RX_gates"] = n_rx
            print(" -----------Final ansatz----------- ")
            print(" *final converged energy iteration is %20.12f" % opt_result.fun)
            result["final_energy_last_iteration"] = opt_result.fun
            break
        # the chosen batch of maximum gradients from the list sorted_mylist_value_without_0
        chosen_batch = sorted_mylist_value_without_0
        # list to store the gradients: gamma_i = grad_i/Norm_1
        gamma1 = []
        # list to store the indices of the operators associated from sorted_index
        sorted_index1 = []
        # initiate current norm to zero
        curr_norm1 = 0
        for z in chosen_batch:
            curr_norm1 += z * z
        curr_norm1 = np.sqrt(curr_norm1)
        for i in range(n_max_grads):
            gamma1.append(chosen_batch[i] / curr_norm1)
            sorted_index1.append(sorted_index[i])
        #         print('N_max_grads: ', N_max_grads)
        #         print('gamma1: ', gamma1)
        print("sorted_index1: ", sorted_index1)
        # there are two types of initial thetas: either zero, of from gamma_1 list
        for j in range(len(sorted_index1)):
            #             parameters_ansatz.append(gamma1[j])
            #             parameters_ansatz.append(0.0)
            parameters_ansatz.append(0.01)
            # for UCCSD: we must multiply by i
            #             ansatz_ops.append(cluster_ops_sp[sorted_index1[j]])
            ansatz_ops.append(complex(0.0, 1.0) * cluster_ops_sp[sorted_index1[j]])
            op_indices.append(sorted_index1[j])
            ansatz_mat.append(cluster_ops_sparse[sorted_index1[j]])
        #         print('parameters_ansatz: ', parameters_ansatz)
        #         print('op_indices: ', op_indices)
        #         print('ansatz_mat: ', ansatz_mat)
        #         print('ansatz_ops: ', ansatz_ops)
        # print('cluster_ops_sparse: ', cluster_ops_sparse)
        # minimize and optimize from simulator
        opt_result = scipy.optimize.minimize(
            lambda parameters: ucc_action(
                hamiltonian_sp, ansatz_ops, hf_init_sp, parameters
            ),
            x0=parameters_ansatz,
            method=optimizer,
            tol=tolerance,
            options={"maxiter": 100000, "disp": True},
        )
        xlist = opt_result.x
        print(" Finished energy iteration_i: %20.12f" % opt_result.fun)
        #         print('length of ansatz_ops', len(ansatz_ops))
        #         print('length of xlist', len(xlist))
        print(" -----------New ansatz created----------- ")
        print(" %4s \t%s \t%s" % ("#", "Coefficients", "Term"))
        parameters_ansatz = []

        for si in range(len(ansatz_ops)):
            print(" %4i \t%f \t%s" % (si, xlist[si], op_indices[si]))
            parameters_ansatz.append(xlist[si])
        curr_state = prepare_state_ansatz(
            hamiltonian_sp, ansatz_ops, hf_init_sp, parameters_ansatz
        )
        curr_state_open_f = prepare_state(reference_ket, ansatz_mat, parameters_ansatz)
        prev_norm = curr_norm
        gates = curr_state.ops
        cnot = count("CNOT", gates)
        hadamard = count("H", gates)
        ry = count("_4", gates)
        rx = count("_2", gates)
        iterations["energies"].append(opt_result.fun)
        iterations["energies_substracted_from_FCI"].append(abs(opt_result.fun - fci))
        iterations["norms"].append(curr_norm1)
        iterations["Max_gradients"].append(sorted_mylist_value_without_0[0])
        iterations["fidelity"].append(fid)
        iterations["CNOTs"].append(cnot)
        iterations["Hadamard"].append(hadamard)
        iterations["RY"].append(ry)
        iterations["RX"].append(rx)
    return iterations, result


# In[ ]:
