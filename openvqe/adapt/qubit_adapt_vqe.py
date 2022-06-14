import warnings

import scipy.optimize
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter("ignore", SparseEfficiencyWarning)
import math

import numpy as np
import scipy
from numpy import binary_repr
from qat.fermion.chemistry.ucc_deprecated import build_ucc_ansatz
from qat.lang.AQASM import Program, X
from qat.qpus import get_default_qpu
from scipy import sparse

from ..common_files.sorted_gradient import value_without_0, index_without_0, abs_sort_desc, corresponding_index
from ..common_files.circuit import count

def prepare_adapt_state(reference_state, ansatz, coefficients, qubit_number):
    # Initialize the state vector with the reference state.\n",
    state = reference_state
    #     print(\"state\",state)
    # Apply the ansatz operators one by one to obtain the state as optimized\n",
    # by the last iteration\n",
    for (i, operator) in enumerate(ansatz):
        # Multiply the operator by the variational parameter\n",
        #         operator = coefficients[i]*operator
        # Obtain the sparse matrix representing the operator\n",
        sparse_operator = term_to_matrix_sparse(operator, qubit_number)
        sparse_operator = coefficients[i] * sparse_operator
        # Exponentiate the operator\n"
        exp_operator = scipy.sparse.linalg.expm(-1j * sparse_operator)
        #         dim, _ = sparseOperator.shape
        #         expOperator = expm_multiply(dim,coefficients[i],-1j*sparseOperator)
        # Act on the state with the operator\n"
        state = exp_operator * state
    return state


def expm_multiply(dim, a, g):
    x = math.cos(a) * scipy.sparse.identity(dim, float) + math.sin(a) * g
    #     X = cosm(g) + 1j*sinm(-g)
    #     print(X)
    return x


# def exact_adapt_energy(coef_vect,operators,reference_state,hamiltonian,n_el):
#  # Find the number of qubits of the system (2**qubitNumber = dimension)
#     dimension, _ = hamiltonian.shape
# #     print("dimension",dimension)
# #     print(\"dimension\",dimension)
#     qubit_number = int(np.log(dimension)/np.log(2))
# #     print("qubitnumberfromexact", qubitNumber)
#   # Transform reference vector into a Compressed Sparse Column matrix\n",
#     reference_state= jw_hartree_fock_state(n_el,qubit_number)
# #     print("referenceState",referenceState)
#     ket = scipy.sparse.csr_matrix(reference_state,dtype=complex).transpose()
# #     print("ket",ket)
# #     ket = referenceState
#       # Apply e ** (coefficient * operator) to the state (ket) for each operator in\n",
#      #the ansatz, following the order of the list\n",
#     for (coefficient,operator) in zip(coef_vect,operators):
#         operator_sparse = term_to_matrix_sparse(operator,qubit_number)
# #         print("operator_sparse1",operator_sparse)
# #         operator_sparse = coefficient*operator_sparse
# #         expOperator =  expm_multiply(dimension,coefficient,operator_sparse)
# #         sparseOperator = get_sparse_operator(operator,qubitNumber)
#         exp_operator  = scipy.sparse.linalg.expm(-1j*coefficient*operator_sparse)
#         ket = exp_operator * ket
# #     print("ket",ket)
#      # Get the corresponding bra and calculate the energy: |<bra| H |ket>|\n",
#     bra = ket.transpose().conj()
#     energy = (bra * hamiltonian * ket)[0,0].real
# #     print('Energy finished: ', energy)
# #     print('parameterssssss: ', coefVect,operators,referenceState,hamiltonian)
#     return energy


def term_to_matrix_sparse(Poolmix, nbqbits):
    X = sparse.csr_matrix(np.array([[0, 1], [1, 0]]))
    Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]]))
    Z = sparse.csr_matrix(np.diag([1, -1]))
    I = sparse.csr_matrix(np.diag([1, 1]))
    #     print("X",X)
    #     print("Y",Y)
    #     print("Z",Z)
    #     print("I",I)
    #     print(Y)\n",
    dic_Pauli = {"I": I, "X": X, "Y": Y, "Z": Z}
    matrix_final = 0
    # for i in range(len(Poolmix)):
    for term in Poolmix.terms:
        result_one_term = 0
        # print(term.op, term.qbits)
        char_term = [char for char in term.op]
        # print(term.op)
        qb_term = term.qbits
        dic_op = {}
        for n in range(nbqbits):
            dic_op[n] = I
        #         print(dic_op[n])\n",
        for n in range(len(term.qbits)):
            dic_op[qb_term[n]] = dic_Pauli[char_term[n]]
        #     print(dic_op)\n",
        matrix = 0
        for d in dic_op:
            if type(matrix) == int:
                matrix = dic_op[d]
            else:
                matrix = scipy.sparse.kron(matrix, dic_op[d])
        result_one_term = sparse.csr_matrix(matrix * term.coeff)
        # print('matrix-current: ', result_one_term)
        matrix_final += result_one_term
    # print(matrix_final)
    return matrix_final  # matrix*term.coeff


def calculate_gradient(sparse_operator, state, sparse_hamiltonian):
    test_state = sparse_operator * state
    bra = state.transpose().conj()
    gradient = 2 * (np.abs(bra * sparse_hamiltonian * test_state)[0, 0].real)
    return gradient


def prepare_state(cluster_ops_sp, hf_init_sp, parameters, nbqbits):
    reg = 0
    prog = Program()
    reg = prog.qalloc(nbqbits)
    #     qrout = build_ucc_ansatz(cluster_ops_sp,hf_init_sp,n_steps=1)
    #     prog.apply(qrout(parameters), reg)
    for n_term, (term, theta_term) in enumerate(zip(cluster_ops_sp, parameters)):
        init = hf_init_sp if n_term == 0 else 0
        qprog = build_ucc_ansatz([term], init, n_steps=1)
        prog.apply(qprog([theta_term]), reg)
    circ1 = 0
    circ1 = prog.to_circ()
    curr_state = circ1
    return curr_state


def compute_commutator_i(commu, curr_state):
    qpu = 0
    qpu = get_default_qpu()
    # X= -(hamiltonian_sp*oi*(complex(0,1)) - oi*(complex(0,1))*hamiltonian_sp)
    circuit = curr_state
    job = circuit.to_job(job_type="OBS", observable=commu)
    res = qpu.submit(job)
    #     print("gradientcurrent",res)
    return res.value


def prepare_hf_state(hf_init_sp, cluster_ops_sp):
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


def hf_energy(hf_state, hamiltonian_sp):
    qpu = get_default_qpu()
    circ = hf_state
    res = qpu.submit(circ.to_job(job_type="OBS", observable=hamiltonian_sp))
    return res.value


def ucc_action(hamiltonian_sp, cluster_ops_sp, hf_init_sp, theta, iteration):
    #     print("theta",theta)
    qpu = 0
    prog = 0
    reg = 0
    qpu = get_default_qpu()
    prog = Program()
    reg = prog.qalloc(hamiltonian_sp.nbqbits)
    for n_term, (term, theta_term) in enumerate(zip(cluster_ops_sp, theta)):
        init = hf_init_sp if n_term == 0 else 0
        qprog = build_ucc_ansatz([term], init, n_steps=1)
        prog.apply(qprog([theta_term]), reg)
    #     qrout = build_ucc_ansatz(cluster_ops_sp,hf_init_sp,n_steps=1)
    #     prog.apply(qrout(theta), reg)
    circ = prog.to_circ()
    #     if (iteration==10):
    #         for op in circ.iterate_simple():
    #             print("opnew",op)
    #         print(circ)
    job = circ.to_job(job_type="OBS", observable=hamiltonian_sp)
    res = qpu.submit(job)
    return res.value


def qubit_adapt_vqe(
    hamiltonian_sp,
    hamiltonian_sp_sparse,
    reference_ket,
    nqubits,
    pool_mix,
    hf_init_sp,
    fci,
    chosen_grad=2,
    adapt_conver="norm",
    adapt_thresh=1e-08,
    adapt_maxiter=45,
    tolerance_sim=1e-07,
    method_sim="BFGS",
):
    iterations_sim = {
        "energies": [],
        "energies_substracted_from_fci": [],
        "norms": [],
        "Max_gradient": [],
        "CNOTs": [],
        "Hadamard": [],
        "RY": [],
        "RX": [],
    }
    result_sim = {}

    iterations_ana = {
        "energies": [],
        "energies_substracted_from_fci": [],
        "norms": [],
        "Max_gradient": [],
    }
    result_ana = {}

    parameters_sim = []
    parameters_ana = []

    ansatz_ops = []  # SQ operator strings in the ansatz
    #     for i in range(len(cluster_ops_sp)):
    #         parameters.append(0.0)
    # print(len(theta_0))
    #     print("hamiltonian_spfrom ADAPt",hamiltonian_sp)
    curr_state = prepare_hf_state(hf_init_sp, pool_mix)
    ref_energy = hf_energy(curr_state, hamiltonian_sp)
    ref_energy_ana = (
        reference_ket.T.conj().dot(hamiltonian_sp_sparse.dot(reference_ket))[0, 0].real
    )
    print("reference_energy from the simulator:", ref_energy)
    print("reference_energy from the analytical calculations:", ref_energy_ana)
    curr_state_open_f = prepare_adapt_state(
        reference_ket, ansatz_ops, parameters_ana, nqubits
    )
    #     print(" Reference Energy: %12.8f" %ref_energy)
    #     curr_state,_ = simple_circuit_parameters(hamiltonian_sp)
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Start Qubit ADAPT-VQE algorithm:")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    #     chosegrad = 2
    Y = int(chosen_grad)
    print(" ------------------------------------------------------")
    print("        The number of maximum gradients inserted in each iteration:", Y)
    print(" ------------------------------------------------------")
    op_indices = []
    #     print("Compute the commutator for the first time")
    #     listcommutators1= []
    #     listcommutators2= []
    #     # check the points of imaginary part i
    #     for oi in pool_mix:
    #         X= 2.0j*(hamiltonian_sp*oi)
    #         listcommutators2.append(X)
    # #     listcommutators2= []
    # #     for oi in pool_mix:
    # #         Y= 1j*(hamiltonian_sp*oi-oi*hamiltonian_sp)
    # #         listcommutators2.append(Y)
    # method = 'BFGS'
    prev_norm = 0.0
    for n_iter in range(adapt_maxiter):
        print("\n")
        print(
            " --------------------------------------------------------------------------"
        )
        print("                         Qubit ADAPT-VQE iteration: ", n_iter)
        print(
            " --------------------------------------------------------------------------"
        )
        next_deriv = 0
        curr_norm = 0
        #         print(" Check gradient list chronological order")
        list_grad = []
        #         print(" Check each new operator for coupling")
        print("\n")
        print(" ------------------------------------------------------")
        print("        Start the analytical gradient calculation:")
        print(" ------------------------------------------------------")
        for i in range(len(pool_mix)):
            #             print("i",i)
            # gi = #compute_commutator_i(listcommutators2[i],curr_state)
            operator_sparse = term_to_matrix_sparse(pool_mix[i], nqubits)
            gi = calculate_gradient(
                operator_sparse, curr_state_open_f, hamiltonian_sp_sparse
            )
            curr_norm += gi * gi
            list_grad.append(gi)
            if abs(gi) > abs(next_deriv):
                next_deriv = gi
        #             print("list_gradorginal",list_grad)
        mylist_value_without_0 = value_without_0(list_grad)
        #             print('mylist_value_without_0:',mylist_value_without_0)

        mylist_index_without_0 = index_without_0(list_grad)
        #         print('mylist_index_without_0:',mylist_index_without_0)
        sorted_mylist_value_without_0 = abs_sort_desc(
            value_without_0(list_grad)
        )
        print(
            "sorted_mylist_value of gradient_without_0", sorted_mylist_value_without_0
        )
        sorted_index = corresponding_index(
            mylist_value_without_0,
            mylist_index_without_0,
            sorted_mylist_value_without_0,
        )
        #         print('sorted Index: ',sorted_index)
        curr_norm = np.sqrt(curr_norm)
        max_of_gi = next_deriv

        # print("list_grad",X)
        print(" Norm of <[H,A]> = %12.8f" % curr_norm)
        print(" Max  of <[H,A]> = %12.8f" % max_of_gi)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
        # elif adapt_conver == "var":
        #     if abs(var) < adapt_thresh:
        #         #variance
        #         converged = True
        else:
            print(" FAIL: Convergence criterion not defined")
            exit()

        if converged or (abs(curr_norm - prev_norm) < 10 ** (-7)):
            print(" Ansatz Growth Converged!")
            result_sim["optimizer"] = method_sim
            result_sim["final_norm"] = curr_norm
            result_sim["indices"] = op_indices
            result_sim["len_operators"] = len(op_indices)
            result_sim["parameters"] = parameters_sim
            result_sim["final_energy"] = opt_result_sim.fun

            #             result_ana["optimizer"] = method_ana
            #             result_ana["final_norm"] = curr_norm
            #             result_ana["indices"] = op_indices
            #             result_ana["len_operators"] = len(op_indices)
            #             result_ana["parameters"] = parameters_ana
            #             result_ana["final_energy"] = opt_result_ana.fun

            gates = curr_state.ops
            m = count("CNOT", gates)
            H = count("H", gates)
            RX = count("_2", gates)
            print(" -----------Final ansatz----------- ")
            print(" %4s %12s %18s" % ("#", "Coeff", "Term"))
            for si in range(len(ansatz_ops)):
                print(" %4i %12.8f" % (si, parameters_sim[si]))
            break

        #         print("\n")
        chosen_batch = sorted_mylist_value_without_0

        gamma1 = []
        sorted_index1 = []
        curr_norm1 = 0
        for z in chosen_batch:
            curr_norm1 += z * z
            curr_norm1 = np.sqrt(curr_norm1)
        for i in range(Y):
            gamma1.append(chosen_batch[i] / curr_norm1)
            sorted_index1.append(sorted_index[i])
        #             parameters = []
        for m in range(len(gamma1)):
            parameters_sim.append(gamma1[m])
            parameters_ana.append(gamma1[m])
            #             parameters.append(0.0)
            ansatz_ops.append(pool_mix[sorted_index1[m]])
            op_indices.append(sorted_index1[m])
        print("initial parameters", parameters_sim)
        print("op_indices of iteration_%d" % n_iter, op_indices)
        #         opt_result_ana = scipy.optimize.minimize(exact_adapt_energy,
        #                              parameters_ana,
        #                          (ansatz_ops,reference_ket,hamiltonian_sp_sparse,n_el),
        #                          method = method_ana,
        #                          tol =tolerance_ana,
        #                          options = {'gtol': 10**(-5),
        #                         'maxiter': 50000,
        #                         'disp': False})
        #         xlist_ana = opt_result_ana.x
        opt_result_sim = scipy.optimize.minimize(
            lambda theta: ucc_action(
                hamiltonian_sp, ansatz_ops, hf_init_sp, theta, n_iter
            ),
            x0=parameters_sim,
            method=method_sim,
            tol=tolerance_sim,
            options={"maxiter": 100000, "disp": False},
        )
        xlist_sim = opt_result_sim.x
        #         print(" ----------- ansatz from analytical calculations----------- ")
        #         print(" %s\t %s\t\t %s" %("#","Coeff","Term"))
        #         parameters_ana = []
        #         for si in range(len(ansatz_ops)):
        #             print(" %i\t %f\t %s" %(si, xlist_ana[si], op_indices[si]) )
        #             parameters_ana.append(xlist_ana[si])
        #         print(" Energy reached from the analytical calculations: %20.20f" %opt_result_ana.fun)

        #         curr_state_open_f = prepare_adapt_state(reference_ket,ansatz_ops,parameters_ana,nqubits)

        print(" ----------- ansatz from the simulator----------- ")
        print(" %s\t %s\t\t %s" % ("#", "Coeff", "Term"))
        parameters_sim = []
        for si in range(len(ansatz_ops)):
            print(" %i\t %f\t %s" % (si, xlist_sim[si], op_indices[si]))
            parameters_sim.append(xlist_sim[si])
        print(" Energy reached from the simulator: %20.20f" % opt_result_sim.fun)
        curr_state = prepare_state(ansatz_ops, hf_init_sp, parameters_sim, nqubits)
        curr_state_open_f = prepare_adapt_state(
            reference_ket, ansatz_ops, parameters_sim, nqubits
        )

        prev_norm = curr_norm
        gates = curr_state.ops
        cnot = count("CNOT", gates)
        hadamard = count("H", gates)
        ry = count("_4", gates)
        rx = count("_2", gates)
        iterations_sim["energies"].append(opt_result_sim.fun)
        iterations_sim["energies_substracted_from_fci"].append(
            abs(opt_result_sim.fun - fci)
        )
        iterations_sim["norms"].append(curr_norm)
        iterations_sim["Max_gradient"].append(sorted_mylist_value_without_0[0])
        #         iterations_ana["energies"].append(opt_result_ana.fun)
        #         iterations_ana["norms"].append(curr_norm)
        #         iterations_ana["Max_gradient"].append(sorted_mylist_value_without_0[0])
        iterations_sim["CNOTs"].append(cnot)
        iterations_sim["Hadamard"].append(hadamard)
        iterations_sim["RY"].append(ry)
        iterations_sim["RX"].append(rx)
    return iterations_sim, iterations_ana, result_sim, result_ana
