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

def prepare_adapt_state(reference_state, ansatz, coefficients):
    """
    Computes the action of the matrix exponential of qubit sparse operators ("ansatz") on
    the state initiated by "reference_state".

    Parameters
    -----------
    reference_state: ndarray
        the initial state
    
    ansatz: List<transposable linear operator>
        the sparse qubit operators

    coefficients: List<float>
        list of parameters for the "ansatz"

    Returns
    --------
    state: ndarray
        the new state

    """

    # Initialize the state vector with the reference state.\n",
    state = reference_state
    # Apply the ansatz operators one by one to obtain the state as optimized,
    # by the last iteration
    for (i, operator) in enumerate(ansatz):
        # Obtain the sparse matrix representing the operator\n",
        sparse_operator = term_to_matrix_sparse(operator)
        sparse_operator = coefficients[i] * sparse_operator
        # Exponentiate the operator\n"
        exp_operator = scipy.sparse.linalg.expm(-1j * sparse_operator)
        # Act on the state with the operator\n"
        state = exp_operator * state
    return state


# def expm_multiply(dim, a, g):
#     x = math.cos(a) * scipy.sparse.identity(dim, float) + math.sin(a) * g
#     return x


# def exact_adapt_energy(coef_vect,operators,reference_state,hamiltonian,n_el):
#     dimension, _ = hamiltonian.shape
#     qubit_number = int(np.log(dimension)/np.log(2))
#     # Transform reference vector into a Compressed Sparse Column matrix\n",
#     reference_state= jw_hartree_fock_state(n_el,qubit_number)
#     ket = scipy.sparse.csr_matrix(reference_state,dtype=complex).transpose()
#     # Apply e ** (coefficient * operator) to the state (ket) for each operator in\n",
#     #the ansatz, following the order of the list\n",
#     for (coefficient,operator) in zip(coef_vect,operators):
#         operator_sparse = term_to_matrix_sparse(operator)
#         exp_operator  = scipy.sparse.linalg.expm(-1j*coefficient*operator_sparse)
#         ket = exp_operator * ket
#      # Get the corresponding bra and calculate the energy: |<bra| H |ket>|\n",
#     bra = ket.transpose().conj()
#     energy = (bra * hamiltonian * ket)[0,0].real
#     return energy


def term_to_matrix_sparse(spin_operator):
    """
    converts the terms in the spin hamiltonian into a sparse.csr_matrix.

    Parameters
    ----------

    spin_operator: Hamiltonian
        cluster operator in the spin representation
    
    Returns
    --------

    matrix_final: ndarray
        the final matrix of the spin operator


    """
    X = sparse.csr_matrix(np.array([[0, 1], [1, 0]]))
    Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]]))
    Z = sparse.csr_matrix(np.diag([1, -1]))
    I = sparse.csr_matrix(np.diag([1, 1]))
    dic_Pauli = {"I": I, "X": X, "Y": Y, "Z": Z}
    matrix_final = 0
    nbqbits = spin_operator.nbqbits
    for term in spin_operator.terms:
        result_one_term = 0
        char_term = [char for char in term.op]
        qb_term = term.qbits
        dic_op = {}
        for n in range(nbqbits):
            dic_op[n] = I
        for n in range(len(term.qbits)):
            dic_op[qb_term[n]] = dic_Pauli[char_term[n]]
        matrix = 0
        for d in dic_op:
            if type(matrix) == int:
                matrix = dic_op[d]
            else:
                matrix = scipy.sparse.kron(matrix, dic_op[d])
        result_one_term = sparse.csr_matrix(matrix * term.coeff)
        matrix_final += result_one_term
    return matrix_final


def calculate_gradient(sparse_operator, state, sparse_hamiltonian):
    """
    Computation of the gradient 2*<bra|sparse_hamiltonian*sparse_operator|state>

    Parameters
    ----------
    sparse_operator: transposable linear operator
        the sparse qubit operator
    
    state: ndarray
        the current state

    sparse_hamiltonian: ndarray
        the sparsed hamiltonian operator

    Returns
    -------
    gradient: float
        the computed gradient for the qubit operator

    """
    test_state = sparse_operator * state
    bra = state.transpose().conj()
    gradient = 2 * (np.abs(bra * sparse_hamiltonian * test_state)[0, 0].real)
    return gradient


def prepare_state_ansatz(cluster_ops_sp, hf_init_sp, parameters):
    """
    It constructs the trial wave function (ansatz) 

    Parameters
    ----------
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

    prog = Program()
    reg = prog.qalloc(cluster_ops_sp[0].nbqbits)
    for n_term, (term, theta_term) in enumerate(zip(cluster_ops_sp, parameters)):
        init = hf_init_sp if n_term == 0 else 0
        qprog = build_ucc_ansatz([term], init, n_steps=1)
        prog.apply(qprog([theta_term]), reg)
    circ = prog.to_circ()
    curr_state = circ
    return curr_state


def compute_commutator_i(commutator, curr_state):
    """
    computes the expectation value of the commutator of a given qubit operator

    Parameters
    -----------

    commutator: Hamiltonian
        The product of the qubit operator and the spin hamiltonian

    curr_state: qat.core.Circuit
        the ansatz of the current state
    
    Parameters
    -----------
    res.value: float
        the obtained expectation value (gradient)

    """
    qpu = get_default_qpu()
    job = curr_state.to_job(job_type="OBS", observable=commutator)
    res = qpu.submit(job)
    return res.value


def prepare_hf_state(hf_init_sp, cluster_ops_sp):
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
    for j in range(nbqbits):
        if int(list_ket_hf[j] == 1):
            prog.apply(X, qb[j])
    circuit = prog.to_circ()
    return circuit


def hf_energy(hf_state, hamiltonian_sp):
    """
    Returns the Hartee Fock energy

    Parameters
    ----------

    hf_state: qat.core.Circuit
        the circuit representing the HF state

    hamiltonian_sp: Hamiltonian
        Hamiltonian in the spin representation

    
    Returns
    --------
        res.value: float
            the resulted energy

    """
    qpu = get_default_qpu()
    res = qpu.submit(hf_state.to_job(job_type="OBS", observable=hamiltonian_sp))
    return res.value


def ucc_action(hamiltonian_sp, cluster_ops_sp, hf_init_sp, theta_current):
    """
    It maps the exponential of cluster operators ("cluster_ops_sp") associated by their parameters ("theta_current")
    using the CNOTS-staircase method, which is done by "build_ucc_ansatz" which creates the circuit on the top of
    the HF-state ("hf_init_sp"). Then, this function also calculates the expected value of the hamiltonian ("hamiltonian_sp").

    Parameters
    ----------
    hamiltonian_sp: Hamiltonian
        Hamiltonian in the spin representation

    cluster_ops_sp: list[Hamiltonian]
        list of spin cluster operators
    
    hf_init_sp: int
        the integer corresponds to the hf_init (The Hartree-Fock state in integer representation) obtained by using
        "qat.fermion.transforms.record_integer".
    
    theta_current: List<float>
        the Parameters of the cluster operators
    
    Returns
    --------
        res.value: float
            the resulted energy

    """
    qpu = get_default_qpu()
    prog = Program()
    reg = prog.qalloc(hamiltonian_sp.nbqbits)
    for n_term, (term, theta_term) in enumerate(zip(cluster_ops_sp, theta_current)):
        init = hf_init_sp if n_term == 0 else 0
        qprog = build_ucc_ansatz([term], init, n_steps=1)
        prog.apply(qprog([theta_term]), reg)
    circ = prog.to_circ()
    res = qpu.submit(circ.to_job(job_type="OBS", observable=hamiltonian_sp))
    return res.value


def qubit_adapt_vqe(
    hamiltonian_sp,
    hamiltonian_sp_sparse,
    reference_ket,
    nqubits,
    pool_mix,
    hf_init_sp,
    fci,
    n_max_grads=2,
    adapt_conver="norm",
    adapt_thresh=1e-08,
    adapt_maxiter=45,
    tolerance_sim=1e-07,
    method_sim="BFGS",
):
    """
    adapt_conver
    adapt_thresh
    adapt_maxiter
    tolerance_sim
    method_sim

    Runs the loop of making qubit adapt vqe found in this reference in section "Results"
    Grimsley HR, Economou SE, Barnes E, Mayhall NJ. An adaptive variational algorithm for exact molecular simulations
    on a quantum computer. Nature communications 2019; 10(1): 1-9.

    Note: the analytical calculation for the "exact_adapt_energy" are still under developement so that we later can compare
    the results obtained by the qlm simulator and the "exact_adapt_energy" 

    Parameters
    ----------
    
    hamiltonian_sp: Hamiltonian
        Hamiltonian in the spin representation

    hamiltonian_sp_sparse: ndarray
        The sparsed spin hamiltonian
    
    reference_ket: ndarray
        the initial state
    
    nqubits: int
        the number of qubits
    
    pool_mix: List<Hamiltonian>
        list of qubit cluster operators
    
    hf_init_sp: int
        the integer corresponds to the hf_init (The Hartree-Fock state in integer representation) obtained by using
        "qat.fermion.transforms.record_integer".
    
    fci: float
        the full configuration interaction energy
    
    n_max_grads: int
        the number of maximum gradients chosen per internal iteration

    adapt_conver: string
        in our case, "norm" is chosen
    
    adapt_thresh: float
        the norm threshold 

    adapt_maxiter: int
        the number of maximum iteration to perform the whole adaptive loop
    
    tolerance_sim: float
        the tolerance for reaching convergence
    
    method_sim: string
        the type of the optimizer for the qlm simulator
    
    Returns:
    --------

    iterations_sim: Dict
        the following properties:
        energies, energies_substracted_from_fci, norms, Max_gradient, CNOTs, Hadamard, RY, RX

    result_sim: Dict
        the following properties:
        optimizer, final_norm, indices, len_operators, parameters, final_energy,

    iterations_ana: Empty dict (see the note above)
    
    result_ana: Empty dict (see the note above)

    """
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
    curr_state = prepare_hf_state(hf_init_sp, pool_mix)
    ref_energy = hf_energy(curr_state, hamiltonian_sp)
    ref_energy_ana = (
        reference_ket.T.conj().dot(hamiltonian_sp_sparse.dot(reference_ket))[0, 0].real
    )
    print("reference_energy from the simulator:", ref_energy)
    print("reference_energy from the analytical calculations:", ref_energy_ana)
    curr_state_open_f = prepare_adapt_state(
        reference_ket, ansatz_ops, parameters_ana
    )
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    print("                      Start Qubit ADAPT-VQE algorithm:")
    print("                                                          ")
    print(" --------------------------------------------------------------------------")
    print("                                                          ")
    #     chosegrad = 2
    Y = int(n_max_grads)
    print(" ------------------------------------------------------")
    print("        The number of maximum gradients inserted in each iteration:", Y)
    print(" ------------------------------------------------------")
    op_indices = []

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
        list_grad = []
        print("\n")
        print(" ------------------------------------------------------")
        print("        Start the analytical gradient calculation:")
        print(" ------------------------------------------------------")
        for i in range(len(pool_mix)):
            #             print("i",i)
            # gi = #compute_commutator_i(listcommutators2[i],curr_state)
            operator_sparse = term_to_matrix_sparse(pool_mix[i])
            gi = calculate_gradient(
                operator_sparse, curr_state_open_f, hamiltonian_sp_sparse
            )
            curr_norm += gi * gi
            list_grad.append(gi)
            if abs(gi) > abs(next_deriv):
                next_deriv = gi
        mylist_value_without_0 = value_without_0(list_grad)

        mylist_index_without_0 = index_without_0(list_grad)
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
        curr_norm = np.sqrt(curr_norm)
        max_of_gi = next_deriv

        print(" Norm of <[H,A]> = %12.8f" % curr_norm)
        print(" Max  of <[H,A]> = %12.8f" % max_of_gi)

        converged = False
        if adapt_conver == "norm":
            if curr_norm < adapt_thresh:
                converged = True
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
            print(" -----------Final ansatz----------- ")
            print(" %4s %12s %18s" % ("#", "Coeff", "Term"))
            for si in range(len(ansatz_ops)):
                print(" %4i %12.8f" % (si, parameters_sim[si]))
            break

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
                hamiltonian_sp, ansatz_ops, hf_init_sp, theta
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

        #         curr_state_open_f = prepare_adapt_state(reference_ket,ansatz_ops,parameters_ana)

        print(" ----------- ansatz from the simulator----------- ")
        print(" %s\t %s\t\t %s" % ("#", "Coeff", "Term"))
        parameters_sim = []
        for si in range(len(ansatz_ops)):
            print(" %i\t %f\t %s" % (si, xlist_sim[si], op_indices[si]))
            parameters_sim.append(xlist_sim[si])
        print(" Energy reached from the simulator: %20.20f" % opt_result_sim.fun)
        curr_state = prepare_state_ansatz(ansatz_ops, hf_init_sp, parameters_sim)
        curr_state_open_f = prepare_adapt_state(
            reference_ket, ansatz_ops, parameters_sim
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
