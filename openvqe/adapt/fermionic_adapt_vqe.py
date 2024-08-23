import numpy as np
import scipy
import scipy.optimize
from numpy import binary_repr
from qat.fermion.chemistry.ucc_deprecated import build_ucc_ansatz
from qat.lang.AQASM import Program, X
from qat.qpus import get_default_qpu

from ..common_files.sorted_gradient import value_without_0, index_without_0, abs_sort_desc, corresponding_index
from ..common_files.circuit import count

def prepare_adapt_state(reference_ket, spmat_ops, parameters):
    """
    Computes the action of the matrix exponential of fermionic sparse operators ("spmat_ops") on
    the state initiated by "reference_ket".

    Parameters
    -----------
    reference_ket: ndarray
        the initial state
    
    spmat_ops: List<transposable linear operator>
        the sparse fermionic operators

    parameters: List<float>
        list of parameters for the "spmat_ops"
    

    Returns
    --------
    new_state: ndarray
        the new state

    """
    new_state = reference_ket * 1.0
    for k in range(len(parameters)):
        new_state = scipy.sparse.linalg.expm_multiply((parameters[k] * spmat_ops[k]), new_state)
    return new_state


def compute_gradient_i(i, cluster_ops_sparse, v, sig):
    """
    Compute analytically the gradient of the ith operator of "cluster_ops_sparse".

    Parameters
    -----------
    i: int
        the index of the fermionic operator for which the gradient is computed
        
    cluster_ops_sparse: List<transposable linear operator>
        the sparse fermionic operators

    v: ndarray
        the current state
    
    sig: ndarray
        the resultant of the dot product of a hamiltonian operator with the current state


    Returns
    --------
    gi: float
        the gradient of the correponding fermionic operator

    """

    op_a = cluster_ops_sparse[i]
    gi = 2 * (sig.transpose().conj().dot(op_a.dot(v)))
    assert gi.shape == (1, 1)
    gi = gi[0, 0]
    # print('gi values are: ', gi)
    assert np.isclose(gi.imag, 0)
    gi = gi.real
    return gi


def return_gradient_list(cluster_ops_sparse, hamiltonian_sparse, curr_state
):
    """
    Compute analytically the gradient for all fermionic cluster operators ("cluster_ops_sparse").

    Parameters
    -----------
    cluster_ops_sparse: List<transposable linear operator>
        the sparse fermionic operators

    hamiltonian_sparse: ndarray
        the hamiltonian operator
    
    curr_state: ndarray
        the current state


    Returns
    --------
    list_grad: List<float>
        the gradients for all the fermionic cluster operators
    
    curr_norm: float
        the magnitude of the gradients
    
    next_deriv: float
        the maximum gradient (in absolute value)
    
    next_index: int
        the index of the operator with maximum gradient

    """
    
    list_grad = []
    curr_norm = 0
    next_deriv = 0
    next_index = 0
    sig = hamiltonian_sparse.dot(curr_state)
    for oi in range(len(cluster_ops_sparse)):
        gi = compute_gradient_i(oi, cluster_ops_sparse, curr_state, sig)
        list_grad.append(abs(gi))
        curr_norm += gi * gi
        if abs(gi) > abs(next_deriv):
            next_deriv = gi
            next_index = oi
    return list_grad, curr_norm, next_deriv, next_index


# create the function related to myADAPT-VQE:
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


def commutators_calculations(cluster_ops_sp, hamiltonian_sp):
    """
    Compute the commutators [cluster_ops_sp[i], hamiltonian_sp].
    Note: it is under developement

    Parameters
    -----------

    cluster_ops_sp: list[Hamiltonian]
        list of spin cluster operators
    
    hamiltonian_sp: Hamiltonian
        Hamiltonian in the spin representation
    

    Returns
    --------
    list_commutators: list<float>
        list of the resulting commutators (gradients)

    """

    list_commutators = []
    for oi in cluster_ops_sp:
        X = -(
            hamiltonian_sp * oi * (complex(0, 1))
            - oi * (complex(0, 1)) * hamiltonian_sp
        )
        list_commutators.append(X)
    return list_commutators


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


def get_statevector(result, nbqbits):
    """
    Get the statevector from the Result class

    Parameters
    -----------
    result: qat.core.BatchResult
        the result of an executed job
    
    nbqbits: int
        the number of qubits
    
    statevector: ndarray
        the resulting statevector representation

    """
    statevector = np.zeros((2**nbqbits), np.complex128)
    for sample in result:
        statevector[sample.state.int] = sample.amplitude
    return statevector


def fun_fidelity(circ, eigenvalues, eigenvectors, nbqbits):
    """
    Checks the fidelity between the resulted state and exact wave function

    Parameters
    ----------
    circ: qat.core.Circuit
        the circuit corresponding to the resulted state

    eigenvalues: ndarray
        the eigen values of the hamiltonian

    eigenvectors: ndarray
        the eigen vectors of the hamiltonian

    nbqbits: int
        the number of qubits
    
    Returns
    --------

    fid: float
        the fidelity

    """
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
    
    """
    Runs the loop of making fermionic adapt vqe found in this reference in section "Results"
    Grimsley HR, Economou SE, Barnes E, Mayhall NJ. An adaptive variational algorithm for exact molecular simulations
    on a quantum computer. Nature communications 2019; 10(1): 1-9.

    Parameters
    ----------
    
    hamiltonian_sparse: ndarray
        The sparse hamiltonian
    
    cluster_ops_sparse: List<transposable linear operator>
        the sparse fermionic operators

    reference_ket: ndarray
        the initial state
    
    hamiltonian_sp: Hamiltonian
        Hamiltonian in the spin representation
    
    cluster_ops_sp: list[Hamiltonian]
        list of spin cluster operators
    
    hf_init_sp: int
        the integer corresponds to the hf_init (The Hartree-Fock state in integer representation) obtained by using
        "qat.fermion.transforms.record_integer".
    
    n_max_grads: int
        the number of maximum gradients chosen per internal iteration
    
    fci: float
        the full configuration interaction energy
    
    optimizer: string
        the type of the optimizer
    
    tolerance: float
        the tolerance for reaching convergence
    
    type_conver: string
        in our case, "norm" is chosen
    
    threshold_needed: float
        the norm threshold 

    max_external_iterations: int
        the number of maximum iteration to perform the whole adaptive loop
    

    Returns
    --------

    iterations: Dict
        the following properties of the simulation: 
        energies, energies_substracted_from_FCI, norms, Max_gradients, fidelity, CNOTs, Hadamard, RY, and RX.

    result: Dict
        the following properties after convergence:
        indices, Number_operators, final_norm, parameters, Number_CNOT_gates, Number_Hadamard_gates, Number_RX_gates, final_energy_last_iteration,

    """
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
            cluster_ops_sparse, hamiltonian_sparse, curr_state_open_f
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
            print(" *final converged energy iteration is %20.12f" % opt_result.fun)  # noqa: F821
            result["final_energy_last_iteration"] = opt_result.fun # noqa: F821
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
        print("sorted_index1: ", sorted_index1)
        # there are two types of initial thetas: either zero, of from gamma_1 list
        for j in range(len(sorted_index1)):
            parameters_ansatz.append(0.01)
            # for UCCSD: we must multiply by i
            #             ansatz_ops.append(cluster_ops_sp[sorted_index1[j]])
            ansatz_ops.append(complex(0.0, 1.0) * cluster_ops_sp[sorted_index1[j]])
            op_indices.append(sorted_index1[j])
            ansatz_mat.append(cluster_ops_sparse[sorted_index1[j]])
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
        print(" -----------New ansatz created----------- ")
        print(" %4s \t%s \t%s" % ("#", "Coefficients", "Term"))
        parameters_ansatz = []

        for si in range(len(ansatz_ops)):
            print(" %4i \t%f \t%s" % (si, xlist[si], op_indices[si]))
            parameters_ansatz.append(xlist[si])
        curr_state = prepare_state_ansatz(ansatz_ops, hf_init_sp, parameters_ansatz)
        curr_state_open_f = prepare_adapt_state(reference_ket, ansatz_mat, parameters_ansatz)
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
