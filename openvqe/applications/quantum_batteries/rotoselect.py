"""
CS-VQE with Rotoselect as optimizer
"""

import cudaq
import pickle
from time import time
from utils import get_ham_from_dict, rel_err
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

cudaq.set_target("nvidia")


## Helper functions for Rotoselect optimizer
def RGen(kernel, param, generator, wire):
    """
    Applies the given rotation gate `generator` on the qubit `wire`

    Args:
      kernel: The `cudaq.kernel`.
      param: The parameter value of the rotation gate.
      generator: Rotation gate name.
      wire: The qubit on which the rotation gate is applied

    Returns:
      Updated `kernel`.
    """
    if generator == "X":
        kernel.rx(param, wire)
    elif generator == "Y":
        kernel.ry(param, wire)
    elif generator == "Z":
        kernel.rz(param, wire)


def create_kernel(n_qubits):
    """
    Creats a cudaq kernel and allocates qubits

    Args:
      n_qubits: Number of qubits in the circuit.

    Returns:
        qubits: The allocated qubits based on `n_qubits`.
        kernel: The `cudaq.kernel`.
        parameters: Parameters required by the quantum circuit.
    """
    kernel, parameters = cudaq.make_kernel(list)
    qubits = kernel.qalloc(n_qubits)
    return qubits, kernel, parameters


def ansatz(generators, n_qubits, hf):
    """
    The Hardware efficient ansatz with a single rotation layer
    followed by a CNOT ladder

    Args:
      generators: Set of Pauli rotation gates.
      n_qubits: Number of qubits in the circuit.
      hf: Hartree fock state.

    Returns:
        The `cudaq.kernel` with gates applied.
    """
    qubits, kernel, parameters = create_kernel(n_qubits)

    # Prepare the Hartree Fock State.
    if hf is not None:
        for i, q in enumerate(hf):
            if q == "1":
                kernel.x(qubits[i])

    for q in range(n_qubits):
        RGen(kernel, parameters[q], generators[q], qubits[q])
    for q in range(n_qubits - 1):
        kernel.cx(qubits[q], qubits[q + 1])
    return kernel


def cost_fn(params, generators, n_qubits, hamiltonian, hf):
    """
    Calculates the energy expectation value of a given Hamiltonian using a specified
    ansatz and parameters.

    Args:
      params: Parameters required by the quantum circuit.
      generators: Set of Pauli rotation gates.
      hamiltonian: System Hamiltonian.
      hf:  Hartree fock state.

    Returns:
      The energy expectation value calculated based on the provided parameters,
    generators, number of qubits, Hamiltonian, and Hartree-Fock state.
    """
    kernel = ansatz(generators, n_qubits, hf)
    expectation_value = cudaq.observe(kernel, hamiltonian, params).expectation()
    return expectation_value


def rotosolve(
    d, params, generators, cost, M_0, n_qubits, hamiltonian, hf
):  # M_0 only calculated once
    """
    The rotosolve optimizer

    Args:
      d: the index of a parameter in the `params`.
      params: Parameters required by the quantum circuit.
      generators: Set of Pauli rotation gates.
      cost: The energy expectation value function.
      M_0: Energy expectation value whern `params[d]` is zero.
      n_qubits: Number of qubits in the circuit.
      hamiltonian: System Hamiltonian.
      hf: Hartree fock state.

    Returns:
      The energy expectation value with updated parameters.
    """
    params[d] = np.pi / 2.0
    M_0_plus = cost(params, generators, n_qubits, hamiltonian, hf)
    params[d] = -np.pi / 2.0
    M_0_minus = cost(params, generators, n_qubits, hamiltonian, hf)
    a = np.arctan2(
        2.0 * M_0 - M_0_plus - M_0_minus, M_0_plus - M_0_minus
    )  # returns value in (-pi,pi]
    params[d] = -np.pi / 2.0 - a
    if params[d] <= -np.pi:
        params[d] += 2 * np.pi
    return cost(params, generators, n_qubits, hamiltonian, hf)


def optimal_theta_and_gen_helper(
    d, params, generators, cost, n_qubits, hamiltonian, hf
):
    """
    The function iterates over different generators to find the optimal
    parameters and generators for a given cost function.

    Args:
      d: the index of a parameter in the `params`.
      params: Parameters required by the quantum circuit.
      generators: Set of Pauli rotation gates.
      cost: The energy expectation value function.
      n_qubits: Number of qubits in the circuit.
      hamiltonian: System Hamiltonian.
      hf: Hartree fock state.

    Returns:
      The optimal value of the parameter `params[d]` (`params_opt_d`) and
      the corresponding optimal generator choice (`generators_opt_d`).
    """
    params[d] = 0.0
    M_0 = cost(
        params, generators, n_qubits, hamiltonian, hf
    )  # M_0 independent of generator selection
    for generator in ["X", "Y", "Z"]:
        generators[d] = generator
        params_cost = rotosolve(
            d, params, generators, cost, M_0, n_qubits, hamiltonian, hf
        )
        # initialize optimal generator with first item in list, "X", and update if necessary
        if generator == "X" or params_cost <= params_opt_cost:
            params_opt_d = params[d]
            params_opt_cost = params_cost
            generators_opt_d = generator
    return params_opt_d, generators_opt_d


def rotoselect_cycle(cost, params, generators, n_qubits, hamiltonian, hf):
    """
    One cycle of Rotoselect algorithm

    Args:
      cost: The energy expectation value function.
      params: Parameters required by the quantum circuit.
      generators: Set of Pauli rotation gates.
      n_qubits: Number of qubits in the circuit.
      hamiltonian: System Hamiltonian.
      hf: Hartree fock state.

    Returns:
      The updated `params` and `generators`.
    """
    for d in range(len(params)):
        params[d], generators[d] = optimal_theta_and_gen_helper(
            d, params, generators, cost, n_qubits, hamiltonian, hf
        )
    return params, generators


if __name__ == "__main__":
    ccsd_energy = -3688.046308050882  # Reference classical energy

    # Load the contextuals hamiltonians and hartree fock states
    with open("CS_hams.pickle", "rb") as handle:
        hams = pickle.load(handle)

    hamiltonians = []
    hf_states = []
    for n_qubits, val in hams.items():
        ham_dict = hams[n_qubits]["ham"]
        hamiltonians.append(get_ham_from_dict(ham_dict))

        hf_dict = hams[n_qubits]["hf"]
        if hf_dict:
            hf_states.append(list(hf_dict.keys())[0])
        else:
            hf_states.append(None)

    mean_durations = []
    std_durations = []
    mean_rel_errs = []
    std_rel_errs = []

    num_qubits = [
        ham.get_qubit_count() for ham, hf in zip(hamiltonians[::-1], hf_states[::-1])
    ]

    num_iterations = 10  # Number of experiments

    for ham, hf in zip(hamiltonians[::-1], hf_states[::-1]):
        n_qubits = ham.get_qubit_count()
        init_generators = np.array(["Y"] * n_qubits)

        print(f"\nnum qubits = {n_qubits}")

        temp_durations = []
        temp_rel_errs = []

        for _ in range(num_iterations):
            # Initial gate parameters which intialize the qubit in the zero state
            initial_parameters = np.random.uniform(size=n_qubits)

            params_rsel = initial_parameters.copy()
            generators = init_generators.copy()

            n_steps = 50
            parameter_count = n_qubits

            # Optimzer loop
            start = time()
            for i in range(n_steps):
                params_rsel, generators = rotoselect_cycle(
                    cost_fn, params_rsel, generators, n_qubits, ham, hf
                )
            energy = cost_fn(params_rsel, generators, n_qubits, ham, hf)
            end = time()

            # store the relative error and runtime
            temp_rel_errs.append(rel_err(ccsd_energy, energy))
            temp_durations.append(end - start)

        # Calculate the mean and standarad deviations for the
        # relative error and runtime
        mean_durations.append(np.mean(temp_durations))
        mean_rel_errs.append(np.mean(temp_rel_errs))

        std_durations.append(np.std(temp_durations))
        std_rel_errs.append(np.std(temp_rel_errs))

        print(f"minimized <H> = {round(energy,16)}")
        print(f"num params = {parameter_count}")
        print(f"rel_error = {mean_rel_errs[-1]} +- {std_rel_errs[-1]}")
        print(f"duration = {mean_durations[-1]} += {std_durations[-1]}")
        print(f"generators = {generators}")

    # Plot the relative error and runtime
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].errorbar(num_qubits, mean_rel_errs, std_rel_errs, marker="o", capsize=4)
    ax[0].set_xlabel("# qubits")
    ax[0].set_ylabel("Rel Error")
    ax[0].set_xticks(num_qubits)
    ax[0].set_yscale("log")
    ax[0].set_title("Relative error")

    ax[1].errorbar(num_qubits, mean_durations, std_durations, marker="o", capsize=4)
    ax[1].set_xlabel("# qubits")
    ax[1].set_ylabel("durations")
    ax[1].set_xticks(num_qubits)
    ax[1].set_title("Runtime")

    plt.savefig("rotoselect")
