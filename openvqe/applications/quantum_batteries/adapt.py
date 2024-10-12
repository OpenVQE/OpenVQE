"""
CS-ADAPT-VQE: Extension to CS-VQE using ADAPT-VQE as contextual subspace solver
"""

import cudaq
import pickle
from time import time
from utils import get_ham_from_dict, rel_err
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
cudaq.set_target("nvidia")


def single_excitation(kernel, phi, wires):
    """
    Single excitation rotation.
    Reference:
    https://docs.pennylane.ai/en/stable/code/api/pennylane.SingleExcitation.html

    Args:
      kernel: The `cudaq.kernel`.
      phi: rotation angle.
      wires: the qubits the operation acts on.
    """
    kernel.tdg(wires[0])
    kernel.h(wires[0])
    kernel.s(wires[0])
    kernel.tdg(wires[1])
    kernel.sdg(wires[1])
    kernel.h(wires[1])
    kernel.cx(wires[1], wires[0])
    kernel.rz(-phi / 2, wires[0])
    kernel.ry(phi / 2, wires[1])
    kernel.cx(wires[1], wires[0])
    kernel.sdg(wires[0])
    kernel.h(wires[0])
    kernel.t(wires[0])
    kernel.h(wires[1])
    kernel.s(wires[1])
    kernel.t(wires[1])


def double_excitation(kernel, phi, wires):
    """
    Double excitation rotation.
    Reference:
    https://docs.pennylane.ai/en/stable/code/api/pennylane.DoubleExcitation.html

    Args:
      kernel: The `cudaq.kernel`.
      phi: rotation angle.
      wires: the qubits the operation acts on.
    """
    kernel.cx(wires[2], wires[3])
    kernel.cx(wires[0], wires[2])
    kernel.h(wires[3])
    kernel.h(wires[0])
    kernel.cx(wires[2], wires[3])
    kernel.cx(wires[0], wires[1])
    kernel.ry(phi / 8, wires[1])
    kernel.ry(-phi / 8, wires[0])
    kernel.cx(wires[0], wires[3])
    kernel.h(wires[3])
    kernel.cx(wires[3], wires[1])
    kernel.ry(phi / 8, wires[1])
    kernel.ry(-phi / 8, wires[0])
    kernel.cx(wires[2], wires[1])
    kernel.cx(wires[2], wires[0])
    kernel.ry(-phi / 8, wires[1])
    kernel.ry(phi / 8, wires[0])
    kernel.cx(wires[3], wires[1])
    kernel.h(wires[3])
    kernel.cx(wires[0], wires[3])
    kernel.ry(-phi / 8, wires[1])
    kernel.ry(phi / 8, wires[0])
    kernel.cx(wires[0], wires[1])
    kernel.cx(wires[2], wires[0])
    kernel.h(wires[0])
    kernel.h(wires[3])
    kernel.cx(wires[0], wires[2])
    kernel.cx(wires[2], wires[3])


def excitations(electrons, orbitals, delta_sz=0):
    """
    Generate single and double excitations from a Hartree-Fock reference state.
    Reference:
    https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.excitations.html

    Args:
      electrons:  Number of electrons.
      orbitals: Number of *spin* orbitals.
      delta_sz: Specifies the selection rules ``sz[p] - sz[r] = delta_sz`` and
            ``sz[p] + sz[p] - sz[r] - sz[s] = delta_sz`` for the spin-projection ``sz`` of
            the orbitals involved in the single and double excitations, respectively.
            ``delta_sz`` can take the values :math:`0`, :math:`\pm 1` and :math:`\pm 2`.

    Returns:
      lists with the indices of the spin orbitals involved in the single and
      double excitations
    """
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    singles = [
        [r, p]
        for r in range(electrons)
        for p in range(electrons, orbitals)
        if sz[p] - sz[r] == delta_sz
    ]

    doubles = [
        [s, r, q, p]
        for s in range(electrons - 1)
        for r in range(s + 1, electrons)
        for q in range(electrons, orbitals - 1)
        for p in range(q + 1, orbitals)
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    return singles, doubles


# gradient of ith parameter
def parameter_shift_term(kernel, hamiltonian, params, i):
    """
    Calculates the gradient of a given kernel and Hamiltonian with
    respect to the ith parameter using the parameter shift rule.

    Args:
      kernel: The `cudaq.kernel`
      hamiltonian: System Hamiltonian.
      param: Parameters required by the quantum circuit.
      i: Index of the parameter whose gradient is to be computed.

    Returns:
      Gradient of the ith parameter.
    """
    shifted = params.copy()
    shifted[i] += np.pi / 2  # offset
    forward = cudaq.observe(
        kernel, hamiltonian, shifted
    ).expectation()  # forward evaluation
    shifted[i] -= np.pi / 2  # revert back

    shifted[i] += -np.pi / 2  # offset
    backward = cudaq.observe(
        kernel, hamiltonian, shifted
    ).expectation()  # backward evaluation

    return 0.5 * (forward - backward)


def parameter_shift(kernel, hamiltonian, params):
    """
    Calculates the gradients of a given kernel and Hamiltonian with
    respect to the parameters using the parameter shift rule.

    Args:
      kernel: The `cudaq.kernel`
      hamiltonian: System Hamiltonian.
      params: Parameters required by the quantum circuit.

    Returns:
      An array of gradients for each parameter in the `params` array.
    """
    gradients = np.zeros([len(params)])

    for i in range(len(params)):
        gradients[i] = parameter_shift_term(kernel, hamiltonian, params, i)

    return gradients


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
    return kernel, parameters, qubits


def basis_state(kernel, qubits, hf):
    """
    Prepares the Hartree fock state. Updates the
    `kernel` in-place.

    Args:
      kernel: The `cudaq.kernel`
      qubits: list of qubits of the quantum circuit.
      hf:  Hartree fock state.
    """
    if hf is not None:
        for i, q in enumerate(hf):
            if q == "1":
                kernel.x(qubits[i])


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

    # We only focus on the hamiltonian with 7 and 8 qubits
    num_qubits = [
        ham.get_qubit_count()
        for ham, hf in zip(hamiltonians[::-1][-2:], hf_states[::-1][-2:])
    ]

    num_iterations = 20  # Number of experiments

    for ham, hf in zip(hamiltonians[::-1][-2:], hf_states[::-1][-2:]):
        n_qubits = ham.get_qubit_count()
        print(f"\nnum qubits = {n_qubits}")

        # Calculate the number of electrons in the
        # contextual subspace
        electrons = 0
        for i, q in enumerate(hf):
            if q == "1":
                electrons += 1

        print(f"num electrons = {electrons}")

        # Calulate the singles and doubles excitations
        singles, doubles = excitations(electrons, n_qubits)
        print(f"Total number of excitations = {len(singles) + len(doubles)}")

        temp_durations = []
        temp_rel_errs = []

        for _ in range(num_iterations):
            # Adaptive optimization: Manual Construction

            start = time()
            # Compute gradients for all double excitations
            kernel, parameters, qubits = create_kernel(n_qubits)
            basis_state(kernel, qubits, hf)
            for i, wires in enumerate(doubles):
                double_excitation(kernel, parameters[i], [qubits[q] for q in wires])
            init_params = [0.0] * len(doubles)
            grads = parameter_shift(kernel, ham, init_params)

            # In principle, we should select the double excitations
            # with gradients larger than a pre-defined threshold.
            # However we choose the one with maximum absolute gradient
            # value and therefore only one gate is chosen.

            if len(grads):
                doubles_select = doubles[np.argmax(abs(grads))]
            else:
                doubles_select = []
            del kernel, parameters, qubits

            # Perform VQE to obtain the optimized parameters for the selected double excitations.
            if len(doubles_select):
                kernel, parameters, qubits = create_kernel(n_qubits)
                basis_state(kernel, qubits, hf)
                double_excitation(
                    kernel, parameters[0], [qubits[q] for q in doubles_select]
                )
                parameter_count = 1
                optimizer = cudaq.optimizers.NelderMead()
                optimizer.max_iterations = 1000
                optimizer.initial_parameters = np.random.uniform(size=parameter_count)
                energy, params_doubles = cudaq.vqe(
                    kernel, ham, optimizer, parameter_count=parameter_count
                )
                del kernel, parameters, qubits

            # Compute gradients for all single excitations.
            kernel, parameters, qubits = create_kernel(n_qubits)
            basis_state(kernel, qubits, hf)
            if len(doubles_select):
                double_excitation(
                    kernel, params_doubles[0], [qubits[q] for q in doubles_select]
                )
            for i, wires in enumerate(singles):
                single_excitation(kernel, parameters[i], [qubits[q] for q in wires])
            init_params = [0.0] * len(singles)
            grads = parameter_shift(kernel, ham, init_params)

            # Select the single excitation with maximum absolute gradient value
            singles_select = singles[np.argmax(abs(grads))]
            del kernel, parameters, qubits

            # Perform the final VQE optimization with all the selected excitations.
            kernel, parameters, qubits = create_kernel(n_qubits)
            basis_state(kernel, qubits, hf)
            parameter_count = 0
            if len(doubles_select):
                double_excitation(
                    kernel,
                    parameters[parameter_count],
                    [qubits[q] for q in doubles_select],
                )
                parameter_count += 1
            single_excitation(
                kernel, parameters[parameter_count], [qubits[q] for q in singles_select]
            )
            parameter_count += 1
            optimizer = cudaq.optimizers.NelderMead()
            optimizer.max_iterations = 100
            optimizer.initial_parameters = np.random.uniform(size=parameter_count)
            energy, params = cudaq.vqe(
                kernel, ham, optimizer, parameter_count=parameter_count
            )
            del kernel, parameters, qubits

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

    # Plot the relative error and runtime
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].errorbar(
        num_qubits,
        mean_rel_errs,
        std_rel_errs,
        marker="o",
        linestyle="dotted",
        capsize=4,
    )
    ax[0].set_xlabel("# qubits")
    ax[0].set_ylabel("Rel Error")
    ax[0].set_xticks(num_qubits)
    ax[0].set_yscale("log")
    ax[0].set_title("Relative error")

    ax[1].errorbar(
        num_qubits,
        mean_durations,
        std_durations,
        marker="o",
        linestyle="dotted",
        capsize=4,
    )
    ax[1].set_xlabel("# qubits")
    ax[1].set_ylabel("durations")
    ax[1].set_xticks(num_qubits)
    ax[1].set_title("Runtime")

    plt.savefig("adapt")
