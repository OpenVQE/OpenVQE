"""
CS-VQE: VQE algorithm on the contextual hamiltonians of Li2FeSiO4
"""

import cudaq
import pickle
from time import time
from utils import get_ham_from_dict, rel_err
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
cudaq.set_target("nvidia")

# Reference classical energy
ccsd_energy = -3688.046308050882

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

layers = [4, 8, 12]  # We test with these layers for the HEA

mean_durations = {l: [] for l in layers}
std_durations = {l: [] for l in layers}
mean_rel_errs = {l: [] for l in layers}
std_rel_errs = {l: [] for l in layers}

num_qubits = [
    ham.get_qubit_count() for ham, hf in zip(hamiltonians[::-1], hf_states[::-1])
]

num_iterations = 20  # Number of experiments

for num_layers in layers:
    print(f"\nnum layers = {num_layers}")
    for ham, hf in zip(hamiltonians[::-1], hf_states[::-1]):
        n_qubits = ham.get_qubit_count()

        print(f"num qubits = {n_qubits}")

        temp_durations = []
        temp_rel_errs = []

        for _ in range(num_iterations):
            kernel, thetas = cudaq.make_kernel(list)
            qubits = kernel.qalloc(n_qubits)

            # Prepare the Hartree Fock State.
            if hf is not None:
                for i, q in enumerate(hf):
                    if q == "1":
                        kernel.x(qubits[i])

            # Hardware efficient ansatz
            for l in range(num_layers):
                # Rotation layer
                for q in range(n_qubits):
                    kernel.ry(thetas[l * n_qubits + q], qubits[q])

                # Entangling layer
                for q in range(n_qubits - 1):
                    kernel.cx(qubits[q], qubits[q + 1])

            # Final rotation layer
            for q in range(n_qubits):
                kernel.ry(thetas[num_layers * n_qubits + q], qubits[q])

            # Optimizer initialization
            parameter_count = (num_layers + 1) * n_qubits
            optimizer = cudaq.optimizers.NelderMead()
            optimizer.max_iterations = 1000
            optimizer.initial_parameters = np.random.uniform(size=parameter_count)

            # Optimzer loop
            start = time()
            energy, parameters = cudaq.vqe(
                kernel, ham, optimizer, parameter_count=parameter_count
            )
            end = time()

            # store the relative error and runtime
            temp_rel_errs.append(rel_err(ccsd_energy, energy))
            temp_durations.append(end - start)

            del kernel, thetas, qubits, optimizer

        # Calculate the mean and standarad deviations for the
        # relative error and runtime
        mean_durations[num_layers].append(np.mean(temp_durations))
        mean_rel_errs[num_layers].append(np.mean(temp_rel_errs))

        std_durations[num_layers].append(np.std(temp_durations))
        std_rel_errs[num_layers].append(np.std(temp_rel_errs))

        print(f"minimized <H> = {round(energy,16)}")
        print(f"num params = {parameter_count}")
        print(
            f"rel_error = {mean_rel_errs[num_layers][-1]} +- {std_rel_errs[num_layers][-1]}"
        )
        print(
            f"duration = {mean_durations[num_layers][-1]} += {std_durations[num_layers][-1]}"
        )

# Plot the relative error and runtime
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for num_layers in layers:
    ax[0].errorbar(
        num_qubits,
        mean_rel_errs[num_layers],
        std_rel_errs[num_layers],
        marker="o",
        label=f"{num_layers} layers",
        capsize=4,
    )
ax[0].set_xlabel("# qubits")
ax[0].set_ylabel("Rel Error")
ax[0].set_xticks(num_qubits)
ax[0].set_yscale("log")
ax[0].set_title("Relative error")
ax[0].legend()


for num_layers in layers:
    ax[1].errorbar(
        num_qubits,
        mean_durations[num_layers],
        std_durations[num_layers],
        marker="o",
        label=f"{num_layers} layers",
        capsize=4,
    )
ax[1].set_xlabel("# qubits")
ax[1].set_ylabel("durations")
ax[1].set_xticks(num_qubits)
ax[1].set_title("Runtime")
ax[1].legend()

plt.savefig("cs_vqe")
