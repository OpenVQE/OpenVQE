# Simulating lithium-ion batteries on quantum computers

## Team Jetix

Members: Gopal Ramesh Dahale

## Abstract

One of the frontiers for the Noisy Intermediate-Scale Quantum (NISQ) era's practical uses of quantum computers is quantum chemistry. By employing Hybrid Quantum Classical Optimization we aim to investigate the potential of quantum computing for simulating lithium-ion batteries. We implement quantum algorithms that efficiently calculates the ground state energy of materials used in batteries. We use a variant of Variational Quantum Eigensolver (VQE) known as Contextual Subspace VQE (CS-VQE) and focus on exploring the power of hardware-efficient ansatzes to improve the convergence of finding the ground state energy of the materials. Furthermore, techniques like ADAPT-VQE and Rotoselect algorithms are employed to solve for the contextual subspace hamiltonians, hence, presenting an extension to the CS-VQE algorithm. We use these techniques to simulate a realistic cathode material: dilithium iron silicate and were able to reduce the number of qubits required in the algorithm by more than half, proving to be suitable for the ISQ era. Our implementation leverages NVIDIA's GPU and CUDA Quantum Platform which enable us to perform faster simulations. The findings from our study demonstrates the potential of quantum computing to effectively perform quantum simulations of lithium-ion batteries.

## Files

The first is the `contextual_subspace.ipynb` which focusses on calculating the contextual subspace Hamiltonians for the Li$_2$FeSiO$_4$ material (structure is in `Li2FeSiO4.cif`). The CS Hamiltonians are saved in `CS_hams.pickle`.


Using the CS Hamiltonians, we execute different quantum algorithms

1. `cs_vqe.py` has the implements the traditional VQE algorithm with hardware-efficient ansatz for varying number of layers.
2. `rotoselect.py` uses the Rotoselect optimization for the CS-VQE algorithm.
3. `adapt.py` extends the CS-VQE algorithm by using ADAPT-VQE. We refer the new algorithm as CS-ADAPT-VQE.

### Results

- The `logs` directory holds the logs generated during the experiments.
- `plots` has the graphs for different experiments.

## Misc
- `plot_scaling.py` was used to plot he qubits vs ham terms for CS Hamiltonians.
- `utils.py` holds some utility functions common to other files.

## Acknowledgments

We acknowledge the hardware and software support from [CUDA Quantum](https://github.com/NVIDIA/cuda-quantum) and NVIDIA. This project would not have been possible with [Tangelo](https://github.com/goodchemistryco/Tangelo) and [Symmer](https://github.com/UCL-CCS/symmer). [PennyLane](https://github.com/PennyLaneAI/pennylane) and [Qiskit](https://github.com/Qiskit/qiskit) were used for visualization purposes.