OpenVQE: README
=======================

[![License](https://img.shields.io/github/license/OpenVQE/openvqe.svg)](https://opensource.org/licenses/MIT)
[![Current Release](https://img.shields.io/github/v/release/OpenVQE/openvqe.svg)](https://github.com/OpenVQE/openvqe/releases)

**OpenVQE** is an open-source extension of the Variational Quantum Eigensolver (VQE) for quantum chemistry, building on the Quantum Learning Machine (QLM) and developed using tools from [`myqlm-fermion`](https://github.com/myQLM/myqlm-fermion.git). It enhances QLM's capabilities in quantum chemistry computations. This repository produce quantum computing applications. Please follows us as there will be more and more applications to coming.

## Key Publication
For a comprehensive overview, refer to the main OpenVQE paper:

*Open Source Variational Quantum Eigensolver Extension of the Quantum Learning Machine (QLM) for Quantum Chemistry.*  
M. Haidar, M. J. Rančić, T. Ayral, Y. Maday, J.-P. Piquemal, *WIREs Comp. Mol. Sci.*, 2023, e1664 (Open Access)  
DOI: [10.1002/wcms.1664](https://doi.org/10.1002/wcms.1664)


## Modules


1. **UCC Family**: this module consists of different classes and functions to generate the fermionic cluster operators (fermionic pool) and the  qubit pools and to get the optimized energies from VQE in the case of active and non-active orbital selections. For example, UCCSD, QUCCSD, UCCGSD, K-UpCCGSD, spin-complemented pair, singlet and doublet generalized single and double excitations, etc.

2. **ADAPT**: Contains two sub-modules:
    - Fermionic-ADAPT: it contains functions that performs the fermionic ADAPT-VQE algorthmic steps  in the active and non-active space selections.
    - Qubit-ADAPT: it  contains functions that perform the Qubit ADAPT-VQE algorithmic steps calculation in the active and non-active space orbital selections.

3. **Applications**: Contains practical quantum computing applications:
    - **quantum_batteries**: simulation of lithium batteries using quantum computing produced by Gopal Dahale, a Master's student in Quantum Science at EPFL, for integrating the lithium battery application into the repository. Check out his [GitHub](https://github.com/Gopal-Dahale).
4. **Applications/Algorithms**: Regroup promising quantum computing algorithms.
    - **QAOA**: Application of Quantum Approximate Optimization Algorithm (QAOA) dedicated to minimum vertex cover problem. 
    - **quantum_maxcut**: Quantum annealing computing to solve maxcut problem.

Installation
--------------

To install OpenVQE from source:

```bash
git clone https://github.com/OpenVQE/OpenVQE.git
cd OpenVQE
git checkout alpha
pip install .
pip install -r requirements.txt
```

## Troubleshooting

If you have a problem with the installation we recommand using `conda`:

```shell
conda create --name openvqe python=3.11
conda activate openvqe
# Repeat the installation steps above
git clone https://github.com/OpenVQE/OpenVQE.git
cd OpenVQE
git checkout alpha
pip install .
pip install -r requirements.txt
```

To explore the quantum battery application, you will need to install CUDA and have an NVIDIA GPU. For detailed instructions, refer to the    [NVIDIA CUDA Quantum GitHub page](https://github.com/NVIDIA/cuda-quantum). If you want to install it on linux do:
```shell
sudo apt update && sudo apt upgrade -y
sudo apt install cuda-11-8 -y
```

If you have the error: `Qiskit is installed in an invalid environment that has both Qiskit >=1.0 and an earlier version.`. Run the command: `pip uninstall qiskit-terra` inside your terminal. 

## Contributing to the Package

Please read the CONTRIBUTING.md file.

Getting started
----------------

## Video tutorial

Please checkout the lecture of Dr. Mohammad Haidar on how to use openVQE: https://www.youtube.com/watch?v=NkRFcn4LuNs. 

## Notebook 

Tutorials are stored inside jupyter notebooks in the "notebooks" folder. These notebooks contains examples of code using openVQE that you can run locally. The lithium battery application also has an dedicated notebooks. 

Documentation
---------------
The code is based on the well documented code of `myqlm-fermion` framework [link](https://myqlm.github.io/).
The main functions are documented in the code base.
For more information, please refer to our paper: 
> *Open Source Variational Quantum Eigensolver Extension of the Quantum Learning Machine (QLM) for Quantum Chemistry. 
M. Haidar,  M. J. Rančić, T. Ayral, Y. Maday, J.-P. Piquemal, WIREs Comp. Mol. Sci., 2023, e1664 (Open Access)
DOI: 10.1002/wcms.1664*

How to cite
-----------
> *Mohammad Haidar, Marko J. Ranˇci´c, Thomas Ayral, Yvon Maday, and Jean-Philip Piquemal. Open source variational quantum eigensolver extension of the quantum learning machine for quantum chemistry. WIREs Computational Molecular Science, page e1664, 2023*

Getting in touch
-----------
For any questions regarding OpenVQE or related research, contact: mohammadhaidar2016@outlook.com.

License
-----------
OpenVQE is created by Mohammad Haidar and licensed under the MIT License.

References
-----------
* Nooijen, Marcel. "Can the eigenstates of a many-body hamiltonian be represented exactly using a general two-body cluster expansion?." Physical review letters 84.10 (2000): 2108.
* Lee, Joonho, et al. "Generalized unitary coupled cluster wave functions for quantum computation." Journal of chemical theory and computation 15.1 (2018): 311-324.
* Grimsley, Harper R., et al. "An adaptive variational algorithm for exact molecular simulations on a quantum computer." Nature communications 10.1 (2019): 1-9.
* Tang, Ho Lun, et al. "qubit-adapt-vqe: An adaptive algorithm for constructing hardware-efficient ansätze on a quantum processor." PRX Quantum 2.2 (2021): 020310.
* Xia, Rongxin, and Sabre Kais. "Qubit coupled cluster singles and doubles variational quantum eigensolver ansatz for electronic structure calculations." Quantum Science and Technology 6.1 (2020): 015001.
* Shkolnikov, V. O., et al. "Avoiding symmetry roadblocks and minimizing the measurement overhead of adaptive variational quantum eigensolvers." arXiv preprint arXiv:2109.05340 (2021).
