OpenVQE: README
=======================

`OpenVQE` is an Open Source Variational Quantum Eigensolver extension of the Quantum Learning Machine to  Quantum Chemistry. It was developed based on the tools of `myqlm-fermion` 


It consists of two main modules as follows:

- UCC Family: this module consists of different classes and functions to generate the fermionic cluster operators (fermionic pool) and the  qubit pools and to get the optimized energies from VQE in the case of active and non-active orbital selections. For example, UCCSD, QUCCSD, UCCGSD, K-UpCCGSD, spin-complemented pair, etc.

- ADAPT: consists of two sub-modules which are: 
    - Fermionic-ADAPT: it contains functions that performs the fermionic ADAPT-VQE algorthmic steps  in the active and non-active space selections.
    - Qubit-ADAPT: it  contains functions that perform the Qubit ADAPT-VQE algorithmic steps calculation in the active and non-active space orbital selections.


Installation
--------------
### Prerequisites:
#### Install myqlm-fermion:
You can install the fermion module with pip directly from the internet (not supported for now):
```shell
pip install myqlm-fermion
```
or install from the sources directly :
```shell
git clone https://github.com/myQLM/myqlm-fermion.git
cd myqlm-fermion
pip install -r requirements.txt
pip install .
```
### Install OpenVQE
install from the sources directly :
```shell
git clone https://github.com/Haidarmm/OpenVQE.git
git checkout update-ferm
cd OpenVQE
```
Move to the next section for getting started!


Getting started
----------------
```shell
cd python
- to access the first module UCC family:
cd ucc
open with jupyter notebook: main_for_total_ucc.ipynb
- to access module fermionic_adapt:
cd fermionic_adapt
open with jupyter notebook: main_without_open_fermion.ipynb
- to access module qubit_adapt
cd qubit_adapt
open with jupyter notebook: main_qubit_pool_without_open_fermion.ipynb
```

Documentation
---------------
The code is based on the well documented code of `myqlm-fermion` framework [link](https://myqlm.github.io/).
The main functions are documented in the code base.
For more information, please refer to our paper that wil be published soon (the link to be provided).

License
-----------
The code is published under the GNU General Public License v3.0.

References
-----------
* Nooijen, Marcel. "Can the eigenstates of a many-body hamiltonian be represented exactly using a general two-body cluster expansion?." Physical review letters 84.10 (2000): 2108.
* Lee, Joonho, et al. "Generalized unitary coupled cluster wave functions for quantum computation." Journal of chemical theory and computation 15.1 (2018): 311-324.
* Grimsley, Harper R., et al. "An adaptive variational algorithm for exact molecular simulations on a quantum computer." Nature communications 10.1 (2019): 1-9.
* Tang, Ho Lun, et al. "qubit-adapt-vqe: An adaptive algorithm for constructing hardware-efficient ansätze on a quantum processor." PRX Quantum 2.2 (2021): 020310.
* Xia, Rongxin, and Sabre Kais. "Qubit coupled cluster singles and doubles variational quantum eigensolver ansatz for electronic structure calculations." Quantum Science and Technology 6.1 (2020): 015001.
* Shkolnikov, V. O., et al. "Avoiding symmetry roadblocks and minimizing the measurement overhead of adaptive variational quantum eigensolvers." arXiv preprint arXiv:2109.05340 (2021).
