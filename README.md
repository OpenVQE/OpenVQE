OpenVQE: README
=======================

`OpenVQE` is an Open Source Variational Quantum Eigensolver extension of the Quantum Learning Machine to  Quantum Chemistry. It was developed based on the tools of `myqlm-fermion` 


It consists of two main modules as follows:

- UCC Family: this module consists of different classes and functions to generate the fermionic cluster operators (fermionic pool) and the  qubit pools and to get the optimized energies from VQE in the case of active and non-active orbital selections. For example, UCCSD, QUCCSD, UCCGSD, K-UpCCGSD, spin-complemented pair, singlet and doublet generalized single and double excitations, etc.

- ADAPT: consists of two sub-modules which are: 
    - Fermionic-ADAPT: it contains functions that performs the fermionic ADAPT-VQE algorthmic steps  in the active and non-active space selections.
    - Qubit-ADAPT: it  contains functions that perform the Qubit ADAPT-VQE algorithmic steps calculation in the active and non-active space orbital selections.


Installation
--------------

install OpenVQE from source:
```shell
git clone https://github.com/OpenVQE/OpenVQE.git
cd OpenVQE
pip install .
```
Move to the next section for getting started!

## If you want to contribute to the package

Install the depencies of OpenVQE
```shell
pip install -r requirements.txt
```


Getting started
----------------
### Notebooks

Jupyter notebooks are available in the "notebooks" folder.

### Hello world example


```shell
from openvqe.ucc import ...

```

Documentation
---------------
The code is based on the well documented code of `myqlm-fermion` framework [link](https://myqlm.github.io/).
The main functions are documented in the code base.
For more information, please refer to our paper that wil be published soon (the link to be provided).

License
-----------
The code is published under MIT LICENSE.

References
-----------
* Nooijen, Marcel. "Can the eigenstates of a many-body hamiltonian be represented exactly using a general two-body cluster expansion?." Physical review letters 84.10 (2000): 2108.
* Lee, Joonho, et al. "Generalized unitary coupled cluster wave functions for quantum computation." Journal of chemical theory and computation 15.1 (2018): 311-324.
* Grimsley, Harper R., et al. "An adaptive variational algorithm for exact molecular simulations on a quantum computer." Nature communications 10.1 (2019): 1-9.
* Tang, Ho Lun, et al. "qubit-adapt-vqe: An adaptive algorithm for constructing hardware-efficient ansätze on a quantum processor." PRX Quantum 2.2 (2021): 020310.
* Xia, Rongxin, and Sabre Kais. "Qubit coupled cluster singles and doubles variational quantum eigensolver ansatz for electronic structure calculations." Quantum Science and Technology 6.1 (2020): 015001.
* Shkolnikov, V. O., et al. "Avoiding symmetry roadblocks and minimizing the measurement overhead of adaptive variational quantum eigensolvers." arXiv preprint arXiv:2109.05340 (2021).
