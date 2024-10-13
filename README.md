OpenVQE: README
=======================

`OpenVQE` is an Open Source Variational Quantum Eigensolver extension of the Quantum Learning Machine to  Quantum Chemistry. It was developed based on the tools of `myqlm-fermion` (https://github.com/myQLM/myqlm-fermion.git).

Check the main OpenVQE paper:
Open Source Variational Quantum Eigensolver Extension of the Quantum Learning Machine (QLM) for Quantum Chemistry. 
M. Haidar,  M. J. Rančić, T. Ayral, Y. Maday, J.-P. Piquemal, WIREs Comp. Mol. Sci., 2023, e1664 (Open Access)
DOI: 10.1002/wcms.1664

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
git checkout alpha
pip install .
pip install -r requirements.txt
```

## Troubleshooting

If you have a problem with the installation we recommand you to use conda:

```shell
conda create --name openvqe python=3.11
conda activate openvqe
## Repeat the installation steps above
git clone https://github.com/OpenVQE/OpenVQE.git
cd OpenVQE
git checkout alpha
pip install .
pip install -r requirements.txt
```

## Contributing to the Package

Note: OpenVQE is distributed under the MIT license. By contributing code to the package, 
you agree that your work will be licensed under the MIT license.

1. Click on the fork button.
2. Deselect the checkbox saying "Copy the main branch only".
3. Click on "Choose an owner" and select a github profile.
4. Click on create fork.
5. Push your changes to your forked repository.
6. Open a pull request (PR) from your forked repository to the alpha branch of the OpenVQE repository.
7. Send an email to Mohammad Haidar(mohammadhaidar2016@outlook.com) with Nathan Vaneberg in cc(nathanvaneberg@gmail.com), including a link to your pull request, a description of your changes and a confirmation that your contribution will be licensed under the MIT license.

Move to the next section for getting started!

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
For more information, please refer to our paper: 
> *Open Source Variational Quantum Eigensolver Extension of the Quantum Learning Machine (QLM) for Quantum Chemistry. 
M. Haidar,  M. J. Rančić, T. Ayral, Y. Maday, J.-P. Piquemal, WIREs Comp. Mol. Sci., 2023, e1664 (Open Access)
DOI: 10.1002/wcms.1664*

How to cite
-----------
> *Mohammad Haidar, Marko J. Ranˇci´c, Thomas Ayral, Yvon Maday, and Jean-Philip Piquemal. Open source variational quantum eigensolver extension of the quantum learning machine for quantum chemistry. WIREs Computational Molecular Science, page e1664, 2023*

Getting in touch
-----------
For any question about OpenVQE or my research, don't hesitate to get in touch: mohammadhaidar2016@outlook.com

License
-----------
OpenVQE was created by Mohammad Haidar. It is licensed under the terms of the MIT License.

Thanks
-----------

We would like to thank Gopal Dahale, a Master's student in Quantum Science at EPFL, for his integration of the lithium batteries application into the repository. You can check out his GitHub here: https://github.com/Gopal-Dahale.

References
-----------
* Nooijen, Marcel. "Can the eigenstates of a many-body hamiltonian be represented exactly using a general two-body cluster expansion?." Physical review letters 84.10 (2000): 2108.
* Lee, Joonho, et al. "Generalized unitary coupled cluster wave functions for quantum computation." Journal of chemical theory and computation 15.1 (2018): 311-324.
* Grimsley, Harper R., et al. "An adaptive variational algorithm for exact molecular simulations on a quantum computer." Nature communications 10.1 (2019): 1-9.
* Tang, Ho Lun, et al. "qubit-adapt-vqe: An adaptive algorithm for constructing hardware-efficient ansätze on a quantum processor." PRX Quantum 2.2 (2021): 020310.
* Xia, Rongxin, and Sabre Kais. "Qubit coupled cluster singles and doubles variational quantum eigensolver ansatz for electronic structure calculations." Quantum Science and Technology 6.1 (2020): 015001.
* Shkolnikov, V. O., et al. "Avoiding symmetry roadblocks and minimizing the measurement overhead of adaptive variational quantum eigensolvers." arXiv preprint arXiv:2109.05340 (2021).
