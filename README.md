OpenVQE: README
=======================

**OpenVQE** is an open-source extension of the Variational Quantum Eigensolver (VQE) for quantum chemistry, building on the Quantum Learning Machine (QLM) and developed using tools from [`myqlm-fermion`](https://github.com/myQLM/myqlm-fermion.git). It enhances QLM's capabilities in quantum chemistry computations.


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

3. **Applications**: Includes practical quantum computing applications, such as calculating the ground state energy of battery materials.


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

To explore the quantum battery application, you will need to install CUDA and have an NVIDIA GPU. For detailed instructions, refer to the    [NVIDIA CUDA Quantum GitHub page](https://github.com/NVIDIA/cuda-quantum).

## Contributing to the Package

OpenVQE is distributed under the MIT license. By contributing, you agree to license your work under MIT. Here's how to contribute:

- Go to the OpenVQE main page and click the fork button.

![alt text](images/image-6.png)
- Deselect "Copy the main branch only".
- Click on "Choose an owner" and select a github profile.
- Click on create fork.
- Open a terminal.
- Follow the installation protocol above.
- Create a new branch for development: 
```shell
git branch my_amazing_application origin/master`
git checkout my_amazing_application`
```
- Add your new amazing functionnalities and push your changes: 
```shell
git push origin HEAD
```
- Open a pull request (PR) to the alpha branch of the OpenVQE repository.
    - Go to your github fork.
    - Click on the contribute button.

    ![alt text](images/image.png)
    - Click on open pull request.

    ![alt text](images/image-2.png)
    - Open a pull request (PR) from your forked repository to the alpha branch of the OpenVQE repository.

    ![alt text](images/image-3.png)
    - Click on create pull request.

    ![alt text](images/image-4.png)
- Finally, send an email to Mohammad Haidar (mohammadhaidar2016@outlook.com) with Nathan Vaneberg (nathanvaneberg@gmail.com) cc'd. Include a link to your PR, a description of your changes, and confirm your contribution will be licensed under MIT.

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
For any questions regarding OpenVQE or related research, contact: mohammadhaidar2016@outlook.com.

License
-----------
OpenVQE is created by Mohammad Haidar and licensed under the MIT License.

Thanks
-----------

Special thanks to Gopal Dahale, a Master's student in Quantum Science at EPFL, for integrating the lithium battery application into the repository. Check out his [GitHub](https://github.com/Gopal-Dahale).

References
-----------
* Nooijen, Marcel. "Can the eigenstates of a many-body hamiltonian be represented exactly using a general two-body cluster expansion?." Physical review letters 84.10 (2000): 2108.
* Lee, Joonho, et al. "Generalized unitary coupled cluster wave functions for quantum computation." Journal of chemical theory and computation 15.1 (2018): 311-324.
* Grimsley, Harper R., et al. "An adaptive variational algorithm for exact molecular simulations on a quantum computer." Nature communications 10.1 (2019): 1-9.
* Tang, Ho Lun, et al. "qubit-adapt-vqe: An adaptive algorithm for constructing hardware-efficient ansätze on a quantum processor." PRX Quantum 2.2 (2021): 020310.
* Xia, Rongxin, and Sabre Kais. "Qubit coupled cluster singles and doubles variational quantum eigensolver ansatz for electronic structure calculations." Quantum Science and Technology 6.1 (2020): 015001.
* Shkolnikov, V. O., et al. "Avoiding symmetry roadblocks and minimizing the measurement overhead of adaptive variational quantum eigensolvers." arXiv preprint arXiv:2109.05340 (2021).
