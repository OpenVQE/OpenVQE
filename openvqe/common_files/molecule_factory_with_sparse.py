import numpy as np
import scipy
from numpy import binary_repr
from qat.fermion import ElectronicStructureHamiltonian
from qat.fermion.chemistry.pyscf_tools import perform_pyscf_computation
from qat.fermion.chemistry.ucc import (
    convert_to_h_integrals,
    transform_integrals_to_new_basis,
)
from qat.fermion.chemistry.ucc_deprecated import (
    get_active_space_hamiltonian,
    get_cluster_ops_and_init_guess,
)
from qat.fermion.transforms import (
    get_bk_code,
    get_jw_code,
    get_parity_code,
    recode_integer,
    transform_to_bk_basis,
    transform_to_jw_basis,
    transform_to_parity_basis,
)

from .generator_excitations import (
    singlet_gsd,
    singlet_sd,
    singlet_upccgsd,
    spin_complement_gsd,
    spin_complement_gsd_twin,
    uccsd,
)


class MoleculeFactory:
    """
    A class used to represent the molecule  which consists of several main methods: get_parameters, generate_hamiltonian, calculate_uccsd, generate_cluster_ops.
    """

    def get_parameters(self, molecule_symbol):
        """
        This function is responsible for representing the molecule returning its characteristics.

        Parameters
        ----------
        molecule_symbol : string
            the sumbol of the molecule
        Returns
        -------
        r: float
            bone length
        geometry: list<Tuple>
            geometry representation of a molecule (x,y,z) format.
        charge: int
            charge of the molecule
        spin: int
            spin of the molecule
        basis: string
            chemical basis set of the molecule
        """
        if molecule_symbol == "LIH":
            r = 1.45
            geometry = [("Li", (0, 0, 0)), ("h", (0, 0, r))]
            charge = 0
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "H2":
            r = 0.75
            geometry = [("h", (0, 0, 0)), ("h", (0, 0, r))]
            charge = 0
            spin = 0
            #             basis = 'cc-pVDZ'
            basis = "6-31g"

        elif molecule_symbol == "HeH+":
            r = 1.0
            geometry = [("he", (0, 0, 0)), ("h", (0, 0, r))]
            charge = 1
            spin = 0
            basis = "6-31g"
        elif molecule_symbol == "HD+":
            r = 0.75
            geometry = [("h", (0, 0, 0)), ("h", (0, 0, r))]
            charge = 1
            spin = 1
            basis = "6-31g"
        elif molecule_symbol == "H4":
            # h4
            r = 0.85
            geometry = [
                ("h", (0, 0, 0)),
                ("h", (0, 0, 1 * r)),
                ("h", (0, 0, 2 * r)),
                ("h", (0, 0, 3 * r)),
            ]
            charge = 0
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "H6":
            r = 1.5
            geometry = [
                ("h", (0, 0, 0)),
                ("h", (0, 0, 1 * r)),
                ("h", (0, 0, 2 * r)),
                ("h", (0, 0, 3 * r)),
                ("h", (0, 0, 4 * r)),
                ("h", (0, 0, 5 * r)),
            ]
            charge = 0
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "H8":
            r = 1.0
            geometry = [
                ("h", (0, 0, 0)),
                ("h", (0, 0, 1 * r)),
                ("h", (0, 0, 2 * r)),
                ("h", (0, 0, 3 * r)),
                ("h", (0, 0, 4 * r)),
                ("h", (0, 0, 5 * r)),
                ("h", (0, 0, 6 * r)),
                ("h", (0, 0, 7 * r)),
            ]
            charge = 0
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "H10":
            r = 1.0
            geometry = [
                ("h", (0, 0, 0)),
                ("h", (0, 0, 1 * r)),
                ("h", (0, 0, 2 * r)),
                ("h", (0, 0, 3 * r)),
                ("h", (0, 0, 4 * r)),
                ("h", (0, 0, 5 * r)),
                ("h", (0, 0, 6 * r)),
                ("h", (0, 0, 7 * r)),
                ("h", (0, 0, 8 * r)),
                ("h", (0, 0, 9 * r)),
            ]
            charge = 0
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "BeH2":
            r = 1.4
            geometry = [("Be", (0, 0, 0 * r)), ("h", (0, 0, r)), ("h", (0, 0, -r))]
            charge = 0
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "H2O":
            r = 1.0285
            theta = 0.538 * np.pi
            geometry = [
                ("O", (0, 0, 0 * r)),
                ("h", (0, 0, r)),
                ("h", (0, r * np.sin(np.pi - theta), r * np.cos(np.pi - theta))),
            ]
            charge = 0
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "NH3":
            r = 1.0703
            theta = (100.107 / 180) * np.pi
            geometry = [
                ("N", (0, 0, 0 * r)),
                (
                    "h",
                    (
                        0,
                        2 * (np.sin(theta / 2) / np.sqrt(3)) * r,
                        np.sqrt(1 - 4 * np.sin(theta / 2) ** 2 / 3) * r,
                    ),
                ),
                (
                    "h",
                    (
                        np.sin(theta / 2) * r,
                        -np.sin(theta / 2) / np.sqrt(3) * r,
                        np.sqrt(1 - 4 * np.sin(theta / 2) ** 2 / 3) * r,
                    ),
                ),
                (
                    "h",
                    (
                        -np.sin(theta / 2) * r,
                        -np.sin(theta / 2) / np.sqrt(3) * r,
                        np.sqrt(1 - 4 * np.sin(theta / 2) ** 2 / 3) * r,
                    ),
                ),
            ]
            charge = 0
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "HF":
            r = 1.0
            geometry = [("F", (0, 0, 0 * r)), ("h", (0, 0, r))]
            charge = 0
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "HO":
            r = 1.8
            geometry = [("h", (0, 0, 0 * r)), ("O", (0, 0, 1 * r))]
            charge = -1
            spin = 0
            basis = "sto-3g"
        elif molecule_symbol == "CO2":
            r = 1.22
            geometry = [
                ["C", [0.0, 0.0, 8.261342997000753e-07]],
                [
                    "O",
                    [
                        1.0990287608769004e-18,
                        2.7114450405987004e-19,
                        1.2236575813458745,
                    ],
                ],
                [
                    "O",
                    [
                        2.696319376811295e-22,
                        2.4247676462727696e-23,
                        -1.2236561920609494,
                    ],
                ],
            ]
            basis = "sto-3g"
            spin = 0
            charge = 0
        return r, geometry, charge, spin, basis

    def generate_hamiltonian(
        self, molecule_symbol, active=False, transform="JW", display=True
    ):
        """
        This function is responsible for generating the hamiltonian of the molecule in the fermionic and the spin representation.
        It operates in the active and non-active space selections.

        Parameters
        ----------
        molecule_symbol : string
            the sumbol of the molecule
        active: bool
            False if the non-active space is considered
            True if the active space is considered
        transform: string
            type of transformation
        display: bool
            False if no need to print the steps for debugging
            True if debugging is needed (so print the steps)
        Returns
        -------
        In case of active space:
            h_active: Hamiltonian
                electronic structure hamiltonian in the active space selection
            h_active_sparse: np.array
                Active space hamiltonian matrix in the sparse representation
            h_active_sp: Hamiltonian
                Active space hamiltonian in the spin representation
            h_active_sp_sparse: Hamiltonian
                Active space sparsed hamiltonian in the spin representation
            nb_active_els: int
                number of electrons in the active space
            active_noons: list[float]
                list of noons in the active space
            active_orb_energies: list[float]
                list of orbital energies in the active space
            info: dict (dict of str: float)
                dictionary of HF, CCSD, FCI

        In case of non-active space:
            h: Hamiltonian
                electronic structure hamiltonian
            hamiltonian_sparse: np.array
                Hamiltonian matrix in the sparse representation
            hamiltonian_sp: Hamiltonian
                Hamiltonian in the spin representation
            hamiltonian_sp_sparse: Hamiltonian
                Sparse hamiltonian in the spin representation
            n_elec: int
                number of electrons
            noons_full: list[float]
                list of noons
            orb_energies_full: list[float]
                list of orbital energies
            info: dict (dict of str:float)
                dictionary of HF, CCSD, FCI
        """
        r, geometry, charge, spin, basis = self.get_parameters(molecule_symbol)
        (
            rdm1,
            orbital_energies,
            nuclear_repulsion,
            n_elec,
            one_body_integrals,
            two_body_integrals,
            info,
        ) = perform_pyscf_computation(
            geometry=geometry, basis=basis, spin=spin, charge=charge, run_fci=True
        )
        if display:
            print("Number of electrons = ", n_elec)
            print(
                "Number of qubits before active space selection = ", rdm1.shape[0] * 2
            )
            # print("rdm1", rdm1)
            # print(info)
            print("Orbital energies = ", orbital_energies)
            print("Nuclear repulsion = ", nuclear_repulsion)

        # nuclear_repulsion = molbasis.nuclear_repulsion

        hpq, hpqrs = convert_to_h_integrals(one_body_integrals, two_body_integrals)
        if not active:

            hamiltonian = ElectronicStructureHamiltonian(
                hpq, hpqrs, constant_coeff=nuclear_repulsion
            )
            noons, basis_change = np.linalg.eigh(rdm1)
            noons = list(reversed(noons))
            if display:
                print("Noons = ", noons)
            noons_full, orb_energies_full = [], []
            for ind in range(len(noons)):
                noons_full.extend([noons[ind], noons[ind]])
                orb_energies_full.extend([orbital_energies[ind], orbital_energies[ind]])
            hamiltonian_sp = None
            if transform == "JW":
                trafo = transform_to_jw_basis
                hamiltonian_sp = trafo(hamiltonian)
            elif transform == "Bravyi-Kitaev":
                trafo= transform_to_bk_basis
                hamiltonian_sp = trafo(hamiltonian)
            elif transform == "parity_basis":
                trafo = transform_to_parity_basis
                hamiltonian_sp = trafo(hamiltonian)
            hamiltonian_sparse = hamiltonian.get_matrix(sparse=True)
            hamiltonian_sp_sparse = hamiltonian_sp.get_matrix(sparse=True)
            return (
                hamiltonian,
                hamiltonian_sparse,
                hamiltonian_sp,
                hamiltonian_sp_sparse,
                n_elec,
                noons_full,
                orb_energies_full,
                info,
            )

        noons, basis_change = np.linalg.eigh(rdm1)
        noons = list(reversed(noons))
        if display:
            print("Noons = ", noons)

        basis_change = np.flip(basis_change, axis=1)
        one_body_integrals, two_body_integrals = transform_integrals_to_new_basis(
            one_body_integrals, two_body_integrals, basis_change
        )

        threshold_1 = 2 - noons[0]
        if len(noons) < 3:
            threshold_2 = 0.01
        else:
            threshold_2 = noons[-1]
        if display:
            print("threshold_1 chosen = ", threshold_1)
            print("threshold_2 chosen = ", threshold_2)
        hamiltonian_active, active_inds, occ_inds = get_active_space_hamiltonian(
            one_body_integrals,
            two_body_integrals,
            noons,
            n_elec,
            nuclear_repulsion,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
        )
        if display:
            print(
                "Number of qubits after active space selection =",
                hamiltonian_active.nbqbits,
            )

        active_noons, active_orb_energies = [], []
        for ind in active_inds:
            active_noons.extend([noons[ind], noons[ind]])
            active_orb_energies.extend([orbital_energies[ind], orbital_energies[ind]])
        nb_active_els = n_elec - 2 * len(occ_inds)
        if display:
            print("length of active noons: ", len(active_noons))
            print("length of orbital energies: ", len(active_orb_energies))
        # import numpy as np
        # eigen,_ = np.linalg.eigh(hamiltonian_active.get_matrix())
        # print("eigen",eigen)

        # print("eigen",min(eigen))
        hamiltonian_active_sp = None
        if transform == "JW":
            trafo = transform_to_jw_basis
            hamiltonian_active_sp = trafo(hamiltonian_active)
        elif transform == "Bravyi-Kitaev":
            trafo = transform_to_bk_basis
            hamiltonian_active_sp = trafo(hamiltonian_active)
        elif transform == "parity_basis":
            trafo = transform_to_parity_basis
            hamiltonian_active_sp = trafo(hamiltonian_active)
        hamiltonian_active_sparse = hamiltonian_active.get_matrix(sparse=True)
        hamiltonian_active_sp_sparse = hamiltonian_active_sp.get_matrix(sparse=True)
        return (
            hamiltonian_active,
            hamiltonian_active_sparse,
            hamiltonian_active_sp,
            hamiltonian_active_sp_sparse,
            nb_active_els,
            active_noons,
            active_orb_energies,
            info,
        )

    def calculate_uccsd(self, molecule_symbol, transform, active):
        """
        This function is responsible for constructing the cluster operators, the MP2 initial guesses variational parameters of the UCCSD ansatz. It computes the cluster operators in the spin representation and the size of the pool.

        Parameters
        ----------
        molecule_symbol : string
            the sumbol of the molecule
        transform : string
            type of transformation. Either 'JW', 'Bravyi-Kitaev', or 'parity_basis'
        active : bool
            False if the non-active space is considered
            True if the active space is considered

        Returns
        -------
        pool_size: int
            The number of the cluster operators
        cluster_ops: list[Hamiltonian]
            list of fermionic cluster operators
        cluster_ops_sp: list[Hamiltonian]
            list of spin cluster operators
        theta_MP2: list[float]
            list of parameters in the MP2 pre-screening process
        hf_init: int
            the integer corresponding to the occupation of the Hartree-Fock solution
        """
        if not active:
            (
                hamiltonian,
                hamiltonian_sparse,
                hamiltonian_sp,
                hamiltonian_sp_sparse,
                n_elec,
                noons_full,
                orb_energies_full,
                info,
            ) = self.generate_hamiltonian(
                molecule_symbol, active=False, transform=transform, display=False
            )
            pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init = uccsd(
                hamiltonian, n_elec, noons_full, orb_energies_full, transform
            )
            return pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init
        else:
            (
                hamiltonian_active,
                hamiltonian_active_sparse,
                hamiltonian_active_sp,
                hamiltonian_active_sp_sparse,
                nb_active_els,
                active_noons,
                active_orb_energies,
                info,
            ) = self.generate_hamiltonian(
                molecule_symbol, active=True, transform=transform
            )
            pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init = uccsd(
                hamiltonian_active,
                nb_active_els,
                active_noons,
                active_orb_energies,
                transform,
            )
            return pool_size, cluster_ops, cluster_ops_sp, theta_MP2, hf_init

    def find_hf_init(self, hamiltonian, n_elec, noons_full, orb_energies_full):
        _, _, hf_init = get_cluster_ops_and_init_guess(
            n_elec, noons_full, orb_energies_full, hamiltonian.hpqrs
        )
        return hf_init

    def generate_cluster_ops(
        self, molecule_symbol, type_of_generator, transform, active=False
    ):
        """
        This function computes the cluster operators and its size in the fermionic and spin representations for UCC family ansatz in the active and non-active space selection for any molecule within its basis sets. It also generates the cluster operators in the matrix representation using sparse method.

        Parameters
        ----------
        molecule_symbol : string
            the sumbol of the molecule
        type_of_generator: string
            the type of excitation generator the user needs
        transform : string
            type of transform of the molecule
        active : bool
            False if the non-active space is considered
            True if the active space is considered

        Returns
        -------

        In the case of UCCSD:
            pool_size: int
                size of the pool
            cluster_ops: list[Any]
                list of fermionic cluster operators
            cluster_ops_sp: list[Any]
                list of spin cluster operators
            cluster_ops_sparse: np.ndarray
                matrix representation of the cluster operator
            theta_MP2: list[float]
                list of parameters in the MP2 screening process
            hf_init: int
                the integer coressponding to the occupation of the Hartree-Fock solution
        
        In any of the other cases:
            pool_size: int
                size of the pool
            cluster_ops: list[Any]
                list of fermionic cluster operators
            cluster_ops_sp: list[Any]
                list of spin cluster operators
            cluster_ops_sparse: np.ndarray
                matrix representation of the cluster operator
        """

        r, geometry, charge, spin, basis = self.get_parameters(molecule_symbol)
        (
            rdm1,
            orbital_energies,
            nuclear_repulsion,
            n_elec,
            one_body_integrals,
            two_body_integrals,
            info,
        ) = perform_pyscf_computation(
            geometry=geometry, basis=basis, spin=spin, charge=charge, run_fci=True
        )

        orbital_number = len(orbital_energies)

        if active:
            (
                hamiltonian_active,
                hamiltonian_active_sparse,
                hamiltonian_active_sp,
                hamiltonian_active_sp_sparse,
                nb_active_els,
                active_noons,
                active_orb_energies,
                info,
            ) = self.generate_hamiltonian(
                molecule_symbol, active=True, transform=transform, display=False
            )
            # print('here: ', len(active_orb_energies))
            orbital_number = int(len(active_orb_energies) / 2)
            n_elec = nb_active_els

        # orb_energies_full = orbital_energies
        pool_size, cluster_ops, cluster_ops_sp = None, None, None
        if type_of_generator == "singlet_sd":
            pool_size, cluster_ops, cluster_ops_sp = singlet_sd(
                n_elec, orbital_number, transform
            )
        elif type_of_generator == "singlet_gsd":
            pool_size, cluster_ops, cluster_ops_sp = singlet_gsd(
                n_elec, orbital_number, transform
            )
        elif type_of_generator == "spin_complement_gsd":
            pool_size, cluster_ops, cluster_ops_sp = spin_complement_gsd(
                n_elec, orbital_number, transform
            )
        elif type_of_generator == "spin_complement_gsd_twin":
            pool_size, cluster_ops, cluster_ops_sp = spin_complement_gsd_twin(
                n_elec, orbital_number, transform
            )
        elif type_of_generator == "sUPCCGSD":
            pool_size, cluster_ops, cluster_ops_sp = singlet_upccgsd(
                orbital_number, transform
            )
        elif type_of_generator == "UCCSD":
            (
                pool_size,
                cluster_ops,
                cluster_ops_sp,
                theta_MP2,
                hf_init,
            ) = self.calculate_uccsd(molecule_symbol, transform, active=False)
            cluster_ops_sparse = []
            for i in cluster_ops_sp:
                cluster_ops_sparse.append(i.get_matrix(sparse=True))
            return (
                pool_size,
                cluster_ops,
                cluster_ops_sp,
                cluster_ops_sparse,
                theta_MP2,
                hf_init,
            )
        else:
            return
        # print('Result of the generate_cluster_ops: ', pool_size,cluster_ops,cluster_ops_sp)
        cluster_ops_sparse = []
        for i in cluster_ops_sp:
            cluster_ops_sparse.append(i.get_matrix(sparse=True))
        # cluster_ops_sparse = cluster_ops_sp.get_matrix(sparse=True)
        return pool_size, cluster_ops, cluster_ops_sp, cluster_ops_sparse

    # TESTING hOW TO GET ThE REFERENCE KET
    def get_reference_ket(self, hf_init, nbqbits, transform):
        if transform == "JW":
            hf_init_sp = recode_integer(hf_init, get_jw_code(nbqbits))
        elif transform == "Bravyi-Kitaev":
            hf_init_sp = recode_integer(hf_init, get_bk_code(nbqbits))
        elif transform == "parity_basis":
            hf_init_sp = recode_integer(hf_init, get_parity_code(nbqbits))
        ket_hf = binary_repr(hf_init_sp)
        list_ket_hf = [int(c) for c in ket_hf]
        reference_state = self.from_ket_to_vector(list_ket_hf)
        sparse_reference_state = scipy.sparse.csr_matrix(
            reference_state, dtype=complex
        ).transpose()
        return sparse_reference_state, hf_init_sp

    def from_ket_to_vector(self, ket):
        state_vector = [1]
        for i in ket:
            qubit_vector = [not i, i]
            state_vector = np.kron(state_vector, qubit_vector)
        return state_vector
