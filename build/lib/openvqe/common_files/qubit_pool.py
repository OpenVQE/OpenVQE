import numpy as np
import itertools
from qat.core import Term
from qat.fermion.transforms import transform_to_jw_basis
from qat.fermion import SpinHamiltonian as Hamiltonian

class QubitPool:
    """
    This class contains functions that are responsible for generating the qubit cluster operators in a pool
    from two main functions: 'generate_pool_from_cluster' and 'generate_pool_without_cluster'.
    The first function takes as parameters the pool name and return the pool with its size.
    The pools that are accessible for the user are as follows: full, full_without_Z and reduced_without_Z.
    However the function generate_pool_without_cluster() takes as parameters the type of pool, pool generated from cluster operators and returns
    the pool with its size. The type of pool that are considered as options for the user are as follows:
    - YXXX
    - XYXX
    - XXYX
    - XXXY
    - random
    - two
    - four
    - eight
    - without_Z_from_generator
    - minimal
    - pure_with_symmetry 
    Note that the other functions in this class are helper functions.

    """
    def generate_pool(self, cluster_ops):
        """
        generate the qubitPool from cluster_ops which is equivalent to getting cluster_ops_sp
        
        Parameters
        ----------
        cluster_ops: list[Hamiltonian]
            list of fermionic cluster operators        

        Returns
        ----------
        qubit_pool: list[Hamiltonian]
            list of spin cluster operators    
        
        """
        qubit_pool = []
        for i in cluster_ops:
            qubit_op = transform_to_jw_basis(i)
            qubit_pool.append(qubit_op)
        return qubit_pool

    
    def extract_terms(self, qubit_pool):
        """
        Extract terms from qubitPool such as Pauli strings and number of qbits.
        
        Parameters
        ----------
        qubit_pool: list[Hamiltonian]
            list of spin cluster operators  

        Returns
        ----------
        terms: list[string]
            list of extracted terms for each cluster operator 
        
        """

        terms = list()
        for qubit_op in qubit_pool:
            for term in qubit_op.terms:
                term_letters = term.op
                term_qbits = term.qbits
                term_temp = "["
                for index in range(len(term_letters)):
                    term_temp += "%s%s " % (term_letters[index], term_qbits[index])
                term_temp = term_temp[:-1]
                term_temp += "]"
                if term_temp not in terms:
                    terms.append(term_temp)
        return terms

    
    def extract_qubits_operators(self, terms):
        """
        get the qubits and the operators from the terms

        Parameters
        ----------
        terms: list[string]
            list of terms for each cluster operator

        Returns
        ----------
        list_digits: List<List<int>>
            Extracted digits for each term

        list_letters: List<List<string>>
            Extracted letters for each term

        """

        list_digits = []
        list_letters = []

        for term in terms:
            digits = []
            letters = ""
            for char_ind in range(len(term)):
                if term[char_ind].isdigit() and term[char_ind + 1].isdigit():
                    two_digits = "%s%s" % (term[char_ind], term[char_ind + 1])
                    digits.append(int(two_digits))

                elif term[char_ind].isdigit() and not term[char_ind - 1].isdigit():
                    digits.append(int(term[char_ind]))
                elif term[char_ind].isalpha():
                    letters += term[char_ind]
            list_digits.append(digits)
            list_letters.append(letters)
        return list_digits, list_letters

    
    def terms_to_hamiltonian(self, terms, nbqbits):
        """
        transform the qubits and terms to hamiltonian operators, which is the accepted input format for
        generator excitations.


        Parameters
        ----------
        terms: list[string]
            list of terms for each cluster operator
        
        nbqbits: int
            The number of qbits

        Returns
        ----------
        list_hamiltonian: List<Hamiltonian>
            list of spin cluster operators
        
        """
        list_digits, list_letters = self.extract_qubits_operators(terms)
        list_hamiltonian = []
        for i in range(len(list_digits)):
            hamiltonian = Hamiltonian(
                nbqbits, [Term(-1.0, list_letters[i], list_digits[i])]
            )
            list_hamiltonian.append(hamiltonian)
        return list_hamiltonian

    def extract_qubits_operators_without_z(self, terms):
        """
        same as 'extract_qubits_operators' but ommiting the Z Pauli terms

        Parameters
        ----------
        terms: list[string]
            list of terms for each cluster operator

        Returns
        ----------
        list_digits: List<List<int>>
            Extracted digits for each term

        list_letters: List<List<string>>
            Extracted letters for each term

        """

        list_digits = []
        list_letters = []
        skip = False

        for term in terms:
            digits = []
            letters = ""
            for char_ind in range(len(term)):
                if term[char_ind].isdigit() and term[char_ind + 1].isdigit():
                    two_digits = "%s%s" % (term[char_ind], term[char_ind + 1])
                    if skip:
                        skip = False
                        continue
                    digits.append(int(two_digits))

                elif term[char_ind].isdigit() and not term[char_ind - 1].isdigit():
                    if skip:
                        skip = False
                        continue
                    digits.append(int(term[char_ind]))
                elif term[char_ind].isalpha():
                    if term[char_ind] == "Z":
                        skip = True
                        continue
                    letters += term[char_ind]
            list_digits.append(digits)
            list_letters.append(letters)
        return list_digits, list_letters

    def extract_terms_without_z(self, terms):
        """
        Omits all the Z Pauli terms.
        
        Parameters
        ----------
        terms: list[string]
            list of extracted terms for each cluster operator 

        Returns
        ----------
        terms: list[string]
            list of extracted terms for each cluster operator with Z Pauli terms omitted
        
        """

        list_digits, list_letters = self.extract_qubits_operators_without_z(terms)
        terms_z = list()
        index = 0
        for i in range(len(list_digits)):
            term_temp = "["
            for letter in list_letters[i]:

                term_temp += "%s%s " % (letter, list_digits[i][index])
                index += 1
            term_temp = term_temp[:-1]
            term_temp += "]"
            if term_temp not in terms_z:
                terms_z.append(term_temp)
            index = 0
        terms_z = list(terms_z)
        return terms_z


    
    def generate_reduced_qubit_pool(self, terms, nbqbits):
        """
        generate the qubit pools discarding the Z-string responsible for the anticommutation of fermions
        and only keeping the first string acting on each set of spin orbitals

        Parameters
        ----------
        terms: list[string]
            list of terms for each cluster operator
        
        nbqbits: int
            The number of qbits

        Returns
        ----------
        list_hamiltonian: List<Hamiltonian>
            list of spin cluster operators

        """
        list_digits, list_letters = self.extract_qubits_operators(terms)
        ## Reduced Qubit pool
        reduced_qubit_pool = []
        included = []
        for i in range(len(list_digits)):
            operators = list_letters[i]
            qubits = list_digits[i]
            indices = []
            terms = ""
            digits = []
            for j in range(len(list_digits[i])):
                operator = operators[j]
                qubit = qubits[j]
                if operator != "Z":
                    terms += operator
                    digits.append(qubit)
                    indices.append(qubit)

            new_operator_hamiltonian = Hamiltonian(nbqbits, [Term(-1.0, terms, digits)])
            if indices not in included:
                reduced_qubit_pool.append(new_operator_hamiltonian)
                included.append(indices)
        return reduced_qubit_pool

    ############### POOLS ####################
    # generate YXXX pool
    def generate_yxxx_pool(self, nbqbits):
        """
        generate the qubit pools containing the YXXX string with parity sum conditions.


        Parameters
        ----------
        nbqbits: int
            The number of qbits
        
        Returns
        ----------
        length: int
            Number of YXXX operators in a pool

        yxxx_pool: List<Hamiltonian>
            list of spin cluster operators
        
        """
        yxxx_pool = []
        for a, b in itertools.combinations(range(nbqbits), 2):
            parity = (a + b) % 2
            if parity == 0:
                terms = "YX"
                digits = [a, b]
                yxxx_pool.append(Hamiltonian(nbqbits, [Term(-1.0, terms, digits)]))
        for a, b, c, d in itertools.combinations(range(nbqbits), 4):
            paritySum = a % 2 + b % 2 + c % 2 + d % 2

            if paritySum % 2 == 0:
                terms = "YXXX"
                digits = [a, b, c, d]
                yxxx_pool.append(Hamiltonian(nbqbits, [Term(-1.0, terms, digits)]))
        return len(yxxx_pool), yxxx_pool

    # generate XYXX pool
    def generate_xyxx_pool(self, nbqbits):
        """
        generate the qubit pools containing the XYXX string with parity sum conditions.


        Parameters
        ----------
        nbqbits: int
            The number of qbits
        
        Returns
        ----------
        length: int
            Number of XYXX operators in a pool

        xyxx_pool: List<Hamiltonian>
            list of spin cluster operators
        
        """

        xyxx_pool = []
        for a, b in itertools.combinations(range(nbqbits), 2):
            parity = (a + b) % 2
            # print(a, b)
            if parity == 0:
                terms = "YX"
                digits = [a, b]
                xyxx_pool.append(Hamiltonian(nbqbits, [Term(-1.0, terms, digits)]))
        for a, b, c, d in itertools.combinations(range(nbqbits), 4):
            paritySum = a % 2 + b % 2 + c % 2 + d % 2

            if paritySum % 2 == 0:
                terms = "XYXX"
                digits = [a, b, c, d]
                xyxx_pool.append(Hamiltonian(nbqbits, [Term(-1.0, terms, digits)]))
        return len(xyxx_pool), xyxx_pool

    # generate XXYX pool
    def generate_xxyx_pool(self, nbqbits):
        """
        generate the qubit pools containing the XXYX string with parity sum conditions.


        Parameters
        ----------
        nbqbits: int
            The number of qbits
        
        Returns
        ----------
        length: int
            Number of XXYX operators in a pool

        xxyx_pool: List<Hamiltonian>
            list of spin cluster operators
        
        """

        xxyx_pool = []
        for a, b in itertools.combinations(range(nbqbits), 2):
            parity = (a + b) % 2
            # print(a, b)
            if parity == 0:
                # yxxxPool.append(QubitOperator(((a, 'Y'), (b, 'X')), 1j))
                terms = "YX"
                digits = [a, b]
                xxyx_pool.append(Hamiltonian(nbqbits, [Term(-1.0, terms, digits)]))
        for a, b, c, d in itertools.combinations(range(nbqbits), 4):
            paritySum = a % 2 + b % 2 + c % 2 + d % 2

            if paritySum % 2 == 0:
                terms = "XXYX"
                digits = [a, b, c, d]
                xxyx_pool.append(Hamiltonian(nbqbits, [Term(-1.0, terms, digits)]))
        return len(xxyx_pool), xxyx_pool

    # generate XXXY pool
    def generate_xxxy_pool(self, nbqbits):
        """
        generate the qubit pools containing the XXXY string with parity sum conditions.


        Parameters
        ----------
        nbqbits: int
            The number of qbits
        
        Returns
        ----------
        length: int
            Number of XXXY operators in a pool

        xxxy_pool: List<Hamiltonian>
            list of spin cluster operators
        
        """

        xxxy_pool = []
        for a, b in itertools.combinations(range(nbqbits), 2):
            parity = (a + b) % 2
            # print(a, b)
            if parity == 0:
                # yxxxPool.append(QubitOperator(((a, 'Y'), (b, 'X')), 1j))
                terms = "YX"
                digits = [a, b]
                xxxy_pool.append(Hamiltonian(nbqbits, [Term(-1.0, terms, digits)]))
        for a, b, c, d in itertools.combinations(range(nbqbits), 4):
            paritySum = a % 2 + b % 2 + c % 2 + d % 2

            if paritySum % 2 == 0:
                terms = "XXXY"
                digits = [a, b, c, d]
                xxxy_pool.append(Hamiltonian(nbqbits, [Term(-1.0, terms, digits)]))
        return len(xxxy_pool), xxxy_pool

    # generate random pool
    def generate_random_pool(self, yxxx_pool, xyxx_pool, xxyx_pool, xxxy_pool):
        """
        generate randomly chosen qubit pools from YXXX, XYXX, XXYX, and XXXY pools

        Parameters
        ----------
        yxxx_pool: List<Hamiltonian>
            list of YXXX spin cluster operators 
        
        xyxx_pool: List<Hamiltonian>
            list of XYXX spin cluster operators 

        xxyx_pool: List<Hamiltonian>
            list of XXYX spin cluster operators 

        xxxy_pool: List<Hamiltonian>
            list of XXXY spin cluster operators 

        
        Returns
        ----------
        length: int
            Number of operators in the randomly generated pool

        random_pool: List<Hamiltonian>
            list of the randomly chosen spin cluster operators
        
        """
        random_pool = []
        string_options = [yxxx_pool, xyxx_pool, xxyx_pool, xxxy_pool]

        for i in range(len(xxxy_pool)):
            chosen = np.random.randint(0, 4)
            random_pool.append(string_options[chosen][i])

        return len(random_pool), random_pool

    ############# TWO FOUR EIGHT POOLS ###############

    # generate two pools
    def generate_two_pools(self, nbqbits):
        """
        generate the list of qubit pool operators where each operator is a sum of two Pauli strings
        associated with their qubit digits with parity conditions.

        Parameters
        ----------
        nbqbits: int
            The number of qbits
        
        Returns
        ----------
        length: int
            Number of two-pool operators

        two_pool: List<Hamiltonian>
            list of spin cluster operators
        
        """

        base_string = "XXYX"

        op1 = base_string[0]
        op2 = base_string[1]
        op3 = base_string[2]
        op4 = base_string[3]

        two_pool = []

        for a, b in itertools.combinations(range(nbqbits), 2):
            parity = (a + b) % 2

            if parity == 0:
                terms = "YX"
                digits = [a, b]
                operator = Hamiltonian(nbqbits, [Term(-1.0, terms, digits)])
                terms_z = "ZZ"
                digits_z = [a, b]

                terms_empty = "II"
                digits_empty = [a, b]
                z = Hamiltonian(
                    nbqbits,
                    [
                        Term(1.0, terms_empty, digits_empty),
                        Term(-1.0, terms_z, digits_z),
                    ],
                )
                two_pool.append(operator * z)

        for a, b, c, d in itertools.combinations(range(nbqbits), 4):
            parity_sum = a % 2 + b % 2 + c % 2 + d % 2

            if parity_sum % 2 == 0:
                terms = op1 + op2 + op3 + op4
                digits = [a, b, c, d]
                operator = Hamiltonian(nbqbits, [Term(-1.0, terms, digits)])
                terms_empty = "IIII"
                digits_empty = [a, b, c, d]
                terms_z = "ZZZZ"
                digits_z = [a, b, c, d]
                z = Hamiltonian(
                    nbqbits,
                    [
                        Term(1.0, terms_empty, digits_empty),
                        Term(+1.0, terms_z, digits_z),
                    ],
                )
                two_pool.append(operator * z)
        return len(two_pool), two_pool

    # generate four pools
    def generate_four_pools(self, nbqbits):
        """
        generate the list of qubit pool operators where each operator is a sum of four Pauli strings
        associated with their qubit digits with parity conditions.


        Parameters
        ----------
        nbqbits: int
            The number of qbits
        
        Returns
        ----------
        length: int
            Number of four-pool operators

        four_pool: List<Hamiltonian>
            list of spin cluster operators
        
        """

        base_string = "XXYX"

        op1 = base_string[0]
        op2 = base_string[1]
        op3 = base_string[2]
        op4 = base_string[3]

        four_pool = []

        for a, b in itertools.combinations(range(nbqbits), 2):
            parity = (a + b) % 2
            if parity == 0:
                terms = "YX"
                digits = [a, b]
                operator = Hamiltonian(nbqbits, [Term(-1.0, terms, digits)])
                terms_z = "ZZ"
                digits_z = [a, b]
                terms_empty = "II"
                digits_empty = [a, b]
                z = Hamiltonian(
                    nbqbits,
                    [
                        Term(-1.0, terms_empty, digits_empty),
                        Term(+1.0, terms_z, digits_z),
                    ],
                )

                four_pool.append(operator * z)

        for a, b, c, d in itertools.combinations(range(nbqbits), 4):
            parity_sum = a % 2 + b % 2 + c % 2 + d % 2

            if parity_sum % 2 == 0:
                terms = op1 + op2 + op3 + op4
                digits = [a, b, c, d]
                operator = Hamiltonian(nbqbits, [Term(-1.0, terms, digits)])

                terms_empty = "II"
                digits_empty = [a, b]
                terms_z = "ZZZZ"
                digits_z = [a, b, c, d]
                z1 = Hamiltonian(
                    nbqbits,
                    [
                        Term(-1.0, terms_empty, digits_empty),
                        Term(-1.0, terms_z, digits_z),
                    ],
                )
                """
                if (a % 2 == b % 2):
                  # aabb, bbaa, aaaa, bbbb
                  z2 = QubitOperator(()) - QubitOperator((             (c,'Z'),(d,'Z')))

                elif (a % 2 == c % 2):
                  # abab, baba
                  z2 = QubitOperator(()) - QubitOperator((        (b,'Z'),     (d,'Z')))

                else:
                  # abba, baab
                  z2 = QubitOperator(()) - QubitOperator(((a,'Z'),             (d,'Z')))
                """

                if a % 2 == b % 2 and c % 2 == d % 2 and b % 2 == c % 2:
                    # aaaa, bbbb
                    terms_z = "ZZ"
                    digits_z = [c, d]
                    terms_empty = "II"
                    digits_empty = [c, d]
                    z2 = Hamiltonian(
                        nbqbits,
                        [
                            Term(-1.0, terms_empty, digits_empty),
                            Term(+1.0, terms_z, digits_z),
                        ],
                    )
                    four_pool.append(operator * z1 * z2)
                    digits_z = [b, d]
                    digits_empty = [b, d]
                    z2 = Hamiltonian(
                        nbqbits,
                        [
                            Term(-1.0, terms_empty, digits_empty),
                            Term(+1.0, terms_z, digits_z),
                        ],
                    )
                    four_pool.append(operator * z1 * z2)
                    digits_z = [a, d]
                    digits_empty = [a, d]
                    z2 = Hamiltonian(
                        nbqbits,
                        [
                            Term(-1.0, terms_empty, digits_empty),
                            Term(+1.0, terms_z, digits_z),
                        ],
                    )
                elif a % 2 == b % 2:
                    terms_z = "ZZ"
                    digits_z = [c, d]
                    terms_empty = "II"
                    digits_empty = [c, d]
                    z2 = Hamiltonian(
                        nbqbits,
                        [
                            Term(-1.0, terms_empty, digits_empty),
                            Term(+1.0, terms_z, digits_z),
                        ],
                    )
                # aabb, bbaa
                elif a % 2 == c % 2:
                    terms_z = "ZZ"
                    digits_z = [b, d]
                    terms_empty = "II"
                    digits_empty = [b, d]
                    z2 = Hamiltonian(
                        nbqbits,
                        [
                            Term(-1.0, terms_empty, digits_empty),
                            Term(+1.0, terms_z, digits_z),
                        ],
                    )
                else:
                    # abba, baab
                    terms_z = "ZZ"
                    digits_z = [a, d]
                    terms_empty = "II"
                    digits_empty = [a, d]
                    z2 = Hamiltonian(
                        nbqbits,
                        [
                            Term(-1.0, terms_empty, digits_empty),
                            Term(+1.0, terms_z, digits_z),
                        ],
                    )
                four_pool.append(operator * z1 * z2)
        return len(four_pool), four_pool

    # extract the terms grouped by Hamiltonian groups with their coeffs
    def extract_terms_with_coeff(self, qubit_pool):
        """
        Sometimes we have coefficients like MP2 pre-screening parameters are needed.
        So in this function, we extract from the qubit pool the terms with their associated parameters.

                
        Parameters
        ----------
        qubit_pool: list[Hamiltonian]
            list of spin cluster operators  

        Returns
        ----------
        list_terms: list[any]
            list of extracted terms for each cluster operator 
        
        list_coeffs: list[list[floats]]
            list of associated coefficients
        
        """
        list_terms = list()
        list_coeffs = list()

        for qubit_op in qubit_pool:
            terms = list()
            coeffs = list()
            for term in qubit_op.terms:
                term_letters = term.op
                term_qbits = term.qbits
                if term._coeff.complex_p.re == 0:
                    term_coeff = term._coeff.complex_p.im
                else:
                    term_coeff = term._coeff.complex_p.re
                coeffs.append(term_coeff)
                term_temp = "["
                for index in range(len(term_letters)):
                    term_temp += "%s%s " % (term_letters[index], term_qbits[index])
                term_temp = term_temp[:-1]
                term_temp += "]"
                if term_temp not in terms:
                    terms.append(term_temp)
            list_terms.append(terms)
            list_coeffs.append(coeffs)
        return list_terms, list_coeffs

    
    def extract_list_qubits_operators(self, list_terms):
        """
        extract the qubits and operators from the lists and keep them grouped together as they were in
        the original Hamiltonian objects.

        Parameters
        ----------
        list_terms: List[Any]
            list of extracted terms for each cluster operator 

        Returns
        ----------
        list_list_digits: List[List[int]]
            Extracted digits for each term

        list_list_letters: List[List[string]]
            Extracted letters for each term

        """
        list_list_digits = []
        list_list_letters = []

        for terms in list_terms:
            list_digits = []
            list_letters = []
            for term in terms:
                digits = []
                letters = ""
                for char_ind in range(len(term)):
                    if term[char_ind].isdigit() and term[char_ind + 1].isdigit():
                        two_digits = "%s%s" % (term[char_ind], term[char_ind + 1])
                        digits.append(int(two_digits))

                    elif term[char_ind].isdigit() and not term[char_ind - 1].isdigit():
                        digits.append(int(term[char_ind]))
                    elif term[char_ind].isalpha():
                        letters += term[char_ind]
                list_digits.append(digits)
                list_letters.append(letters)
            list_list_digits.append(list_digits)
            list_list_letters.append(list_letters)
        return list_list_digits, list_list_letters

    # generate eight pools
    def generate_eight_pools(self, nbqbits, qubit_pool):
        """
        generate the list of qubit pool operators where each operator is a sum of eight Pauli strings
        associated with their qubit digits with parity conditions.

        Parameters
        ----------
        nbqbits: int
            The number of qbits
        
        Returns
        ----------
        length: int
            Number of eight-pool operators

        eight_pool: List<Hamiltonian>
            list of spin cluster operators
        
        """

        
        eight_pool = []

        list_terms, list_coefs = self.extract_terms_with_coeff(qubit_pool)
        list_list_digits, list_list_letters = self.extract_list_qubits_operators(
            list_terms
        )
        list_new_pauli = []
        list_new_operators = []

        for list_digits, list_letters, coefs in zip(
            list_list_digits, list_list_letters, list_coefs
        ):
            new_operator = Hamiltonian(nbqbits, [Term(0, "II", [0, nbqbits - 1])])
            if not list_digits:
                continue
            for i in range(len(list_digits)):
                coefficient = coefs[i]
                qubit_append = []
                operator_string = ""

                for qubit, operator in zip(list_digits[i], list_letters[i]):
                    if operator != "Z":
                        qubit_append.append(qubit)
                        operator_string += operator

                new_pauli = Hamiltonian(
                    nbqbits, [Term(-1 * coefficient, operator_string, qubit_append)]
                )
                list_new_pauli.append(new_pauli)
                new_operator += new_pauli
                list_new_operators.append(new_operator)
            if new_operator not in eight_pool and -new_operator not in eight_pool:
                eight_pool.append(new_operator)
        return len(eight_pool), eight_pool

        # generate eight pools

    def generate_pool_without_z_from_generator(self, nbqbits, qubit_pool):
        """
        Takes the original Hamiltonian object and returns the Hamiltonian object without Z Pauli terms.
        Note: each double fermionic operator after JW transformation gives eight operators composed of Pauli strings including Z term.
        (This is a second version of generate_eight_pools)

        Parameters
        -----------
        nbqbits: int
            The number of qbits

        qubit_pool: list[Hamiltonian]
            list of spin cluster operators  
        
        Returns
        ----------
        length: int
            Number of eight-pool operators

        eight_pool: List<Hamiltonian>
            list of spin cluster operators

        """
        eight_pool = []

        list_terms, list_coefs = self.extract_terms_with_coeff(qubit_pool)
        list_list_digits, list_list_letters = self.extract_list_qubits_operators(
            list_terms
        )
        list_new_pauli = []
        list_new_operators = []

        for list_digits, list_letters, coefs in zip(
            list_list_digits, list_list_letters, list_coefs
        ):
            new_operator = Hamiltonian(nbqbits, [Term(0, "II", [0, nbqbits - 1])])
            if not list_digits:
                continue
            for i in range(len(list_digits)):
                coefficient = coefs[i]
                qubit_append = []
                operator_string = ""

                for qubit, operator in zip(list_digits[i], list_letters[i]):
                    if operator != "Z":
                        qubit_append.append(qubit)
                        operator_string += operator
                new_pauli = Hamiltonian(
                    nbqbits, [Term(-1 * coefficient, operator_string, qubit_append)]
                )
                list_new_pauli.append(new_pauli)
                new_operator += new_pauli
                list_new_operators.append(new_operator)

            eight_pool.append(new_operator)
        return len(eight_pool), eight_pool

    # generate minimal pool
    def generate_minimal_pool(self, nbqbits):
        """
        generate minimal pool according to this article
        
        '''Tang HL, Shkolnikov V, Barron GS, et al. qubit-adapt-vqe: An adaptive algorithm for constructing hardware-efficient
        ansätze on a quantum processor. PRX Quantum 2021; 2(2): 020310'''

        Parameters
        ----------
        nbqbits: int
            The number of qbits

        Returns
        ----------
        length: int
            Number of minimal pool operators

        minimal_pool4: List<Hamiltonian>
            list of spin cluster operators

        """

        k = nbqbits - 1
        minimal_pool3 = []
        for i in range(nbqbits - 1):
            operator = Hamiltonian(nbqbits, [Term(-1, "YZ", [k - (i + 1), k - i])])
            minimal_pool3.append(operator)
            operator = Hamiltonian(nbqbits, [Term(-1, "Y", [k - i])])
            minimal_pool3.append(operator)

        # V from the qubit-ADAPT article (appendix C)

        minimal_pool4 = []
        for i in range(nbqbits):
            operator_string = "Y"
            qubits = [k - i]
            for j in range(i):
                operator_string += "Z"
                qubits.append(k - j)
            qubits = sorted(qubits)
            operator = Hamiltonian(nbqbits, [Term(-1, operator_string, qubits)])
            minimal_pool4.append(operator)
            if i != 0 and i != nbqbits - 1:
                operator_string = "Y"
                qubits = [k - i]
                for j in range(i - 1):
                    operator_string += "Z"
                    qubits.append(k - j)
                qubits = sorted(qubits)
                operator = Hamiltonian(nbqbits, [Term(-1, operator_string, qubits)])
                minimal_pool4.append(operator)
        return len(minimal_pool4), minimal_pool4

    # generate pure pool with symmetry
    
    def generate_pool_pure_with_symmetry(self, molecule_symbol):
        """
        generate pool pure with symmetry according to this article
        
        '''Shkolnikov V, Mayhall NJ, Economou SE, Barnes E. Avoiding symmetry roadblocks and minimizing the measurement
        overhead of adaptive variational quantum eigensolvers. arXiv preprint arXiv:2109.05340 2021'''
        
        Note: Only H4 is supported. User can extend for other molecules like BeH2 or LiH.

        Parameters
        ----------
        molecule_symbol: string
            The name of the molecule

        Returns
        ----------
        length: int
            Number of minimal pool operators

        pool_pure: List<Hamiltonian>
            list of spin cluster operators

        """
        
        pool_pure = []

        if molecule_symbol == "H4":
            hamiltonian1 = Hamiltonian(
                8, [Term(float(-1.0), "YIXIYIYI", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            # hamiltonian1 = Hamiltonian(8, [ Term(1.0, "IYIYIXIY", [7,6,5,4,3,2,1,0])])
            hamiltonian2 = Hamiltonian(
                8, [Term(float(-1.0), "ZYXIYIZY", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            hamiltonian3 = Hamiltonian(
                8, [Term(float(-1.0), "YIZYXIZY", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            hamiltonian4 = Hamiltonian(
                8, [Term(float(-1.0), "ZZYXYYII", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            hamiltonian5 = Hamiltonian(
                8, [Term(float(1.0), "XXIZIIXY", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            hamiltonian6 = Hamiltonian(
                8, [Term(float(-1.0), "YIZYZXYI", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            hamiltonian7 = Hamiltonian(
                8, [Term(float(-1.0), "XIYZYZYI", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            hamiltonian8 = Hamiltonian(
                8, [Term(float(1.0), "XZIIYZII", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            hamiltonian9 = Hamiltonian(
                8, [Term(float(1.0), "ZXXZZXYI", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            hamiltonian10 = Hamiltonian(
                8, [Term(float(1.0), "XXIIIIXY", [0, 1, 2, 3, 4, 5, 6, 7])]
            )
            hamiltonian11 = Hamiltonian(
                8, [Term(float(-1.0), "IYYZXIZY", [0, 1, 2, 3, 4, 5, 6, 7])]
            )

            # # ZYIZIYZY, ZXXZYYYI, YZIIIXII, YZXYIIXY,
            # # IIXXIIYI, IYYYZZII, IYXZIYZY, ZXZIIXYI,
            # # YYZZZIYI, YIXYZZXY, IIXXXIYI, IYXXIYXY,
            # # ZYIXIXII, XYXIZZII.
            # #ComplexNumber(re=0.7071067811865475, im=0.0)
            pool_pure.append(hamiltonian1)
            pool_pure.append(hamiltonian2)
            pool_pure.append(hamiltonian3)
            pool_pure.append(hamiltonian4)
            pool_pure.append(hamiltonian5)
            pool_pure.append(hamiltonian6)
            pool_pure.append(hamiltonian7)
            pool_pure.append(hamiltonian8)
            pool_pure.append(hamiltonian9)
            pool_pure.append(hamiltonian10)
            pool_pure.append(hamiltonian11)

        return len(pool_pure), pool_pure

    ############### Qubit Excitations ####################
    def single_position_generator(self, nos_qubits):
        """
        Returns the permutation list of indices [i1, i2] such that i1 < i2 and i2 <= nos_qubits

        Parameters
        ----------
        nos_qubits: int
            number of qbits

        Returns
        ----------
        store: List[List[int]]
            the permutation list

        """

        store = []
        x = [i for i in range(nos_qubits)]
        ls = list(itertools.permutations(x, 2))
        for p in range(len(ls)):
            i, k = ls[p]
            if i < k:
                store.append(ls[p])
        return store

    def double_position_generator(self, nos_qubits):
        """
        Returns the permutation list of indices [i1, i2, i3, i4] such that i1 < i2 < i3 < i4 and i4 <= nos_qubits

        Parameters
        ----------
        nos_qubits: int
            number of qbits

        Returns
        --------
        store: List[List[int]]
            the permutation list

        """

        store = []
        x = [i for i in range(nos_qubits)]
        ls = list(itertools.permutations(x, 4))
        for p in range(len(ls)):
            i, j, k, l = ls[p]
            if i < j < k < l:
                store.append(ls[p])
        return store

    def generate_excitations(self, nbqbits, s, d):
        """
        generates the single (s) and double (d) qubit excitations according to this article:

        '''Yordanov YS, Armaos V, Barnes CH, Arvidsson-Shukur DR. Qubit-excitation-based adaptive variational quantum
        eigensolver. Communications Physics 2021; 4(1): 1-1'''


        Parameters
        ----------
        nbqbits: int
            number of qbits
        
        s: List[List[int]]
            list of two-indices lists

        d: List[List[int]]
            list of four-indices lists

        Returns
        ----------
        length: int
            the number of qubit excitations

        qubit_excitation: List<Hamiltonian>
            The list of spin cluster operators of qubit excitations

        """

        qubit_excitation = []
        for i in s:
            hamiltonian = Hamiltonian(
                nbqbits,
                [
                    Term(float(-1.0 / 2), "XY", list(i)),
                    Term(float(+1.0 / 2), "YX", list(i)),
                ],
            )
            qubit_excitation.append(hamiltonian)

        for i in d:
            hamiltonian = Hamiltonian(
                nbqbits,
                [
                    Term(float(-1.0 / 8), "XYXX", list(i)),
                    Term(float(-1.0 / 8), "YXXX", list(i)),
                    Term(float(-1.0 / 8), "YYYX", list(i)),
                    Term(float(-1.0 / 8), "YYXY", list(i)),
                    Term(float(+1.0 / 8), "XXYX", list(i)),
                    Term(float(+1.0 / 8), "XXXY", list(i)),
                    Term(float(+1.0 / 8), "YXYY", list(i)),
                    Term(float(+1.0 / 8), "XYYY", list(i)),
                ],
            )
            qubit_excitation.append(hamiltonian)

        return len(qubit_excitation), qubit_excitation

    def qubit_excitations(self, nbqbits):
        """
        Combines the functions 'single_position_generator', 'double_position_generator', and 'generate_excitations'

        Parameters
        ----------
        nbqbits: int
            number of qbits
        
        Returns

        s: List[List[int]]
            list of two-indices lists

        d: List[List[int]]
            list of four-indices lists
        
        len_qubit_excitation: int
            the number of qubit excitations

        qubit_excitation: List<Hamiltonian>
            The list of spin cluster operators of qubit excitations
        ----------


        """
        s = self.single_position_generator(nbqbits)
        d = self.double_position_generator(nbqbits)
        len_qubit_excitation, qubit_excitation = self.generate_excitations(
            nbqbits, s, d
        )
        return s, d, len_qubit_excitation, qubit_excitation

    def generate_pool_without_cluster(
        self, pool_type, nbqbits=12, qubit_pool=None, molecule_symbol="H4"
    ):
        """
        This function calls the following type of pools:
        - YXXX
        - XYXX
        - XXYX
        - XXXY
        - random
        - two
        - four
        - eight
        - without_Z_from_generator
        - minimal
        - pure_with_symmetry
        
        Parameters
        -----------
        pool_type: string
            The pool type

        nbqbits: int
            The number of qbits. Defaults to 12.

        qubit_pool: list[Hamiltonian]
            list of spin cluster operators. Defaults to None. Only used for the following pool types:
                - eight
                - without_Z_from_generator

        molecule_symbol: string
            The name of the molecule. Defaults to 'H4'. Only used for pure_with_symmetry pool type
        

        Returns
        ----------
        len_returned_pool: int
            Number of the pool operators

        returned_pool: List<Hamiltonian>
            list of spin cluster operators

        """

        returned_pool = None
        len_returned_pool = None
        print("The current pool is", pool_type)
        if pool_type == "YXXX":
            len_returned_pool, returned_pool = self.generate_yxxx_pool(nbqbits=nbqbits)
        elif pool_type == "XYXX":
            len_returned_pool, returned_pool = self.generate_xyxx_pool(nbqbits=nbqbits)
        elif pool_type == "XXYX":
            len_returned_pool, returned_pool = self.generate_xxyx_pool(nbqbits=nbqbits)
        elif pool_type == "XXXY":
            len_returned_pool, returned_pool = self.generate_xxxy_pool(nbqbits=nbqbits)
        elif pool_type == "random":
            _, yxxx_pool = self.generate_yxxx_pool(nbqbits=nbqbits)
            _, xyxx_pool = self.generate_xyxx_pool(nbqbits=nbqbits)
            _, xxyx_pool = self.generate_xxyx_pool(nbqbits=nbqbits)
            _, xxxy_pool = self.generate_xxxy_pool(nbqbits=nbqbits)
            len_returned_pool, returned_pool = self.generate_random_pool(
                yxxx_pool, xyxx_pool, xxyx_pool, xxxy_pool
            )
        elif pool_type == "two":
            len_returned_pool, returned_pool = self.generate_two_pools(nbqbits=nbqbits)
        elif pool_type == "four":
            len_returned_pool, returned_pool = self.generate_four_pools(nbqbits=nbqbits)
        elif pool_type == "eight":
            len_returned_pool, returned_pool = self.generate_eight_pools(
                nbqbits, qubit_pool
            )
        elif pool_type == "without_Z_from_generator":
            (
                len_returned_pool,
                returned_pool,
            ) = self.generate_pool_without_z_from_generator(nbqbits, qubit_pool)

        elif pool_type == "minimal":
            len_returned_pool, returned_pool = self.generate_minimal_pool(nbqbits)
        elif pool_type == "pure_with_symmetry":
            len_returned_pool, returned_pool = self.generate_pool_pure_with_symmetry(
                molecule_symbol=molecule_symbol
            )

        return len_returned_pool, returned_pool

    def generate_pool_from_cluster(self, pool_condition, cluster_ops, nbqbits):
        """
        This function generates the following type of pools:
            - full
            - full_without_Z
            - reduced_without_Z
        
        Parameters
        ------------
        pool_condition: string
            The pool type

        cluster_ops: list[Hamiltonian]
            list of fermionic cluster operators
        
        nbqbits: int
            The number of qbits.

        
        Returns
        ----------
        len_hamiltonian_pool: int
            Number of the pool operators

        hamiltonian_pool: List<Hamiltonian>
            list of spin cluster operators

        """
        qubit_pool = self.generate_pool(cluster_ops)
        terms = self.extract_terms(qubit_pool)
        hamiltonian_pool = None
        len_hamiltonian_pool = None
        print("The current pool is", pool_condition)
        if pool_condition == "full":
            hamiltonian_pool = self.terms_to_hamiltonian(terms, nbqbits=nbqbits)
            len_hamiltonian_pool = len(hamiltonian_pool)
        elif pool_condition == "full_without_Z":
            terms_without_z = self.extract_terms_without_z(terms)
            hamiltonian_pool = self.terms_to_hamiltonian(
                terms_without_z, nbqbits=nbqbits
            )
            len_hamiltonian_pool = len(hamiltonian_pool)
        elif pool_condition == "reduced_without_Z":
            hamiltonian_pool = self.generate_reduced_qubit_pool(terms, nbqbits=nbqbits)
            len_hamiltonian_pool = len(hamiltonian_pool)

        return len_hamiltonian_pool, hamiltonian_pool
