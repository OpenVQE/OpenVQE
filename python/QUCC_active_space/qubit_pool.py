import numpy as np
import itertools
from qat.core import Term
from qat.fermion.transforms import transform_to_jw_basis
from qat.fermion import Hamiltonian


class QubitPool:
    # generate the qubitPool from cluster_ops
    def generate_pool(self, cluster_ops):
        qubit_pool = []
        for i in cluster_ops:
            qubit_op = transform_to_jw_basis(i)
            qubit_pool.append(qubit_op)
        return qubit_pool

    # extract terms from qubitPool
    def extract_terms(self, qubit_pool):
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

    # get the qubits and the operators from the terms
    def extract_qubits_operators(self, terms):
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

    # transform the qubits and operators to hamiltonian
    def terms_to_hamiltonian(self, terms, nbqbits):
        list_digits, list_letters = self.extract_qubits_operators(terms)
        list_hamiltonian = []
        for i in range(len(list_digits)):
            hamiltonian = Hamiltonian(
                nbqbits, [Term(-1.0, list_letters[i], list_digits[i])]
            )
            list_hamiltonian.append(hamiltonian)
        return list_hamiltonian

    # extract qubits and operators without the 'Z' (remove the Z)
    def extract_qubits_operators_without_z(self, terms):
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

    # generate the terms without the 'Z' operator without duplicates
    def extract_terms_without_z(self, terms):
        list_digits, list_letters = self.extract_qubits_operators_without_z(terms)
        terms_z = list()
        index = 0
        for i in range(len(list_digits)):
            term_temp = "["
            for letter in list_letters[i]:

                term_temp += "%s%s " % (letter, list_digits[i][index])
                # print('%s%s' %(letter, list_digits[i][index]))
                index += 1
            term_temp = term_temp[:-1]
            term_temp += "]"
            if term_temp not in terms_z:
                terms_z.append(term_temp)
            index = 0
        terms_z = list(terms_z)
        return terms_z

    # generate the reduced qubits from terms containing Z's qubits and excluding 'Z' operator
    def generate_reduced_qubit_pool(self, terms, nbqbits):
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
            # print(newOperator_hamiltonian)
            if indices not in included:
                reduced_qubit_pool.append(new_operator_hamiltonian)
                included.append(indices)
        return reduced_qubit_pool

    ############### POOLS ####################
    # generate YXXX pool
    def generate_yxxx_pool(self, nbqbits):
        yxxx_pool = []
        for a, b in itertools.combinations(range(nbqbits), 2):
            parity = (a + b) % 2
            # print(a, b)
            if parity == 0:
                # yxxxPool.append(QubitOperator(((a, 'Y'), (b, 'X')), 1j))
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
        random_pool = []
        string_options = [yxxx_pool, xyxx_pool, xxyx_pool, xxxy_pool]

        for i in range(len(xxxy_pool)):
            chosen = np.random.randint(0, 4)
            random_pool.append(string_options[chosen][i])

        return len(random_pool), random_pool

    ############# TWO FOUR EIGHT POOLS ###############

    # generate two pools
    def generate_two_pools(self, nbqbits):
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
        list_terms = list()
        list_coeffs = list()

        for qubit_op in qubit_pool:
            terms = list()
            coeffs = list()
            for term in qubit_op.terms:
                # print(term)
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
                    # print('%s%s' %(term_letters[index], term_qbits[index]))
                term_temp = term_temp[:-1]
                term_temp += "]"

                # print(term_temp)
                # terms.add(term_temp)
                if term_temp not in terms:
                    terms.append(term_temp)
            list_terms.append(terms)
            list_coeffs.append(coeffs)
        # terms = list(terms)
        return list_terms, list_coeffs

    # extract the qubits and operators from the lists and keep them grouped together as they were in
    # Hamiltonian objects
    def extract_list_qubits_operators(self, list_terms):
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
                indices = []
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
                """
                if indices not in included:
                eightPool.append(newOperator)
                included.append(indices)
                print("2\n",newOperator,"\n")
                """
            if new_operator not in eight_pool and -new_operator not in eight_pool:
                eight_pool.append(new_operator)
        # print("length of new Pauli: ", len(list_new_pauli))
        #         print("The number of current operators in the pool: ", len(list_new_operators))
        return len(eight_pool), eight_pool

        # generate eight pools

    def generate_pool_without_z_from_generator(self, nbqbits, qubit_pool):
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
                indices = []
                qubit_append = []
                operator_string = ""

                for qubit, operator in zip(list_digits[i], list_letters[i]):
                    if operator != "Z":
                        qubit_append.append(qubit)
                        operator_string += operator
                #                 print(operator_string, coefficient, qubit_append)
                new_pauli = Hamiltonian(
                    nbqbits, [Term(-1 * coefficient, operator_string, qubit_append)]
                )
                #                 print(newPauli)
                list_new_pauli.append(new_pauli)
                new_operator += new_pauli
                list_new_operators.append(new_operator)
                """
                if indices not in included:
                eightPool.append(newOperator)
                included.append(indices)
                print("2\n",newOperator,"\n")
                """
            eight_pool.append(new_operator)
        # print("length of new Pauli: ", len(list_new_pauli))
        #             print(newOperator)
        #         print("The number of current operators in the pool: ", len(list_new_operators))
        return len(eight_pool), eight_pool

    # generate minimal pool
    def generate_minimal_pool(self, nbqbits):
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
        store = []
        x = [i for i in range(nos_qubits)]
        ls = list(itertools.permutations(x, 2))
        for p in range(len(ls)):
            i, k = ls[p]
            if i < k:
                store.append(ls[p])
        return store

    def double_position_generator(self, nos_qubits):
        store = []
        x = [i for i in range(nos_qubits)]
        ls = list(itertools.permutations(x, 4))
        for p in range(len(ls)):
            i, j, k, l = ls[p]
            if i < j < k < l:
                store.append(ls[p])
        return store

    def generate_excitations(self, nbqbits, s, d):
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
        s = self.singlePositionGenerator(nbqbits)
        d = self.doublePositionGenerator(nbqbits)
        len_qubit_excitation, qubit_excitation = self.generate_excitations(
            nbqbits, s, d
        )
        return s, d, len_qubit_excitation, qubit_excitation

    def generate_pool_without_cluster(
        self, pool_type, nbqbits=12, qubit_pool=None, molecule_symbol="H4"
    ):
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
            len_yxxx_pool, yxxx_pool = self.generate_yxxx_pool(nbqbits=nbqbits)
            len_xyxx_pool, xyxx_pool = self.generate_xyxx_pool(nbqbits=nbqbits)
            len_xxyx_pool, xxyx_pool = self.generate_xxyx_pool(nbqbits=nbqbits)
            len_xxxy_pool, xxxy_pool = self.generate_xxxy_pool(nbqbits=nbqbits)
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
                molecule_symbol="H4"
            )
        # elif pool_type == 'not_pure_with_symmetry':
        # len_returned_pool, returned_pool = quPool.generate_Pool_not_pure_with_symmetry(molecule_symbol='H4')

        return len_returned_pool, returned_pool

    def generate_pool_from_cluster(self, pool_condition, cluster_ops, nbqbits):
        qubit_pool = self.generate_pool(cluster_ops)
        # print(len(qubitPool))
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
