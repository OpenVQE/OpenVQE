# Copyright Atos 2021

from qat.core import Term


def permute_fermionic_operator(fermionic_term, ind):
    """
    Perform the permutation of the two operators in index ind and ind + 1 in a fermionic Term pauli string

    Args:
        fermionic_term (Term): the fermionic term which operators we seek to permute
        ind (int): the lower index of the two consecutive creation or annihilation operators we seek to permute

    Returns:
        list_terms (list<Term>): the list of fermionic terms resulting of the permutation
    """
    coeff = fermionic_term.coeff
    pauli_op = fermionic_term.op
    qbits = fermionic_term.qbits

    if ind >= len(pauli_op) - 1:
        raise IndexError
    permuted_pauli_op = (
        pauli_op[:ind] + pauli_op[ind + 1] + pauli_op[ind] + pauli_op[ind + 2 :]
    )
    permuted_qbits = qbits[:]
    permuted_qbits[ind], permuted_qbits[ind + 1] = (
        permuted_qbits[ind + 1],
        permuted_qbits[ind],
    )
    if (
        "c" in pauli_op[ind : ind + 2]
        and "C" in pauli_op[ind : ind + 2]
        and qbits[ind] == qbits[ind + 1]
    ):
        return [
            Term(
                coefficient=coeff,
                pauli_op=pauli_op[:ind] + pauli_op[ind + 2 :],
                qbits=qbits[:ind] + qbits[ind + 2 :],
            ),
            Term(coefficient=-coeff, pauli_op=permuted_pauli_op, qbits=permuted_qbits),
        ]
    else:
        return [
            Term(coefficient=-coeff, pauli_op=permuted_pauli_op, qbits=permuted_qbits)
        ]


def order_qubits(fermionic_term):
    """
    Takes a fermionic term which pauli_op is supposed to be ordered properly, and reorder it increasing qbit numbers

    Args:
        fermionic_term (Term): the term to reorder

    Returns:
        ordered_term (Term): the reordered term
    """
    coeff = fermionic_term.coeff
    pauli_op = fermionic_term.op
    qbits = fermionic_term.qbits

    ind_c = pauli_op.index("c")
    qbits_C = qbits[:ind_c]
    qbits_c = qbits[ind_c:]
    new_qbits = []

    for qbits_op in [qbits_C, qbits_c]:
        qbits_temp = qbits_op[:]
        ordered = False
        while not ordered:
            ind = 0
            while ind < len(qbits_temp) - 1 and qbits_temp[ind] <= qbits_temp[ind + 1]:
                if qbits_temp[ind] == qbits_temp[ind + 1]:
                    return
                ind += 1
            if ind < len(qbits_temp) - 1:
                ind += 1
                new_ind = 0
                while qbits_temp[new_ind] < qbits_temp[ind]:
                    new_ind += 1
                elt_not_in_order = qbits_temp.pop(ind)
                qbits_temp.insert(new_ind, elt_not_in_order)
                coeff *= (-1) ** (ind - new_ind)
            else:
                ordered = True
        new_qbits += qbits_temp
    return Term(coefficient=coeff, pauli_op=pauli_op, qbits=new_qbits)


def order_fermionic_ops(fermionic_term):
    """
    Order the operators list of a fermionic_term by putting the creations operators on the left and the annihilation operators on the right, with respect to the fermionic anticommutation relations.

    Args:
         fermionic_term (Term): the term to order

    Returns:
        ordered_fermionic_terms (list<Term>): the list of ordered fermionic terms
    """
    coeff = fermionic_term.coeff
    pauli_op = fermionic_term.op
    qbits = fermionic_term.qbits

    ind_c = pauli_op.index("c")
    try:
        ind_C = pauli_op[ind_c:].index("C") + ind_c
    except ValueError:
        new_terms = [fermionic_term]
        ordered_pauli_op = True
    else:
        new_terms = []
        for new_fermionic_term in permute_fermionic_operator(fermionic_term, ind_C - 1):
            new_terms += order_fermionic_term(new_fermionic_term)
    return new_terms


def order_fermionic_term(fermionic_term):
    """
    Order any fermionic term by putting the creation operators on the left, ordered by increasing qubit numbers, and the annihilation operators on the right, ordered y increasing qubit numbers, with respect to the fermionic anticommutation relations.

    Args:
        fermionic_term (Term): the term to order

    Returns:
        ordered_fermionic_terms (list<Term>): the list of ordered fermionic terms
    """
    new_terms = order_fermionic_ops(fermionic_term)
    ordered_terms = []
    for new_term in new_terms:
        ordered_term = order_qubits(new_term)
        if ordered_term:
            ordered_terms.append(ordered_term)
    return ordered_terms
