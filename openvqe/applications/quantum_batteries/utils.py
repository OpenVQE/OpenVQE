"""
Utility functions
"""

from cudaq import spin
from functools import reduce


# String to cudaq spin map
op_map = {"I": spin.i, "X": spin.x, "Y": spin.y, "Z": spin.z}


def pauli_string_to_op(pauli_string):
    """
    Convert Pauli string to cudaq spin operator
    Args:
      pauli_string: Pauli String comprising I, X, Y, Z.

    Returns:
      cudaq spin operator of the given `pauli_string`.
    """
    return reduce(
        lambda a, b: a * b, [op_map[op](q) for q, op in enumerate(pauli_string)]
    )


def get_ham_from_dict(ham_dict):
    """
    Convert a Hamiltonian given in form of dictionary
    to cudaq spin operator

    Args:
      ham_dict: Dictionary from pauli strings to coefficients.

    Returns:
      cudaq spin operator of the given dictionary Hamiltonian.
    """
    return reduce(
        lambda a, b: a + b,
        [
            coeff.real * pauli_string_to_op(pauli_string)
            for pauli_string, coeff in ham_dict.items()
        ],
    )


def rel_err(target, measured):
    """
    Calculates the relative error between a target value and a measured value.

    Args:
      target: The `target` parameter typically refers to the true or expected value.
      measured: The `measured` parameter represents the value that was actually measured in
      a particular experiment.

    Returns:
      The relative error between a target value and a measured value.
    """
    return abs((target - measured) / target)
