import numpy as np
import itertools
from mrh.util import params
from mrh.util.la import matrix_eigen_control_options, assert_matrix_square
from mrh.util.basis import represent_operator_in_basis, get_complementary_states, get_overlapping_states, get_complete_basis, is_basis_orthonormal, is_basis_orthonormal_and_complete, compute_nelec_in_subspace
from math import factorial

def symmetrize_tensor_conj (tensor):
    # tensors are by default in Mulliken/chemist's order
    # The even indices are bra and the odd indices are ket
    # So the permutation is simply [1, 0, 3, 2, 5, 4, ...]
    perm = tuple(sum ([[x+1, x] for x in range (0, len(tensor.shape), 2)],[]))
    tensor += np.transpose (np.conjugate (tensor), perm)
    tensor /= 2
    return tensor

def symmetrize_tensor_elec (tensor):
    # tensors are by default in Mulliken/chemists order: (pq|rs), etc.
    # the permutations are, eg, [2, 3, 0, 1] and [4, 5, 0, 1, 2, 3] etc.
    nelec = len (tensor.shape) // 2
    elec_perms = itertools.islice (itertools.permutations (range (nelec)), 1, None)
    orb_perms = (sum (tuple ((2*x, 2*x+1) for x in perm), ()) for perm in elec_perms)
    tensor += sum (np.transpose (tensor, perm) for perm in orb_perms)
    tensor /= factorial (nelec)
    return tensor

def symmetrize_tensor (tensor):
    return symmetrize_tensor_elec (symmetrize_tensor_conj (tensor))

