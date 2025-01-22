# A collection of useful manipulations of basis sets (i.e., rectangular matrices) and operators (square matrices)

import sys
import numpy as np
from scipy import linalg
from mrh.util.io import prettyprint_ndarray
from mrh.util.la import is_matrix_zero, is_matrix_eye, is_matrix_idempotent, matrix_eigen_control_options, matrix_svd_control_options, align_vecs
from mrh.util import params
from itertools import combinations
from math import sqrt
import copy

################    basic queries and assertions for basis-set-related objects    ################



def assert_vector_statelist (test_vector, max_element=None, max_length=None, min_length=0):
    if (max_length == None):
        max_length = test_vector.shape[0] + 1
    if (max_element == None):
        max_element = np.amax (test_vector) + 1
    err_str = "vector not 1d array of unique nonnegative integers with {0} <= length <= {1} and maximum element < {2}\n({3})".format (min_length, max_length, max_element, test_vector)
    cond_isvec = (test_vector.ndim == 1)
    cond_min = (np.amin (test_vector) >= 0)
    cond_max = (np.amax (test_vector) <= max_element)
    cond_length = (test_vector.shape[0] <= max_length) and (test_vector.shape[0] >= min_length)
    cond_int = np.all (np.mod (test_vector, 1.0) == 0.0)
    u, counts = np.unique (test_vector, return_counts=True)
    cond_uniq = np.all (counts == 1)
    assert (cond_isvec and cond_min and cond_max and cond_length and cond_int and cond_uniq), err_str
    return test_vector.shape[0]

def assert_vector_stateis (test_vector, vecdim=None):
    err_str = "vector not 1d boolean array"
    cond_isvec = (test_vector.ndim == 1)
    cond_length = (test_vector.shape[0] == vecdim) or (not vecdim)
    if not cond_length:
        err_str = err_str + " (vector has length {0}, should be {1})".format (test_vector.shape[0], vecdim)
    cond_bool = np.all (np.logical_or (test_vector == 1, test_vector == 0))
    assert (cond_isvec and cond_len and cond_bool), err_str

def measure_basis_nonorthonormality (the_basis, ovlp=1):
    cOc = np.atleast_2d (ovlp) 
    c2b = np.asarray (the_basis)
    b2c = c2b.conjugate ().T
    nbas = c2b.shape[1]
    try:
        test_matrix = b2c @ cOc @ c2b
    except ValueError:
        test_matrix = (b2c * cOc[0,0]) @ c2b
    test_matrix -= np.eye (nbas)
    return np.amax (np.abs (test_matrix)), linalg.norm (test_matrix)

def is_basis_orthonormal (the_basis, ovlp=1, rtol=params.num_zero_rtol, atol=params.num_zero_rtol):
    cOc = np.atleast_2d (ovlp) 
    c2b = np.asarray (the_basis)
    b2c = c2b.conjugate ().T
    try:
        test_matrix = b2c @ cOc @ c2b
    except ValueError:
        test_matrix = (b2c * cOc[0,0]) @ c2b
    rtol *= test_matrix.shape[0]
    atol *= test_matrix.shape[0]
    return is_matrix_eye (test_matrix, rtol=rtol, atol=atol)

def is_basis_orthonormal_and_complete (the_basis, rtol=params.num_zero_rtol, atol=params.num_zero_rtol):
    return (is_basis_orthonormal (the_basis, rtol=rtol, atol=atol) and (the_basis.shape[1] == the_basis.shape[0]))

def are_bases_orthogonal (bra_basis, ket_basis, ovlp=1, rtol=params.num_zero_ltol, atol=params.num_zero_ltol):
    test_matrix = basis_olap (bra_basis, ket_basis, ovlp)
    rtol *= sqrt (bra_basis.shape[1] * ket_basis.shape[1])
    atol *= sqrt (bra_basis.shape[1] * ket_basis.shape[1])
    return is_matrix_zero (test_matrix, rtol=rtol, atol=atol), test_matrix

def are_bases_equivalent (bra_basis, ket_basis, ovlp=1, rtol=params.num_zero_ltol, atol=params.num_zero_ltol):
    bra_basis = orthonormalize_a_basis (bra_basis)
    ket_basis = orthonormalize_a_basis (ket_basis)
    if bra_basis.shape[1] != ket_basis.shape[1]: return False
    svals = get_overlapping_states (bra_basis, ket_basis, only_nonzero_vals=True)[2]
    rtol *= sqrt (bra_basis.shape[1] * ket_basis.shape[1])
    atol *= sqrt (bra_basis.shape[1] * ket_basis.shape[1])
    return np.allclose (svals, 1, rtol=rtol, atol=atol)
    

################    simple manipulations and common calculations        ################



def enforce_maxel_positive (the_basis):
    '''Multiply coefficients for states with negative largest coefficient by -1'''
    idx0 = np.asarray (np.abs (the_basis)).argmax (axis=0)
    idx1 = list(range(the_basis.shape[1]))
    cols = np.where (np.asarray (the_basis)[idx0,idx1]<0)[0]
    the_basis[:,cols] *= -1
    return the_basis

def sort_states_by_diag_maxabs (the_basis):
    '''Sort states so that the coefficients in each with the maximum absolute value are on the diagonal'''
    cols = np.asarray (np.abs (the_basis)).argmax (axis=0).argsort ()
    the_basis = the_basis[:,cols]
    return the_basis

def basis_olap (bra_basis, ket_basis, ovlp=1):
    c2p = np.asarray (bra_basis)
    c2q = np.asarray (ket_basis)
    p2c = c2p.conjugate ().T
    cOc = np.atleast_2d (ovlp)
    try:
        return np.asarray (p2c @ cOc @ c2q)
    except ValueError:
        return np.asarray ((p2c * cOc[0,0]) @ c2q)

def represent_operator_in_basis (braOket, bra1_basis = None, ket1_basis = None, bra2_basis = None, ket2_basis = None):
    # This CHANGES the basis that braOket is stored in
    allbases = [i for i in [bra1_basis, ket1_basis, bra2_basis, ket2_basis] if i is not None]
    if len (allbases) == 0:
        raise RuntimeError ("needs at least one basis")
    bra1_basis = allbases[0]
    ket1_basis = bra1_basis if ket1_basis is None else ket1_basis
    bra2_basis = bra1_basis if bra2_basis is None else bra2_basis
    ket2_basis = bra2_basis if ket2_basis is None else ket2_basis
    the_bases = [bra1_basis, ket1_basis, bra2_basis, ket2_basis]
    if any ([i.shape[0] == 0 for i in the_bases]):
        newshape = tuple ([i.shape[1] for i in the_bases])
        if len (braOket.shape) == 2: newshape = tuple (newshape[:2])
        return np.zeros (newshape, dtype=braOket.dtype)
    if all ([is_matrix_eye (i) for i in the_bases]):
        return braOket
    if len (braOket.shape) == 2:
        return represent_operator_in_basis_1body (braOket, bra1_basis, ket1_basis)
    elif len (braOket.shape) == 3 and braOket.shape[0] == 2:
        return np.stack ([represent_operator_in_basis_1body (O, bra1_basis, ket1_basis) for O in braOket], axis=0)
    elif len (braOket.shape) == 4:
        return represent_operator_in_basis_2body (braOket, bra1_basis, ket1_basis, bra2_basis, ket2_basis)
    else:
        raise ValueError ("Only one- and two-body operators (two- and four-index arrays) or pairs of one-body operators supported")

def represent_operator_in_basis_1body (braOket, bra_basis, ket_basis):
    lOr = np.asarray (braOket)
    l2p = np.asarray (bra_basis)
    r2q = np.asarray (ket_basis)
    p2l = l2p.conjugate ().T
    try:
        return np.asarray (p2l @ lOr @ r2q)
    except ValueError as err:
        print (p2l.shape)
        print (lOr.shape)
        print (r2q.shape)
        raise (err)


def represent_operator_in_basis_2body (braOket, bra1_basis, ket1_basis, bra2_basis, ket2_basis):
    abcd = braOket
    az = np.conj (bra1_basis)
    bz = ket1_basis
    cz = np.conj (bra2_basis)
    dz = ket2_basis


    #abcd = np.einsum ('abcd,az->zbcd',abcd,az)
    #abcd = np.einsum ('abcd,bz->azcd',abcd,bz)
    #abcd = np.einsum ('abcd,cz->abzd',abcd,cz)
    #abcd = np.einsum ('abcd,dz->abcz',abcd,dz)
    # Order matters when doing this with tensordot! It puts the remaining
    # axes in the order that the tensors are supplied as arguments.
    abcd = np.tensordot (bz, abcd, axes=(0,1)) # xacd 
    abcd = np.tensordot (az, abcd, axes=(0,1)) # wxcd
    abcd = np.tensordot (abcd, cz, axes=(2,0)) # wxdy
    abcd = np.tensordot (abcd, dz, axes=(2,0)) # wxyz
    return abcd

def project_operator_into_subspace (braOket, ket1_basis = None, bra1_basis = None, ket2_basis = None, bra2_basis = None):
    # This DOESN'T CHANGE the basis that braOket is stored in
    allbases = [i for i in [bra1_basis, ket1_basis, bra2_basis, ket2_basis] if i is not None]
    if len (allbases) == 0:
        raise RuntimeError ("needs at least one basis")
    bra1_basis = allbases[0]
    ket1_basis = bra1_basis if ket1_basis is None else ket1_basis
    bra2_basis = bra1_basis if bra2_basis is None else bra2_basis
    ket2_basis = bra2_basis if ket2_basis is None else ket2_basis
    the_bases = [bra1_basis, ket1_basis, bra2_basis, ket2_basis]
    if any ([basis.shape[1] == 0 for basis in the_bases]):
        return np.zeros_like (braOket)
    if all ([is_matrix_eye (basis) for basis in the_bases]):
        return braOket
    if len (braOket.shape) == 2:
        return project_operator_into_subspace_1body (braOket, bra1_basis, ket1_basis)
    elif len (braOket.shape) == 3 and braOket.shape[0] == 2:
        return np.stack ([project_operator_into_subspace_1body (O, bra1_basis, ket1_basis) for O in braOket], axis=0)
    elif len (braOket.shape) == 4:
        return project_operator_into_subspace_2body (braOket, bra1_basis, ket1_basis, bra2_basis, ket2_basis)
    else:
        raise ValueError ("Only one- and two-body operators (two- and four-index arrays) or pairs of one-body operators supported")

def project_operator_into_subspace_1body (braOket, bra_basis, ket_basis):
    lOr = np.asarray (braOket)

    l2p = np.asarray (bra_basis)
    p2l = l2p.conjugate ().T
    r2q = np.asarray (ket_basis)
    q2r = r2q.conjugate ().T

    lPl = l2p @ p2l
    rPr = r2q @ q2r
    return np.asarray (lPl @ lOr @ rPr)

def project_operator_into_subspace_2body (braOket, bra1_basis, ket1_basis, bra2_basis, ket2_basis):
    abcd_l = np.asarray (braOket)
    l2a = np.asarray (bra1_basis)
    l2b = np.asarray (ket1_basis)
    l2c = np.asarray (bra2_basis)
    l2d = np.asarray (ket2_basis)
    a2l = l2a.conjugate ().T
    b2l = l2b.conjugate ().T
    c2l = l2c.conjugate ().T
    d2l = l2d.conjugate ().T

    #abcd = np.einsum ('abcd,az->zbcd', abcd, np.asarray (l2a * l2a.H))
    #abcd = np.einsum ('abcd,bz->azcd', abcd, np.asarray (l2b * l2b.H))
    #abcd = np.einsum ('abcd,cz->abzd', abcd, np.asarray (l2c * l2c.H))
    #abcd = np.einsum ('abcd,dz->abcz', abcd, np.asarray (l2d * l2d.H))
    # Order matters when doing this with tensordot! It puts the remaining
    # axes in the order that the tensors are supplied as arguments.
    abcd = np.tensordot (l2b, np.tensordot (b2l, abcd_l, axes=(1,1))) # xacd
    abcd = np.tensordot (l2a, np.tensordot (a2l, abcd_l, axes=(1,1))) # wxcd
    abcd = np.tensordot (np.tensordot (abcd_l, l2c, axes=(2,0)), c2l) # wxdy
    abcd = np.tensordot (np.tensordot (abcd_l, l2d, axes=(2,0)), d2l) # wxyz
    return abcd



def compute_operator_trace_in_subset (the_operator, the_subset_basis):
    return np.trace (represent_operator_in_basis (the_operator, the_subset_basis))

compute_nelec_in_subspace = compute_operator_trace_in_subset



################    More complicated basis manipulation functions        ################



def get_overlapping_states (bra_basis, ket_basis, across_operator=None, inner_symmetry=None, outer_symmetry=(None, None), enforce_symmetry=False,
        max_nrvecs=0, max_nlvecs=0, num_zero_atol=params.num_zero_atol, only_nonzero_vals=True, full_matrices=False):
    c2p = np.asarray (bra_basis)
    c2q = np.asarray (ket_basis)
    cOc = 1 if across_operator is None else np.asarray (across_operator)
    assert (c2p.shape[0] == c2q.shape[0]), "you need to give the two spaces in the same basis"
    assert (c2p.shape[1] <= c2p.shape[0]), "you need to give the first state in a complete basis (c2p). Did you accidentally transpose it?"
    assert (c2q.shape[1] <= c2q.shape[0]), "you need to give the second state in a complete basis (c2q). Did you accidentally transpose it?"
    assert (max_nlvecs <= c2p.shape[1]), "you can't ask for more left states than are in your left space"
    assert (max_nrvecs <= c2q.shape[1]), "you can't ask for more right states than are in your right space"
    if np.any (across_operator):
        assert (c2p.shape[0] == cOc.shape[0] and c2p.shape[0] == cOc.shape[1]), "when specifying an across_operator, it's dimensions need to be the same as the external basis"
    get_labels = (not (inner_symmetry is None)) or (not (outer_symmetry[0] is None)) or (not (outer_symmetry[1] is None))

    try:
        rets = matrix_svd_control_options (cOc, lspace=c2p, rspace=c2q, full_matrices=full_matrices,
            symmetry=inner_symmetry,
            lspace_symmetry=outer_symmetry[0],
            rspace_symmetry=outer_symmetry[1],
            strong_symm=enforce_symmetry,
            sort_vecs=-1, only_nonzero_vals=only_nonzero_vals, num_zero_atol=num_zero_atol)
        c2l, svals, c2r = rets[:3]
        if get_labels: llab, rlab = rets[3:]
    except linalg.LinAlgError as e:
        print ("LinAlgError in SVD! Analyzing...")
        if isinstance (cOc, np.ndarray):
            print ("Shape of across_operator: {}".format (cOc.shape))
            print ("Any NANs in across_operator? {}".format (np.count_nonzero (np.isnan (cOc))))
            print ("Any INFs in across_operator? {}".format (np.count_nonzero (np.isinf (cOc))))
            print ("min/max across_operator: {}/{}".format (np.amin (cOc), np.amax (cOc)))
        print ("Shape of bra_basis: {}".format (c2p.shape))
        print ("Any NANs in bra_basis? {}".format (np.count_nonzero (np.isnan (c2p))))
        print ("Any INFs in bra_basis? {}".format (np.count_nonzero (np.isinf (c2p))))
        print ("min/max bra_basis: {}/{}".format (np.amin (c2p), np.amax (c2p)))
        print ("Shape of ket_basis: {}".format (c2p.shape))
        print ("Any NANs in ket_basis? {}".format (np.count_nonzero (np.isnan (c2p))))
        print ("Any INFs in ket_basis? {}".format (np.count_nonzero (np.isinf (c2p))))
        print ("min/max ket_basis: {}/{}".format (np.amin (c2p), np.amax (c2p)))
        proj_l = c2p @ c2p.conjugate ().T
        if isinstance (cOc, np.ndarray):
            proj_l = cOc @ proj_l @ cOc
        r_symmetry = inner_symmetry if outer_symmetry[1] is None else outer_symmetry[1]
        rets = matrix_eigen_control_options (proj_l, subspace=c2q, symmetry=r_symmetry, strong_symm=enforce_symmetry, sort_vecs=-1,
            only_nonzero_vals=False, num_zero_atol=num_zero_atol)
        evals_r, c2r = rets[:2]
        if get_labels: rlab = rets[2]
        proj_r = c2q @ c2q.conjugate ().T
        if isinstance (cOc, np.ndarray):
            proj_r = cOc @ proj_r @ cOc 
        l_symmetry = inner_symmetry if outer_symmetry[0] is None else outer_symmetry[0]
        rets = matrix_eigen_control_options (proj_r, subspace=c2p, symmetry=l_symmetry, strong_symm=enforce_symmetry, sort_vecs=-1,
            only_nonzero_vals=False, num_zero_atol=num_zero_atol)
        evals_l, c2l = rets[:2]
        if get_labels: llab = rets[2]
        print ("These pairs of eigenvalues should be equal and all positive:")
        for el, er in zip (evals_l, evals_r):
            print (el, er)
        mlen = min (len (evals_l), len (evals_r))
        if len (evals_l) > mlen: print ("More left-hand eigenvalues: {}".format (evals_l[mlen:]))
        if len (evals_r) > mlen: print ("More left-hand eigenvalues: {}".format (evals_r[mlen:]))
        raise (e)

    # Truncate the basis if requested
    max_nlvecs = max_nlvecs or c2l.shape[1]
    max_nrvecs = max_nrvecs or c2r.shape[1]

    # But you can't truncate it smaller than it already is
    max_nlvecs = min (max_nlvecs, c2l.shape[1])
    max_nrvecs = min (max_nrvecs, c2r.shape[1])
    c2l = c2l[:,:max_nlvecs]
    c2r = c2r[:,:max_nrvecs]

    if get_labels: return c2l, c2r, svals, llab, rlab
    return c2l, c2r, svals
    
def measure_basis_olap (bra_basis, ket_basis):
    if bra_basis.shape[1] == 0 or ket_basis.shape[1] == 0:
        return 0, 0
    svals = get_overlapping_states (bra_basis, ket_basis)[2]
    olap_ndf = len (svals)
    olap_mag = np.sum (svals * svals)
    return olap_mag, svals

def count_linind_states (the_states, ovlp=1, num_zero_atol=params.num_zero_atol):
    c2b = np.asarray (the_states)
    b2c = c2b.conjugate ().T
    cOc = np.asarray (ovlp)
    nbas = c2b.shape[0]
    nstates = c2b.shape[1]
    bOb = b2c @ cOc @ c2b if cOc.shape == ((nbas, nbas)) else b2c @ c2b
    if is_matrix_zero (bOb) or np.abs (np.trace (bOb)) <= num_zero_atol: return 0
    evals = matrix_eigen_control_options (bOb, only_nonzero_vals=True)[0]
    return len (evals)

def orthonormalize_a_basis (overlapping_basis, ovlp=1, num_zero_atol=params.num_zero_ltol, symmetry=None, enforce_symmetry=False):
    if (is_basis_orthonormal (overlapping_basis)):
        return overlapping_basis
    c2b = np.asarray (overlapping_basis)
    cOc = np.asarray (ovlp)
    if enforce_symmetry:
        c2n = np.zeros ((overlapping_basis.shape[0], 0), dtype=overlapping_basis.dtype)
        for c2s in symmetry:
            s2c = c2s.conjugate ().T
            s2b = s2c @ c2b
            sOs = s2c @ cOc @ c2s if cOc.shape == ((c2b.shape[0], c2b.shape[0])) else (s2c * cOc) @ c2s
            s2n = orthonormalize_a_basis (s2b, ovlp=sOs, num_zero_atol=num_zero_atol, symmetry=None, enforce_symmetry=False)
            c2n = np.append (c2n, c2s @ s2n, axis=1)
        return (c2n)

    b2c = c2b.conjugate ().T
    bOb = b2c @ cOc @ c2b if cOc.shape == ((c2b.shape[0], c2b.shape[0])) else (b2c * cOc) @ c2b
    assert (not is_matrix_zero (bOb)), "overlap matrix is zero! problem with basis?"
    assert (np.allclose (bOb, bOb.conjugate ().T)), "overlap matrix not hermitian! problem with basis?"
    assert (np.abs (np.trace (bOb)) > num_zero_atol), "overlap matrix zero or negative trace! problem with basis?"
     
    evals, evecs = matrix_eigen_control_options (bOb, sort_vecs=-1, only_nonzero_vals=True, num_zero_atol=num_zero_atol)
    if len (evals) == 0:
        return np.zeros ((c2b.shape[0], 0), dtype=c2b.dtype)
    p2x = np.asarray (evecs)
    c2x = c2b @ p2x 
    assert (not np.any (evals < 0)), "overlap matrix has negative eigenvalues! problem with basis?"

    # I want c2n = c2x * x2n
    # x2n defined such that n2c * c2n = I
    # n2x * x2c * c2x * x2n = n2x * evals_xx * x2n = I
    # therefore
    # x2n = evals_xx^{-1/2}
    x2n = np.asarray (np.diag (np.reciprocal (np.sqrt (evals))))
    c2n = c2x @ x2n
    n2c = c2n.conjugate ().T
    nOn = n2c @ cOc @ c2n if cOc.shape == ((c2b.shape[0], c2b.shape[0])) else (n2c * cOc) @ c2n
    if not is_basis_orthonormal (c2n):
        # Assuming numerical problem due to massive degeneracy; remove constant from diagonal to improve solver?
        assert (np.all (np.isclose (np.diag (nOn), 1))), np.diag (nOn) - 1
        nOn[np.diag_indices_from (nOn)] -= 1
        evals, evecs = matrix_eigen_control_options (nOn, sort_vecs=-1, only_nonzero_vals=False)
        n2x = np.asarray (evecs)
        c2x = c2n @ n2x
        x2n = np.asarray (np.diag (np.reciprocal (np.sqrt (evals + 1))))
        c2n = c2x @ x2n
        n2c = c2n.conjugate ().T
        nOn = n2c @ cOc @ c2n if cOc.shape == ((c2b.shape[0], c2b.shape[0])) else (n2c * cOc) @ c2n
        assert (is_basis_orthonormal (c2n)), "failed to orthonormalize basis even after two tries somehow\n" + str (
            prettyprint_ndarray (nOn)) + "\n" + str (np.linalg.norm (nOn - np.eye (c2n.shape[1]))) + "\n" + str (evals)

    return np.asarray (c2n)

def get_states_from_projector (the_projector, num_zero_atol=params.num_zero_atol):
    proj_cc = np.asarray (the_projector)
    assert (np.allclose (proj_cc, proj_cc.H)), "projector must be hermitian\n" + str (np.linalg.norm (proj_cc - proj_cc.conjugate ().T))
    assert (is_matrix_idempotent (proj_cc)), "projector must be idempotent\n" + str (np.linalg.norm ((proj_cc @ proj_cc) - proj_cc))
    evals, evecs = matrix_eigen_control_options (proj_cc, sort_vecs=-1, only_nonzero_vals=True, num_zero_atol=num_zero_atol)
    idx = np.isclose (evals, 1)
    return evecs[:,idx]

def get_complementary_states (incomplete_basis, already_complete_warning=True, atol=params.num_zero_atol, symmetry=None, enforce_symmetry=False):
    if symmetry is None: enforce_symmetry = False
    if incomplete_basis.shape[1] == 0:
        if symmetry is None:
            return np.eye (incomplete_basis.shape[0])
        else:
            return np.concatenate (symmetry, axis=1)
    orthonormal_basis = orthonormalize_a_basis (incomplete_basis, symmetry=symmetry, enforce_symmetry=enforce_symmetry)
    print ("did I gain an active orbital? {}, {}".format (incomplete_basis.shape[1], orthonormal_basis.shape[1]))
    if is_basis_orthonormal_and_complete (orthonormal_basis):
        if already_complete_warning:
            print ("warning: tried to construct a complement for a basis that was already complete")
        return np.zeros ((incomplete_basis.shape[0], 0))
    if enforce_symmetry:
        c2p = np.zeros ((orthonormal_basis.shape[0], 0), dtype=orthonormal_basis.dtype)
        for c2s in symmetry:
            s2b = c2s.conjugate ().T @ orthonormal_basis
            if not count_linind_states (s2b):
                c2p = np.append (c2p, c2s, axis=1)
                continue
            s2p = get_complementary_states (s2b, atol=atol, already_complete_warning=False, symmetry=None, enforce_symmetry=False)
            c2p = np.append (c2p, c2s @ s2p, axis=1)
        return (orthonormalize_a_basis (c2p, symmetry=symmetry, enforce_symmetry=True))

    # Kernel
    nbas = orthonormal_basis.shape[1]
    Q, R = linalg.qr (orthonormal_basis)
    assert (are_bases_equivalent (Q[:,:nbas], orthonormal_basis)), 'basis overlap = {}'.format (measure_basis_olap (Q[:,:nbas], orthonormal_basis))
    assert (are_bases_orthogonal (Q[:,nbas:], orthonormal_basis)), 'basis overlap = {}'.format (measure_basis_olap (Q[:,nbas:], orthonormal_basis))
    '''
    err = linalg.norm (ovlp[:nbas,:].T @ ovlp[:nbas,:]) - np.eye (nbas)) / nbas
    assert (abs (err) < 1e-8), err
    err = linalg.norm (ovlp[nbas:,:]) / nbas
    assert (abs (err) < 1e-8), err
    '''
    return orthonormalize_a_basis (Q[:,nbas:])


def get_complete_basis (incomplete_basis, symmetry=None, enforce_symmetry=False):
    complementary_states = get_complementary_states (incomplete_basis, already_complete_warning = False, symmetry=symmetry, enforce_symmetry=enforce_symmetry)
    if np.any (complementary_states):
        return np.append (incomplete_basis, complementary_states, axis=1)
    else:
        return incomplete_basis

def get_projector_from_states (the_states):
    l2p = np.asarray (the_states)
    p2l = l2p.conjugate ().T
    return l2p @ p2l




################    symmetry block manipulations   ################



# Should work with overlapping states!
def is_operator_block_adapted (the_operator, the_blocks, tol=params.num_zero_atol):
    tol *= the_operator.shape[0]
    if isinstance (the_blocks[0], np.ndarray):
        umat = np.concatenate (the_blocks, axis=1)
        assert (is_basis_orthonormal_and_complete (umat)), 'Symmetry blocks must be orthonormal and complete, {}'.format (len (the_blocks))
        operator_block = represent_operator_in_basis (the_operator, umat)
        labels = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (the_blocks)])
        return is_operator_block_adapted (operator_block, labels)
    iterable = the_blocks if isinstance (the_blocks[0], np.ndarray) else np.unique (the_blocks)
    offblk_operator = the_operator.copy ()
    for blk in np.unique (the_blocks):
        offblk_operator[np.ix_(the_blocks==blk,the_blocks==blk)] = 0
    return is_matrix_zero (offblk_operator, atol=tol)

# Should work with overlapping states!
def is_subspace_block_adapted (the_basis, the_blocks, tol=params.num_zero_atol):
    return is_operator_block_adapted (the_basis @ the_basis.conjugate ().T, the_blocks, tol=tol)

# Should work with overlapping states!
def are_states_block_adapted (the_basis, the_blocks, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    if not is_subspace_block_adapted (the_basis, the_blocks, tol=atol): return False
    atol *= the_basis.shape[0]
    rtol *= the_basis.shape[0]
    for blk in the_blocks:
        projector = blk @ blk.conjugate ().T
        is_symm = ((projector @ the_basis) * the_basis).sum (0)
        if not (np.all (np.logical_or (is_symm < rtol, # Better alternative to np.isclose (is_symm, 0) b/c of the way the latter works
                                       np.isclose (is_symm, 1, atol=atol, rtol=rtol)))): return False
    return True

# Should work with overlapping states!
def assign_blocks (the_basis, the_blocks, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    assert (is_subspace_block_adapted (the_basis, the_blocks, tol=atol)), 'Basis space must be block-adapted before assigning states'
    labels = -np.ones (the_basis.shape[1], dtype=int)
    for idx, blk in enumerate (the_blocks):
        projector = blk @ blk.conjugate ().T
        is_symm = ((projector @ the_basis) * the_basis).sum (0)
        check = np.all (np.logical_or (is_symm < rtol, np.isclose (is_symm, 1, atol=atol, rtol=rtol)))
        assert (check), 'Basis states must be individually block-adapted before being assigned (is_symm = {} for label {})'.format (is_symm, idx)
        labels[np.isclose(is_symm, 1, atol=atol, rtol=rtol)] = idx
    assert (np.all (labels>=0)), 'Failed to assign states {}'.format (np.where (labels<0)[0])
    return labels.astype (int)
    
def symmetrize_basis (the_basis, the_blocks, sorting_metric=None, sort_vecs=1, do_eigh_metric=True, check_metric_block_adapted=True, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    atol_scl = atol * the_basis.shape[0]
    rtol_scl = rtol * the_basis.shape[0]
    if the_blocks is None: the_blocks=[np.eye (the_basis.shape[0])]
    assert (is_subspace_block_adapted (the_basis, the_blocks, tol=atol)), 'Basis space must be block-adapted before blockifying states'
    symmetrized_basis = align_states (the_basis, the_blocks)
    labels = assign_blocks (symmetrized_basis, the_blocks)
    assert (is_basis_orthonormal (symmetrized_basis, atol=atol, rtol=rtol)), "? labels = {}".format (labels)

    if sorting_metric is None:
        return symmetrized_basis, labels
    else:
        if sorting_metric.shape[0] == the_basis.shape[0]:
            metric_symm = represent_operator_in_basis (sorting_metric, symmetrized_basis)
        else:
            assert (sorting_metric.shape[0] == the_basis.shape[1]), 'The sorting metric must be in either the row or column basis of the orbital matrix that is being symmetrized'
            metric_symm = represent_operator_in_basis (sorting_metric, the_basis.conjugate ().T @ symmetrized_basis)
        if check_metric_block_adapted: assert (is_operator_block_adapted (metric_symm, labels, tol=atol))
        metric_evals, evecs, labels = matrix_eigen_control_options (metric_symm, symm_blocks=labels, sort_vecs=sort_vecs, only_nonzero_vals=False, num_zero_atol=atol)
        symmetrized_basis = symmetrized_basis @ evecs
        return symmetrized_basis, labels, metric_evals

def align_states (unaligned_states, the_blocks, sorting_metric=None, sort_vecs=1, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    ''' Symmbreak-tolerant '''
    unaligned_states = orthonormalize_a_basis (unaligned_states)
    if the_blocks is None: the_blocks=[np.eye (unaligned_states.shape[0])]
    block_umat = np.concatenate (the_blocks, axis=1)
    assert (is_basis_orthonormal_and_complete (block_umat)), 'Symmetry blocks must be orthonormal and complete, {}'.format (len (the_blocks))

    if sorting_metric is None: sorting_metric=np.diag (np.arange (unaligned_states.shape[1]))
    if sorting_metric.shape[0] == unaligned_states.shape[1]: sorting_metric=represent_operator_in_basis (sorting_metric, unaligned_states.conjugate ().T)
    block_idx = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (the_blocks)])

    c2u = unaligned_states
    c2s = block_umat
    s2c = c2s.conjugate ().T
    s2u = s2c @ c2u
    s2a = align_vecs (s2u, block_idx)[0]
    c2a = c2s @ s2a
    aligned_states = c2a

    sortval = ((sorting_metric @ aligned_states) * aligned_states).sum (0)
    aligned_states = aligned_states[:,np.argsort (sortval)[::sort_vecs]]
    assert (are_bases_equivalent (unaligned_states, aligned_states)), linalg.norm (ovlp - np.eye (ovlp.shape[0]))
    return aligned_states

def eigen_weaksymm (the_matrix, the_blocks, subspace=None, sort_vecs=1, only_nonzero_vals=False, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    if the_blocks is None: the_blocks=[np.eye (the_matrix.shape[0])]
    if subspace is None: subspace = np.eye (the_matrix.shape[0])
    subspace_matrix = represent_operator_in_basis (the_matrix, subspace)
    evals, evecs = matrix_eigen_control_options (subspace_matrix, symm_blocks=None, sort_vecs=sort_vecs, only_nonzero_vals=only_nonzero_vals,
        num_zero_atol=atol)
    evecs = subspace @ evecs
    idx_unchk = np.ones (len (evals), dtype=np.bool_)
    while np.count_nonzero (idx_unchk > 0):
        chk_1st_eval = evals[idx_unchk][0]
        idx_degen = np.isclose (evals, chk_1st_eval, rtol=rtol, atol=atol)
        if np.count_nonzero (idx_degen) > 1:
            evecs[:,idx_degen] = align_states (evecs[:,idx_degen], the_blocks, atol=atol, rtol=rtol)
        idx_unchk[idx_degen] = False
    return evals, evecs, assign_blocks_weakly (evecs, the_blocks)

def get_block_weights (the_states, the_blocks):
    s2p = np.asarray (the_states)
    if isinstance (the_blocks[0], np.ndarray):
        c2p = s2p
        c2s = np.concatenate (the_blocks, axis=1)
        s2c = c2s.conjugate ().T
        s2p = s2c @ c2p
        labels = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (the_blocks)]).astype (int)
    else:
        labels = np.asarray (the_blocks)
    wgts = s2p.conjugate () * s2p
    wgts = np.stack ([wgts[labels==lbl,:].sum (0) for lbl in np.unique (labels)], axis=-1)
    return wgts

def assign_blocks_weakly (the_states, the_blocks):
    wgts = get_block_weights (the_states, the_blocks)
    return np.argmax (wgts, axis=1)

def cleanup_operator_symmetry (the_operator, the_blocks):
    if the_blocks is None or len (the_blocks) == 1:
        return the_operator
    if not isinstance (the_blocks[0], np.ndarray):
        dummy_blocks = [np.eye (the_operator.shape[0])[:,the_blocks==lbl] for lbl in np.unique (the_blocks)]
        return cleanup_operator_symmetry (the_operator, dummy_blocks)
    trashbin = np.zeros_like (the_operator)
    for blk1, blk2 in combinations (the_blocks, 2):
        trashbin += project_operator_into_subspace (the_operator, blk1, blk2)
        trashbin += project_operator_into_subspace (the_operator, blk2, blk1)
    print ("Norm of matrix elements thrown in the trash: {}".format (linalg.norm (trashbin)))
    the_operator -= trashbin
    assert (is_operator_block_adapted (the_operator, the_blocks))
    return the_operator

def analyze_operator_blockbreaking (the_operator, the_blocks, block_labels=None):
    if block_labels is None: block_labels = np.arange (len (the_blocks), dtype=int)
    if isinstance (the_blocks[0], np.ndarray):
        c2s = np.concatenate (the_blocks, axis=1)
        assert (is_basis_orthonormal_and_complete (c2s)), "Symmetry block problem? Not a complete, orthonormal basis."
        blocked_operator = represent_operator_in_basis (the_operator, c2s)
        blocked_idx = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (the_blocks)])
        c2l, op_svals, c2r = analyze_operator_blockbreaking (blocked_operator, blocked_idx, block_labels=block_labels)
        c2l = [c2s @ s2l for s2l in c2l]
        c2r = [c2s @ s2r for s2r in c2r]
        return c2l, op_svals, c2r
    elif np.asarray (the_blocks).dtype == np.asarray (block_labels).dtype:
        the_indices = np.empty (len (the_blocks), dtype=int)
        for idx, lbl in enumerate (block_labels):
            idx_indices = (the_blocks == lbl)
            the_indices[idx_indices] = idx
        the_blocks = the_indices
    c2l = []
    c2r = []
    op_svals = []
    norbs = the_operator.shape[0]
    my_range = [idx for idx, bl in enumerate (block_labels) if idx in the_blocks]
    for idx1, idx2 in combinations (my_range, 2):
        blk1 = block_labels[idx1]
        blk2 = block_labels[idx2]
        idx12 = np.ix_(the_blocks==idx1, the_blocks==idx2)
        lvecs = np.eye (norbs, dtype=the_operator.dtype)[:,the_blocks==idx1]
        rvecs = np.eye (norbs, dtype=the_operator.dtype)[:,the_blocks==idx2]
        mat12 = the_operator[idx12]
        if is_matrix_zero (mat12):
            c2l.append (np.zeros ((norbs,0), dtype=the_operator.dtype))
            c2r.append (np.zeros ((norbs,0), dtype=the_operator.dtype))
            op_svals.append (np.zeros ((0), dtype=the_operator.dtype))
            continue
        try:
            vecs1, svals, vecs2 = matrix_svd_control_options (mat12, sort_vecs=-1, only_nonzero_vals=False)
            lvecs = lvecs @ vecs1
            rvecs = rvecs @ vecs2
        except ValueError as e:
            if the_operator[idx12].size > 0: raise (e)
            c2l.append (np.zeros ((norbs,0), dtype=the_operator.dtype))
            c2r.append (np.zeros ((norbs,0), dtype=the_operator.dtype))
            op_svals.append (np.zeros ((0), dtype=the_operator.dtype))
            continue
        #print ("Coupling between {} and {}: {} svals, norm = {}".format (idx1, idx2, len (svals), linalg.norm (svals)))
        c2l.append (lvecs)
        c2r.append (rvecs)
        op_svals.append (svals)
    return c2l, op_svals, c2r

def measure_operator_blockbreaking (the_operator, the_blocks, block_labels=None):
    op_s = np.asarray (the_operator).copy ()
    if isinstance (the_blocks[0], np.ndarray):
        op_c = op_s
        c2s = np.concatenate (the_blocks, axis=1)
        s2c = c2s.conjugate ().T
        op_s = s2c @ op_c @ c2s
        labels = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (the_blocks)]).astype (int)
    else:
        labels = np.asarray (the_blocks)
    for lbl in np.unique (labels):
        idx = labels==lbl
        idx = np.ix_(idx,idx)
        op_s[idx] = 0
    try:
        return np.amax (np.abs (op_s)), linalg.norm (op_s)
    except ValueError as e:
        assert (op_s.size == 0), e
        return 0.0, 0.0

def analyze_subspace_blockbreaking (the_basis, the_blocks, block_labels=None):
    projector = the_basis @ the_basis.conjugate ().T
    return analyze_operator_blockbreaking (projector, the_blocks, block_labels=block_labels)

def measure_subspace_blockbreaking (the_subspace, the_blocks, block_labels=None):
    ''' Returns 4 numbers. The first 2 are the largest and norm of the deviation of individual states from being symmetry-adapted.
    The last 2 are the largest and norm of the subspace projector from being symmetry-adapted. If the third and fourth are close
    to zero but the first and second aren't, the subspace can be described by symmetry-adapted states by rotating the individual states
    among themselves to expose the underlying symmetry. '''
    # This function MUST assume that symmetry labels apply to the complete basis on the first axis of the subspace
    projector = the_subspace @ the_subspace.conjugate ().T
    projmax, projnorm = measure_operator_blockbreaking (projector, the_blocks, block_labels=block_labels)
    wgts = get_block_weights (the_subspace, the_blocks)
    idx = wgts > 0.5
    wgts[idx] = 1 - wgts[idx]
    try:
        return np.amax (np.abs (wgts)), linalg.norm (wgts), projmax, projnorm
    except ValueError as e:
        assert (wgts.size == 0), e
        return 0.0, 0.0, 0.0, 0.0

def get_subspace_symmetry_blocks (the_subspace, the_blocks, atol=params.num_zero_atol, rtol=params.num_zero_rtol):
    c2p = np.asarray (the_subspace)
    new_blocks = []
    remaining_space=None
    for idx, c2s in enumerate (the_blocks):
        s2c = c2s.conjugate ().T 
        s2p = s2c @ c2p * (1 + np.arange (c2p.shape[-1]))[None,:] # Prevent degeneracy in the svals
        svals, p2s = matrix_svd_control_options (s2p, rspace=remaining_space, only_nonzero_vals=True, full_matrices=True, sort_vecs=-1, num_zero_atol=rtol)[1:3]
        wgts = (p2s * p2s).sum (0)[:len(svals)]
        assert (np.all (np.isclose (wgts, 1, atol=atol, rtol=rtol))), 'Subspace may not be symmetry-adapted: svals for {}th block: {}'.format (idx, wgts)
        new_blocks.append (p2s[:,:len(svals)])
        remaining_space = p2s[:,len(svals):]
    p2s = np.concatenate (new_blocks, axis=1)    
    assert (is_basis_orthonormal_and_complete (p2s)), measure_basis_nonorthonormality (p2s)
    return new_blocks

def cleanup_subspace_symmetry (the_subspace, the_blocks):
    rlab = assign_blocks_weakly (the_subspace, the_blocks)
    if isinstance (the_blocks[0], np.ndarray):
        c2p = np.asarray (the_subspace)
        c2s = np.concatenate (the_blocks, axis=1)
        s2c = c2s.conjugate ().T
        s2p = s2c @ c2p
        llab = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (the_blocks)]).astype (int)
    else:
        s2p = np.asarray (the_subspace)
        llab = np.asarray (the_blocks)
    for lbl in np.unique (llab):
        ir2p = linalg.qr (s2p[np.ix_(llab==lbl,rlab==lbl)])[0]
        s2p[:,rlab==lbl] = 0
        s2p[np.ix_(llab==lbl,rlab==lbl)] = ir2p
    if isinstance (the_blocks[0], np.ndarray):
        c2p = c2s @ s2p
    else:
        c2p = s2p
    return c2p




