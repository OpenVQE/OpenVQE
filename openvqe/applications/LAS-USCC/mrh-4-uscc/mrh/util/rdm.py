import numpy as np
import itertools
from scipy import linalg
from math import factorial, sqrt
from mrh.util.la import *
from mrh.util.basis import *
from mrh.util.my_math import is_close_to_integer
from mrh.util import params
from mrh.util.io import warnings

def get_1RDM_from_OEI (one_electron_hamiltonian, nocc, subspace=None, symmetry=None, strong_symm=None):
    evals, evecs = matrix_eigen_control_options (one_electron_hamiltonian, sort_vecs=1, subspace=subspace,
        symmetry=symmetry, strong_symm=strong_symm, only_nonzero_vals=False)[:2]
    l2p = evecs[:,:nocc]
    p2l = l2p.conjugate ().T
    return l2p @ p2l

def get_1RDM_from_OEI_in_subspace (one_electron_hamiltonian, subspace_basis, nocc_subspace, num_zero_atol):
    l2w = np.asmatrix (subspace_basis)
    w2l = l2w.H
    OEI_wrk = represent_operator_in_basis (one_electron_hamiltonian, l2w)
    oneRDM_wrk = get_1RDM_from_OEI (OEI_wrk, nocc_subspace)
    oneRDM_loc = represent_operator_in_basis (oneRDM_wrk, w2l)
    return oneRDM_loc
    
def Schmidt_decompose_1RDM (the_1RDM, loc2frag, norbs_bath_max, symmetry=None, fock_helper=None, enforce_symmetry=False,
        bath_tol=params.num_zero_atol, num_zero_atol=params.num_zero_atol, num_zero_rtol=params.num_zero_rtol):
    get_labels = (not (symmetry is None)) or (not (frag_symm is None))
    labels = frag_symm = env_symm = ufrag_labels = efrag_labels = bath_labels = core_labels = None
    norbs_tot = assert_matrix_square (the_1RDM)
    norbs_frag = loc2frag.shape[1]
    assert (norbs_tot >= norbs_frag and loc2frag.shape[0] == norbs_tot)
    assert (is_basis_orthonormal (loc2frag)), linalg.norm (np.dot (loc2frag.T, loc2frag) - np.eye (loc2frag.shape[1]))
    norbs_env = norbs_tot - norbs_frag
    nelec_tot = np.trace (the_1RDM)
    loc2frag_inp = loc2frag.copy ()

    def _analyze_intermediates (loc2int, tag):
        try:
            loc2int = align_states (loc2int, symmetry)
            int_labels = assign_blocks_weakly (loc2int, symmetry)
            err = measure_subspace_blockbreaking (loc2int, symmetry, np.arange (len (symmetry), dtype=int))
            labeldict = dict (zip (*np.unique (int_labels, return_counts=True)))
            print ("{} irreps: {}, err = {}".format (tag, labeldict, err))
        except:
            print ("Analysis failed")
        return
    def _analyze_orth_problem (l2p, lbl):
        for ir in np.unique (lbl):
            print (ir, measure_basis_nonorthonormality (l2p[:,lbl==ir]))
        for ir1, ir2 in itertools.combinations (np.unique (lbl), 2):
            print (ir1, ir2, measure_basis_nonorthonormality (l2p[:,np.logical_or (lbl==ir1,lbl==ir2)]))
        return

    # We need to SVD the environment-fragment block of the 1RDM
    # The bath states are from the left-singular vectors corresponding to nonzero singular value
    # The fragment semi-natural orbitals are from the right-singular vectors of ~any~ singular value
    # Note that only ~entangled~ fragment orbitals are returned so don't overwrite loc2frag!
    print ("Entry to Schmidt_decompose_1RDM")
    _analyze_intermediates (loc2frag, 'Fragment')
    assert (loc2frag.shape == (norbs_tot, norbs_frag)), loc2frag.shape
    loc2env = get_complementary_states (loc2frag, symmetry=symmetry, enforce_symmetry=enforce_symmetry)
    print ("After get_complementary_states")
    _analyze_intermediates (loc2env, 'Environment')
    if enforce_symmetry:
        loc2frag = orthonormalize_a_basis (loc2frag, symmetry=symmetry, enforce_symmetry=enforce_symmetry)
        frag_symm = assign_blocks_weakly (loc2frag, symmetry)
        env_symm = assign_blocks_weakly (loc2env, symmetry)
    print ("After orthonormalize_a_basis")
    _analyze_intermediates (loc2frag, 'Fragment')
    assert (loc2frag.shape == (norbs_tot, norbs_frag)), loc2frag.shape
    assert (loc2env.shape == (norbs_tot, norbs_env)), loc2env.shape
    rets = get_overlapping_states (loc2env, loc2frag, across_operator=the_1RDM, inner_symmetry=symmetry, outer_symmetry=(env_symm, frag_symm), enforce_symmetry=enforce_symmetry,
        only_nonzero_vals=True, full_matrices=True)
    loc2env, loc2frag, svals = rets[:3]
    if get_labels: env_labels, frag_labels = rets[3:]
    print ("Coming right out of SVD")
    _analyze_intermediates (loc2frag, 'Fragment')
    _analyze_intermediates (loc2env, 'Environment')
    assert (is_basis_orthonormal (loc2frag)), measure_basis_nonorthonormality (loc2frag)
    assert (is_basis_orthonormal (loc2env)), measure_basis_nonorthonormality (loc2env)
    norbs_bath = len (svals) #np.count_nonzero (svals > bath_tol)
    norbs_core = norbs_env - norbs_bath
    norbs_ufrag = norbs_frag - norbs_bath
    print ("{} of {} possible bath orbitals found, leaving {} unentangled fragment and {} core orbitals".format (
        norbs_bath, norbs_frag, norbs_ufrag, norbs_core))
    assert (loc2frag.shape == (norbs_tot, norbs_frag)), loc2frag.shape
    assert (loc2env.shape == (norbs_tot, norbs_env)), loc2env.shape
    loc2bath = loc2env[:,:norbs_bath]
    loc2core = loc2env[:,norbs_bath:]
    loc2efrag = loc2frag[:,:norbs_bath]
    loc2ufrag = loc2frag[:,norbs_bath:]
    assert (is_basis_orthonormal (loc2core)), measure_basis_nonorthonormality (loc2core)
    assert (is_basis_orthonormal (loc2ufrag)), measure_basis_nonorthonormality (loc2ufrag)
    assert (is_basis_orthonormal (loc2efrag)), measure_basis_nonorthonormality (loc2efrag)
    assert (is_basis_orthonormal (loc2bath)), measure_basis_nonorthonormality (loc2bath)
    if get_labels:
        efrag_labels = frag_labels[:norbs_bath]
        ufrag_labels = frag_labels[norbs_bath:]
        bath_labels = env_labels[:norbs_bath]
        core_labels = env_labels[norbs_bath:]
    if norbs_ufrag > 0:
        # Problem with strict symmetry enforcement: large possibility of rounding errors accidentally overlapping left null to right non-null or vice-versa
        # Need to correct for this numerically
        if enforce_symmetry:
            loc2ent = np.append (loc2efrag, loc2bath, axis=1)
            proj = loc2ent @ loc2ent.conjugate ().T
            loc2ufrag -= proj @ loc2ufrag
        mat = the_1RDM if fock_helper is None else fock_helper
        rets = matrix_eigen_control_options (mat, subspace=loc2ufrag, symmetry=symmetry, strong_symm=enforce_symmetry, sort_vecs=-1, only_nonzero_vals=False)
        loc2ufrag = rets[1]
        if get_labels: ufrag_labels = rets[2]
    print ("After separating fragment to entangled and unentangled sectors")
    _analyze_intermediates (loc2efrag, 'Entangled fragment')
    _analyze_intermediates (loc2ufrag, 'Unentangled fragment')
    loc2imp = np.concatenate ([loc2ufrag, loc2efrag, loc2bath], axis=1)
    nelec_imp = ((the_1RDM @ loc2imp) * loc2imp).sum ()
    if norbs_core > 0:
        # Problem with strict symmetry enforcement: large possibility of rounding errors accidentally overlapping left null to right non-null or vice-versa
        # Need to correct for this numerically
        if enforce_symmetry:
            proj = loc2imp @ loc2imp.conjugate ().T
            loc2core -= proj @ loc2core
        mat = the_1RDM if fock_helper is None else fock_helper
        rets = matrix_eigen_control_options (mat, subspace=loc2core, symmetry=symmetry, strong_symm=enforce_symmetry, sort_vecs=-1, only_nonzero_vals=False)
        loc2core = rets[1]
        if get_labels: core_labels = rets[2]
    print ("After making canonical or natural core orbitals")
    _analyze_intermediates (loc2core, 'Core')
    loc2emb = np.append (loc2imp, loc2core, axis=1)
    if get_labels: labels = np.concatenate ((ufrag_labels, efrag_labels, bath_labels, core_labels))
    try:
        assert (is_basis_orthonormal (loc2emb)), measure_basis_nonorthonormality (loc2emb)
        assert (is_basis_orthonormal (loc2core)), measure_basis_nonorthonormality (loc2core)
        assert (is_basis_orthonormal (loc2ufrag)), measure_basis_nonorthonormality (loc2ufrag)
        assert (is_basis_orthonormal (loc2imp)), measure_basis_nonorthonormality (loc2imp)
    except AssertionError as e:
        norbs_imp = norbs_frag + norbs_bath
        print ("Embedding basis")
        _analyze_orth_problem (loc2emb, labels)
        print ("Fragment orbitals")
        _analyze_orth_problem (loc2emb[:,:norbs_frag], labels[:norbs_frag])
        print ("Environment orbitals")
        _analyze_orth_problem (loc2emb[:,norbs_frag:], labels[norbs_frag:])
        print ("Impurity orbitals")
        _analyze_orth_problem (loc2emb[:,:norbs_imp], labels[:norbs_imp])
        print ("Core orbitals")
        _analyze_orth_problem (loc2emb[:,norbs_imp:], labels[norbs_imp:])
        print ("Entangled impurity orbitals")
        _analyze_orth_problem (loc2emb[:,norbs_ufrag:norbs_imp], labels[norbs_ufrag:norbs_imp])
        raise (e)
    return loc2emb, norbs_bath, nelec_imp, labels

def electronic_energy_orbital_decomposition (norbs_tot, OEI=None, oneRDM=None, TEI=None, twoRDM=None):
    E_bas = np.zeros (norbs_tot)
    if (OEI is not None) and (oneRDM is not None):
        # Let's make sure that matrix-multiplication doesn't mess me up
        OEI     = np.asarray (OEI)
        oneRDM  = np.asarray (oneRDM)
        prod    = OEI * oneRDM
        E_bas  += 0.5 * np.einsum ('ij->i', prod)[:norbs_tot]
        E_bas  += 0.5 * np.einsum ('ij->j', prod)[:norbs_tot]
    if (TEI is not None) and (twoRDM is not None):
        # Let's make sure that matrix-multiplication doesn't mess me up
        TEI    = np.asarray (TEI)
        twoRDM = np.asarray (twoRDM)
        prod = TEI * twoRDM
        E_bas += (0.125 * np.einsum ('ijkl->i', prod))[:norbs_tot]
        E_bas += (0.125 * np.einsum ('ijkl->j', prod))[:norbs_tot]
        E_bas += (0.125 * np.einsum ('ijkl->k', prod))[:norbs_tot]
        E_bas += (0.125 * np.einsum ('ijkl->l', prod))[:norbs_tot]
    return E_bas

def get_E_from_RDMs (EIs, RDMs):
    energy = 0.0
    for EI, RDM in zip (EIs, RDMs):
        pref    = 1.0 / factorial (len (EI.shape))
        EI      = np.ravel (np.asarray (EI))
        RDM     = np.ravel (np.asarray (RDM))
        energy += pref * np.dot (EI, RDM)
    return energy

def idempotize_1RDM (oneRDM, thresh):
    evals, evecs = linalg.eigh (oneRDM)
    diff_evals = (2.0 * np.around (evals / 2.0)) - evals
    # Only allow evals to change by at most +-thresh
    idx_floor = np.where (diff_evals < -abs (thresh))[0]
    idx_ceil  = np.where (diff_evals >  abs (thresh))[0]
    diff_evals[idx_floor] = -abs(thresh)
    diff_evals[idx_ceil]  =  abs(thresh)
    nelec_diff = np.sum (diff_evals)
    new_evals = evals + diff_evals
    new_oneRDM = represent_operator_in_basis (np.diag (new_evals), evecs.T)
    return new_oneRDM, nelec_diff

def Schmidt_decomposition_idempotent_wrapper (working_1RDM, loc2wfrag, norbs_bath_max, symmetry=None, fock_helper=None, enforce_symmetry=False,
        bath_tol=params.num_zero_atol, idempotize_thresh=0, num_zero_atol=params.num_zero_atol):
    norbs_tot = loc2wfrag.shape[0]
    norbs_wfrag = loc2wfrag.shape[1]
    loc2wemb, norbs_wbath, nelec_wimp, labels = Schmidt_decompose_1RDM (working_1RDM, loc2wfrag, norbs_bath_max, fock_helper=fock_helper,
        bath_tol=bath_tol, symmetry=symmetry, enforce_symmetry=enforce_symmetry)
    norbs_wimp  = norbs_wfrag + norbs_wbath
    norbs_wcore = norbs_tot - norbs_wimp
    loc2wimp  = loc2wemb[:,:norbs_wimp]
    loc2wcore = loc2wemb[:,norbs_wimp:]
    print ("Schmidt decomposition found {0} bath orbitals for this fragment, of an allowed total of {1}".format (norbs_wbath, norbs_bath_max))
    print ("Schmidt decomposition found {0} electrons in this impurity".format (nelec_wimp))
    working_1RDM_core = np.zeros(working_1RDM.shape)
    if norbs_wcore > 0:
        working_1RDM_core = project_operator_into_subspace (working_1RDM, loc2wcore)
        if abs (idempotize_thresh) > num_zero_atol:
            working_1RDM_core, nelec_wcore_diff = idempotize_1RDM (working_1RDM_core, idempotize_thresh)
            nelec_wimp -= nelec_wcore_diff
            print ("After attempting to idempotize the core (part of the putatively idempotent guide) 1RDM with a threshold of "
            + "{0}, {1} electrons were found in the impurity".format (idempotize_thresh, nelec_wimp))
    if not np.isclose (nelec_wimp, round(nelec_wimp), atol=num_zero_atol, rtol=1e-5):
        raise RuntimeError ("Can't solve impurity problems without integer number of electrons! nelec_wimp={0}".format (nelec_wimp))
    return loc2wemb, norbs_wbath, int (round (nelec_wimp)), working_1RDM_core, labels

def get_2CDM_from_2RDM (twoRDM, oneRDMs):
    oneRDMs = np.asarray (oneRDMs)
    if len (oneRDMs.shape) < 3:
        #warnings.warn ("requires spin-separated 1-RDM - approximating as [1/2 dm, 1/2 dm]", RuntimeWarning)
        oneRDM = oneRDMs.copy ()
        oneRDMs = oneRDM / 2
        oneRDMs = np.stack ((oneRDMs, oneRDMs), axis=0)
    else:
        oneRDM = oneRDMs[0] + oneRDMs[1]
    #twoCDM  = twoRDM - np.einsum ('pq,rs->pqrs', oneRDM, oneRDM)
    #twoCDM +=    0.5 * np.einsum ('ps,rq->pqrs', oneRDM, oneRDM)
    twoCDM  = twoRDM.copy ()
    twoCDM -= np.multiply.outer (oneRDM, oneRDM)
    twoCDM += np.multiply.outer (oneRDMs[0], oneRDMs[0]).transpose (0, 3, 2, 1)
    twoCDM += np.multiply.outer (oneRDMs[1], oneRDMs[1]).transpose (0, 3, 2, 1)
    return twoCDM

def get_2CDMs_from_2RDMs (twoRDM, oneRDMs):
    ''' PySCF stores spin-separated twoRDMs as (aa, ab, bb) '''
    oneRDMs = np.asarray (oneRDMs)
    if len (oneRDMs.shape) < 3:
        #warnings.warn ("requires spin-separated 1-RDM - approximating as [1/2 dm, 1/2 dm]", RuntimeWarning)
        oneRDM = oneRDMs.copy ()
        oneRDMs = oneRDM / 2
        oneRDMs = np.stack ((oneRDMs, oneRDMs), axis=0)
    #twoCDM  = twoRDM - np.einsum ('pq,rs->pqrs', oneRDM, oneRDM)
    #twoCDM +=    0.5 * np.einsum ('ps,rq->pqrs', oneRDM, oneRDM)
    twoCDM = [i.copy () for i in twoRDM]
    twoCDM[0] -= np.multiply.outer (oneRDMs[0], oneRDMs[0])
    twoCDM[1] -= np.multiply.outer (oneRDMs[0], oneRDMs[1]) 
    twoCDM[2] -= np.multiply.outer (oneRDMs[1], oneRDMs[1])
    twoCDM[0] += np.multiply.outer (oneRDMs[0], oneRDMs[0]).transpose (0, 3, 2, 1)
    twoCDM[2] += np.multiply.outer (oneRDMs[1], oneRDMs[1]).transpose (0, 3, 2, 1)
    return tuple (twoCDM)

def get_2RDM_from_2CDM (twoCDM, oneRDMs):
    oneRDMs = np.asarray (oneRDMs)
    if len (oneRDMs.shape) < 3:
        #warnings.warn ("requires spin-separated 1-RDM - approximating as [1/2 dm, 1/2 dm]", RuntimeWarning)
        oneRDM = oneRDMs.copy ()
        oneRDMs = oneRDM / 2
        oneRDMs = np.stack ((oneRDMs, oneRDMs), axis=0)
    else:
        oneRDM = oneRDMs[0] + oneRDMs[1]
    #twoRDM  = twoCDM + np.einsum ('pq,rs->pqrs', oneRDM, oneRDM)
    #twoRDM -=    0.5 * np.einsum ('ps,rq->pqrs', oneRDM, oneRDM)
    twoRDM  = twoCDM.copy ()
    twoRDM += np.multiply.outer (oneRDM, oneRDM)
    twoRDM -= np.multiply.outer (oneRDMs[0], oneRDMs[0]).transpose (0, 3, 2, 1)
    twoRDM -= np.multiply.outer (oneRDMs[1], oneRDMs[1]).transpose (0, 3, 2, 1)
    return twoRDM

def get_2RDMs_from_2CDMs (twoCDM, oneRDMs):
    ''' PySCF stores spin-separated twoRDMs as (aa, ab, bb) '''
    oneRDMs = np.asarray (oneRDMs)
    if len (oneRDMs.shape) < 3:
        #warnings.warn ("requires spin-separated 1-RDM - approximating as [1/2 dm, 1/2 dm]", RuntimeWarning)
        oneRDM = oneRDMs.copy ()
        oneRDMs = oneRDM / 2
        oneRDMs = np.stack ((oneRDMs, oneRDMs), axis=0)
    #twoCDM  = twoRDM - np.einsum ('pq,rs->pqrs', oneRDM, oneRDM)
    #twoCDM +=    0.5 * np.einsum ('ps,rq->pqrs', oneRDM, oneRDM)
    twoRDM = [i.copy () for i in twoCDM]
    twoRDM[0] += np.multiply.outer (oneRDMs[0], oneRDMs[0])
    twoRDM[1] += np.multiply.outer (oneRDMs[0], oneRDMs[1])
    twoRDM[2] += np.multiply.outer (oneRDMs[1], oneRDMs[1])
    twoRDM[0] -= np.multiply.outer (oneRDMs[0], oneRDMs[0]).transpose (0, 3, 2, 1)
    twoRDM[2] -= np.multiply.outer (oneRDMs[1], oneRDMs[1]).transpose (0, 3, 2, 1)
    return tuple (twoRDM)



def S2_exptval (oneDM, twoDM, Nelec=None, cumulant=False):
    ''' Evaluate S^2 expectation value from spin-summed one- and two-body density matrices.
        <S^2> = 1/4 (3N - sum_pq [2P_pqqp + P_ppqq])
              = Tr[D-(D**2)/2] - 1/2 sum_pq L_pqqp

        Args:

        oneDM: ndarray of shape = (M,M)
            spin-summed one-body density matrix

        twoDM: ndarray of shape = (M,M,M,M)
            spin-summed two-body density matrix if cumulant == False or density cumulant if cumulant == True

        Kwargs:

        Nelec: int, default = None
            if not supplied, is computed as trace of oneDM

        cumulant: bool, default = False
            whether the cumulant expansion is used for oneDM and twoDM
    '''

    if not cumulant:
        twoDM = get_2CDM_from_2RDM (twoDM, oneDM)

    return np.sum (np.diag (oneDM) - np.einsum ('pq,qp->p', oneDM, oneDM)/2 - np.einsum ('pqqp->p', twoDM)/2)



