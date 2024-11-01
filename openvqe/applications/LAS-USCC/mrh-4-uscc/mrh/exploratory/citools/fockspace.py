import numpy as np
from scipy import linalg
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import _unpack_nelec
from itertools import product
from mrh.my_pyscf.lassi.op_o1 import fermion_spin_shuffle as _fss
from mrh.my_pyscf.fci.csfstring import CSFTransformer

MAX_NORB = 26 # for now
ADDRS_NELEC = np.array ([0], dtype=np.uint8)
for i in range (MAX_NORB):
    ADDRS_NELEC = np.append (ADDRS_NELEC, ADDRS_NELEC+1)

def hilbert2fock (ci, norb, nelec):
    assert (norb <= MAX_NORB)
    nelec = _unpack_nelec (nelec)
    ndeta = cistring.num_strings (norb, nelec[0])
    ndetb = cistring.num_strings (norb, nelec[1])
    ci = np.asarray (ci).reshape (-1, ndeta, ndetb)
    nroots = ci.shape[0]
    ci1 = np.zeros ((nroots, 2**norb, 2**norb), dtype=ci.dtype)
    strsa = cistring.addrs2str (norb, nelec[0], list(range(ndeta)))
    strsb = cistring.addrs2str (norb, nelec[1], list(range(ndetb)))
    ci1[:,strsa[:,None],strsb[:]] = ci[:,:,:]
    return ci1

def fock2hilbert (ci, norb, nelec):
    assert (norb <= MAX_NORB)
    nelec = _unpack_nelec (nelec)
    ci = np.asarray (ci).reshape (-1, 2**norb, 2**norb)
    nroots = ci.shape[0]
    ndeta = cistring.num_strings (norb, nelec[0])
    ndetb = cistring.num_strings (norb, nelec[1])
    ci1 = np.empty ((nroots, ndeta, ndetb), dtype=ci.dtype)
    strsa = cistring.addrs2str (norb, nelec[0], list(range(ndeta)))
    strsb = cistring.addrs2str (norb, nelec[1], list(range(ndetb)))
    ci1[:,:,:] = ci[:,strsa[:,None],strsb]
    return ci1

def fermion_spin_shuffle (norb, norb_f):
    ''' Compute the sign factors corresponding to the convention
    difference between

    ... a2' a1' a0' ... b2' b1' b0' |vac>

    and

    ... a2' b2' a1' b1' a0' b0' |vac>

    in the context of a Fock-space FCI vector where 0,1,2,etc.
    denote clusters of spatial orbitals ("fragments"). These factors
    must by multiplied into a direct-product wave function if the
    individual factors span different particle-number sectors.
    Compare my_pyscf.lassi.op_o1.fermion_spin_shuffle

    Args:
        norb : integer
            Number of spatial orbitals
        norb_f : nfrag integers
            number of orbitals in each fragment, presented
            in the order in which the corresponding field
            operators act on the physical vacuum to create
            a determinant

    Returns:
        sgn : array of shape (2**(2*norb))
            +- 1 for each determinant in the Fock space FCI
            vector
    '''
    # Extremely suboptimal Python implementation
    assert (norb*2 <= MAX_NORB)
    nfrag = len (norb_f)
    ndet = 2**norb
    ndet_f = [2**n for n in norb_f]
    sgn = np.zeros ((ndet,ndet), dtype=np.int8)
    # The lowest-index fragment is the fastest-changing fragment
    addrs = np.arange (ndet, dtype=np.uint64)
    nelec_f = np.zeros ((ndet, nfrag), dtype=np.int8)
    for ifrag, n in enumerate (ndet_f):
        addrs, fragaddrs = np.divmod (addrs, n)
        nelec_f[:,ifrag] = ADDRS_NELEC[fragaddrs]
    ne_f, ne_f_idx = np.unique (nelec_f, return_inverse=True, axis=0)
    for (ia, na_f), (ib, nb_f) in product (enumerate (ne_f), repeat=2):
        idx_a = ne_f_idx == ia
        idx_b = ne_f_idx == ib
        idx = np.ix_(idx_a,idx_b)
        sgn[idx] = _fss (na_f, nb_f)
    assert (np.count_nonzero (sgn==0) == 0)
    return sgn

def fermion_frag_shuffle (norb, i, j):
    ''' Compute the sign factors incurred upon bringing any
    field operators in the orbital range [i,j) to the left side
    of the field-operator product; i.e., from

    ... cj' c(j-l)' *** ci' c(i-k)' ... |vac>

    to

    c(j-l)' *** ci' ... cj' c(i-k)' ... |vac>

    in the context of a spinless Fock-space FCI vector.
    These factors must be multiplied into a wave function when
    projecting into the smaller Fock space of [i,j<norb).
    Compare my_pyscf.lassi.op_o1.fermion_frag_shuffle

    Args:
        norb : integer
            Number of orbitals. This function assumes spinless
            fermions.
        i : integer
            Start of fragment range; inclusive
        j : integer
            End of fragment range; exclusive

    Returns:
        sgn : array of shape (2**norb)
            +- 1 for each determinant in the spinless Fock
            space FCI vector
    '''
    # Extremely suboptimal Python implementation
    assert (norb <= MAX_NORB)
    assert (j>i)
    ndet = 2**norb
    if j==norb: return 1 # trivial result
    sgn = np.zeros (ndet, dtype=np.int8)
    # Lower orbital indices change faster than higher ones
    addrs_p = np.arange (ndet, dtype=np.uint64) // (2**i)
    addrs_q, addrs_p = np.divmod (addrs_p, 2**(j-i))
    nperms = ADDRS_NELEC[addrs_p] * ADDRS_NELEC[addrs_q]
    sgn = np.array ([1,-1], dtype=np.int8)[nperms%2]
    assert (sgn.size==ndet)
    return sgn

def onv_str (stra, strb, norb):
    ''' Generate a printable string of the type 0a2b from spin-up and spin-down
    strings for norb orbitals '''
    s = ''
    dbl = stra & strb
    vir = (~stra) & (~strb)
    alp = stra & (~strb)
    bet = (~stra) & strb
    for iorb in range (norb):
        chk = (1 << iorb)
        if dbl & (1 << iorb): s = '2' + s
        if vir & (1 << iorb): s = '0' + s
        if alp & (1 << iorb): s = 'a' + s
        if bet & (1 << iorb): s = 'b' + s
    assert len (s) == norb
    return s

def hilbert_sector_weight (ci, norb, nelec, smult, is_hilbert=False):
    if np.asarray (nelec).size == 1:
        nelec = _unpack_nelec (nelec,spin=(smult-1))
    if not is_hilbert:
        ci = fock2hilbert (ci, norb, nelec)
    norm = CSFTransformer (norb, nelec[0], nelec[1],
        smult).vec_det2csf (ci, normalize=False, return_norm=True)[1]
    return norm**2

def hilbert_sector_weights (ci, norb):
    sectors = []
    n_weights = []
    for nelec in product (range (norb+1), repeat=2):
        spin = nelec[0] - nelec[1]
        smin = abs (spin) + 1
        smax = min (sum (nelec), 2*norb - sum (nelec)) + 1
        ci_h = fock2hilbert (ci, norb, nelec)
        n_weights.append ((ci_h*ci_h).sum((1,2)))
        for smult in range (smin, smax+1, 2):
            w = hilbert_sector_weight (ci_h, norb, nelec, smult, is_hilbert=True)
            sectors.append ([nelec, smult, w])
    weights = np.asarray ([w[-1] for w in sectors])
    assert (np.all (np.abs (1.0 - weights.sum (0)) < 1e-4)), '{}'.format (weights.sum (0))
    return sectors

def number_operator (ci, norb):
    ''' Two returns: neleca|ci> and nelecb|ci> '''
    naci = np.zeros_like (ci)
    nbci = np.zeros_like (ci)
    for (na,nb) in product (range (norb+1), repeat=2):
        ci_h = fock2hilbert (ci, norb, (na,nb))
        naci += hilbert2fock (na*ci_h, norb, (na,nb)).reshape (*naci.shape)
        nbci += hilbert2fock (nb*ci_h, norb, (na,nb)).reshape (*nbci.shape)
    return naci, nbci

if __name__ == '__main__':
    nroots = 3
    norb = 8
    ci_f = np.random.rand (nroots, (2**norb)**2)
    s = np.diag (np.dot (ci_f.conj (), ci_f.T))
    ci_f /= np.sqrt (s)[:,None]
    ci_f = ci_f.reshape (nroots, (2**norb), (2**norb))
    ci_fhf = ci_f.copy ()
    norms = []
    neleca_avg = []
    nelecb_avg = []
    for (neleca, nelecb) in product (list (range (norb+1)), repeat=2):
        ci_h = fock2hilbert (ci_f, norb, (neleca, nelecb))
        n = (ci_h.conj () * ci_h).sum ((1,2))
        print ("na =",neleca,", nb=",nelecb,"subspace has shape =",ci_h.shape,"and weights =",n)
        norms.append (n)
        neleca_avg.append (n*neleca)
        nelecb_avg.append (n*nelecb)
        ci_fhf -= hilbert2fock (ci_h, norb, (neleca, nelecb))
    print ("This should be zero:",np.amax (np.abs (ci_fhf)))
    norms = np.stack (norms, axis=0).sum (0)
    print ("These should be ones:", norms)
    neleca_avg = np.stack (neleca_avg, axis=0).sum (0)
    nelecb_avg = np.stack (nelecb_avg, axis=0).sum (0)
    print ("<neleca>, <nelecb> should be close to 4 (statistically) =", neleca_avg, nelecb_avg)
    weights = hilbert_sector_weights (ci_f, norb)
    for row in weights:
        print ("nelec = {}, smult = {}, weight = {}".format (*row))
    print ("Trying out smult = 9 manually:", hilbert_sector_weight (ci_f, 8, 8, 9))

    # Test outer-product stuff
    from mrh.my_pyscf.lassi.op_o0 import _ci_outer_product
    ci_h_f = [np.random.rand (6,6), np.random.rand (2,2), np.random.rand (6,6)]
    ci_h_f = [c / linalg.norm (c) for c in ci_h_f]
    ci_h = _ci_outer_product (ci_h_f, [4,2,4], [[2,2],[1,1],[2,2]])
    ci_f = np.multiply.outer (np.squeeze (hilbert2fock (ci_h_f[2], 4, (2,2))),
                              np.squeeze (hilbert2fock (ci_h_f[1], 2, (1,1))))
    ci_f = ci_f.transpose (0,2,1,3).reshape (2**6, 2**6)
    ci_f = np.multiply.outer (ci_f, np.squeeze (hilbert2fock (ci_h_f[0], 4, (2,2))))
    ci_f = ci_f.transpose (0,2,1,3).reshape (2**10, 2**10)
    print ("Outer-product comparison with old implementation - this should be zero:",
        np.amax (np.abs (ci_h - fock2hilbert (ci_f, 10, (5,5)))))

    sgn_fss = fermion_spin_shuffle (10, (4,2,4))
    sgn_ffs = fermion_frag_shuffle (20, 8, 12)
    print (sgn_fss.shape)
    print (sgn_ffs.shape)
