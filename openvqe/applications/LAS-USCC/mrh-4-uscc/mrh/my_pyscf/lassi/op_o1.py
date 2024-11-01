import numpy as np
from pyscf import lib, fci
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.addons import cre_a, cre_b, des_a, des_b
from itertools import product, combinations
import time

# NOTE: PySCF has a strange convention where
# dm1[p,q] = <q'p>, but
# dm2[p,q,r,s] = <p'r'sq>
# The return values of make_stdm12s and roots_make_rdm12s observe this convention, but
# everywhere else in this file, the more sensible convention
# dm1[p,q] = <p'q>,
# dm2[p,q,r,s] = <p'r'sq>
# is used.

def fermion_spin_shuffle (na_list, nb_list):
    ''' Compute the sign factor corresponding to the convention
        difference between

        ... a2' a1' a0' ... b2' b1' b0' |vac>

        and

        ... a2' b2' a1' b1' a0' b0' |vac>

        where subspaces 0, 1, 2, etc. have arbitrary numbers of spin-up
        and spin-down electrons 

        Args:
            na: list of up-spin electrons for each subspace
            nb: list of down-spin electrons for each subspace

        Returns:
            sgn: +-1
    '''
    assert (len (na_list) == len (nb_list))
    nperms = 0
    for ix, nb in enumerate (nb_list[1:]):
        na = sum(na_list[:ix+1])
        nperms += na * nb
    return (1,-1)[nperms%2]

def fermion_frag_shuffle (nelec_f, frag_list):
    ''' Compute the sign factor associated with the isolation of
        particular fragments in a product of fermion field operators;
        i.e., the difference between

        ... c2' ... c1' ... c0' ... |vac>

        and

        ... c2' c1' c0' ... |vac>  

        Args:
            nelec_f: list of electron numbers per fragment for the
                whole state
            frag_list: list of fragments to coalesce

        Returns:
            sgn: +- 1
    '''

    frag_list = list (set (frag_list))
    nperms = 0
    nbtwn = 0
    for ix, frag in enumerate (frag_list[1:]):
        lfrag = frag_list[ix]
        if (frag - lfrag) > 1:
            nbtwn += sum ([nelec_f[jx] for jx in range (lfrag+1,frag)])
        if nbtwn:
            nperms += nelec_f[frag] * nbtwn
    return (1,-1)[nperms%2]

def fermion_des_shuffle (nelec_f, frag_list, i):
    ''' Compute the sign factor associated with anticommuting a destruction
        operator past creation operators of unrelated fragments, i.e.,    
        
        ci ... cj' ci' ch' .. |vac> -> ... cj' ci ci' ch' ... |vac>

        Args:
            nelec_f: list of electron numbers per fragment for the whole state
            frag_list: list of fragment numbers actually involved in a given
                transfer; i.e., the argument 'frag_list' of a recent call to
                fermion_frag_shuffle
            i: fragment of the destruction operator to commute foward

        Returns:
            sgn: +- 1
        
    '''
    assert (i in frag_list)
    # Assuming that low orbital indices touch the vacuum first,
    # the destruction operator commutes past the high-index field
    # operators first -> reverse the order of frag_list
    frag_list = list (set (frag_list))[::-1]
    nelec_f = [nelec_f[ix] for ix in frag_list]
    i = frag_list.index (i)
    nperms = sum (nelec_f[:i]) if i else 0
    return (1,-1)[nperms%2]

def lst_hopping_index (fciboxes, nlas, nelelas, nelec_frs):
    ''' Build the LAS state transition hopping index

        Args:
            fciboxes: list of h1e_zipped_fcisolvers
            nlas: list of norbs for each fragment
            nelelas: list of neleca + nelecb for each fragment
            nelec_frs : ndarray of shape (nfrags,nroots,2)
                Number of electrons of each spin in each rootspace in each
                fragment

        Returns:
            hopping_index: ndarray of ints of shape (nfrags, 2, nroots, nroots)
                element [i,j,k,l] reports the change of number of electrons of
                spin j in fragment i between LAS states k and l
            zerop_index: ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS states are
                connected by a null excitation; i.e., no electron, pair,
                or spin hopping or pair splitting/coalescence. This implies
                nonzero 1- and 2-body transition density matrices within
                all fragments.
            onep_index: ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS states
                are connected by exactly one electron hop from i to j or vice
                versa, implying nonzero 1-body transition density matrices
                within spectator fragments and phh/pph modes within
                source/dest fragments.
    '''
    nelelas = [sum (_unpack_nelec (ne)) for ne in nelelas]
    nelec_fsr = nelec_frs.transpose (0,2,1)
    hopping_index = np.array ([[np.subtract.outer (spin, spin)
        for spin in frag] for frag in nelec_fsr])
    symm_index = np.all (hopping_index.sum (0) == 0, axis=0)
    zerop_index = symm_index & (np.count_nonzero (hopping_index, axis=(0,1)) == 0)
    onep_index = symm_index & (np.abs (hopping_index).sum ((0,1)) == 2)
    return hopping_index, zerop_index, onep_index

class LSTDMint1 (object):
    ''' LAS state transition density matrix intermediate 1: fragment-local data.

        Quasi-sparse-memory storage for LAS-state transition density matrix single-fragment
        intermediates. Stores all local transition density matrix factors. Run the `kernel` method
        to compute all data, and run the get_* methods to return the computed intermediates:

        (s and t are spin: a,b for 1 operator; aa, ab, bb for 2 operators
        s is a spin argument passed to the "get" function
        t is a spin index on the returned array)

        get_h (i,j,s): <i|s|j>
        get_p (i,j,s): <i|s'|j> = conj (<j|s|i>)
        get_dm1 (i,j): <i|t't|j>
        get_hh (i,j,s): <i|s2s1|j>
        get_pp (i,j,s): <i|s1's2'|j> = conj (<j|s2s1|i>)
        get_sm (i,j): <i|b'a|j>
        get_sp (i,j): <i|a'b|j> = conj (<j|b'a|i>)
        get_phh (i,j): <i|t'ts|j>
        get_pph (i,j): <i|s't't|j> = conj (<j|t'ts|i>)
        get_dm2 (i,j): <i|t1't2't2t1|j>

        TODO: two-electron spin-broken components
            <i|a'b'bb|j> & h.c. & a<->b
            <i|a'a'bb|j> & a<->b
        Req'd for 2e- relativistic (i.e., spin-breaking) operators

        Args:
            fcibox: instance of :class:`H1EZipFCISolver`
                fcisolvers for the current fragment
            norb : integer
                number of active orbitals in the current fragment
            nelec : integer or sequence of length 2
                base number of electrons in the current fragment
            nroots : integer
                number of states considered
            hopping_index: ndarray of ints of shape (2, nroots, nroots)
                element [i,j,k] reports the change of number of electrons of
                spin i in the current fragment between LAS states j and k
            idx_frag : integer
                index label of current fragment

        Kwargs:
            dtype : instance of np.dtype
                Currently not used
    '''

    def __init__(self, fcibox, norb, nelec, nroots, hopping_index, idx_frag,
                 dtype=np.float64):
        # I'm not sure I need linkstrl
        self.linkstrl = fcibox.states_gen_linkstr (norb, nelec, tril=True)
        self.linkstr = fcibox.states_gen_linkstr (norb, nelec, tril=False)
        self.fcisolvers = fcibox.fcisolvers
        self.norb = norb
        self.nelec = nelec
        self.nroots = nroots
        self.ovlp = np.zeros ((nroots, nroots), dtype=dtype)
        self.nelec_r = [_unpack_nelec (fcibox._get_nelec (solver, nelec))
                        for solver in self.fcisolvers]
        self._h = [[[None for i in range (nroots)] for j in range (nroots)] for s in (0,1)]
        self._hh = [[[None for i in range (nroots)] for j in range (nroots)] for s in (-1,0,1)] 
        self._phh = [[[None for i in range (nroots)] for j in range (nroots)] for s in (0,1)]
        self._sm = [[None for i in range (nroots)] for j in range (nroots)]
        self.dm1 = [[None for i in range (nroots)] for j in range (nroots)]
        self.dm2 = [[None for i in range (nroots)] for j in range (nroots)]
        self.hopping_index = hopping_index
        self.idx_frag = idx_frag

    # Exception catching

    def try_get (self, tab, *args):
        if len (args) == 3: return self.try_get_tdm (tab, *args)
        elif len (args) == 2: return self.try_get_dm (tab, *args)
        else: raise RuntimeError (str (len (args)))

    def try_get_dm (self, tab, i, j):
        try:
            assert (tab[i][j] is not None)
            return tab[i][j]
        except Exception as e:
            errstr = 'frag {} failure to get element {},{}'.format (self.idx_frag, i, j)
            errstr = errstr + '\nhopping_index entry: {}'.format (self.hopping_index[:,i,j])
            raise RuntimeError (errstr)

    def try_get_tdm (self, tab, s, i, j):
        try:
            assert (tab[s][i][j] is not None)
            return tab[s][i][j]
        except Exception as e:
            errstr = 'frag {} failure to get element {},{} w spin {}'.format (
                self.idx_frag, i, j, s)
            errstr = errstr + '\nhopping_index entry: {}'.format (self.hopping_index[:,i,j])
            raise RuntimeError (errstr)

    # 1-particle 1-operator intermediate

    def get_h (self, i, j, s):
        return self.try_get (self._h, s, i, j)
        #return self._h[s][i][j]

    def set_h (self, i, j, s, x):
        self._h[s][i][j] = x
        return x

    def get_p (self, i, j, s):
        return self.try_get (self._h, s, j, i).conj ()
        #return self._h[s][j][i].conj ()

    # 2-particle intermediate

    def get_hh (self, i, j, s):
        return self.try_get (self._hh, s, i, j)
        #return self._hh[s][i][j]

    def set_hh (self, i, j, s, x):
        self._hh[s][i][j] = x
        return x

    def get_pp (self, i, j, s):
        return self.try_get (self._hh, s, j, i).conj ().T
        #return self._hh[s][j][i].conj ().T

    # 1-particle 3-operator intermediate

    def get_phh (self, i, j, s):
        return self.try_get (self._phh, s, i, j)
        #return self._phh[s][i][j]

    def set_phh (self, i, j, s, x):
        self._phh[s][i][j] = x
        return x

    def get_pph (self, i, j, s):
        return self.try_get (self._phh, s, j, i).conj ().transpose (0,3,2,1)
        #return self._phh[s][j][i].conj ().transpose (0,3,2,1)

    # spin-hop intermediate

    def get_sm (self, i, j):
        return self.try_get (self._sm, i, j)
        #return self._sm[i][j]

    def set_sm (self, i, j, x):
        self._sm[i][j] = x
        return x

    def get_sp (self, i, j):
        return self.try_get (self._sm, j, i).conj ().T
        #return self._sm[j][i].conj ().T

    # 1-density intermediate

    def get_dm1 (self, i, j):
        if j > i:
            return self.try_get (self.dm1, j, i).conj ().transpose (0, 2, 1)
            #return self.dm1[j][i].conj ().transpose (0, 2, 1)
        return self.try_get (self.dm1, i, j)
        #return self.dm1[i][j]

    def set_dm1 (self, i, j, x):
        if j > i:
            self.dm1[j][i] = x.conj ().transpose (0, 2, 1)
        else:
            self.dm1[i][j] = x

    # 2-density intermediate

    def get_dm2 (self, i, j):
        k, l = max (i, j), min (i, j)
        return self.try_get (self.dm2, k, l)
        #return self.dm2[k][l]

    def set_dm2 (self, i, j, x):
        if j > i:
            self.dm2[j][i] = x.conj ().transpose (0, 2, 1, 4, 3)
        else:
            self.dm2[i][j] = x

    def kernel (self, ci, hopping_index, zerop_index, onep_index):
        ''' Compute the transition density matrix factors.

        Args:
            ci : list of ndarray of length nroots
                Contains CI vectors for the current fragment
            hopping_index: ndarray of ints of shape (2, nroots, nroots)
                element [i,j,k] reports the change of number of electrons of
                spin i in the current fragment between LAS states j and k
            zerop_index : ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS states are
                connected by a null excitation; i.e., no electron, pair,
                or spin hopping or pair splitting/coalescence. This implies
                nonzero 1- and 2-body transition density matrices within
                all fragments.
            onep_index : ndarray of bools of shape (nroots, nroots)
                element [i,j] is true where the ith and jth LAS states
                are connected by exactly one electron hop from i to j or vice
                versa, implying nonzero 1-body transition density matrices
                within spectator fragments and phh/pph modes within
                source/dest fragments.

        Returns:
            t0 : tuple of length 2
                timestamp of entry into this function, for profiling by caller
        '''

        nroots, norb = self.nroots, self.norb
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())

        # Overlap matrix
        for i, j in combinations (range (self.nroots), 2):
            if self.nelec_r[i] == self.nelec_r[j]:
                self.ovlp[i,j] = ci[i].conj ().ravel ().dot (ci[j].ravel ())
        self.ovlp += self.ovlp.T
        for i in range (self.nroots):
            self.ovlp[i,i] = ci[i].conj ().ravel ().dot (ci[i].ravel ())

        # Spectator fragment contribution
        spectator_index = np.all (hopping_index == 0, axis=0)
        spectator_index[np.triu_indices (self.nroots, k=1)] = False
        spectator_index = np.stack (np.where (spectator_index), axis=1)
        for i, j in spectator_index:
            solver = self.fcisolvers[j]
            linkstr = self.linkstr[j]
            nelec = self.nelec_r[j]
            dm1s, dm2s = solver.trans_rdm12s (ci[i], ci[j], norb, nelec, link_index=linkstr) 
            self.set_dm1 (i, j, np.stack (dm1s, axis=0).transpose (0,2,1))
            # Transpose based on docstring of direct_spin1.trans_rdm12s
            if zerop_index[i,j]: self.set_dm2 (i, j, dm2s)

        # Cache some b_p|i> beforehand for the sake of the spin-flip intermediate 
        hidx_ket_a = np.where (np.any (hopping_index[0] < 0, axis=0))[0]
        hidx_ket_b = np.where (np.any (hopping_index[1] < 0, axis=0))[0]
        bpvec_list = [None for ket in range (nroots)]
        for ket in hidx_ket_b:
            if np.any (np.all (hopping_index[:,:,ket] == np.array ([1,-1])[:,None], axis=0)):
                bpvec_list[ket] = np.stack ([des_b (ci[ket], norb, self.nelec_r[ket], p)
                                             for p in range (norb)], axis=0)

        # a_p|i>
        for ket in hidx_ket_a:
            nelec = self.nelec_r[ket]
            apket = np.stack ([des_a (ci[ket], norb, nelec, p) for p in range (norb)], axis=0)
            nelec = (nelec[0]-1, nelec[1])
            for bra in np.where (hopping_index[0,:,ket] < 0)[0]:
                bravec = ci[bra].ravel ()
                # <j|a_p|i>
                if np.all (hopping_index[:,bra,ket] == [-1,0]):
                    self.set_h (bra, ket, 0, bravec.dot (apket.reshape (norb,-1).T))
                    # <j|a'_q a_r a_p|i>, <j|b'_q b_r a_p|i> - how to tell if consistent sign rule?
                    if onep_index[bra,ket]:
                        solver = self.fcisolvers[bra]
                        linkstr = self.linkstr[bra]
                        phh = np.stack ([solver.trans_rdm12s (ketmat, ci[bra], norb,
                            self.nelec_r[bra], link_index=linkstr)[0] for ketmat in apket],
                            axis=-1)# Arg order switched cf. docstring of direct_spin1.trans_rdm12s
                        err = np.abs (phh[0] + phh[0].transpose (0,2,1))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err)) 
                        # ^ Passing this assert proves that I have the correct index
                        # and argument ordering for the call and return of trans_rdm12s
                        self.set_phh (bra, ket, 0, phh)
                # <j|b'_q a_p|i> = <j|s-|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,1]):
                    bqbra = bpvec_list[bra].reshape (norb, -1).conj ()
                    self.set_sm (bra, ket, np.dot (bqbra, apket.reshape (norb, -1).T))
                # <j|b_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-1,-1]):
                    hh = np.array ([[np.dot (bravec, des_b (pket, norb, nelec, q).ravel ())
                        for pket in apket] for q in range (norb)])
                    self.set_hh (bra, ket, 1, hh)
                # <j|a_q a_p|i>
                elif np.all (hopping_index[:,bra,ket] == [-2,0]):
                    hh_triu = [bravec.dot (des_a (apket[p], norb, nelec, q).ravel ())
                        for q, p in combinations (range (norb), 2)] 
                    hh = np.zeros ((norb, norb), dtype = apket.dtype)
                    hh[np.triu_indices (norb, k=1)] = hh_triu
                    hh -= hh.T
                    self.set_hh (bra, ket, 0, hh)                
                
        # b_p|i>
        for ket in hidx_ket_b:
            nelec = self.nelec_r[ket]
            bpket = np.stack ([des_b (ci[ket], norb, nelec, p)
                for p in range (norb)], axis=0) if bpvec_list[ket] is None else bpvec_list[ket]
            nelec = (nelec[0], nelec[1]-1)
            for bra in np.where (hopping_index[1,:,ket] < 0)[0]:
                bravec = ci[bra].ravel ()
                # <j|b_p|i>
                if np.all (hopping_index[:,bra,ket] == [0,-1]):
                    self.set_h (bra, ket, 1, bravec.dot (bpket.reshape (norb,-1).T))
                    # <j|a'_q a_r b_p|i>, <j|b'_q b_r b_p|i> - how to tell if consistent sign rule?
                    if onep_index[bra,ket]:
                        solver = self.fcisolvers[bra]
                        linkstr = self.linkstr[bra]
                        phh = np.stack ([solver.trans_rdm12s (ketmat, ci[bra], norb,
                            self.nelec_r[bra], link_index=linkstr)[0] for ketmat in bpket],
                            axis=-1) # Arg order switched cf. docstring direct_spin1.trans_rdm12s
                        err = np.abs (phh[1] + phh[1].transpose (0,2,1))
                        assert (np.amax (err) < 1e-8), '{}'.format (np.amax (err))
                        # ^ Passing this assert proves that I have the correct index
                        # and argument ordering for the call and return of trans_rdm12s
                        self.set_phh (bra, ket, 1, phh)
                # <j|b_q b_p|i>
                elif np.all (hopping_index[:,bra,ket] == [0,-2]):
                    hh_triu = [bravec.dot (des_b (bpket[p], norb, nelec, q).ravel ())
                        for q, p in combinations (range (norb), 2)]
                    hh = np.zeros ((norb, norb), dtype = bpket.dtype)
                    hh[np.triu_indices (norb, k=1)] = hh_triu
                    hh -= hh.T
                    self.set_hh (bra, ket, 2, hh)                
        
        return t0

class LSTDMint2 (object):
    ''' LAS state transition density matrix intermediate 2 - whole-system DMs
        Carry out multiplications such as

            <I|sp'sq|J> = <I|sp'|J> * <I|sq|J>
            <I|s1p's2p's2p s1q|J> = <I|s1p's2p's2p|J> * <I|s1q|J>

        and so forth, where `p` and `q` are on different fragments. The parent class stores the
        entire nroots-by-nroots 1- and 2-body transition density matrix arrays (see make_stdm12s
        below), which is computed and returned by calling the `kernel` method.

        The initializer categorizes all possible interactions among a set of LAS states as
        "null" (no electrons move), "1c" (one charge unit hops; cp'cq), "1s" (one spin unit hops;
        ap'bq'bp aq), "1s1c", (a coupled spin flip of between one fragment and a charge unit which
        is hopping between two other fragments; ap'br'bq ar) "2c" (every case in which two charge
        units move among any two, three, or four fragments).

        The heart of the class is "_crunch_all_", which iterates over all listed interactions,
        builds the corresponding transition density matrices, and passes them into the "_put_D1_"
        and "_put_D2_" methods, which are overwritten in child classes to make the operator or
        reduced density matrices as appropriate.

        Subclass the __init__, __??t_D?_, __add_transpose__, and kernel methods to do various
        different things which rely on LAS-state tdm12s as intermediates without cacheing the whole
        things (i.e. operators or DMs in different basis).

        Args:
            ints : list of length nfrags of instances of :class:`LSTDMint1`
                fragment-local intermediates
            nlas : list of length nfrags of integers
                numbers of active orbitals in each fragment
            hopping_index: ndarray of ints of shape (nfrags, 2, nroots, nroots)
                element [i,j,k,l] reports the change of number of electrons of
                spin j in fragment i between LAS states k and l

        Kwargs:
            dtype : instance of np.dtype
                Currently not used; TODO: generalize to ms-broken fragment-local states?
        '''
    # TODO: SO-LASSI o1 implementation: a SOMF implementation using spin-pure LAS product states
    # states as a basis requires the sz-breaking sector of the 1-body stdm1 to be added here. I.E.,
    # in addition to the interactions listed above, we also need "sm" (total spin lowering; ap'bq)
    # (N.B.: "sp" is just the adjoint of "sm"). 
    # TODO: at some point, if it ever becomes rate-limiting, make this multithread better

    def __init__(self, ints, nlas, hopping_index, dtype=np.float64):
        self.ints = ints
        self.nlas = nlas
        self.norb = sum (nlas)
        self.hopping_index = hopping_index
        self.nfrags, _, self.nroots, _ = nfrags, _, nroots, _ = hopping_index.shape
        self.dtype = dtype
        self.tdm1s = self.tdm2s = None

        # The primary index arrays
        # The nth column of each array is the (n+1)th argument of the corresponding _crunch_*_
        # member function below. The first two columns are always the bra and the ket. Further
        # columns identify fragments whose quantum numbers are changed by the interaction. If
        # necessary (i.e., for 1c and 2c), the last column identifies spin case.
        self.exc_null = np.empty ((0,2), dtype=int)
        self.exc_1c = np.empty ((0,5), dtype=int)
        self.exc_1s = np.empty ((0,4), dtype=int)
        self.exc_1s1c = np.empty ((0,5), dtype=int)
        self.exc_2c = np.empty ((0,7), dtype=int)

        # Process connectivity data to quickly distinguish interactions

        # Should probably be all == true anyway if I call this by symmetry blocks
        conserv_index = np.all (hopping_index.sum (0) == 0, axis=0)

        # Number of field operators involved in a given interaction
        nsop = np.abs (hopping_index).sum (0) # 0,0 , 2,0 , 0,2 , 2,2 , 4,0 , 0,4
        nop = nsop.sum (0) # 0, 2, 4
        ispin = nsop[1,:,:] // 2
        # This last ^ is somewhat magical, but notice that it corresponds to the mapping
        #   2,0 ; 4,0 -> 0 -> a or aa
        #   0,2 ; 2,2 -> 1 -> b or ab
        #   0,4       -> 2 -> bb

        # For each interaction, the change to each fragment of
        charge_index = hopping_index.sum (1) # charge
        spin_index = hopping_index[:,0] - hopping_index[:,1] # spin (*2)

        # Upon a given interaction, count the number of fragments which:
        ncharge_index = np.count_nonzero (charge_index, axis=0) # change in charge
        nspin_index = np.count_nonzero (spin_index, axis=0) # change in spin

        findf = np.argsort ((3*hopping_index[:,0]) + hopping_index[:,1], axis=0, kind='stable')
        # This is an array of shape (nfrags, nroots, nroots) such that findf[:,i,j]
        # is list of fragment indices sorted first by the number of spin-up electrons
        # gained (>0) or lost (<0), and then by the number of spin-down electrons gained
        # or lost in the interaction between states "i" and "j". Because at most 2
        # des/creation ops are involved, the factor of 3 sets up the order a'b'ba without
        # creating confusion between spin and charge of freedom. The 'stable' sort keeps
        # relative order -> sign convention!
        #
        # Throughout the below, we use element-wise logical operators to generate mask
        # index arrays addressing elements of the last two dimensions of "findf" that
        # are consistent with a state interaction of a specific type. We then use the
        # fragment index lists thus specified to identify the source and destination
        # fragments of the charge or spin units that are transferred in that interaction,
        # and store those fragment indices along with the state indices.


        # Zero-electron interactions
        tril_index = np.zeros_like (conserv_index)
        tril_index[np.tril_indices (self.nroots,k=-1)] = True
        idx = conserv_index & tril_index & (nop == 0)
        self.exc_null = np.vstack (list (np.where (idx))).T
 
        # One-electron interactions
        idx = conserv_index & (nop == 2) & tril_index
        if nfrags > 1: self.exc_1c = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx], ispin[idx]]
        ).T

        # Unsymmetric two-electron interactions: full square
        idx_2e = conserv_index & (nop == 4)

        # Two-electron interaction: ii -> jk ("split").
        idx = idx_2e & (ncharge_index == 3) & (np.amin (charge_index, axis=0) == -2)
        if nfrags > 2: exc_split = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-2][idx], findf[0][idx],
            ispin[idx]]
        ).T

        # Two-electron interaction: k(a)j(b) -> i(a)k(b) ("1s1c")
        idx = idx_2e & (nspin_index==3) & (ncharge_index==2) & (np.amin(spin_index,axis=0)==-2)
        if nfrags > 2: self.exc_1s1c = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[1][idx], findf[0][idx]]
        ).T

        # Symmetric two-electron interactions: lower triangle only
        idx_2e = idx_2e & tril_index

        # Two-electron interaction: i(a)j(b) -> j(a)i(b) ("1s") 
        idx = idx_2e & (ncharge_index == 0) & (nspin_index == 2)
        if nfrags > 1: self.exc_1s = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx]]
        ).T

        # Two-electron interaction: ii -> jj ("pair") 
        idx = idx_2e & (ncharge_index == 2) & (nspin_index < 3)
        if nfrags > 1: exc_pair = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-1][idx], findf[0][idx],
            ispin[idx]]
        ).T

        # Two-electron interaction: ij -> kl ("scatter")
        idx = idx_2e & (ncharge_index == 4)
        if nfrags > 3: exc_scatter = np.vstack (
            list (np.where (idx)) + [findf[-1][idx], findf[0][idx], findf[-2][idx], findf[1][idx], 
            ispin[idx]]
        ).T

        # Combine "split", "pair", and "scatter" into "2c"
        if nfrags > 1: self.exc_2c = exc_pair
        if nfrags > 2: self.exc_2c = np.vstack ((self.exc_2c, exc_split))
        if nfrags > 3: self.exc_2c = np.vstack ((self.exc_2c, exc_scatter))

        # overlap tensor
        self.ovlp = np.stack ([i.ovlp for i in ints], axis=-1)

        # spin-shuffle sign vector
        self.nelec_rf = np.asarray ([[list (i.nelec_r[ket]) for i in ints]
                                     for ket in range (self.nroots)]).transpose (0,2,1)
        self.spin_shuffle = [fermion_spin_shuffle (nelec_sf[0], nelec_sf[1])
                             for nelec_sf in self.nelec_rf]
        self.nelec_rf = self.nelec_rf.sum (1)

    def get_range (self, i):
        p = sum (self.nlas[:i])
        q = p + self.nlas[i]
        return p, q

    def get_ovlp_fac (self, bra, ket, *inv):
        idx = np.ones (self.nfrags, dtype=np.bool_)
        idx[list (inv)] = False
        wgt = np.prod (self.ovlp[bra,ket,idx])
        uniq_frags = list (set (inv))
        wgt *= self.spin_shuffle[bra] * self.spin_shuffle[ket]
        wgt *= fermion_frag_shuffle (self.nelec_rf[bra], uniq_frags)
        wgt *= fermion_frag_shuffle (self.nelec_rf[ket], uniq_frags)
        return wgt

    def _get_D1_(self, bra, ket):
        return self.tdm1s[bra,ket]

    def _put_D1_(self, bra, ket, D1):
        pass

    def _get_D2_(self, bra, ket):
        return self.tdm2s[bra,ket]

    def _put_D2_(self, bra, ket, D2):
        pass

    # Cruncher functions
    def _crunch_null_(self, bra, ket):
        '''Compute the reduced density matrix elements between states bra and ket which have the
        the same spin-up and spin-down electron numbers on all fragments (For instance, bra=ket)
        '''
        d1 = self._get_D1_(bra, ket)
        d2 = self._get_D2_(bra, ket)
        nlas = self.nlas
        for i, inti in enumerate (self.ints):
            p = sum (nlas[:i])
            q = p + nlas[i]
            d1_s_ii = inti.get_dm1 (bra, ket)
            fac = self.get_ovlp_fac (bra, ket, i)
            d1[:,p:q,p:q] = fac * np.asarray (d1_s_ii)
            d2[:,p:q,p:q,p:q,p:q] = fac * np.asarray (inti.get_dm2 (bra, ket))
            for j, intj in enumerate (self.ints[:i]):
                assert (i>j)
                r = sum (nlas[:j])
                s = r + nlas[j]
                d1_s_jj = intj.get_dm1 (bra, ket)
                d2_s_iijj = np.multiply.outer (d1_s_ii, d1_s_jj).transpose (0,3,1,2,4,5)
                d2_s_iijj = d2_s_iijj.reshape (4, q-p, q-p, s-r, s-r)
                d2_s_iijj *= self.get_ovlp_fac (bra, ket, i, j)
                d2[:,p:q,p:q,r:s,r:s] = d2_s_iijj
                d2[(0,3),r:s,r:s,p:q,p:q] = d2_s_iijj[(0,3),...].transpose (0,3,4,1,2)
                d2[(1,2),r:s,r:s,p:q,p:q] = d2_s_iijj[(2,1),...].transpose (0,3,4,1,2)
                d2[(0,3),p:q,r:s,r:s,p:q] = -d2_s_iijj[(0,3),...].transpose (0,1,4,3,2)
                d2[(0,3),r:s,p:q,p:q,r:s] = -d2_s_iijj[(0,3),...].transpose (0,3,2,1,4)
        self._put_D1_(bra, ket, d1)
        self._put_D2_(bra, ket, d2)

    def _crunch_1c_(self, bra, ket, i, j, s1):
        '''Compute the reduced density matrix elements of a single electron hop; i.e.,

        <bra|j'(s1)i(s1)|ket>

        i.e.,

        j ---s1---> i

        and conjugate transpose
        '''
        d1 = self._get_D1_(bra, ket)
        d2 = self._get_D2_(bra, ket)
        inti, intj = self.ints[i], self.ints[j]
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        fac = 1
        fac = self.get_ovlp_fac (bra, ket, i, j)
        fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j), i)
        fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j), j)
        d1_ij = np.multiply.outer (self.ints[i].get_p (bra, ket, s1),
                                   self.ints[j].get_h (bra, ket, s1))
        d1[s1,p:q,r:s] = fac * d1_ij
        s12l = s1 * 2   # aa: 0 OR ba: 2
        s12h = s12l + 1 # ab: 1 OR bb: 3 
        s21l = s1       # aa: 0 OR ab: 1
        s21h = s21l + 2 # ba: 2 OR bb: 3
        s1s1 = s1 * 3   # aa: 0 OR bb: 3
        def _crunch_1c_tdm2 (d2_ijkk, i0, i1, j0, j1, k0, k1):
            d2[(s12l,s12h), i0:i1, j0:j1, k0:k1, k0:k1] = d2_ijkk
            d2[(s21l,s21h), k0:k1, k0:k1, i0:i1, j0:j1] = d2_ijkk.transpose (0,3,4,1,2)
            d2[s1s1, i0:i1, k0:k1, k0:k1, j0:j1] = -d2_ijkk[s1,...].transpose (0,3,2,1)
            d2[s1s1, k0:k1, j0:j1, i0:i1, k0:k1] = -d2_ijkk[s1,...].transpose (2,1,0,3)
        # pph (transpose from Dirac order to Mulliken order)
        d2_ijii = fac * np.multiply.outer (self.ints[i].get_pph (bra,ket,s1),
                                           self.ints[j].get_h (bra,ket,s1)).transpose (0,1,4,2,3)
        _crunch_1c_tdm2 (d2_ijii, p, q, r, s, p, q)
        # phh (transpose to bring spin to outside and then from Dirac order to Mulliken order)
        d2_ijjj = fac * np.multiply.outer (self.ints[i].get_p (bra,ket,s1),
                                           self.ints[j].get_phh (bra,ket,s1)).transpose (1,0,4,2,3)
        _crunch_1c_tdm2 (d2_ijjj, p, q, r, s, r, s)
        # spectator fragment mean-field (should automatically be in Mulliken order)
        for k in range (self.nfrags):
            if k in (i, j): continue
            fac = self.get_ovlp_fac (bra, ket, i, j, k)
            fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j, k), i)
            fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j, k), j)
            t, u = self.get_range (k)
            d1_skk = self.ints[k].get_dm1 (bra, ket)
            d2_ijkk = fac * np.multiply.outer (d1_ij, d1_skk).transpose (2,0,1,3,4)
            _crunch_1c_tdm2 (d2_ijkk, p, q, r, s, t, u)
        self._put_D1_(bra, ket, d1)
        self._put_D2_(bra, ket, d2)

    def _crunch_1s_(self, bra, ket, i, j):
        '''Compute the reduced density matrix elements of a spin unit hop; i.e.,

        <bra|i'(a)j'(b)i(b)j(a)|ket>

        i.e.,

        j ---a---> i
        i ---b---> j

        and conjugate transpose
        '''
        d2 = self._get_D2_(bra, ket) # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        y, z = min (i, j), max (i, j)
        fac = -1 * self.get_ovlp_fac (bra, ket, i, j)
        d2_spsm = fac * np.multiply.outer (self.ints[i].get_sp (bra, ket),
                                           self.ints[j].get_sm (bra, ket))
        d2[1,p:q,r:s,r:s,p:q] = d2_spsm.transpose (0,3,2,1)
        d2[2,r:s,p:q,p:q,r:s] = d2_spsm.transpose (2,1,0,3)
        self._put_D2_(bra, ket, d2)

    def _crunch_1s1c_(self, bra, ket, i, j, k):
        '''Compute the reduced density matrix elements of a spin-charge unit hop; i.e.,

        <bra|i'(a)k'(b)j(b)k(a)|ket>

        i.e.,

        k ---a---> i
        j ---b---> k

        and conjugate transpose
        '''
        d2 = self._get_D2_(bra, ket) # aa, ab, ba, bb -> 0, 1, 2, 3
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k)
        fac = -1 * self.get_ovlp_fac (bra, ket, i, j, k) # a'bb'a -> a'ab'b sign
        fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j, k), i)
        fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j, k), j)
        sp = np.multiply.outer (self.ints[i].get_p (bra, ket, 0), self.ints[j].get_h (bra, ket, 1))
        sm = self.ints[k].get_sm (bra, ket)
        d2_ikkj = fac * np.multiply.outer (sp, sm).transpose (0,3,2,1) # a'bb'a -> a'ab'b transpose
        d2[1,p:q,t:u,t:u,r:s] = d2_ikkj
        d2[2,t:u,r:s,p:q,t:u] = d2_ikkj.transpose (2,3,0,1)
        self._put_D2_(bra, ket, d2)

    def _crunch_2c_(self, bra, ket, i, j, k, l, s2lt):
        '''Compute the reduced density matrix elements of a two-electron hop; i.e.,

        <bra|i'(s1)k'(s2)l(s2)j(s1)|ket>

        i.e.,

        j ---s1---> i
        l ---s2---> k

        with

        s2lt = 0, 1, 2
        s1   = a, a, b
        s2   = a, b, b

        and conjugate transpose

        Note that this includes i=k and/or j=l cases, but no other coincident fragment indices. Any
        other coincident fragment index (that is, any coincident index between the bra and the ket)
        turns this into one of the other interactions implemented in the above _crunch_ functions:
        s1 = s2  AND SORT (ik) = SORT (jl)                 : _crunch_null_
        s1 = s2  AND (i = j XOR i = l XOR j = k XOR k = l) : _crunch_1c_
        s1 != s2 AND (i = l AND j = k)                     : _crunch_1s_
        s1 != s2 AND (i = l XOR j = k)                     : _crunch_1s1c_
        '''
        # s2lt: 0, 1, 2 -> aa, ab, bb
        # s2: 0, 1, 2, 3 -> aa, ab, ba, bb
        s2  = (0, 1, 3)[s2lt] # aa, ab, bb
        s2T = (0, 2, 3)[s2lt] # aa, ba, bb -> when you populate the e1 <-> e2 permutation
        s11 = s2 // 2
        s12 = s2 % 2
        d2 = self._get_D2_(bra, ket)
        fac = self.get_ovlp_fac (bra, ket, i, j, k, l)
        if i == k:
            pp = self.ints[i].get_pp (bra, ket, s2lt)
            if s2lt != 1: assert (np.all (np.abs (pp + pp.T)) < 1e-8), '{}'.format (
                np.amax (np.abs (pp + pp.T)))
        else:
            pp = np.multiply.outer (self.ints[i].get_p (bra, ket, s11),
                                    self.ints[k].get_p (bra, ket, s12))
            fac *= (1,-1)[int (i>k)]
            fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j, k, l), i)
            fac *= fermion_des_shuffle (self.nelec_rf[bra], (i, j, k, l), k)
        if j == l:
            hh = self.ints[j].get_hh (bra, ket, s2lt)
            if s2lt != 1: assert (np.all (np.abs (hh + hh.T)) < 1e-8), '{}'.format (
                np.amax (np.abs (hh + hh.T)))
        else:
            hh = np.multiply.outer (self.ints[l].get_h (bra, ket, s12),
                                    self.ints[j].get_h (bra, ket, s11))
            fac *= (1,-1)[int (j>l)]
            fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j, k, l), j)
            fac *= fermion_des_shuffle (self.nelec_rf[ket], (i, j, k, l), l)
        d2_ijkl = fac * np.multiply.outer (pp, hh).transpose (0,3,1,2) # Dirac -> Mulliken transp
        p, q = self.get_range (i)
        r, s = self.get_range (j)
        t, u = self.get_range (k) 
        v, w = self.get_range (l)
        d2[s2, p:q,r:s,t:u,v:w] = d2_ijkl
        d2[s2T,t:u,v:w,p:q,r:s] = d2_ijkl.transpose (2,3,0,1)
        if s2 == s2T: # same-spin only: exchange happens
            d2[s2,p:q,v:w,t:u,r:s] = -d2_ijkl.transpose (0,3,2,1)
            d2[s2,t:u,r:s,p:q,v:w] = -d2_ijkl.transpose (2,1,0,3)
        self._put_D2_(bra, ket, d2)

    def _crunch_all_(self):
        for row in self.exc_null: self._crunch_null_(*row)
        for row in self.exc_1c: self._crunch_1c_(*row)
        for row in self.exc_1s: self._crunch_1s_(*row)
        for row in self.exc_1s1c: self._crunch_1s1c_(*row)
        for row in self.exc_2c: self._crunch_2c_(*row)
        self._add_transpose_()
        for state in range (self.nroots): self._crunch_null_(state, state)

    def _add_transpose_(self):
        self.tdm1s += self.tdm1s.conj ().transpose (1,0,2,4,3)
        self.tdm2s += self.tdm2s.conj ().transpose (1,0,2,4,3,6,5)

    def kernel (self):
        ''' Main driver method of class.

        Returns:
            stdm1s : ndarray of shape (nroots,nroots,2,ncas,ncas)
                1-body spin-separated LAS-state transition density matrices
            stdm2s : ndarray of shape (nroots,nroots,4,ncas,ncas,ncas,ncas)
                2-body spin-separated LAS-state transition density matrices
            t0 : tuple of length 2
                timestamp of entry into this function, for profiling by caller
        '''
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.tdm1s = np.zeros ([self.nroots,]*2 + [2,] + [self.norb,]*2, dtype=self.dtype)
        self.tdm2s = np.zeros ([self.nroots,]*2 + [4,] + [self.norb,]*4, dtype=self.dtype)
        self._crunch_all_()
        return self.tdm1s, self.tdm2s, t0

class HamS2ovlpint (LSTDMint2):
    __doc__ = LSTDMint2.__doc__ + '''

    SUBCLASS: Hamiltonian, spin-squared, and overlap matrices

    `kernel` call returns operator matrices without cacheing stdm12s array

    Additional args:
        h1 : ndarray of size ncas**2
            Contains effective 1-electron Hamiltonian amplitudes in second quantization
        h2 : ndarray of size ncas**4
            Contains 2-electron Hamiltonian amplitudes in second quantization
    '''
    # TODO: SO-LASSI o1 implementation: the one-body spin-orbit coupling part of the
    # Hamiltonian in addition to h1 and h2, which are spin-symmetric

    def __init__(self, ints, nlas, hopping_index, h1, h2, dtype=np.float64):
        LSTDMint2.__init__(self, ints, nlas, hopping_index, dtype=dtype)
        self.h1 = h1.ravel ()
        self.h2 = h2.ravel ()

    def _get_D1_(self, bra, ket):
        self.d1[:] = 0.0
        return self.d1

    def _get_D2_(self, bra, ket):
        self.d2[:] = 0.0
        return self.d2

    def _put_D1_(self, bra, ket, D1):
        M1 = D1[0] - D1[1]
        D1 = D1.sum (0)
        self.ham[bra,ket] += np.dot (self.h1, D1.ravel ())
        self.s2[bra,ket] += (np.trace (M1)/2)**2 + np.trace (D1)/2

    def _put_D2_(self, bra, ket, D2):
        self.ham[bra,ket] += np.dot (self.h2, D2.sum (0).ravel ()) / 2
        self.s2[bra,ket] -= np.einsum ('pqqp->', D2[1] + D2[2]) / 2

    def _add_transpose_(self):
        self.ham += self.ham.T
        self.s2 += self.s2.T

    def kernel (self):
        ''' Main driver method of class.

        Returns:
            ham : ndarray of shape (nroots,nroots)
                Hamiltonian in LAS product state basis
            s2 : ndarray of shape (nroots,nroots)
                Spin-squared operator in LAS product state basis
            ovlp : ndarray of shape (nroots,nroots)
                Overlap matrix of LAS product states
            t0 : tuple of length 2
                timestamp of entry into this function, for profiling by caller
        '''
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.d1 = np.zeros ([2,]+[self.norb,]*2, dtype=self.dtype)
        self.d2 = np.zeros ([4,]+[self.norb,]*4, dtype=self.dtype)
        self.ham = np.zeros ([self.nroots,]*2, dtype=self.dtype)
        self.s2 = np.zeros ([self.nroots,]*2, dtype=self.dtype)
        self._crunch_all_()
        ovlp = np.prod (self.ovlp, axis=-1)
        ovlp *= np.multiply.outer (self.spin_shuffle, self.spin_shuffle)
        return self.ham, self.s2, ovlp, t0

class LRRDMint (LSTDMint2):
    __doc__ = LSTDMint2.__doc__ + '''

    SUBCLASS: LASSI-root reduced density matrix

    `kernel` call returns reduced density matrices for LASSI eigenstates (or linear combinations of
    product states generally) without cacheing stdm12s array

    Additional args:
        si : ndarray of shape (nroots,nroots_si)
            Contains LASSI eigenvectors
    '''
    # TODO: at some point, if it ever becomes rate-limiting, make this multithread better
    # TODO: SO-LASSI o1 implementation: these density matrices can only be defined in the full
    # spinorbital basis

    def __init__(self, ints, nlas, hopping_index, si, dtype=np.float64):
        LSTDMint2.__init__(self, ints, nlas, hopping_index, dtype=dtype)
        self.nroots_si = si.shape[-1]
        self.si_dm = np.stack ([np.dot (si[:,i:i+1],si[:,i:i+1].conj ().T)
            for i in range (self.nroots_si)], axis=-1)

    def _get_D1_(self, bra, ket):
        self.d1[:] = 0.0
        return self.d1

    def _get_D2_(self, bra, ket):
        self.d2[:] = 0.0
        return self.d2

    def _put_D1_(self, bra, ket, D1):
        self.rdm1s[:] += np.multiply.outer (self.si_dm[bra,ket,:], D1)

    def _put_D2_(self, bra, ket, D2):
        self.rdm2s[:] += np.multiply.outer (self.si_dm[bra,ket,:], D2)

    def _add_transpose_(self):
        self.rdm1s += self.rdm1s.conj ().transpose (0,1,3,2)
        self.rdm2s += self.rdm2s.conj ().transpose (0,1,3,2,5,4)

    def kernel (self):
        ''' Main driver method of class.

        Returns:
            rdm1s : ndarray of shape (nroots_si,2,ncas,ncas)
                Spin-separated 1-body reduced density matrices of LASSI states
            rdm2s : ndarray of shape (nroots_si,4,ncas,ncas,ncas,ncas)
                Spin-separated 2-body reduced density matrices of LASSI states
            t0 : tuple of length 2
                timestamp of entry into this function, for profiling by caller
        '''
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.d1 = np.zeros ([2,]+[self.norb,]*2, dtype=self.dtype)
        self.d2 = np.zeros ([4,]+[self.norb,]*4, dtype=self.dtype)
        self.rdm1s = np.zeros ([self.nroots_si,] + list (self.d1.shape), dtype=self.dtype)
        self.rdm2s = np.zeros ([self.nroots_si,] + list (self.d2.shape), dtype=self.dtype)
        self._crunch_all_()
        return self.rdm1s, self.rdm2s, t0

def make_ints (las, ci, nelec_frs):
    ''' Build fragment-local intermediates (`LSTDMint1`) for LASSI o1

    Args:
        las : instance of :class:`LASCINoSymm`
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Returns:
        hopping_index : ndarray of ints of shape (nfrags, 2, nroots, nroots)
            element [i,j,k,l] reports the change of number of electrons of
            spin j in fragment i between LAS states k and l
        ints : list of length nfrags of instances of :class:`LSTDMint1`
    '''
    fciboxes = las.fciboxes
    nfrags = len (fciboxes)
    nroots = nelec_frs.shape[1]
    nlas = las.ncas_sub
    nelelas = [sum (_unpack_nelec (ne)) for ne in las.nelecas_sub]
    lroots = np.array ([[1 if ci_ij.ndim<3 else ci_ij.shape[0]
                         for ci_ij in ci_i]
                        for ci_i in ci])
    if np.any(lroots>1): raise NotImplementedError ("LASSI o1 algorithm w/ local excitations")
    hopping_index, zerop_index, onep_index = lst_hopping_index (fciboxes, nlas, nelelas, nelec_frs)
    ints = []
    for ifrag in range (nfrags):
        tdmint = LSTDMint1 (fciboxes[ifrag], nlas[ifrag], nelelas[ifrag], nroots,
                            hopping_index[ifrag], ifrag)
        t0 = tdmint.kernel (ci[ifrag], hopping_index[ifrag], zerop_index, onep_index)
        lib.logger.timer (las, 'LAS-state TDM12s fragment {} intermediate crunching'.format (
            ifrag), *t0)
        ints.append (tdmint)
    return hopping_index, ints

def make_stdm12s (las, ci, nelec_frs, **kwargs):
    ''' Build spin-separated LAS product-state 1- and 2-body transition density matrices

    Args:
        las : instance of :class:`LASCINoSymm`
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Returns:
        tdm1s : ndarray of shape (nroots,2,ncas,ncas,nroots)
            Contains 1-body LAS state transition density matrices
        tdm2s : ndarray of shape (nroots,2,ncas,ncas,2,ncas,ncas,nroots)
            Contains 2-body LAS state transition density matrices
    '''
    nlas = las.ncas_sub
    ncas = las.ncas
    nroots = nelec_frs.shape[1]

    # First pass: single-fragment intermediates
    hopping_index, ints = make_ints (las, ci, nelec_frs)

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    outerprod = LSTDMint2 (ints, nlas, hopping_index, dtype=ci[0][0].dtype)
    lib.logger.timer (las, 'LAS-state TDM12s second intermediate indexing setup', *t0)        
    tdm1s, tdm2s, t0 = outerprod.kernel ()
    lib.logger.timer (las, 'LAS-state TDM12s second intermediate crunching', *t0)        

    # Put tdm1s in PySCF convention: [p,q] -> q'p
    tdm1s = tdm1s.transpose (0,2,4,3,1)
    tdm2s = tdm2s.reshape (nroots,nroots,2,2,ncas,ncas,ncas,ncas).transpose (0,2,4,5,3,6,7,1)
    return tdm1s, tdm2s

def ham (las, h1, h2, ci, nelec_frs, **kwargs):
    ''' Build Hamiltonian, spin-squared, and overlap matrices in LAS product state basis

    Args:
        las : instance of :class:`LASCINoSymm`
        h1 : ndarray of size ncas**2
            Contains effective 1-electron Hamiltonian amplitudes in second quantization
        h2 : ndarray of size ncas**4
            Contains 2-electron Hamiltonian amplitudes in second quantization
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Returns:
        ham : ndarray of shape (nroots,nroots)
            Hamiltonian in LAS product state basis
        s2 : ndarray of shape (nroots,nroots)
            Spin-squared operator in LAS product state basis
        ovlp : ndarray of shape (nroots,nroots)
            Overlap matrix of LAS product states
    '''
    nlas = las.ncas_sub

    # First pass: single-fragment intermediates
    hopping_index, ints = make_ints (las, ci, nelec_frs)

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    outerprod = HamS2ovlpint (ints, nlas, hopping_index, h1, h2, dtype=ci[0][0].dtype)
    lib.logger.timer (las, 'LASSI Hamiltonian second intermediate indexing setup', *t0)        
    ham, s2, ovlp, t0 = outerprod.kernel ()
    lib.logger.timer (las, 'LASSI Hamiltonian second intermediate crunching', *t0)        
    return ham, s2, ovlp


def roots_make_rdm12s (las, ci, nelec_frs, si, **kwargs):
    ''' Build spin-separated LASSI 1- and 2-body reduced density matrices

    Args:
        las : instance of :class:`LASCINoSymm`
        ci : list of list of ndarrays
            Contains all CI vectors
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si : ndarray of shape (nroots,nroots_si)
            Contains LASSI eigenvectors

    Returns:
        rdm1s : ndarray of shape (nroots_si,2,ncas,ncas)
            Spin-separated 1-body reduced density matrices of LASSI states
        rdm2s : ndarray of shape (nroots_si,2,ncas,ncas,2,ncas,ncas)
            Spin-separated 2-body reduced density matrices of LASSI states
    '''
    nlas = las.ncas_sub
    ncas = las.ncas
    nroots_si = si.shape[-1]

    # First pass: single-fragment intermediates
    hopping_index, ints = make_ints (las, ci, nelec_frs)

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    outerprod = LRRDMint (ints, nlas, hopping_index, si, dtype=ci[0][0].dtype)
    lib.logger.timer (las, 'LASSI root RDM12s second intermediate indexing setup', *t0)        
    rdm1s, rdm2s, t0 = outerprod.kernel ()
    lib.logger.timer (las, 'LASSI root RDM12s second intermediate crunching', *t0)

    # Put rdm1s in PySCF convention: [p,q] -> q'p
    rdm1s = rdm1s.transpose (0,1,3,2)
    rdm2s = rdm2s.reshape (nroots_si, 2, 2, ncas, ncas, ncas, ncas).transpose (0,1,3,4,2,5,6)
    return rdm1s, rdm2s

