import sys
import numpy as np
from scipy import linalg
from pyscf.fci import cistring
from pyscf import fci, lib
from pyscf.fci.direct_nosym import contract_1e as contract_1e_nosym
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import contract_ss, spin_square
from pyscf.data import nist
from itertools import combinations
from mrh.my_pyscf.mcscf import soc_int as soc_int
from mrh.my_pyscf.lassi import dms as lassi_dms

def memcheck (las, ci, soc=None):
    '''Check if the system has enough memory to run these functions! ONLY checks
    if the CI vectors can be stored in memory!!!'''
    nfrags = len (ci)
    nroots = len (ci[0])
    assert (all ([len (c) == nroots for c in ci]))
    lroots_fr = np.array ([[1 if c.ndim<3 else c.shape[0]
                            for c in ci_r]
                           for ci_r in ci])
    lroots_r = np.product (lroots_fr, axis=0)
    nelec_frs = np.array ([[list (_unpack_nelec (fcibox._get_nelec (solver, nelecas)))
                            for solver in fcibox.fcisolvers]
                           for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)])
    nelec_rs = np.unique (nelec_frs.sum (0), axis=0)
    ndet_spinfree = max ([cistring.num_strings (las.ncas, na) 
                          *cistring.num_strings (las.ncas, nb)
                          for na, nb in nelec_rs])
    ndet_soc = max ([cistring.num_strings (2*las.ncas, nelec) for nelec in nelec_rs.sum (1)])
    nbytes_per_sfvec = ndet_spinfree * np.dtype (float).itemsize 
    nbytes_per_sovec = ndet_soc * np.dtype (complex).itemsize
    # 3 vectors: the bra (from a generator), the ket (from a generator), and op|bra>
    # for SOC, each generator must also store the spinfree version
    if soc:
        nbytes = 2*nbytes_per_sfvec + 3*nbytes_per_sovec
    else:
        nbytes = 3*nbytes_per_sfvec
    # memory load of ci_dp vectors
    nbytes += sum ([np.prod ([c[iroot].size for c in ci])
                    * np.amax ([c[iroot].dtype.itemsize for c in ci])
                    for iroot in range (nroots)])
    safety_factor = 1.2
    mem = nbytes * safety_factor / 1e6
    max_memory = las.max_memory - lib.current_memory ()[0]
    lib.logger.info (las,
        "LASSI op_o0 memory check: {} MB needed of {} MB available ({} MB max)".format (mem,\
        max_memory, las.max_memory))
    return mem < max_memory

def civec_spinless_repr_generator (ci0_r, norb, nelec_r):
    '''Put CI vectors in the spinless representation; i.e., map
        norb -> 2 * norb
        (neleca, nelecb) -> (neleca+nelecb, 0)
    This permits linear combinations of CI vectors with different
    M == neleca-nelecb at the price of higher memory cost. This function
    does NOT change the datatype.

    Args:
        ci0_r: sequence or generator of ndarray of length nprods
            CAS-CI vectors in the spin-pure representation
        norb: integer
            Number of orbitals
        nelec_r: sequence of tuple of length (2)
            (neleca, nelecb) for each element of ci0_r

    Returns:
        ci1_r_gen: callable that returns a generator of length nprods
            generates spinless CAS-CI vectors
        reverser: callable
            Perform the reverse operation on a spinless CAS-CI vector
            Args:
                ci2: ndarray
                    spinless CAS-CI vector
                ne: tuple of length 2
                    neleca, nelecb target Hilbert space
            Returns:
                ci3: ndarray
                    CAS-CI vector of ci2 in the (neleca, nelecb) Hilbert space
    '''
    nelec_r_tot = [sum (n) for n in nelec_r]
    if len (set (nelec_r_tot)) > 1:
        raise NotImplementedError ("Different particle-number subspaces")
    nelec = nelec_r_tot[0]
    addrs = {}
    ndet_sp = {}
    for ne in set (nelec_r):
        neleca, nelecb = _unpack_nelec (ne)
        ndeta = cistring.num_strings (norb, neleca)
        ndetb = cistring.num_strings (norb, nelecb)
        strsa = cistring.addrs2str (norb, neleca, list(range(ndeta)))
        strsb = cistring.addrs2str (norb, nelecb, list(range(ndetb)))
        strs = np.add.outer (strsa, np.left_shift (strsb, norb)).ravel ()
        addrs[ne] = cistring.strs2addr (2*norb, nelec, strs)
        ndet_sp[ne] = tuple((ndeta,ndetb))
    strs = strsa = strsb = None
    ndet = cistring.num_strings (2*norb, nelec)
    nstates = len (nelec_r)
    def ci1_r_gen (buf=None):
        if callable (ci0_r):
            ci0_r_gen = ci0_r ()
        else:
            ci0_r_gen = (c for c in ci0_r)
        ci0 = next (ci0_r_gen)
        if buf is None:
            ci1 = np.empty (ndet, dtype=ci0.dtype)
        else:
            ci1 = np.asarray (buf).flat[:ndet]
        for istate, ne in enumerate (nelec_r):
            ci1[:] = 0.0
            ci1[addrs[ne]] = ci0[:,:].ravel ()
            neleca, nelecb = _unpack_nelec (ne)
            if abs(neleca*nelecb)%2: ci1[:] *= -1
            # Sign comes from changing representation:
            # ... a2' a1' a0' ... b2' b1' b0' |vac>
            # ->
            # ... b2' b1' b0' .. a2' a1' a0' |vac>
            # i.e., strictly decreasing from left to right
            # (the ordinality of spin-down is conventionally greater than spin-up)
            yield ci1[:,None]
            if istate+1==nstates: break
            ci0 = next (ci0_r_gen)
    def reverser (ci2, ne):
        ''' Generate the spin-separated CI vector in a particular M
        Hilbert space from a spinless CI vector '''
        ci3 = ci2[addrs[ne]].reshape (ndet_sp[ne])
        neleca, nelecb = _unpack_nelec (ne)
        if abs(neleca*nelecb)%2: ci3[:] *= -1
        return ci3
    return ci1_r_gen, reverser

def civec_spinless_repr (ci0_r, norb, nelec_r):
    ci1_r_gen, _ = civec_spinless_repr_generator (ci0_r, norb, nelec_r)
    ci1_r = np.stack ([x.copy () for x in ci1_r_gen ()], axis=0)
    return ci1_r

def addr_outer_product (norb_f, nelec_f):
    '''Build index arrays for reshaping a direct product of LAS CI
    vectors into the appropriate orbital ordering for a CAS CI vector'''
    norb = sum (norb_f)
    nelec = sum (nelec_f)
    # Must skip over cases where there are no electrons of a specific spin in a particular subspace
    norbrange = np.cumsum (norb_f)
    addrs = []
    for i in range (0, len (norbrange)):
        irange = range (norbrange[i]-norb_f[i], norbrange[i])
        new_addrs = cistring.sub_addrs (norb, nelec, irange, nelec_f[i]) if nelec_f[i] else []
        if len (addrs) == 0:
            addrs = new_addrs
        elif len (new_addrs) > 0:
            addrs = np.intersect1d (addrs, new_addrs)
    if not len (addrs): addrs=[0] # No beta electrons edge case
    return addrs

def _ci_outer_product (ci_f, norb_f, nelec_f):
    '''Compute outer-product CI vector for one space table from fragment LAS CI vectors.
    See "ci_outer_product"'''
    nfrags = len (norb_f)
    neleca_f = [ne[0] for ne in nelec_f]
    nelecb_f = [ne[1] for ne in nelec_f]
    lroots_f = [1 if ci.ndim<3 else ci.shape[0] for ci in ci_f]
    nprods = np.prod (lroots_f)
    #addrs_las = np.stack (np.meshgrid (*[np.arange(l) for l in lroots_f[::-1]],
    #                                   indexing='ij'), axis=0)
    #addrs_las = addrs_las.reshape (nfrags, nprods)[::-1,:].T
    shape_f = [(lroots, cistring.num_strings (norb, neleca), cistring.num_strings (norb, nelecb))
              for lroots, norb, neleca, nelecb in zip (lroots_f, norb_f, neleca_f, nelecb_f)]
    addrs_a = addr_outer_product (norb_f, neleca_f)
    addrs_b = addr_outer_product (norb_f, nelecb_f)
    idx = np.ix_(addrs_a,addrs_b)
    addrs_a = addrs_b = None
    neleca = sum (neleca_f)
    nelecb = sum (nelecb_f)
    nelec = tuple ((neleca, nelecb))
    ndet_a = cistring.num_strings (sum (norb_f), neleca)
    ndet_b = cistring.num_strings (sum (norb_f), nelecb)
    ci_dp = ci_f[-1].reshape (shape_f[-1])
    for ci_r, shape in zip (ci_f[-2::-1], shape_f[-2::-1]):
        lroots, ndeta, ndetb = ci_dp.shape
        ci_dp = np.multiply.outer (ci_dp, ci_r.reshape (shape))
        ci_dp = ci_dp.transpose (0,3,1,4,2,5).reshape (
            lroots*shape[0], ndeta*shape[1], ndetb*shape[2]
        )
    norm_dp = linalg.norm (ci_dp.reshape (ci_dp.shape[0],-1), axis=1)
    ci_dp /= norm_dp[:,None,None]
    def gen_ci_dp (buf=None):
        if buf is None:
            ci = np.empty ((ndet_a,ndet_b), dtype=ci_f[-1].dtype)
        else:
            ci = np.asarray (buf.flat[:ndet_a*ndet_b]).reshape (ndet_a, ndet_b)
        for vec in ci_dp:
            ci[:,:] = 0.0
            ci[idx] = vec[:,:]
            yield ci
    def dotter (c1, nelec1):
        if nelec1 != nelec: return np.zeros (nprods)
        return np.dot (ci_dp.reshape (nprods, -1),
                       c1[idx].ravel ())
    return gen_ci_dp, nprods, dotter

def ci_outer_product_generator (ci_fr, norb_f, nelec_fr):
    '''Compute outer-product CI vectors from fragment LAS CI vectors and return
    result as a generator

    Args:
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (lroots[i,j],ndeta[i,j],ndetb[i,j])
        norb_f : list of length (nfrags)
            Number of orbitals in each fragment
        nelec_fr : ndarray-like of shape (nfrags, nroots, 2)
            Number of spin-up and spin-down electrons in each fragment
            and root

    Returns:
        ci_r_gen : callable that returns a generator of length (nprods)
            Generates all direct-product CAS CI vectors
        nelec_r : list of length (nroots) of tuple of length 2
            (neleca, nelecb) for each product state
        dotter : callable
            Performs the dot product in the outer product basis
            on a CAS CI vector, without explicitly constructing
            any direct-product CAS CI vectors (again).
            Args:
                c1 : ndarray
                    contains CAS-CI vector
                nelec1 : tuple of length 2
                    neleca, nelecb
            Kwargs:
                reverser : callable
                    Converts ci back from the spinless representation
                    into the (neleca,nelecb) representation. Takes c1
                    and nelec1 and returns a new ci vector
            Returns:
                ndarray of length (nprods)
                    Expansion coefficients for c1 in terms of direct-
                    product states of ci_fr
    '''

    norb = sum (norb_f)
    ndet = max ([cistring.num_strings (norb, ne[0]) * cistring.num_strings (norb, ne[1])
                for ne in np.sum (nelec_fr, axis=0)])
    gen_ci_r = []
    nelec_r = []
    dotter_r = []
    for space in range (len (ci_fr[0])):
        ci_f = [ci[space] for ci in ci_fr]
        nelec_f = [nelec[space] for nelec in nelec_fr]
        nelec = (sum ([ne[0] for ne in nelec_f]), sum ([ne[1] for ne in nelec_f]))
        gen_ci, nprods, dotter = _ci_outer_product (ci_f, norb_f, nelec_f)
        gen_ci_r.append (gen_ci)
        nelec_r.extend ([nelec,]*nprods)
        dotter_r.append ([dotter, nelec, nprods])
    def ci_r_gen (buf=None):
        if buf is None:
            buf1 = np.empty (ndet, dtype=ci_fr[-1][0].dtype)
        else:
            buf1 = np.asarray (buf.flat[:ndet])
        for gen_ci in gen_ci_r:
            for x in gen_ci (buf=buf1):
                yield x
    def dotter (c1, nelec1, reverser=None):
        vec = []
        for dot, ne, nprods in dotter_r:
            wc1 = c1
            if callable (reverser):
                if sum (ne) == sum (nelec1):
                    wc1 = reverser (c1, ne)
                    vec.append (dot (wc1, ne))
                else:
                    vec.append (np.zeros (nprods))
            elif ne==nelec1:
                vec.append (dot (wc1, nelec1))
            else:
                vec.append (np.zeros (nprods))
        return np.concatenate (vec)
    return ci_r_gen, nelec_r, dotter

def ci_outer_product (ci_fr, norb_f, nelec_fr):
    '''Compute outer-product CI vectors from fragment LAS CI vectors.

    Args:
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (lroots[i,j],ndeta[i,j],ndetb[i,j])
        norb_f : list of length (nfrags)
            Number of orbitals in each fragment
        nelec_fr : ndarray-like of shape (nfrags, nroots, 2)
            Number of spin-up and spin-down electrons in each fragment
            and root

    Returns:
        ci_r : list of length (nroots)
            Contains full CAS CI vector
        nelec_r : list of length (nroots) of tuple of length 2
            (neleca, nelecb) for each product state
    '''
    ci_r_gen, nelec_r, _ = ci_outer_product_generator (ci_fr, norb_f, nelec_fr)
    ci_r = [x.copy () for x in ci_r_gen ()]
    return ci_r, nelec_r

#def si_soc (las, h1, ci, nelec, norb):
#
#### function adapted from github.com/hczhai/fci-siso/blob/master/fcisiso.py ###
#
##    au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
#    nroots = len(ci)
#    hsiso = np.zeros((nroots, nroots), dtype=complex)
#    ncas = las.ncas
#    hso_m1 = h1[ncas:2*ncas,0:ncas]
#    hso_p1 = h1[0:ncas,ncas:2*ncas]
#    hso_ze = (h1[0:ncas,0:ncas] - h1[ncas:2*ncas,ncas:2*ncas])/2 
#
#    for istate, (ici, inelec) in enumerate(zip(ci, nelec)):
#        for jstate, (jci, jnelec) in enumerate(zip(ci, nelec)):
#            if jstate > istate:
#                continue
#
#            tp1 = lassi_dms.make_trans(1, ici, jci, norb, inelec, jnelec)
#            tze = lassi_dms.make_trans(0, ici, jci, norb, inelec, jnelec)
#            tm1 = lassi_dms.make_trans(-1, ici, jci, norb, inelec, jnelec)
#
#            if tp1.shape == ():
#                tp1 = np.zeros((ncas,ncas))
#            if tze.shape == ():
#                tze = np.zeros((ncas,ncas))
#            if tm1.shape == ():
#                tm1 = np.zeros((ncas,ncas))
#
#            somat = np.einsum('ri, ir ->', tm1, hso_m1)
#            somat += np.einsum('ri, ir ->', tp1, hso_p1)
#            #somat = somat/2
#            somat += np.einsum('ri, ir ->', tze, hso_ze)
#
#            hsiso[jstate, istate] = somat
#            if istate!= jstate:
#                hsiso[istate, jstate] = somat.conj()
##            somat *= au2cm
#
#    #heigso, hvecso = np.linalg.eigh(hsiso)
#
#    return hsiso

def ham (las, h1, h2, ci_fr, nelec_frs, soc=0, orbsym=None, wfnsym=None):
    '''Build LAS state interaction Hamiltonian, S2, and ovlp matrices

    Args:
        las : instance of class LASSCF
        h1 : ndarray of shape (ncas, ncas)
            Spin-orbit-free one-body CAS Hamiltonian
        h2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-orbit-free two-body CAS Hamiltonian
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        ham_eff : square ndarray of length (nroots)
            Spin-orbit-free Hamiltonian in state-interaction basis
        s2_eff : square ndarray of length (nroots)
            S2 operator matrix in state-interaction basis
        ovlp_eff : square ndarray of length (nroots)
            Overlap matrix in state-interaction basis
    '''
    if soc>1:
        raise NotImplementedError ("Two-electron spin-orbit coupling")
    mol = las.mol
    norb_f = las.ncas_sub
    norb = sum (norb_f)

    # The function below is the main workhorse of this whole implementation
    ci_r_generator, nelec_r, dotter = ci_outer_product_generator (ci_fr, norb_f, nelec_frs)
    nroots = len(nelec_r)
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r]
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    # S2 best taken care of before "spinless representation"
    reverser = None
    s2_eff = np.zeros ((nroots,nroots))
    for i, c, nelec_ket in zip(range(nroots), ci_r_generator (), nelec_r):
        s2c = contract_ss (c, norb, nelec_ket)
        s2_eff[i,:] = dotter (s2c, nelec_ket, reverser=reverser)
    # Hamiltonian may be complex
    h1_re = h1.real
    h2_re = h2.real
    h1_im = None
    if soc:
        h1_im = h1.imag
        h2_re = np.zeros ([2,norb,]*4, dtype=h1_re.dtype)
        h2_re[0,:,0,:,0,:,0,:] = h2[:]
        h2_re[1,:,1,:,0,:,0,:] = h2[:]
        h2_re[0,:,0,:,1,:,1,:] = h2[:]
        h2_re[1,:,1,:,1,:,1,:] = h2[:]
        h2_re = h2_re.reshape ([2*norb,]*4)
        ci_r_generator, reverser = civec_spinless_repr_generator (ci_r_generator, norb, nelec_r)
        nelec_r = nelec_r_spinless
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2

    solver = fci.solver (mol, symm=(wfnsym is not None)).set (orbsym=orbsym, wfnsym=wfnsym)
    ham_eff = np.zeros ((nroots, nroots), dtype=h1.dtype)
    ovlp_eff = np.zeros ((nroots, nroots))
    for i, ket, nelec_ket in zip(range(nroots), ci_r_generator (), nelec_r):
        ovlp_eff[i,:] = dotter (ket, nelec_ket, reverser=reverser)
        h2eff = solver.absorb_h1e (h1_re, h2_re, norb, nelec_ket, 0.5)
        hket = solver.contract_2e (h2eff, ket, norb, nelec_ket)
        if h1_im is not None:
            hket = hket + 1j*contract_1e_nosym (h1_im, ket, norb, nelec_ket)
        ham_eff[i,:] = dotter (hket, nelec_ket, reverser=reverser)
    
    return ham_eff, s2_eff, ovlp_eff

def make_stdm12s (las, ci_fr, nelec_frs, orbsym=None, wfnsym=None):
    '''Build LAS state interaction transition density matrices

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        stdm1s : ndarray of shape (nroots,2,ncas,ncas,nroots) OR (nroots,2*ncas,2*ncas,nroots)
            One-body transition density matrices between LAS states.
            If states with different spin projections (i.e., neleca-nelecb) are present, the 4d
            spinorbital array is returned. Otherwise, the 5d spatial-orbital array is returned.
        stdm2s : ndarray of shape [nroots,]+ [2,ncas,ncas,]*2 + [nroots,]
            Two-body transition density matrices between LAS states
    '''
    mol = las.mol
    norb_f = las.ncas_sub
    norb = sum (norb_f) 
    ci_r_generator, nelec_r, dotter = ci_outer_product_generator (ci_fr, norb_f, nelec_frs)
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r]
    nroots = len (nelec_r)
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    spin_pure = len (set (nelec_r)) == 1

    dtype = ci_fr[-1][0].dtype
    if not spin_pure:
        # Map to "spinless electrons": 
        ci_r_generator, reverser = civec_spinless_repr_generator (ci_r_generator, norb, nelec_r)
        nelec_r = nelec_r_spinless
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2

    solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    stdm1s = np.zeros ((nroots, nroots, 2, norb, norb),
        dtype=dtype).transpose (0,2,3,4,1)
    stdm2s = np.zeros ((nroots, nroots, 2, norb, norb, 2, norb, norb),
        dtype=dtype).transpose (0,2,3,4,5,6,7,1)
    for i, (ci, ne) in enumerate (zip (ci_r_generator (), nelec_r)):
        rdm1s, rdm2s = solver.make_rdm12s (ci, norb, ne)
        stdm1s[i,0,:,:,i] = rdm1s[0]
        stdm1s[i,1,:,:,i] = rdm1s[1]
        stdm2s[i,0,:,:,0,:,:,i] = rdm2s[0]
        stdm2s[i,0,:,:,1,:,:,i] = rdm2s[1]
        stdm2s[i,1,:,:,0,:,:,i] = rdm2s[1].transpose (2,3,0,1)
        stdm2s[i,1,:,:,1,:,:,i] = rdm2s[2]

    spin_sector_offset = np.zeros ((nroots,nroots))
    for i, (ci_bra, ne_bra) in enumerate (zip (ci_r_generator (), nelec_r)):
        for j, (ci_ket, ne_ket) in enumerate (zip (ci_r_generator (), nelec_r)):
            if j==i: break
            M_bra = ne_bra[1] - ne_bra[0]
            M_ket = ne_ket[0] - ne_ket[1]
            N_bra = sum (ne_bra)
            N_ket = sum (ne_ket)
            if ne_bra == ne_ket:
                tdm1s, tdm2s = solver.trans_rdm12s (ci_bra, ci_ket, norb, ne_bra)
                stdm1s[i,0,:,:,j] = tdm1s[0]
                stdm1s[i,1,:,:,j] = tdm1s[1]
                stdm1s[j,0,:,:,i] = tdm1s[0].T
                stdm1s[j,1,:,:,i] = tdm1s[1].T
                for spin, tdm2 in enumerate (tdm2s):
                    p = spin // 2
                    q = spin % 2
                    stdm2s[i,p,:,:,q,:,:,j] = tdm2
                    stdm2s[j,p,:,:,q,:,:,i] = tdm2.transpose (1,0,3,2)

    if not spin_pure: # cleanup the "spinless mapping"
        stdm1s = stdm1s[:,0,:,:,:]
        # TODO: 2e- spin-orbit coupling support in caller
        n = norb // 2
        stdm2s_ = np.zeros ((nroots, nroots, 2, n, n, 2, n, n),
            dtype=dtype).transpose (0,2,3,4,5,6,7,1)
        stdm2s_[:,0,:,:,0,:,:,:] = stdm2s[:,0,:n,:n,0,:n,:n,:]
        stdm2s_[:,0,:,:,1,:,:,:] = stdm2s[:,0,:n,:n,0,n:,n:,:]
        stdm2s_[:,1,:,:,0,:,:,:] = stdm2s[:,0,n:,n:,0,:n,:n,:]
        stdm2s_[:,1,:,:,1,:,:,:] = stdm2s[:,0,n:,n:,0,n:,n:,:]
        stdm2s = stdm2s_

    return stdm1s, stdm2s 

def root_make_rdm12s (las, ci_fr, nelec_frs, si, ix, orbsym=None, wfnsym=None):
    '''Build LAS state interaction reduced density matrices for 1 final
    LASSI eigenstate.

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states
        ix : integer
            Index of column of si to use

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        rdm1s : ndarray of shape (2, ncas, ncas) OR (2*ncas, 2*ncas)
            One-body transition density matrices between LAS states
            If states with different spin projections (i.e., neleca-nelecb) are present, the 2d
            spinorbital array is returned. Otherwise, the 3d spatial-orbital array is returned.
        rdm2s : ndarray of length (2, ncas, ncas, 2, ncas, ncas)
            Two-body transition density matrices between LAS states
    '''
    mol = las.mol
    norb_f = las.ncas_sub
    ci_r_gen, nelec_r, dotter = ci_outer_product_generator (ci_fr, norb_f, nelec_frs)
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r]
    nroots = len (nelec_r)
    norb = sum (norb_f)
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    spin_pure = len (set (nelec_r)) == 1
    if not spin_pure:
        # Map to "spinless electrons": 
        ci_r_gen, reverser = civec_spinless_repr_generator (ci_r_gen, norb, nelec_r)
        nelec_r = nelec_r_spinless
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2
    nelec_r = nelec_r[0]

    ndeta = cistring.num_strings (norb, nelec_r[0])
    ndetb = cistring.num_strings (norb, nelec_r[1])
    ci_r = np.zeros ((ndeta,ndetb), dtype=si.dtype)
    for coeff, c in zip (si[:,ix], ci_r_gen ()):
        try:
            ci_r[:,:] += coeff * c
        except ValueError as err:
            print (ci_r.shape, ndeta, ndetb, c.shape)
            raise (err)
    ci_r_real = np.ascontiguousarray (ci_r.real)
    rdm1s = np.zeros ((2, norb, norb), dtype=ci_r.dtype)
    rdm2s = np.zeros ((2, norb, norb, 2, norb, norb), dtype=ci_r.dtype)
    is_complex = np.iscomplexobj (ci_r)
    if is_complex:
        #solver = fci.fci_dhf_slow.FCISolver (mol)
        #for ix, ci in enumerate (ci_r):
        #    d1, d2 = solver.make_rdm12 (ci, norb, sum(nelec_r))
        #    rdm1s[ix,0,:,:] = d1[:]
        #    rdm2s[ix,0,:,:,0,:,:] = d2[:]
        # ^ this is WAY too slow!
        ci_r_imag = np.ascontiguousarray (ci_r.imag)
    else:
        ci_r_imag = [0,]*nroots
        #solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    d1s, d2s = solver.make_rdm12s (ci_r_real, norb, nelec_r)
    d2s = (d2s[0], d2s[1], d2s[1].transpose (2,3,0,1), d2s[2])
    if is_complex:
        d1s = np.asarray (d1s, dtype=complex)
        d2s = np.asarray (d2s, dtype=complex)
        d1s2, d2s2 = solver.make_rdm12s (ci_r_imag, norb, nelec_r)
        d2s2 = (d2s2[0], d2s2[1], d2s2[1].transpose (2,3,0,1), d2s2[2])
        d1s += np.asarray (d1s2)
        d2s += np.asarray (d2s2)
        d1s2, d2s2 = solver.trans_rdm12s (ci_r_real, ci_r_imag, norb, nelec_r)
        d1s2 -= np.asarray (d1s2).transpose (0,2,1)
        d2s2 -= np.asarray (d2s2).transpose (0,2,1,4,3)
        d1s -= 1j * d1s2 
        d2s += 1j * d2s2
    rdm1s[0,:,:] = d1s[0]
    rdm1s[1,:,:] = d1s[1]
    rdm2s[0,:,:,0,:,:] = d2s[0]
    rdm2s[0,:,:,1,:,:] = d2s[1]
    rdm2s[1,:,:,0,:,:] = d2s[2]
    rdm2s[1,:,:,1,:,:] = d2s[3]

    if not spin_pure: # cleanup the "spinless mapping"
        rdm1s = rdm1s[0,:,:]
        # TODO: 2e- SOC
        n = norb // 2
        rdm2s_ = np.zeros ((2, n, n, 2, n, n), dtype=ci_r.dtype)
        rdm2s_[0,:,:,0,:,:] = rdm2s[0,:n,:n,0,:n,:n]
        rdm2s_[0,:,:,1,:,:] = rdm2s[0,:n,:n,0,n:,n:]
        rdm2s_[1,:,:,0,:,:] = rdm2s[0,n:,n:,0,:n,:n]
        rdm2s_[1,:,:,1,:,:] = rdm2s[0,n:,n:,0,n:,n:]
        rdm2s = rdm2s_

    return rdm1s, rdm2s

def roots_make_rdm12s (las, ci_fr, nelec_frs, si, orbsym=None, wfnsym=None):
    '''Build LAS state interaction reduced density matrices for final
    LASSI eigenstates.

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        rdm1s : ndarray of shape (nroots, 2, ncas, ncas) OR (nroots, 2*ncas, 2*ncas)
            One-body transition density matrices between LAS states
            If states with different spin projections (i.e., neleca-nelecb) are present, the 3d
            spinorbital array is returned. Otherwise, the 4d spatial-orbital array is returned.
        rdm2s : ndarray of length (nroots, 2, ncas, ncas, 2, ncas, ncas)
            Two-body transition density matrices between LAS states
    '''
    rdm1s = []
    rdm2s = []
    for ix in range (si.shape[1]):
        d1, d2 = root_make_rdm12s (las, ci_fr, nelec_frs, si, ix, orbsym=orbsym, wfnsym=wfnsym)
        rdm1s.append (d1)
        rdm2s.append (d2)
    return np.stack (rdm1s, axis=0), np.stack (rdm2s, axis=0)

if __name__ == '__main__':
    from pyscf import scf, lib
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
    import os
    class cd:
        """Context manager for changing the current working directory"""
        def __init__(self, newPath):
            self.newPath = os.path.expanduser(newPath)

        def __enter__(self):
            self.savedPath = os.getcwd()
            os.chdir(self.newPath)

        def __exit__(self, etype, value, traceback):
            os.chdir(self.savedPath)
    from mrh.examples.lasscf.c2h6n4.c2h6n4_struct import structure as struct
    with cd ("/home/herme068/gits/mrh/examples/lasscf/c2h6n4"):
        mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG
    mol.output = 'sa_lasscf_slow_ham.log'
    mol.build ()
    mf = scf.RHF (mol).run ()
    tol = 1e-6 if len (sys.argv) < 2 else float (sys.argv[1])
    las = LASSCF (mf, (4,4), (4,4)).set (conv_tol_grad = tol)
    mo = las.localize_init_guess ((list(range(3)),list(range(9,12))), mo_coeff=mf.mo_coeff)
    las.state_average_(weights = [0.5, 0.5], spins=[[0,0],[2,-2]])
    h2eff_sub, veff = las.kernel (mo)[-2:]
    e_states = las.e_states

    ncore, ncas, nocc = las.ncore, las.ncas, las.ncore + las.ncas
    mo_coeff = las.mo_coeff
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    e0 = las._scf.energy_nuc () + 2 * (((las._scf.get_hcore () + veff.c/2) @ mo_core) * mo_core).sum () 
    h1 = mo_cas.conj ().T @ (las._scf.get_hcore () + veff.c) @ mo_cas
    h2 = h2eff_sub[ncore:nocc].reshape (ncas*ncas, ncas * (ncas+1) // 2)
    h2 = lib.numpy_helper.unpack_tril (h2).reshape (ncas, ncas, ncas, ncas)
    nelec_fr = []
    for fcibox, nelec in zip (las.fciboxes, las.nelecas_sub):
        ne = sum (nelec)
        nelec_fr.append ([_unpack_nelec (fcibox._get_nelec (solver, ne)) for solver in fcibox.fcisolvers])
    ham_eff = slow_ham (las.mol, h1, h2, las.ci, las.ncas_sub, nelec_fr)[0]
    print (las.converged, e_states - (e0 + np.diag (ham_eff)))


