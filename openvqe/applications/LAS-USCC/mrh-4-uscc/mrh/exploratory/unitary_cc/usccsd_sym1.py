import numpy as np
import math, ctypes
from scipy import linalg
from pyscf.lib import logger
from mrh.lib.helper import load_library
from mrh.exploratory.unitary_cc import uccsd_sym0
from mrh.exploratory.unitary_cc import usccsd_sym0
from itertools import combinations, permutations, product, combinations_with_replacement

libfsucc = load_library ('libfsucc')

# Enforce n, sz, and first-order s**2 symmetry in unitary generators. First-
# order means t1_alpha = t2_beta. Unfortunately, I don't know how to ensure
# general s**2 symmetry

def spincases (p_idxs, norb):
    ''' Compute all unique spinorbital indices corresponding to all spin cases
        of a set of field operators acting on a specified list of spatial
        orbitals. The different spin cases are returned 'column-major order':

        aaa...
        baa...
        aba...
        bba...
        aab...

        The index of a given spincase string ('aba...', etc.) can be computed
        as

        p_spin = int (spincase[::-1].replace ('a','0').replace ('b','1'), 2)

        Args:
            p_idxs : ndarray of shape (nelec,)
                Spatial orbital indices
            norb : integer
                Total number of spatial orbitals

        Returns:
            p_idxs : ndarray of shape (2^nelec, nelec)
                Rows contain unique spinorbital cases of the input spatial
                orbitals
            m : ndarray of shape (2^nelec,)
                Number of beta (spin-down) orbitals in each spin case
    '''
    nelec = len (p_idxs)
    p_idxs = p_idxs[None,:]
    m = np.array ([0])
    for ielec in range (nelec):
        q_idxs = p_idxs.copy ()
        q_idxs[:,ielec] += norb
        p_idxs = np.append (p_idxs, q_idxs, axis=0)
        m = np.append (m, m+1)
    p_sorted = np.stack ([np.sort (prow) for prow in p_idxs], axis=0)
    idx_uniq = np.unique (p_sorted, return_index=True, axis=0)[1]
    p_idxs = p_idxs[idx_uniq]
    m = m[idx_uniq]
    return p_idxs, m

class FSUCCOperator (usccsd_sym0.FSUCCOperator):
    __doc__ = uccsd_sym0.FSUCCOperator.__doc__ + '\n\n\n' + \
    ''' uccsd_sym1 child class:

        THE MEANING OF CONSTRUCTOR ARGS a_idxs AND i_idxs IS *CHANGED* IN THIS
        CHILD CLASS. READ THE BELOW CAREFULLY!!!!

        A callable spin-adapted (Sz only) unitary coupled cluster
        operator. For single-excitation operators, spin-up and spin-down
        amplitudes are constrained to be equal. All spin cases for a given
        spatial-orbital excitation pattern (from i_idxs to a_idxs) are grouped
        together and applied to the ket in ascending order of the index

        (a_spin) * nelec + i_spin

        where 'a_spin' and 'i_spin' are the ordinal indices of the spin
        cases returned by the function 'spincases' for a_idxs (creation
        operators) and i_idxs (annihilation operators) respectively, and nelec
        is the order of the generator (1=singles, 2=doubles, etc.) Nilpotent
        or undefined spin cases (i.e., because of spatial-orbital index
        collisions) are omitted.

        The spatial-orbital excitation patterns are applied to the ket in
        ascending order of their ordinal positions in the 'a_idxs' and 'i_idxs'
        lists provided to the constructor.

        Extra Constructor Kwargs:
            s2sym : logical
                Whether to apply partial S**2 symmetry (only implemented for
                single excitations; higher-order excitations may still break
                S**2 symmetry even if True).
    '''

    def __init__(self, norb, a_idxs, i_idxs, s2sym=True):
        # Up to two equal indices in one generator are allowed
        # However, we still can't have any equal generators
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs
        self.symtab = []                       
        self.symtab_uscc = []
        for a, i in zip(a_idxs, i_idxs):
            a = np.ascontiguousarray(a, dtype=np.uint8)
            i = np.ascontiguousarray(i, dtype=np.uint8)
            
            if len(a) == 1:
                if s2sym:
                    symrow_uscc = [len(self.symtab_uscc), len(self.symtab_uscc)+1]
                    self.symtab_uscc.append(symrow_uscc)
                else:
                    self.symtab_uscc.append([len(self.symtab_uscc)])
            else:
                self.symtab_uscc.append([len(self.symtab_uscc)])

        assert len(self.symtab_uscc) == len(a_idxs), "Mismatch in lengths"
        print("len(self.symtab_uscc)", len(self.symtab_uscc))
        self.norb = 2*norb
        self.ngen = len (self.a_idxs)
        assert (len (self.i_idxs) == self.ngen)
        self.uniq_gen_idx = np.array ([x[0] for x in self.symtab_uscc])
        self.amps = np.zeros (self.ngen)
        self.assert_sanity ()

    def assert_sanity (self):
        norb = self.norb // 2
        # print("norb",norb)
        # print("self.a_idxs",self.a_idxs)
        usccsd_sym0.FSUCCOperator.assert_sanity (self)
        for a, i in zip (self.a_idxs, self.i_idxs):
            errstr = 'a,i={},{} breaks sz symmetry'.format (a, i)
            assert (np.sum (a//norb) == np.sum (i//norb)), errstr

    def get_uniq_amps (self):
        ''' Amplitude getter

        Returns:
            x : ndarray of length ngen_uniq
                Current amplitudes of each unique generator
        '''
        return self.amps[self.uniq_gen_idx]
    
    # def set_uniq_amps_(self, x):
        # ''' Amplitude setter

        # Args:
            # x : ndarray of length ngen_uniq
                # Amplitudes for each unique generator
        # '''
        # for symrow_uscc, xi in zip(self.symtab_uscc, x):
            # try:
                # # Check if symrow is a list, and if so, use the first element.
                # index = symrow_uscc[0] if isinstance(symrow_uscc, list) else symrow_uscc
                # self.amps[index] = xi
            # except (ValueError, IndexError) as e:  
                # print(f"Error: {type(e).__name__}, symrow_uscc={symrow_uscc}, xi={xi}, amps.shape={self.amps.shape}")
                # raise e


        # return self
        
        
        
    def set_uniq_amps_(self, x):
        ''' Amplitude setter

        Args:
            x : ndarray of length ngen_uniq
                Amplitudes for each unique generator
        '''
        for symrow_uscc, xi in zip (self.symtab_uscc, x):
            try:
                self.amps[symrow_uscc] = xi
            except ValueError as e:
                print (symrow_uscc, xi, x.shape)
                raise (e)
        return self

    # def product_rule_pack (self, g):
        # ''' Pack a vector of size self.ngen according to the product
            # rule; i.e., for a gradient wrt generator amplitudes '''
        # assert (len (g) == self.ngen)
        # g = np.asarray (g)
        # g_uniq = np.empty (self.ngen_uniq)
        # for ix, symrow in enumerate (self.symtab):
            # g_uniq[ix] = g[symrow].sum ()
        # return g_uniq
        
        
    def product_rule_pack (self, g):
        ''' Pack a vector of size self.ngen according to the product
            rule; i.e., for a gradient wrt generator amplitudes '''
        assert (len (g) == self.ngen)
        g = np.asarray (g)
        g_uniq = np.empty (self.ngen_uniq)
        for ix, symrow_uscc in enumerate (self.symtab_uscc):
            g_uniq[ix] = g[symrow_uscc].sum ()
        return g_uniq

    @property
    def ngen_uniq (self):
        ''' subclass me to apply s**2 or irrep symmetries '''
        return len (self.symtab_uscc)

    def print_tab (self, _print_fn=print):
        norb = self.norb // 2
        for ix in range (self.ngen_uniq): self.print_uniq (ix,
            _print_fn=_print_fn)

    def print_uniq (self, ix, _print_fn=print):
        norb = self.norb // 2
        symrow = self.symtab[ix]
        _print_fn ("Unique amplitude {}".format (ix))
        for gen in symrow:
            ab = self.a_idxs[gen]
            ij = self.i_idxs[gen]
            ptstr = "   {:12.5e} (".format (self.amps[gen])
            for i in ij: ptstr += "{}{}".format (i%norb,('a','b')[i//norb])
            ptstr += '->'
            for a in ab: ptstr += "{}{}".format (a%norb,('a','b')[a//norb])
            ptstr += ')'
            _print_fn (ptstr)

def get_uccs_op (norb, t1=None, freeze_mask=None):
    t1_idx = np.zeros ((norb, norb), dtype=np.bool_)
    t1_idx[np.tril_indices (norb, k=-1)] = True
    if freeze_mask is not None:
        t1_idx[freeze_mask] = False
    t1_idx = np.where (t1_idx)
    a, i = list (t1_idx[0]), list (t1_idx[1])
    uop = FSUCCOperator (norb, a, i)
    if t1 is not None:
        uop.set_uniq_amps_(t1[t1_idx])
    return uop

def get_uccsd_op (norb, t1=None, t2=None, s2sym=True):
    ''' Construct and optionally initialize semi-spin-adapted unitary CC
        correlator with singles and doubles spanning a single undifferentiated
        orbital range. Excitations from spatial orbital(s) i(, j) to spatial
        orbital(s) a(, b) are applied to the ket in the order

        U|ket> = u^n(n-1)_nn u^n(n-1)_n(n-1) u^n(n-2)_nn ... u^11_22 u^11_21
                 ... u^n_(n-1) u^n_(n-2) ... u^3_2 u^3_1 u^2_1 |ket>

        where ^ indicates creation operators (a, b; rows) and _ indicates
        annihilation operators (i, j; columns). The doubles amplitudes are
        arbitrarily chosen in the upper-triangular space (a,b <= i,j), but the
        lower-triangular space is used for the individual double pairs
        (a > b, i > j) and for the singles amplitudes (a > i). In all cases,
        row-major ordering is employed.

        The spin cases of a given set of orbitals a, b, i, j are grouped 
        together. For singles, spin-up (a) and spin-down (b) amplitudes are
        constrained to be equal and the spin-up operator is on the right (i.e.,
        is applied first). For doubles, the spin case order is

        u|ket> -> ^bb_bb ^ab_ab ^ab_ba ^ba_ab ^ba_ba ^aa_aa |ket>

        For spatial orbital cases in which the same index appears more than
        once, spin cases that correspond to nilpotent (eg., ^pp_qr ^aa_aa),
        undefined (eg., ^pq_pq ^ab_ab), or redundant (eg., ^pq_pq ^ab_ba)
        operators are omitted.

        Args:
            norb : integer
                Total number of spatial orbitals. (0.5 * #spinorbitals)

        Kwargs:
            t1 : ndarray of shape (norb,norb)
                Amplitudes at which to initialize the singles operators
            t2 : None
                NOT IMPLEMENTED. Amplitudes at which to initialize the doubles
                operators
            s2sym : logical
                If false, alpha and beta t1 amplitudes are allowed to differ

        Returns:
            uop : object of class FSUCCOperator
                The callable UCCSD operator
    '''
    t1_idx = np.tril_indices (norb, k=-1)
    ab_idxs, ij_idxs = list (t1_idx[0]), list (t1_idx[1])
    pq = [(p, q) for p, q in zip (*np.tril_indices (norb))]
    for ab, ij in combinations_with_replacement (pq, 2):
        ab_idxs.append (ab)
        ij_idxs.append (ij)
    uop = FSUCCOperator (norb, ab_idxs, ij_idxs, s2sym=s2sym)
    x0 = uop.get_uniq_amps ()
    if t1 is not None: x0[:len (t1_idx[0])] = t1[t1_idx]
    if t2 is not None: raise NotImplementedError ("t2 initialization")
    uop.set_uniq_amps_(x0)
    return uop

def contract_s2 (psi, norb):
    assert (psi.size == 2**(2*norb))
    s2psi = np.zeros_like (psi)
    psi_ptr = psi.ctypes.data_as (ctypes.c_void_p)
    s2psi_ptr = s2psi.ctypes.data_as (ctypes.c_void_p)
    libfsucc.FSUCCcontractS2 (psi_ptr, s2psi_ptr,
        ctypes.c_uint (norb))
    return s2psi

def spin_square (psi, norb):
    ss = psi.dot (contract_s2 (psi, norb))
    s = np.sqrt (ss+0.25) - 0.5
    multip = s*2 + 1
    return ss, multip

class UCCS (uccsd_sym0.UCCS):
    def get_uop (self):
        return get_uccs_op (self.norb)

    def rotate_mo (self, mo_coeff=None, x=None):
        if mo_coeff is None: mo_coeff=self.mo_coeff
        if x is None: x=self.x
        norb = self.norb
        t1 = np.zeros ((norb, norb), dtype=x.dtype)
        t1[np.tril_indices (norb, k=-1)] = x[:]
        t1 -= t1.T
        umat = linalg.expm (t1)
        return mo_coeff @ umat

    def kernel (self, mo_coeff=None, psi0=None, x=None):
        self.e_tot, self.mo_coeff, self.x, self.conv = super().kernel (mo_coeff=mo_coeff, psi0=psi0, x=x)
        log = logger.new_logger (self, self.verbose)
        uop = self.get_uop ()
        uop.set_uniq_amps_(self.x)
        uop.print_tab (_print_fn=log.info)
        return self.e_tot, self.mo_coeff, self.x, self.conv

class UCCSD (UCCS):
    def get_uop (self):
        return get_uccsd_op (self.norb)


if __name__ == '__main__':
    # norb = 4
    # nelec = 4
    def pbin (n, k=norb):
        s = bin (n)[2:]
        m = (2*k) - len (s)
        if m: s = ''.join (['0',]*m) + s
        return s
    psi = np.zeros (2**(2*norb))
    psi[51] = 1.0

    from pyscf.fci import cistring, spin_op
    from mrh.exploratory.citools import fockspace

    t1_rand = np.random.rand (norb,norb)
    t2_rand = np.random.rand (norb,norb,norb,norb)
    uop_s = get_uccs_op (norb, t1=t1_rand)
    upsi = uop_s (psi)
    upsi_h = fockspace.fock2hilbert (upsi, norb, nelec)
    uTupsi = uop_s (upsi, transpose=True)
    for ix in range (2**(2*norb)):
        if np.any (np.abs ([psi[ix], upsi[ix], uTupsi[ix]]) > 1e-8):
            print (pbin (ix), psi[ix], upsi[ix], uTupsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))
    print ("<psi|S**2|psi> =",spin_square (psi, norb)[0],
           "<psi|U'S**2U|psi> =",spin_square (upsi, norb)[0],spin_op.spin_square (upsi_h, norb, nelec)[0])

    uop_sd = get_uccsd_op (norb)
    x_rand = (1 - 2*np.random.rand (uop_sd.ngen_uniq)) * math.pi/4
    uop_sd.set_uniq_amps_(x_rand)
    upsi = uop_sd (psi)
    upsi_h = fockspace.fock2hilbert (upsi, norb, nelec)
    uTupsi = uop_sd (upsi, transpose=True)
    for ix in range (2**(2*norb)):
        if np.any (np.abs ([psi[ix], upsi[ix], uTupsi[ix]]) > 1e-8):
            print (pbin (ix), psi[ix], upsi[ix], uTupsi[ix])
    print ("<psi|psi> =",psi.dot (psi), "<psi|U|psi> =",psi.dot (upsi),"<psi|U'U|psi> =",upsi.dot (upsi))
    print ("<psi|S**2|psi> =",spin_square (psi, norb)[0],
           "<psi|U'S**2U|psi> =",spin_square (upsi, norb)[0], spin_op.spin_square (upsi_h, norb, nelec)[0])

    ndet = cistring.num_strings (norb, nelec//2)
    np.random.seed (0)
    tpsi = 1-(2*np.random.rand (ndet))
    tpsi = np.multiply.outer (tpsi, tpsi).ravel ()
    tpsi /= linalg.norm (tpsi)
    tpsi = fockspace.hilbert2fock (tpsi, norb, (nelec//2, nelec//2)).ravel ()
    from scipy import optimize
    def obj_test (uop_test):
        def obj_fun (x):
            uop_test.set_uniq_amps_(x)
            upsi = uop_test (psi)
            ut = upsi.dot (tpsi)
            err = upsi.dot (upsi) - (ut**2)
            jac = np.zeros (uop_test.ngen)
            for ix, dupsi in enumerate (uop_test.gen_deriv1 (psi)):
                jac[ix] += 2*dupsi.dot (upsi - ut*tpsi) 
            jac = uop_test.product_rule_pack (jac)
            print (err, linalg.norm (jac))
            return err, jac

        res = optimize.minimize (obj_fun, uop_test.get_uniq_amps (), method='BFGS', jac=True)

        print (res.success)
        uop_test.set_uniq_amps_(res.x)
        upsi = uop_test (psi)
        uTupsi = uop_test (upsi, transpose=True)
        for ix in range (2**(2*norb)):
            if np.any (np.abs ([psi[ix], upsi[ix], tpsi[ix]]) > 1e-8):
                print (pbin (ix), psi[ix], upsi[ix], tpsi[ix])
        print ("<psi|psi> =",psi.dot (psi), "<tpsi|psi> =",tpsi.dot (psi),"<tpsi|U|psi> =",tpsi.dot (upsi))
        print ("<psi|S**2|psi> =",spin_square (psi, norb)[0],
               "<psi|U'S**2U|psi> =",spin_square (upsi, norb)[0])

    uop_s.set_uniq_amps_(np.zeros (uop_s.ngen_uniq))
    print ("Testing singles...")
    obj_test (uop_s)
    print ('Testing singles and doubles...')
    x = np.zeros (uop_sd.ngen_uniq)
    #x[:uop_s.ngen_uniq] = uop_s.get_uniq_amps ()
    uop_sd.set_uniq_amps_(x)
    obj_test (uop_sd)

    from pyscf import gto, scf, lib
    mol = gto.M (atom = 'H 0 0 0; H 1.2 0 0', basis='6-31g', verbose=lib.logger.DEBUG, output='uccsd_sym0.log')
    rhf = scf.RHF (mol).run ()
    uccs = UCCS (mol).run ()
    print ("The actual test result is:", uccs.e_tot-rhf.e_tot, linalg.norm (uccs.x))
    nmo = mol.nao_nr ()
    hf_mo = rhf.mo_coeff
    uccs_mo0 = uccs.mo_coeff
    uccs_mo1 = uccs.rotate_mo ()
    s0 = mol.intor_symmetric ('int1e_ovlp')
    print ("hf MOs vs UCCS frame:\n", np.diag (hf_mo.T @ s0 @ uccs_mo0))
    print ("hf MOs vs UCCS opt:\n", np.diag (hf_mo.T @ s0 @ uccs_mo1))
    rhf.mo_coeff[:,:] = uccs_mo1[:,:]
    print (rhf.energy_tot (), uccs.e_tot)


