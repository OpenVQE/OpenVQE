# Essentially a deoptimized implementation of LASSCF which is just a
# FCISolver object that does everything in the Fock-space FCI basis
# with no constraint for spin or charge.
#
# Must define:
#   kernel = approx_kernel
#   make_rdm12
#   get_init_guess
#
# In these functions, except for get_init_guess, "nelec" is ignored

import time, math
import numpy as np
from scipy import linalg, optimize
from pyscf import lib, ao2mo
from itertools import combinations
from pyscf.fci import direct_spin1, cistring, spin_op
from pyscf.fci import addons as fci_addons
from itertools import product
from mrh.exploratory.citools import fockspace, addons
from mrh.exploratory.unitary_cc.uccsd_sym1 import get_uccs_op
from mrh.my_pyscf.mcscf.lasci_sync import all_nonredundant_idx
from mrh.my_pyscf.fci import csf_solver
from itertools import product
# from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccs_op
from mrh.exploratory.unitary_cc.uccsd_sym0 import get_uccsd_op

verbose_lbjfgs = [-1,-1,-1,0,50,99,100,101,101,101]
GLOBAL_MAX_CYCLE = None
def _n_m_s (dm1s, dm2s, _print_fn=print):
    neleca = np.trace (dm1s[0])
    nelecb = np.trace (dm1s[1])
    n = neleca+nelecb
    m = (neleca-nelecb)/2.
    ss = m*m + (n/2.) - np.einsum ('pqqp->pq', dm2s[1]).sum ()
    _print_fn ('<N>,<Sz>,<S^2> = %f, %f, %f', n, m, ss)

def kernel (fci, h1, h2, norb, nelec, norb_f=None, ci0_f=None,
            tol=1e-10, gtol=1e-4, max_cycle=None, 
            orbsym=None, wfnsym=None, ecore=0, **kwargs):
    if max_cycle is None:
        max_cycle = GLOBAL_MAX_CYCLE if GLOBAL_MAX_CYCLE is not None else 15000
    if norb_f is None: norb_f = getattr (fci, 'norb_f', [norb])
    if ci0_f is None: ci0_f = fci.get_init_guess (norb, nelec, norb_f, h1, h2)
    verbose = kwargs.get ('verbose', getattr (fci, 'verbose', 0))
    if isinstance (verbose, lib.logger.Logger):
        log = verbose
        verbose = log.verbose
    else:
        log = lib.logger.new_logger (fci, verbose)
    frozen = getattr (fci, 'frozen', None)
    psi = getattr (fci, 'psi', fci.build_psi (ci0_f, norb, norb_f, nelec,
        log=log, frozen=frozen))
    assert (psi.check_ci0_constr)
    psi_options = {'gtol':     gtol,
                   'maxiter':  max_cycle,
                   'disp':     verbose>lib.logger.DEBUG}
    log.info ('LASCI object has %d degrees of freedom', psi.nvar)
    h = [ecore, h1, h2]
    psi_callback = psi.get_solver_callback (h)
    # print("psi.e_de",psi.e_de)
    # print("psi.x",psi.x)
    res = optimize.minimize (psi.e_de, psi.x, args=(h,), method='BFGS',
        jac=True, callback=psi_callback, options=psi_options)

    fci.converged = res.success
    e_tot = psi.energy_tot (res.x, h)
    ci1 = psi.get_fcivec (res.x)
    if verbose>=lib.logger.DEBUG:
        #psi.uop.print_tab (_print_fn=log.debug)
        psi.print_x (res.x, h, _print_fn=log.debug)
    if verbose>=lib.logger.INFO:
        dm1s, dm2s = fci.make_rdm12s (ci1, norb, nelec)
        dm1s = np.stack (dm1s, axis=0)
        dm2s = np.stack (dm2s, axis=0)
        for ix, j in enumerate (np.cumsum (norb_f)):
            i = j - norb_f[ix]
            log.info ('Fragment %d local quantum numbers', ix)
            _n_m_s (dm1s[:,i:j,i:j], dm2s[:,i:j,i:j,i:j,i:j], _print_fn=log.info)
        log.info ('Whole system quantum numbers')
        _n_m_s (dm1s, dm2s, _print_fn=log.info)
    psi.x = res.x
    psi.converged = res.success
    psi.finalize_()
    fci.psi = psi
    return e_tot, ci1


def make_rdm1 (fci, fcivec, norb, nelec, **kwargs):
    dm1 = np.zeros ((norb, norb))
    for nelec in product (range (norb+1), repeat=2):
        ci = fockspace.fock2hilbert (fcivec, norb, nelec)
        d = direct_spin1.make_rdm1 (ci, norb, nelec, **kwargs)
        dm1 += d
    return dm1

def make_rdm1s (fci, fcivec, norb, nelec, **kwargs):
    dm1a = np.zeros ((norb,norb))
    dm1b = np.zeros ((norb,norb))
    for nelec in product (range (norb+1), repeat=2):
        ci = fockspace.fock2hilbert (fcivec, norb, nelec)
        da, db = direct_spin1.make_rdm1s (ci, norb, nelec, **kwargs)
        dm1a += da
        dm1b += db
    return dm1a, dm1b

def make_rdm12 (fci, fcivec, norb, nelec, **kwargs):
    dm1 = np.zeros ((norb,norb))
    dm2 = np.zeros ((norb,norb,norb,norb))
    for nelec in product (range (norb+1), repeat=2):
        ci = fockspace.fock2hilbert (fcivec, norb, nelec)
        d1, d2 = direct_spin1.make_rdm12 (ci, norb, nelec, **kwargs)
        dm1 += d1
        dm2 += d2
    return dm1, dm2

def make_rdm12s (fci, fcivec, norb, nelec, **kwargs):
    dm1 = np.zeros ((2,norb,norb))
    dm2 = np.zeros ((3,norb,norb,norb,norb))
    for nelec in product (range (norb+1), repeat=2):
        ci = fockspace.fock2hilbert (fcivec, norb, nelec)
        d1, d2 = direct_spin1.make_rdm12s (ci, norb, nelec, **kwargs)
        dm1 += np.stack (d1, axis=0)
        dm2 += np.stack (d2, axis=0)
    return tuple(dm1), tuple(dm2)

#def get_init_guess (fci, norb, nelec, norb_f, h1, h2, ci0_f=None, nelec_f=None, smult_f=None):
#    log = lib.logger.new_logger (fci, fci.verbose)
#    if ci0_f is None: ci0_f = _get_init_guess_ci0_f (fci, norb, nelec, norb_f,
#        h1, h2, nelelas=None, smult=None)

def get_init_guess (fci, norb, nelec, norb_f, h1, h2, nelec_f=None, smult_f=None):
    if nelec_f is None:
        nelec_f = getattr (fci, 'nelec_f', _guess_nelec_f (fci, norb, nelec, norb_f, h1, h2))
    if smult_f is None:
        smult_f = [abs(n[0]-n[1])+1 for n in nelec_f]
    h2 = ao2mo.restore (1, h2, norb)
    i = 0
    ci0_f = []
    for no, ne, s in zip (norb_f, nelec_f, smult_f):
        j = i + no
        h1_i = h1[i:j,i:j]
        h2_i = h2[i:j,i:j,i:j,i:j]
        i = j
        csf = csf_solver (fci.mol, smult=s)
        hdiag = csf.make_hdiag_csf (h1_i, h2_i, no, ne)
        ci = csf.get_init_guess (no, ne, 1, hdiag)[0]
        ci = np.squeeze (fockspace.hilbert2fock (ci, no, ne))
        ci0_f.append (ci)
    return ci0_f

def _guess_nelec_f (fci, norb, nelec, norb_f, h1, h2):
    # Pick electron distribution by the lowest-energy single determinant
    nelec = direct_spin1._unpack_nelec (nelec)
    hdiag = fci.make_hdiag (h1, h2, norb, nelec)
    ndetb = cistring.num_strings (norb, nelec[1])
    addr_dp = np.divmod (np.argmin (hdiag), ndetb)
    str_dp = [cistring.addr2str (norb, n, c) for n,c in zip (nelec, addr_dp)]
    nelec_f = []
    for n in norb_f:
        ndet = 2**n
        c = np.zeros ((ndet, ndet))
        det = np.array ([str_dp[0] % ndet, str_dp[1] % ndet], dtype=np.int64)
        str_dp = [str_dp[0] // ndet, str_dp[1] // ndet]
        ne = [0,0]
        for spin in range (2):
            for iorb in range (n):
                ne[spin] += int (bool ((det[spin] & (1 << iorb))))
        nelec_f.append (ne)
    return nelec_f

def spin_square (fci, fcivec, norb, nelec):
    ss = 0.0
    for ne in product (range (norb+1), repeat=2):
        c = fockspace.fock2hilbert (fcivec, norb, ne)
        ssc = spin_op.contract_ss (c, norb, ne)
        ss += c.conj ().ravel ().dot (ssc.ravel ())
    s = np.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip

def contract_ss (fci, ci, norb, nelec):
    fcivec = np.zeros_like (ci)
    for ne in product (range (norb+1), repeat=2):
        c = fockspace.fock2hilbert (ci, norb, ne)
        ssc = spin_op.contract_ss (c, norb, ne)
        ssci += np.squeeze (fockspace.hilbert2fock (ssc, norb, ne))
    return fcivec

def transform_ci_for_orbital_rotation (fci, ci, norb, nelec, umat):
    fcivec = np.zeros_like (ci) 
    for ne in product (range (norb+1), repeat=2):
        c = np.squeeze (fockspace.fock2hilbert (ci, norb, ne))
        c = fci_addons.transform_ci_for_orbital_rotation (c, norb, ne, umat)
        fcivec += np.squeeze (fockspace.hilbert2fock (c, norb, ne))
    return fcivec

class LASUCCTrialState (object):
    ''' Evaluate the energy and Jacobian of a LASSCF trial function parameterized in terms
        of unitary CC singles amplitudes and CI transfer operators. '''

    def __init__(self, fcisolver, ci0_f, norb, norb_f, nelec, epsilon=0.0, log=None, frozen=None):
        self.fcisolver = fcisolver
        self.epsilon = epsilon
        self.ci_f = [ci.copy () for ci in ci0_f]
        self.norb = norb
        self.norb_f = norb_f = np.asarray (norb_f)
        self.nelec = sum (direct_spin1._unpack_nelec (nelec))
        self.nfrag = len (norb_f)
        assert (sum (norb_f) == norb)
        self.uniq_orb_idx = all_nonredundant_idx (norb, 0, norb_f)
        self.nconstr = 1 # Total charge only
        self.log = log if log is not None else lib.logger.new_logger (fcisolver, fcisolver.verbose)
        self.it_cnt = 0
        self.uop = fcisolver.get_uop (norb, norb_f)
        self.var_mask = np.ones (self.nvar_tot, dtype=np.bool_)
        if frozen is not None:
            if frozen.upper () == 'CI':
                i = self.nconstr + self.uop.ngen_uniq
                self.var_mask[i:] = False
            else:
                self.var_mask[frozen] = False
        self.x = np.zeros (self.nvar)
        self.converged = False
        self._e_last = None
        self._jac_last = None

    def fermion_spin_shuffle (self, c, norb=None, norb_f=None):
        if norb is None: norb = self.norb
        if norb_f is None: norb_f = self.norb_f
        return c * fockspace.fermion_spin_shuffle (norb, norb_f)

    def fermion_frag_shuffle (self, c, i, j, norb=None):
        # TODO: fix this!
        if norb is None: norb=self.norb
        c_shape = c.shape
        c = c.ravel ()
        sgn = fockspace.fermion_frag_shuffle (norb, i, j)
        sgn = np.multiply.outer (sgn, sgn).ravel ()
        #if isinstance (sgn, np.ndarray):
        #    print (sgn.shape)
        #    flip_idx = sgn<0
        #    if np.count_nonzero (flip_idx):
        #        c_flip = c.copy ()
        #        c_flip[~flip_idx] = 0.0
        #        det_flp = np.argsort (-np.abs (c_flip))
        #        for det in det_flp[:10]:
        #            deta, detb = divmod (det, 2**norb)
        #            print (fockspace.onv_str (deta, detb, norb), c_flip[det])
        return (c * sgn).reshape (c_shape)

    def pack (self, xconstr, xcc, xci_f):
        x = [xconstr, xcc]
        ci0_f = self.ci_f
        for xci, ci0 in zip (xci_f, ci0_f):
            cHx = ci0.conj ().ravel ().dot (xci.ravel ())
            x.append ((xci - (ci0*cHx)).ravel ())
        x = np.concatenate (x)
        return x[self.var_mask]

    def unpack (self, x_):
        x = np.zeros (self.nvar_tot)
        x[self.var_mask] = x_[:]
        xconstr, x = x[:self.nconstr], x[self.nconstr:]

        xcc = x[:self.uop.ngen_uniq] 
        x = x[self.uop.ngen_uniq:]

        xci = []
        for n in self.norb_f:
            xci.append (x[:2**(2*n)].reshape (2**n, 2**n))
            x = x[2**(2*n):]

        return xconstr, xcc, xci

    @property
    def nvar_tot (self):
        return self.nconstr + self.uop.ngen_uniq + sum ([c.size for c in self.ci_f])

    @property
    def nvar (self):
        return np.count_nonzero (self.var_mask)

    def e_de (self, x, h):
        log = self.log
        t0 = (time.process_time (), time.time ())
        c, uc, huc, uhuc, c_f = self.hc_x (x, h)
        e_tot = self.energy_tot (x, h, uc=uc, huc=huc)
        jac = self.jac (x, h, c=c, uc=uc, huc=huc, uhuc=uhuc, c_f=c_f)
        log.timer ('las_obj full ene+jac eval', *t0)
        return e_tot, jac

    def energy_tot (self, x, h, uc=None, huc=None):
        log = self.log
        norm_x = linalg.norm (x)
        t0 = (time.process_time (), time.time ())
        if (uc is None) or (huc is None):
            uc, huc = self.hc_x (x, h)[1:3]
        uc, huc = uc.ravel (), huc.ravel ()
        cu = uc.conj ()
        cuuc = cu.dot (uc)
        cuhuc = cu.dot (huc)
        e_tot = cuhuc/cuuc
        log.timer ('las_obj energy eval', *t0)
        log.debug ('energy value = %f, norm value = %e, |x| = %e', e_tot, cuuc, norm_x)
        if log.verbose > lib.logger.DEBUG: self.check_x_change (x, e_tot0=e_tot)
        self._e_last = e_tot
        return e_tot

    def jac (self, x, h, c=None, uc=None, huc=None, uhuc=None, c_f=None):
        norm_x = linalg.norm (x)
        log = self.log
        t0 = (time.process_time (), time.time ())
        if any ([x is None for x in [c, uc, huc, uhuc, c_f]]):
            c, uc, huc, uhuc, c_f = self.hc_x (x, h)
        # Revisit the first line below if t ever breaks
        # number symmetry
        jacconstr = self.get_jac_constr (uc)
        t1 = log.timer ('las_obj constr jac', *t0)
        jact1 = self.get_jac_t1 (x, h, c=c, huc=huc, uhuc=uhuc)
        t1 = log.timer ('las_obj ucc jac', *t1)
        jacci_f = self.get_jac_ci (x, h, uhuc=uhuc, uci_f=c_f)
        t1 = log.timer ('las_obj ci jac', *t1)
        log.timer ('las_obj jac eval', *t0)
        g = self.pack (jacconstr, jact1, jacci_f)
        norm_g = linalg.norm (g)
        log.debug ('|gradient| = %e, |x| = %e',norm_g, norm_x)
        self._jac_last = g
        return g

    def hc_x (self, x, h):
        xconstr, xcc, xci = self.unpack (x)
        self.uop.set_uniq_amps_(xcc)
        h = self.constr_h (xconstr, h)
        c_f = self.rotate_ci0 (xci)
        c = self.dp_ci (c_f)
        uc = self.uop (c)
        huc = self.contract_h2 (h, uc)
        uhuc = self.uop (huc, transpose=True)
        return c, uc, huc, uhuc, c_f
        
    def contract_h2 (self, h, ci, norb=None):
        if norb is None: norb = self.norb
        hci = h[0] * ci
        for neleca, nelecb in product (range (norb+1), repeat=2):
            nelec = (neleca, nelecb)
            h2eff = self.fcisolver.absorb_h1e (h[1], h[2], norb, nelec, 0.5)
            ci_h = np.squeeze (fockspace.fock2hilbert (ci, norb, nelec))
            hc = direct_spin1.contract_2e (h2eff, ci_h, norb, nelec)
            hci += np.squeeze (fockspace.hilbert2fock (hc, norb, nelec))
        return hci
            
    def dp_ci (self, ci_f):
        norb, norb_f = self.norb, self.norb_f
        ci = np.ones ([1,1], dtype=ci_f[0].dtype)
        for ix, c in enumerate(ci_f):
            ndet = 2**sum(norb_f[:ix+1])
            ci = np.multiply.outer (c, ci).transpose (0,2,1,3).reshape (ndet, ndet)
        ci = self.fermion_spin_shuffle (ci, norb=norb, norb_f=norb_f)
        return ci

    def constr_h (self, xconstr, h):
        x = xconstr[0]
        norb, nelec = self.norb, self.nelec
        h = [h[0] - (x*nelec), h[1] + (x*np.eye (self.norb)), h[2]]
        return h 

    def rotate_ci0 (self, xci_f):
        ci0, norb = self.ci_f, self.norb
        ci1 = []
        for dc, c in zip (xci_f, ci0):
            dc -= c * c.conj ().ravel ().dot (dc.ravel ())
            phi = linalg.norm (dc)
            cosp = np.cos (phi)
            if np.abs (phi) > 1e-8: sinp = np.sin (phi) / phi
            else: sinp = 1 # as precise as it can be w/ 64 bits
            ci1.append (cosp*c + sinp*dc)
        return ci1

    def project_frag (self, ifrag, vec, ci0_f=None):
        ''' Integrate a vector over all fragments other than ifrag '''
        if ci0_f is None: ci0_f = self.ci_f
        vec = self.fermion_spin_shuffle (vec.copy ())
        norb, norb_f = self.norb, self.norb_f
        norb0, ndet0 = norb, 2**norb
        norb2, ndet2 = 0, 1
        for jfrag, (ci, norb1) in enumerate (zip (ci0_f, norb_f)):
            norb0 -= norb1
            if (jfrag==ifrag):
                norb2 = norb1
                ndet2 = 2**norb1
                continue
            # norb0, norb1, and norb2 are the number of orbitals in the sectors arranged
            # in major-axis order: the slower-moving orbital indices we haven't touched yet,
            # the orbitals we are integrating over in this particular cycle of the for loop,
            # and the fast-moving orbital indices that we have to keep uncontracted
            # because they correspond to the outer for loop.
            # We want to move the field operators corresponding to the generation of the
            # norb1 set to the front of the operator products in order to integrate
            # with the correct sign.
            vec = self.fermion_frag_shuffle (vec, norb2, norb2+norb1, norb=norb0+norb1+norb2)
            ndet0 = 2**norb0
            ndet1 = 2**norb1
            vec = vec.reshape (ndet0, ndet1, ndet2, ndet0, ndet1, ndet2)
            vec = np.tensordot (vec, ci, axes=((1,4),(0,1))).reshape (ndet0*ndet2, ndet0*ndet2)
        return vec

    def get_jac_constr (self, ci):
        dm1 = self.fcisolver.make_rdm12 (ci, self.norb, 0)[0]
        return np.array ([np.trace (dm1) - self.nelec])

    def get_jac_t1 (self, x, h, c=None, huc=None, uhuc=None):
        g = []
        xconstr, xcc, xci_f = self.unpack (x)
        self.uop.set_uniq_amps_(xcc)
        if (c is None) or (uhuc is None):
            c, _, _, uhuc = self.hc_x (x, h)[:4] 
        for duc, uhuc_i in zip (self.uop.gen_deriv1 (c, _full=False), self.uop.gen_partial (uhuc)):
            g.append (2*duc.ravel ().dot (uhuc_i.ravel ()))
        g = self.uop.product_rule_pack (g)
        return np.asarray (g)
               
    def get_grad_t1(self, x, h, c=None, huc=None, uhuc=None, epsilon=0.0):
        """
        Compute the gradients and relevant indices based on input values.

        Parameters:
        - x: array-like
            Input array to unpack and set unique amplitudes.
            
        - h: array-like
            Some form of input data to be used for computation.
            
        - c (optional): array-like
            Precomputed value; if not provided, it will be computed using hc_x.
            
        - huc (optional): array-like
            Precomputed value; if not provided, it will be computed using hc_x.
            
        - uhuc (optional): array-like
            Precomputed value; if not provided, it will be computed using hc_x.
            
        - epsilon (optional): float, default=0.0
            Threshold value for considering a gradient. If epsilon is 0, all gradients are considered.

        Returns:
        - tuple
            all_g: list of all computed gradients
            g: list of gradients above the epsilon threshold
            gen_indices: list of indices representing a_idx and i_idx
            a_idxs_lst: list of a_idx values
            i_idxs_lst: list of i_idx values
            len(a_idxs_lst): length of a_idx list
            len(i_idxs_lst): length of i_idx list
        """

        g = []
        all_g = []
        
        # Unpack and set unique amplitudes
        xconstr, xcc, xci_f = self.unpack(x)
        self.uop.set_uniq_amps_(xcc)

        # Compute 'c' and 'uhuc' if not provided
        if (c is None) or (uhuc is None):
            c, _, _, uhuc = self.hc_x(x, h)[:4]
        
        gen_indices = []
        a_idxs_lst = []
        i_idxs_lst = []
        # print("self.uop.init_a_idxs[i]",self.uop.init_a_idxs)
        for i, (duc, uhuc_i) in enumerate(zip(self.uop.gen_deriv1(c, _full=False), self.uop.gen_partial(uhuc))):
            gradient = 2 * duc.ravel().dot(uhuc_i.ravel())
            all_g.append((gradient, i))
            
            # Allow all gradients if epsilon is 0, else use the abs gradient condition
            if epsilon == 0.0 or abs(gradient) > epsilon:
                g.append((gradient, i))
                a_idx = self.uop.a_idxs[i]
                i_idx = self.uop.i_idxs[i]

                gen_indices.append((a_idx, i_idx))
                a_idxs_lst.append(a_idx)
                i_idxs_lst.append(i_idx)

        return all_g, g, gen_indices, a_idxs_lst, i_idxs_lst, len(a_idxs_lst), len(i_idxs_lst)


      

    def get_jac_ci (self, x, h, uhuc=None, uci_f=None):
        # "uhuc": e^-T1 H e^T1 U|ci0>
        # "jacci": Jacobian elements for the CI degrees of freedom
        # subscript_f means a list over fragments
        # subscript_i means this is not a list but it applies to a particular fragment
        xconstr, xcc, xci_f = self.unpack (x)
        if uhuc is None or uci_f is None:
            uhuc, uci_f = self.hc_x (x, h)[3:]
        #uhuc = self.fermion_spin_shuffle (uhuc)
        norb, norb_f, ci0_f = self.norb, self.norb_f, self.ci_f
        jacci_f = []
        for ifrag, (ci0, xci) in enumerate (zip (ci0_f, xci_f)):
            uhuc_i = self.project_frag (ifrag, uhuc, ci0_f=uci_f)
            # Given three orthonormal basis states |0>, |p>, and |q>,
            # with U = exp [xp (|p><0| - |0><p|) + xq (|q><0| - |0><q|)],
            # we have @ xq = 0, xp != 0:
            # U|0>      = cos (xp) |0> + sin (xp) |p>
            # dU|0>/dxp = cos (xp) |p> - sin (xp) |0>
            # dU|0>/dxq = sin (xp) |q> / xp
            cuhuc_i = ci0.conj ().ravel ().dot (uhuc_i.ravel ())
            uhuc_i -= ci0 * cuhuc_i # subtract component along |0>
            xp = linalg.norm (xci)
            # if xp > 1e-8:
            if xp > 1e-10:
                xci = xci / xp
                puhuc_i = xci.conj ().ravel ().dot (uhuc_i.ravel ())
                uhuc_i -= xci * puhuc_i # subtract component along |p>
                uhuc_i *= math.sin (xp) / xp # evaluate jac along |q>
                dU_dxp  = math.cos (xp) * puhuc_i
                dU_dxp -= math.sin (xp) * cuhuc_i
                uhuc_i += dU_dxp * xci # evaluate jac along |p>
            jacci_f.append (2*uhuc_i)

        return jacci_f

    def get_solver_callback (self, h):
        self.it_cnt = 0
        log = self.log
        def my_call (x):
            t0 = (time.process_time (), time.time ())
            norm_x = linalg.norm (x)
            e, de = self._e_last, self._jac_last
            if (e is None) or (de is None): e, de = self.e_de (x, h)
            norm_g = linalg.norm (de)
            log.info ('iteration %d, E = %f, |x| = %e, |g| = %e', self.it_cnt, e, norm_x, norm_g)
            if log.verbose >= lib.logger.DEBUG:
                self.check_x_symm (x, h, e_tot0=e)
            self.it_cnt += 1
            log.timer ('callback', *t0)
        return my_call
        
    # def get_solver_callback(self, h):
        # self.it_cnt = 0
        # log = self.log

        # def my_call(x):
            # t0 = (time.process_time(), time.time())
            # norm_x = linalg.norm(x)
            # e, de = self._e_last, self._jac_last
            # if (e is None) or (de is None): e, de = self.e_de(x, h)
            # norm_g = linalg.norm(de)

            ####Stopping criterion based on gradient norm
            # if norm_g < 1e-8:  # setting the threshold to 10^-8
                # raise Exception("Gradient norm below threshold, stopping optimization!")

            # log.info('iteration %d, E = %f, |x| = %e, |g| = %e', self.it_cnt, e, norm_x, norm_g)
            # if log.verbose >= lib.logger.DEBUG:
                # self.check_x_symm(x, h, e_tot0=e)
            # self.it_cnt += 1
            # log.timer('callback', *t0)

        # return my_call


    def get_fcivec (self, x=None):
        if x is None: x = self.x
        xconstr, xcc, xci_f = self.unpack (x)
        uc_f = self.rotate_ci0 (xci_f)
        uc = self.dp_ci (uc_f)
        self.uop.set_uniq_amps_(xcc)
        uc = self.uop (uc)
        return uc / linalg.norm (uc)

    def check_ci0_constr (self):
        norb, nelec = self.norb, self.nelec
        ci0 = self.dp_ci (self.ci_f)
        neleca_min = max (0, nelec-norb)
        neleca_max = min (norb, nelec)
        w = 0.0
        for neleca in range (neleca_min, neleca_max+1):
            nelecb = nelec - neleca
            c = fockspace.fock2hilbert (ci0, norb, (neleca,nelecb)).ravel ()
            w += c.conj ().dot (c)
        return w>1e-8

    def check_x_change (self, x, e_tot0=None):
        norb, nelec, log = self.norb, self.nelec, self.log
        log.debug ('<x|x_last>/<x|x> = %e', x.dot (self.x) / x.dot (x))
        self.x = x.copy ()

    def check_x_symm (self, x, h, e_tot0=None):
        norb, nelec, log = self.norb, self.nelec, self.log
        if e_tot0 is None: e_tot0 = self.energy_tot (x, h)
        xconstr = self.unpack (x)[0]
        ci1 = self.get_fcivec (x)
        ss = self.fcisolver.spin_square (ci1, norb, nelec)[0]
        n = np.trace (self.fcisolver.make_rdm12 (ci1, norb, nelec)[0])
        h = self.constr_h (xconstr, h)
        hc1 = self.contract_h2 (h, ci1).ravel ()
        ci1 = ci1.ravel ()
        cc = ci1.conj ().dot (ci1)
        e_tot1 = ci1.conj ().dot (hc1) / cc
        log.debug ('<Psi|[1,S**2,N]|Psi> = %e, %e, %e ; mu = %e', cc, ss, n, xconstr[0])
        log.debug ('These two energies should be the same: %e - %e = %e',
            e_tot0, e_tot1, e_tot0-e_tot1)

    def print_x (self, x, h, _print_fn=print, ci_maxlines=10, jac=None):
        norb, norb_f = self.norb, self.norb_f
        if jac is None: jac = self.jac (x, h)
        xconstr, xcc, xci_f = self.unpack (x)
        jconstr, jcc, jci_f = self.unpack (jac)
        _print_fn ('xconstr = {}'.format (xconstr))
        #kappa = np.zeros ((norb, norb), dtype=xcc.dtype)
        #kappa[self.uniq_orb_idx] = xcc[:]
        #kappa -= kappa.T
        #umat = linalg.expm (kappa)
        ci1_f = self.rotate_ci0 (xci_f)
        #_print_fn ('umat:')
        #fmt_str = ' '.join (['{:10.7f}',]*norb)
        #for row in umat: _print_fn (fmt_str.format (*row))
        for ix, (xci, ci1, jci, n) in enumerate (zip (xci_f, ci1_f, jci_f, norb_f)):
            _print_fn ('Fragment {} x and ci1 leading elements'.format (ix))
            fmt_det = '{:>' + str (max(4,n)) + 's}'
            fmt_str = ' '.join ([fmt_det, '{:>10s}', fmt_det, '{:>10s}', fmt_det, '{:>10s}'])
            _print_fn (fmt_str.format ('xdet', 'xcoeff', 'cdet', 'ccoeff', 'jdet', 'jcoeff'))
            strs_x = np.argsort (-np.abs (xci).ravel ())
            strs_c = np.argsort (-np.abs (ci1).ravel ())
            strs_j = np.argsort (-np.abs (jci).ravel ())
            strsa_x, strsb_x = np.divmod (strs_x, 2**n)
            strsa_c, strsb_c = np.divmod (strs_c, 2**n)
            strsa_j, strsb_j = np.divmod (strs_j, 2**n)
            fmt_str = ' '.join ([fmt_det, '{:10.3e}', fmt_det, '{:10.3e}', fmt_det, '{:10.3e}'])
            for irow, (sa, sb, ca, cb, ja, jb) in enumerate (zip (strsa_x, strsb_x, strsa_c, strsb_c, strsa_j, strsb_j)):
                if irow==ci_maxlines: break
                sdet = fockspace.onv_str (sa, sb, n)
                cdet = fockspace.onv_str (ca, cb, n)
                jdet = fockspace.onv_str (ja, jb, n)
                _print_fn (fmt_str.format (sdet, xci[sa,sb], cdet, ci1[ca,cb], jdet, jci[ja,jb]))

    def finalize_(self):
        ''' Update self.ci_f and set the corresponding part of self.x to zero '''
        xconstr, xcc, xci_f = self.unpack (self.x)
        self.ci_f = self.rotate_ci0 (xci_f) 
        for xc in xci_f: xc[:] = 0.0
        self.x = self.pack (xconstr, xcc, xci_f)
        return self.x

    gen_frag_basis = addons.gen_frag_basis
    get_dense_heff = addons.get_dense_heff

class FCISolver (direct_spin1.FCISolver):
    kernel = kernel
    approx_kernel = kernel
    make_rdm1 = make_rdm1
    make_rdm1s = make_rdm1s
    make_rdm12 = make_rdm12
    make_rdm12s = make_rdm12s
    get_init_guess = get_init_guess
    spin_square = spin_square
    contract_ss = contract_ss
    transform_ci_for_orbital_rotation = transform_ci_for_orbital_rotation
    def build_psi (self, *args, **kwargs):
        return LASUCCTrialState (self, *args, **kwargs)
    def save_psi (self, fname, psi):
        psi_raw = np.concatenate ([psi.x,] 
            + [c.ravel () for c in psi.ci_f])
        np.save (fname, psi_raw)
    def load_psi (self, fname, norb, nelec, norb_f=None, **kwargs):
        if norb_f is None: norb_f = getattr (fci, 'norb_f', [norb])
        raw = np.load (fname)
        ci0_f = [np.empty ((2**n,2**n), dtype=raw.dtype) for n in norb_f]
        psi = self.build_psi (ci0_f, norb, norb_f, nelec, **kwargs)
        n = psi.nvar
        psi.x[:], raw = raw[:n], raw[n:]
        for ci in psi.ci_f:
            n = ci.size
            s = ci.shape
            ci[:,:], raw = raw[:n].reshape (s), raw[n:]
        return psi
    def get_uop (self, norb, norb_f):
        freeze_mask = np.zeros ((norb, norb), dtype=np.bool_)
        for i,j in zip (np.cumsum (norb_f)-norb_f, np.cumsum(norb_f)):
            freeze_mask[i:j,i:j] = True
        return get_uccs_op (norb, freeze_mask=freeze_mask)
        # return get_uccs_op (norb)


