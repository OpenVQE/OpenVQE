import time
import numpy as np
from pyscf import ao2mo
from pyscf.lib import current_memory, numpy_helper
from pyscf.mcscf.mc1step import gen_g_hop
from mrh.util.basis import represent_operator_in_basis, is_basis_orthonormal, measure_basis_olap
from mrh.util.basis import orthonormalize_a_basis, get_complementary_states, get_overlapping_states
from mrh.util.basis import is_basis_orthonormal_and_complete
from mrh.util.rdm import get_2CDM_from_2RDM
from mrh.my_pyscf.df.sparse_df import sparsedf_array
from scipy import linalg
from itertools import product

''' Always remember: ~spin-restricted orbitals~ means the unitary group generator is spin-symmetric
regardless of the wave function! This means you NEVER handle an alpha fock matrix and a beta fock
matrix separately and you must do the "fake" semi-cumulant decomposition
    d^pr_qs = l^pr_qs + D^p_q D^r_s - D^p_s D^r_q / 2
without the spin-density terms! For now fix this by setting dm1s = [dm1/2, dm1/2] and focka = fockb
= fock. '''

class HessianCalculator (object):
    ''' Calculate elements of an orbital-rotation Hessian corresponding to particular orbital
    ranges in a CASSCF or LASSCF wave function. This is not designed to be efficient in orbital
    optimization; it is designed to collect slices of the Hessian stored explicitly for some kind
    of spectral analysis. '''

    def get_operator (self, r, s):
        return HessianOperator (self, r, s)

    def __init__(self, mf, oneRDMs, twoCDM, ao2amo):
        ''' Args:
                mf: pyscf.scf.hf object
                    Must also have an orthonormal complete set of orbitals in mo_coeff (they don't
                    have to be optimized in any way, they just have to be complete and orthonormal)
                oneRDMs: ndarray or list of ndarray with overall shape (2,nao,nao)
                    spin-separated one-body density matrix in the AO basis.
                twoCDM: ndarray or list of ndarray with overall shape (*,ncas,ncas,ncas,ncas) 
                    two-body cumulant density matrix of or list of two-body cumulant density
                    matrices for the active orbital space(s). Has multiple elements in LASSCF (in
                    general)
                ao2amo: ndarray of list of ndarray with overall shape (*,nao,ncas)
                    molecular orbital coefficients for the active space(s)
        '''
        self.scf = mf
        self.oneRDMs = np.asarray (oneRDMs)
        #if self.oneRDMs.ndim == 3: #== 2:
        #    dm = sum (self.oneRDMs) / 2
        #    self.oneRDMs /= 2
        #    self.oneRDMs = np.asarray ([dm, dm])#self.oneRDMs, self.oneRDMs])

        mo = self.scf.mo_coeff
        Smo = self.scf.get_ovlp () @ mo
        moH = mo.conjugate ().T
        moHS = moH @ self.scf.get_ovlp ()
        self.mo = mo
        self.Smo = Smo 
        self.moH = moH 
        self.moHS = moHS
        self.nao, self.nmo = mo.shape

        if isinstance (twoCDM, (list,tuple,)):
            self.twoCDM = twoCDM
        elif twoCDM.ndim == 5:
            self.twoCDM = [t.copy () for t in twoCDM]
        else:
            self.twoCDM = [twoCDM]
        self.nas = len (self.twoCDM)
        self.ncas = [t.shape[0] for t in self.twoCDM]
 
        if isinstance (ao2amo, (list,tuple,)):
            self.mo2amo = ao2amo
        elif ao2amo.ndim == 3:
            self.mo2amo = [l for l in ao2amo]
        else:
            self.mo2amo = [ao2amo]
        self.mo2amo = [self.moHS @ ao2a for ao2a in self.mo2amo]
        assert (len (self.mo2amo) == self.nas), "Same number of mo2amo's and twoCDM's required"

        # Precalculate (full,a|a,a) for fast gradients
        self.faaa = []
        f = np.eye (self.nmo)
        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            assert (t.shape == (n, n, n, n)), "twoCDM array size problem"
            assert (a.shape == (self.nmo, n)), "mo2amo array size problem"
            assert (is_basis_orthonormal (a)), 'problem putting active orbs in orthonormal basis'       
            self.faaa.append (self._get_eri ([f,a,a,a]))

        # Precalculate the fock matrix 
        vj, vk = self.scf.get_jk (dm=self.oneRDMs)
        fock = self.scf.get_hcore () + vj[0] + vj[1]
        self.fock = [fock - vk[0], fock - vk[1]]

        # Put 1rdm and fock in the orthonormal basis
        self.fock = [moH @ f @ mo for f in self.fock]
        self.oneRDMs = [moHS @ D @ Smo for D in self.oneRDMs]

    def __call__(self, *args, **kwargs):
        if len (args) == 0:
            ''' If no orbital ranges are passed, return the full mo-basis Hessian '''
            return self._call_fullrange (self.mo)
        elif len (args) == 1:
            ''' If 1 orbital range is passed, return Hessian with all 4 indices in that range'''
            return self._call_fullrange (args[0])
        elif len (args) == 2:
            ''' If 2 orbital ranges are passed, I assume that you are asking ONLY for the diag'''
            return self._call_diag (args[0], args[1])
        elif len (args) == 3:
            ''' No interpretation; raise an error '''
            raise RuntimeError ("Can't interpret 3 orbital ranges; pass 0, 1, 2, or 4")
        elif len (args) == 4:
            ''' If all 4 orbital ranges are passed, return the Hessian so specified. No permutation
                symmetry can be exploited. '''
            return self._call_general (args[0], args[1], args[2], args[3])
        else:
            raise RuntimeError (("Orbital Hessian has 4 orbital indices; you passed {} orbital "
                                 "ranges").format (len (args)))

    def _call_fullrange (self, p):
        '''Use full permutation symmetry to accelerate it!'''
        norb = [p.shape[-1], p.shape[-1], p.shape[-1], p.shape[-1]]
        if 0 in norb: return np.zeros (norb)
        # Put the orbital ranges in the orthonormal basis for fun and profit
        p = self.moHS @ p
        eris = HessianERITransformer (self, p, p, p, p)
        hess = self._get_Fock2 (p, p, p, p, eris)
        hess -= hess.transpose (1, 0, 2, 3)
        hess -= hess.transpose (0, 1, 3, 2)
        return hess / 2

    def _call_diag (self, p, q):
        lp, lq = p.shape[-1], q.shape[-1]
        hess = self._call_general (p, q, p, q).reshape (lp*lq, lp*lq)
        hess = np.diag (hess).reshape (lp, lq)
        return hess

    def _call_general (self, p, q, r, s):
        ''' The Hessian E2^pr_qs is F2^pr_qs - F2^qr_ps - F2^ps_qr + F2^qs_pr. 
        Since the orbitals are segmented into separate ranges, you can't necessarily just calculate
        one of these and transpose. '''
        norb = [p.shape[-1], q.shape[-1], r.shape[-1], s.shape[-1]]
        if 0 in norb: return np.zeros (norb)
        # Put the orbital ranges in the orthonormal basis for fun and profit
        p = self.moHS @ p
        q = self.moHS @ q
        r = self.moHS @ r
        s = self.moHS @ s
        eris = HessianERITransformer (self, p, q, r, s)
        hess  = self._get_Fock2 (p, q, r, s, eris)
        hess -= self._get_Fock2 (q, p, r, s, eris).transpose (1,0,2,3)
        hess -= self._get_Fock2 (p, q, s, r, eris).transpose (0,1,3,2)
        hess += self._get_Fock2 (q, p, s, r, eris).transpose (1,0,3,2)
        return hess / 2

    def _get_eri (self, orbs_list, compact=False):
        if isinstance (orbs_list, np.ndarray) and orbs_list.ndim == 2:
            orbs_list = [orbs_list, orbs_list, orbs_list, orbs_list]
        # Tragically, I have to go back to the AO basis to interact with PySCF's eri modules.
        # This is the greatest form of racism.
        orbs_list = [self.mo @ o for o in orbs_list]
        if self.scf._eri is not None:
            eri = ao2mo.incore.general (self.scf._eri, orbs_list, compact=compact) 
        elif self.with_df is not None:
            eri = self.with_df.ao2mo (orbs_list, compact=compact)
        else:
            eri = ao2mo.outcore.general_iofree (self.scf.mol, orbs_list, compact=compact)
        norbs = [o.shape[1] for o in orbs_list]
        if not compact: eri = eri.reshape (*norbs)
        return eri

    def _get_collective_basis (self, *args):
        p = np.concatenate (args, axis=-1)
        try: # Linear algebra problem sometimes if one of the arguments is a complete basis?
             # Can I exception-handle my way out of this?
            q = orthonormalize_a_basis (p)
        except np.linalg.LinAlgError as e:
            q = None
            for r in args:
                if is_basis_orthonormal_and_complete (r): q = r
            if q is None: raise (e)
        qH = q.conjugate ().T
        q2p = [qH @ arg for arg in args]
        return q, q2p

    def _get_Fock2 (self, p, q, r, s, eris):
        ''' This calculates one of the terms F2^pr_qs '''

        # Easiest term: 2 f^p_r D^q_s
        f_pr = [p.conjugate ().T @ f @ r for f in self.fock]
        D_qs = [q.conjugate ().T @ D @ s for D in self.oneRDMs]
        hess = 2 * sum ([np.multiply.outer (f, D) for f, D in zip (f_pr, D_qs)])
        hess = hess.transpose (0,2,1,3) # 'pr,qs->pqrs'

        # Generalized Fock matrix terms: delta_qr (F^p_s + F^s_p)
        ovlp_qr = q.conjugate ().T @ r
        if np.amax (np.abs (ovlp_qr)) > 1e-8: # skip if there is no ovlp between the q and r ranges
            gf_ps = self._get_Fock1 (p, s) + self._get_Fock1 (s, p).T
            hess += np.multiply.outer (ovlp_qr, gf_ps).transpose (2,0,1,3) # 'qr,ps->pqrs'

        # Explicit CDM contributions:  2 v^pu_rv l^qu_sv  +  2 v^pr_uv (l^qs_uv + l^qv_us)        
        t0, w0 = time.process_time (), time.time ()
        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            a2q = a.conjugate ().T @ q
            a2s = a.conjugate ().T @ s
            # If either q or s has no weight on the current active space, skip
            if np.amax (np.abs (a2q)) < 1e-8 or np.amax (np.abs (a2s)) < 1e-8:
                continue
            thess  = np.tensordot (eris (p,r,a,a), t, axes=((2,3),(2,3)))
            thess += np.tensordot (eris (p,a,r,a), t + t.transpose (0,1,3,2), axes=((1,3),(1,3)))
            thess = np.tensordot (thess, a2q, axes=(2,0)) # 'prab,aq->prbq'
            thess = np.tensordot (thess, a2s, axes=(2,0)) # 'prbq,bs->prqs'
            hess += 2 * thess.transpose (0, 2, 1, 3) # 'prqs->pqrs'

        # Weirdo split-coulomb and split-exchange terms
        # u,v are supersets of q,s 
        u = self._append_entangled (q)
        v = self._append_entangled (s)
        hess += 4 * self._get_splitc (p, q, r, s, sum (self.oneRDMs), u, v, eris) #perm_pq, perm_rs
        for dm in self.oneRDMs:
            hess -= 2 * self._get_splitx (p, q, r, s, dm, u, v, eris) #perm_pq, perm_rs
        return hess

    def _get_Fock1 (self, p, q):
        ''' Calculate the "generalized fock matrix" for orbital ranges p and q '''
        gfock = sum ([f @ D for f, D in zip (self.fock, self.oneRDMs)])
        gfock = p.conjugate ().T @ gfock @ q
        pH = p.conjugate ().T
        for t, a, n, faaa in zip (self.twoCDM, self.mo2amo, self.ncas, self.faaa):
            a2q = a.conjugate ().T @ q
            # If q has no weight on the current active space, skip
            if (not a2q.size) or np.amax (np.abs (a2q)) < 1e-8:
                continue
            eri = np.tensordot (pH, faaa, axes=1)
            gfock += np.tensordot (eri, t, axes=((1,2,3),(1,2,3))) @ a2q
        return gfock 

    def _get_splitc (self, p, q, r, s, dm, u, v, eris): #perm_pq, perm_rs):
        ''' v^pr_uv D^q_u D^s_v
        It shows up because some of the cumulant decomps put q and s on different 1rdm factors '''
        if u.shape[1] == 0 or v.shape[1] == 0: return 0
        D_uq = u.conjugate ().T @ dm @ q
        D_vs = v.conjugate ().T @ dm @ s
        if np.amax (np.abs (D_uq)) < 1e-8 or np.amax (np.abs (D_vs)) < 1e-8: return 0
        hess = eris (p,u,r,v)
        hess = np.tensordot (hess, D_uq, axes=(1,0)) # 'purv,uq->prvq'
        hess = np.tensordot (hess, D_vs, axes=(2,0)) # 'prvq,vs->prqs'
        return hess.transpose (0,2,1,3) # 'prqs->pqrs'

    def _get_splitx (self, p, q, r, s, dm, u, v, eris): #perm_pq, perm_rs):
        ''' (v^pv_ru + v^pr_vu) g^q_u g^s_v = v^pr_vu g^q_u g^s_v - v^pv_su g^q_u g^r_v
        It shows up because some of the cumulant decompositions put q and s on different 1rdm
        factors. Pay VERY CLOSE ATTENTION to the order of the indices! Remember p-q, r-s are the
        degrees of freedom and the contractions should resemble an exchange diagram from mp2! (I've
        exploited elsewhere the fact that v^pv_ru = v^pu_rv because that's just complex
        conjugation, but I wrote it as v^pv_ru here to make the point because you cannot swap u and
        v in the other one.)'''
        if u.shape[1] == 0 or v.shape[1] == 0: return 0
        D_uq = u.conjugate ().T @ dm @ q
        D_vs = v.conjugate ().T @ dm @ s
        if np.amax (np.abs (D_uq)) < 1e-8 or np.amax (np.abs (D_vs)) < 1e-8: return 0
        hess = eris (p,r,v,u) + eris (p,v,r,u).transpose (0,2,1,3)
        hess = np.tensordot (hess, D_uq, axes=(3,0)) # 'prvu,uq->prvq'
        hess = np.tensordot (hess, D_vs, axes=(2,0)) # 'prvq,vs->prqs'
        return hess.transpose (0,2,1,3) # 'prqs->pqrs'

    def _append_entangled (self, p):
        ''' Do SVD of 1-rdms to get a small number of orbitals that you need to actually pay
        attention to when computing splitc and splitx eris. Append these extras to the end of p '''
        q = get_complementary_states (p)
        if 0 in q.shape: return p
        qH = q.conjugate ().T
        lvecs = []
        for dm in self.oneRDMs + [sum (self.oneRDMs)]:
            lvec, sigma, rvec = linalg.svd (qH @ dm @ p, full_matrices=False)
            idx = np.abs (sigma) > 1e-8
            if np.count_nonzero (idx): lvecs.append (lvec[:,idx])
        if len (lvecs):
            lvec = self._get_collective_basis (*lvecs)[0]
            u = q @ lvec
            return np.append (p, u, axis=1)
        return p

    def get_diagonal_step (self, p, q):
        ''' Obtain a gradient-descent approximation for the relaxation of orbitals p in range q
        using the gradient and diagonal elements of the Hessian, x^p_q = -E1^p_q / E2^pp_qq '''
        # First, get the gradient and svd to obtain conjugate orbitals of p in q
        grad = self._get_Fock1 (p, q) - self._get_Fock1 (q, p).T
        lvec, e1, rvecH = linalg.svd (grad, full_matrices=False)
        rvec = rvecH.conjugate ().T
        p = p @ lvec
        q = q @ rvec
        lp = p.shape[-1]
        lq = q.shape[-1]
        lpq = lp * lq
        # Zero gradient escape
        if not np.count_nonzero (np.abs (e1) > 1e-8): return p, np.zeros (lp), q
        # Because this ^ is faster than sectioning it and even with 20+20+20 active/inactive/ext,
        # it's still only 100 MB
        f_pp = np.stack ([((f @ p) * p).sum (0) for f in self.fock], 0)
        f_qq = np.stack ([((f @ q) * q).sum (0) for f in self.fock], 0)
        f_pq = np.stack ([((f @ p) * q).sum (0) for f in self.fock], 0)
        d_pp = np.stack ([((d @ p) * p).sum (0) for d in self.oneRDMs], 0)
        d_qq = np.stack ([((d @ q) * q).sum (0) for d in self.oneRDMs], 0)
        d_pq = np.stack ([((d @ p) * q).sum (0) for d in self.oneRDMs], 0)
        e2 = ((f_pp * d_qq) + (f_qq * d_pp) - 2 * (f_pq * d_pq)).sum (0)
        #e2 = self.__call__(p, q, p, q)
        #e2 = np.diag (np.diag (e2.reshape (lpq, lpq)).reshape (lp, lq))
        return p, -e1 / e2, q

    def get_impurity_conjugate_gradient (self, i, c, a):
        g, x_ga, a = self.get_diagonal_step (i, a)
        Hop = self.get_operator (g, a)
        iH = i.conjugate ().T
        e2 = Hop (c, i, x_ga)
        return c @ (self._get_Fock1 (c, i) - self._get_Fock1 (i, c).T + e2) @ iH

    def get_conjugate_gradient (self, pq_pairs, r, s):
        ''' OLD CODE, NOT USED '''
        ''' Obtain the gradient for ranges p->q after making approx gradient-descent step in r->s:
        E1'^p_q = E1^p_q - E2^pr_qs * x^r_s = E1^p_q + E2^pr_qs * E1^r_s / E2^rr_ss '''
        t0, w0 = time.process_time (), time.time ()
        r, x_rs, s = self.get_diagonal_step (r, s)
        print ("Time to get diagonal step: {:.3f} s clock, {:.3f} wall".format (
            time.process_time () - t0, time.time () - w0))
        e1 = np.zeros ((self.nao,self.nao), dtype=np.float64)
        for p, q in pq_pairs:
            qH = q.conjugate ().T
            e1pq = self._get_Fock1 (p, q) - self._get_Fock1 (q, p).T
            e1 += p @ e1pq @ qH
        # Zero step escape
        if not np.count_nonzero (np.abs (x_rs) > 1e-8): return e1
        lr = r.shape[-1]
        ls = s.shape[-1]
        diag_idx = np.arange (lr, dtype=int)
        diag_idx = (diag_idx * lr) + diag_idx
        #t0, w0 = time.process_time (), time.time ()
        Hop = self.get_operator (r, s)
        #print ("Time to make Hop: {:.3f} s clock, {:.3f} wall".format (time.process_time () - t0,
        #   time.time () - w0))
        #p = self._get_collective_basis (pq_pairs[0][0], pq_pairs[1][0])[0]
        #q = self._get_collective_basis (pq_pairs[0][1], pq_pairs[1][1])[0]
        #qH = q.conjugate ().T
        #e1_comp = p @ (self._get_Fock1 (p, q) - self._get_Fock1 (q, p).T + Hop (p, q, x_rs)) @ qH
        for p, q in pq_pairs:
            qH = q.conjugate ().T
            e2 = Hop (p, q, x_rs)
            e1 += p @ e2 @ qH
            e2 = None
        #for p, q in pq_pairs:
        #    pH = p.conjugate ().T
        #    errmat = pH @ (e1_comp - e1) @ q
        #    print ("Error in jointly-calculated gradient: {}".format (linalg.norm (errmat)))
        return e1

    def get_veff (self, dm1s):
        if dm1s.ndim == 4:
            ndmat = dm1s.shape[0]
            dm1s = dm1s.reshape (dm1s.shape[0]*dm1s.shape[1], dm1s.shape[2], dm1s.shape[3])
        else:
            errstr = 'must provide an even number of density matrices (a1, b1, a2, b2, ...)'
            assert (dm1s.ndim == 3 and dm1s.shape[0] % 2 == 0), errstr
            ndmat = dm1s.shape[0] // 2
        vj, vk = self.get_jk (dm1s)
        vj = vj.reshape (ndmat, 2, dm1s.shape[-2], dm1s.shape[-1])
        vk = vk.reshape (ndmat, 2, dm1s.shape[-2], dm1s.shape[-1])
        return vj.sum (1)[:,None,:,:] - vk

    def get_jk (self, dm1s):
        if dm1s.ndim == 2: dm1s = dm1s[None,:,:]
        dm1s = np.dot (self.mo, np.dot (dm1s, self.moH)).transpose (1,0,2)
        vj, vk = self.scf.get_jk (dm=dm1s)
        vj = np.dot (self.moH, np.dot (vj, self.mo)).transpose (1,0,2)
        vk = np.dot (self.moH, np.dot (vk, self.mo)).transpose (1,0,2)
        return vj, vk

class CASSCFHessianTester (object):
    ''' Use pyscf.mcscf.mc1step.gen_g_hop to test HessianCalculator.
    There are 3 nonredundant orbital rotation sectors: ui, ai, and au
    Therefore there are 6 nonredundant Hessian sectors: uiui, uiai,
    uiau, aiai, aiau, and auau. Note that PySCF chooses to store the
    lower-triangular (p>q) part. Sadly I can't use permutation symmetry
    to accelerate any of these because any E2^pr_qr would necessarily
    refer to a redundant rotation.'''

    def __init__(self, mc):
        oneRDMs = mc.make_rdm1s ()
        casdm1s = mc.fcisolver.make_rdm1s (mc.ci, mc.ncas, mc.nelecas)
        casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)
        twoCDM = get_2CDM_from_2RDM (casdm2, casdm1s)
        ao2amo = mc.mo_coeff[:,mc.ncore:][:,:mc.ncas]
        self.calculator = HessianCalculator (mc._scf, oneRDMs, twoCDM, ao2amo)
        self.cas = mc
        self.cas_mo = mc.mo_coeff
        self.ncore, self.ncas, self.nelecas = mc.ncore, mc.ncas, mc.nelecas
        self.nocc = self.ncore + self.ncas
        self.nmo = self.cas_mo.shape[1]
        self.hop, self.hdiag = gen_g_hop (mc, self.cas_mo, 1, casdm1, casdm2, mc.ao2mo (self.cas_mo))[2:]

    def __call__(self, pq, rs=None):
        ''' pq, rs = 0 (ui), 1 (ai), 2 (au) '''
        if rs is None: return self._call_diag (pq)
        
        p, q, prange, qrange, np, nq = self._parse_range (pq)
        r, s, rrange, srange, nr, ns = self._parse_range (rs)

        my_hess = self.calculator (p, q, r, s)
        print ("{:8s} {:13s} {:13s} {:13s} {:13}".format ('Idx', 'Mine', "PySCF's", 'Difference',
                                                          'Ratio'))
        fmt_str = "{0:d},{1:d},{2:d},{3:d} {4:13.6e} {5:13.6e} {6:13.6e} {7:13.6e}"

        for (ixp, pi), (ixq, qi) in product (enumerate (prange), enumerate (qrange)):
            Py_hess = self._pyscf_hop_call (pi, qi, rrange, srange)
            for (ixr, ri), (ixs, si) in product (enumerate (rrange), enumerate (srange)):
                diff = my_hess[ixp,ixq,ixr,ixs] - Py_hess[ixr,ixs]
                rat = my_hess[ixp,ixq,ixr,ixs] / Py_hess[ixr,ixs]
                print (fmt_str.format (pi, qi, ri, si, my_hess[ixp,ixq,ixr,ixs], Py_hess[ixr,ixs],
                                       diff, rat))

    def _call_diag (self, pq):
        offs = ix = np = nq = 0
        while ix < pq:
            np, nq = self._parse_range (ix)[4:]
            offs += np*nq
            ix += 1
        p, q, prange, qrange, np, nq = self._parse_range (pq)
        Py_hdiag = self.hdiag[offs:][:np*nq].reshape (np, nq)
        my_hdiag = self.calculator (p, q)

        print ("{:8s} {:13s} {:13s} {:13s} {:13}".format ('Idx', 'Mine', "PySCF's", 'Difference',
                                                          'Ratio'))
        fmt_str = "{0:d},{1:d},{0:d},{1:d} {2:13.6e} {3:13.6e} {4:13.6e} {5:13.6e}"
        for (ixp, pi), (ixq, qi) in product (enumerate (prange), enumerate (qrange)):
            diff = my_hdiag[ixp,ixq] - Py_hdiag[ixp,ixq]
            rat = my_hdiag[ixp,ixq] / Py_hdiag[ixp,ixq]
            print (fmt_str.format (pi, qi, my_hdiag[ixp,ixq], Py_hdiag[ixp,ixq], diff, rat))

    def _parse_range (self, pq):
        if pq == 0: # ui
            p = self.cas_mo[:,self.ncore:self.nocc]
            q = self.cas_mo[:,:self.ncore]
            prange = range (self.ncore,self.nocc)
            qrange = range (self.ncore)
            np = self.ncas
            nq = self.ncore
        elif pq == 1: # ai
            p = self.cas_mo[:,self.nocc:]
            q = self.cas_mo[:,:self.ncore]
            prange = range (self.nocc, self.nmo)
            qrange = range (self.ncore)
            np = self.nmo - self.nocc
            nq = self.ncore
        elif pq == 2: # au
            p = self.cas_mo[:,self.nocc:]
            q = self.cas_mo[:,self.ncore:self.nocc]
            prange = range (self.nocc, self.nmo)
            qrange = range (self.ncore,self.nocc)
            np = self.nmo - self.nocc
            nq = self.ncas
        else: 
            raise RuntimeError ("Undefined range {}".format (pq))
        return p, q, prange, qrange, np, nq

    def _pyscf_hop_call (self, ip, iq, rrange, srange):
        kappa = np.zeros ([self.nmo, self.nmo])
        kappa[ip,iq] = 1
        kappa = self.cas.pack_uniq_var (kappa)
        py_hess = self.hop (kappa)
        py_hess = self.cas.unpack_uniq_var (py_hess)
        return py_hess[rrange,:][:,srange]

class LASSCFHessianCalculator (HessianCalculator):

    def get_operator (self, r, s):
        if getattr (self.ints, 'with_df', None):
            if self.Hop_noxc:
                return DFLASSCFHessianOperator_noxc (self, r, s)
            else:
                return DFLASSCFHessianOperator (self, r, s)
        else:
            return LASSCFHessianOperator (self, r, s)

    def __init__(self, ints, oneRDM_loc, all_frags, fock_c, fock_s, Hop_noxc=False):
        self.ints = ints
        self.Hop_noxc = Hop_noxc
        active_frags = [f for f in all_frags if f.norbs_as]

        # Global things. fock_s is zero because of the semi-cumulant decomposition; this only works
        # because I don't call this constructer for intersubspace ranges!
        self.nmo = self.nao = ints.norbs_tot
        self.mo = self.moH = self.Smo = self.moHS = np.eye (self.nmo)
        oneSDM_loc = sum ([f.oneSDMas_loc for f in active_frags])
        self.oneRDMs = [(oneRDM_loc + oneSDM_loc)/2, (oneRDM_loc - oneSDM_loc)/2]
        #fock_c = ints.loc_rhf_fock_bis (oneRDM_loc)
        #fock_s = -ints.loc_rhf_k_bis (oneSDM_loc)/2 if isinstance (oneSDM_loc, np.ndarray) else 0
        #print ("Error: {} charge; {} spin".format (linalg.norm (fock_c - fock_c_check),
        #                                           linalg.norm (fock_s - fock_s_check)))
        self.fock = [fock_c + fock_s, fock_c - fock_s]

        # Fragment things
        self.mo2amo = [f.loc2amo for f in active_frags]
        self.twoCDM = [f.twoCDMimp_amo for f in active_frags]
        self.ncas = [f.norbs_as for f in active_frags]
        self.faaa = [f.eri_gradient for f in active_frags]

        # Fix cumulant decomposition
        for ix, (mo, dm2, ncas) in enumerate (zip (self.mo2amo, self.twoCDM, self.ncas)):
            moH = mo.conjugate ().T
            dm1s = [moH @ dm @ mo for dm in self.oneRDMs]
            dm1 = dm1s[0] + dm1s[1]
            correction = sum ([np.multiply.outer (dm, dm) for dm in dm1s])
            correction -= np.multiply.outer (dm1, dm1) / 2
            dm2 = dm2 + correction.transpose (0,3,2,1)
            self.twoCDM[ix] = dm2

    def _get_eri (self, orbs_list, compact=False):
        return self.ints.general_tei (orbs_list, compact=compact)
        
    def get_jk (self, dm1s):
        if dm1s.ndim == 2: dm1s = dm1s[None,:,:]
        mo = self.ints.ao2loc
        moH = mo.conjugate ().T
        dm1s = np.dot (mo, np.dot (dm1s, moH)).transpose (1,0,2)
        vj, vk = self.ints.get_jk_ao (dm=dm1s)
        # Note: the below _works_!
        #dm = dm1s.sum (0)
        #dm += dm.T
        #dm[np.diag_indices_from (dm)] /= 2
        #dm = numpy_helper.pack_tril (dm)
        #vj_test = np.zeros_like (vj.sum (0))
        #for eri1 in self.ints.with_df.loop ():
        #    rho = np.dot (eri1, dm)
        #    vj_test += numpy_helper.unpack_tril (np.dot (rho, eri1))
        #    print (vj_test)
        #print ("Comparing vj to manual vj recalculation using cderi: {}".format (vj.sum (0)
        #                                                                         - vj_test))
        vj = np.dot (moH, np.dot (vj, mo)).transpose (1,0,2)
        vk = np.dot (moH, np.dot (vk, mo)).transpose (1,0,2)
        return vj, vk

    def get_diagonal_step (self, p, q):
        ''' Obtain a gradient-descent approximation for the relaxation of orbitals p in range q
        using the gradient and diagonal elements of the Hessian, x^p_q = -E1^p_q / E2^pp_qq '''
        # First, get the gradient and svd to obtain conjugate orbitals of p in q
        grad = self._get_Fock1 (p, q) - self._get_Fock1 (q, p).T
        lvec, e1, rvecH = linalg.svd (grad, full_matrices=False)
        rvec = rvecH.conjugate ().T
        p = p @ lvec
        q = q @ rvec
        lp = p.shape[-1]
        lq = q.shape[-1]
        lpq = lp * lq
        # Zero gradient escape
        if not np.count_nonzero (np.abs (e1) > 1e-8): return p, np.zeros (lp), q
        # Because this ^ is faster than sectioning it and even with 20+20+20 active/inactive/ext,
        # it's still only 100 MB
        if self.Hop_noxc:
            f_pp = np.stack ([((f @ p) * p).sum (0) for f in self.fock], 0)
            f_qq = np.stack ([((f @ q) * q).sum (0) for f in self.fock], 0)
            f_pq = np.stack ([((f @ p) * q).sum (0) for f in self.fock], 0)
            d_pp = np.stack ([((d @ p) * p).sum (0) for d in self.oneRDMs], 0)
            d_qq = np.stack ([((d @ q) * q).sum (0) for d in self.oneRDMs], 0)
            d_pq = np.stack ([((d @ p) * q).sum (0) for d in self.oneRDMs], 0)
            e2 = ((f_pp * d_qq) + (f_qq * d_pp) - 2 * (f_pq * d_pq)).sum (0)
        else:
            e2 = self.__call__(p, q, p, q)
            e2 = np.diag (np.diag (e2.reshape (lpq, lpq)).reshape (lp, lq))
        return p, -e1 / e2, q

class HessianERITransformer (object):

    def __init__(self, parent, p, q, r, s): 
        ''' (wx|yz)
                w: a & p
                x: a & s & r & q 
                y: a & s & r
                z: a & s & q
            Based on the idea that p is the largest orbital range and s is the smallest,
            so p appears only once and s appears three times. Given permutation symmetries
            (wx|yz) = (yz|wx) = (xw|yz) = (wx|zy), I can generate all the eris I need for the
            Hessian calculation from this cache. Since this calls _get_eri, it will also
            automatically take advantage of _eri_kernel if it's available.
        '''
        p,q,r,s = (parent._append_entangled (z) for z in (p,q,r,s))
        a = np.concatenate (parent.mo2amo, axis=1)
        self.w = w = parent._get_collective_basis (a, p)[0]
        self.x = x = parent._get_collective_basis (a, s, r, q)[0] 
        self.y = y = parent._get_collective_basis (a, s, r)[0]
        self.z = z = parent._get_collective_basis (a, s, q)[0]
        self._eri = parent._get_eri ([w,x,y,z])
        return

    def __call__(self, p, q, r, s, _first_call=True):
        ''' Because of several necessary index permutations, I cannot know in advance which of
        w,x,y,z encloses each of p, q, r, s, but I should have prepared it so that any call I make
        can be carried out. yz is the more restrictive pair in my cache, so first see if r, s is in
        yz and if not, flip pq<->rs. wx contains all pairs that I should ever need so. ''' 
        rs_yz, rs_correct = self.pq_in_cd (self.y, self.z, r, s)
        pq_wx, pq_correct = self.pq_in_cd (self.w, self.x, p, q)
        if _first_call and (not (pq_wx and rs_yz)):
            return self.__call__(r, s, p, q, _first_call=False).transpose (2, 3, 0, 1)
        try:
            assert (pq_wx and rs_yz), "Can't place orbital sets in this eri array"
        except AssertionError as e:
            print (p.shape, q.shape, r.shape, s.shape)
            print (self.w.shape, self.x.shape, self.y.shape, self.z.shape)
            for name1, arr1 in zip (('w','x','y','z'), (self.w, self.x, self.y, self.z)):
                for name2, arr2 in zip (('p','q','r','s'), (p, q, r, s)):
                    print ("Is {} in {}? {}".format (name2, name1, self.p_in_c (
                        arr1, arr2, _return_numbers=True)))
            print ("Is p orthonormal? {}".format (linalg.eigh (p.conjugate ().T @ p)[0]))
            print ("Is q orthonormal? {}".format (linalg.eigh (q.conjugate ().T @ q)[0]))
            print ("Is r orthonormal? {}".format (linalg.eigh (r.conjugate ().T @ r)[0]))
            print ("Is s orthonormal? {}".format (linalg.eigh (s.conjugate ().T @ s)[0]))
            print ("Is w orthonormal? {}".format (linalg.eigh (self.w.conjugate ().T @ self.w)[0]))
            print ("Is x orthonormal? {}".format (linalg.eigh (self.x.conjugate ().T @ self.x)[0]))
            print ("Is y orthonormal? {}".format (linalg.eigh (self.y.conjugate ().T @ self.y)[0]))
            print ("Is z orthonormal? {}".format (linalg.eigh (self.z.conjugate ().T @ self.z)[0]))
            raise (e)
        # Permute the order of the pairs individually
        if pq_correct and rs_correct: return self._grind (p, q, r, s)
        elif pq_correct: return self._grind (p, q, s, r).transpose (0, 1, 3, 2)
        elif rs_correct: return self._grind (q, p, r, s).transpose (1, 0, 2, 3)
        else: return self._grind (q, p, s, r).transpose (1, 0, 3, 2)

    def _grind (self, p, q, r, s):
        assert (self.p_in_c (self.w, p)), 'p not in w after permuting!'
        assert (self.p_in_c (self.x, q)), 'q not in x after permuting!'
        assert (self.p_in_c (self.y, r)), 'r not in y after permuting!'
        assert (self.p_in_c (self.z, s)), 's not in z after permuting!'
        p2w = p.conjugate ().T @ self.w
        x2q = self.x.conjugate ().T @ q
        y2r = self.y.conjugate ().T @ r
        z2s = self.z.conjugate ().T @ s
        pqrs = np.tensordot (p2w, self._eri, axes=1)
        pqrs = np.tensordot (pqrs, x2q, axes=((1),(0)))
        pqrs = np.tensordot (pqrs, y2r, axes=((1),(0)))
        pqrs = np.tensordot (pqrs, z2s, axes=((1),(0)))
        return pqrs

    def p_in_c (self, c, p, _return_numbers=False):
        ''' Return c == complete basis for p '''
        svals = linalg.svd (c.conjugate ().T @ p)[1]
        val = np.count_nonzero (np.isclose (svals, 1, rtol=1e-3)) == p.shape[1]
        if _return_numbers: return val, svals-1, p.shape[1]
        else: return val

    def pq_in_cd (self, c, d, p, q):
        testmat = np.asarray ([[self.p_in_c (e, r) for e in (c, d)] for r in (p, q)])
        if testmat[0,0] and testmat[1,1]: return True, True # c contains d and p contains q
        elif testmat[1,0] and testmat[0,1]: return True, False # c contains q and p contains d
        else: return False, False # Wrong pair

class HessianOperator (HessianCalculator):

    def __init__(self, calculator, r, s):
        self.__dict__.update (calculator.__dict__)
        self.r = r
        self.s = s
        rH = self.r.conjugate ().T
        sH = self.s.conjugate ().T
        a = np.concatenate (self.mo2amo, axis=-1)
        do_r = sum (abs (linalg.svd (sH @ a)[1])) > 1e-8
        do_s = sum (abs (linalg.svd (rH @ a)[1])) > 1e-8
        if do_r and do_s: self.k = k = self._get_collective_basis (self.r, self.s)[0]
        elif do_r: self.k = k = r
        elif do_s: self.k = k = s
        i = np.eye (self.nmo)
        self.eris = HessianERITransformer (self, i, k, a, a) if (do_r or do_s) else None

    def _unpack_x (self, x_rs, diag=False):
        if diag: # Unpack diagonal elements only
            kappa = np.diag (x_rs)
        else:
            kappa = x_rs.reshape (self.r.shape[-1], self.s.shape[-1])
        sH = self.s.conjugate ().T
        kappa = self.r @ kappa @ sH
        kappa = kappa - kappa.T
        return kappa

    def _get_tFock1_1b (self, kappa):
        tdm1s = np.stack ([kappa @ dm - dm @ kappa for dm in self.oneRDMs], axis=0)
        tFock1 = np.tensordot (self.fock, tdm1s, axes=((0,1),(0,2)))
        veff = np.squeeze (self.get_veff (tdm1s))
        dm1s = np.stack (self.oneRDMs, axis=0)
        tFock1 += np.tensordot (veff, dm1s, axes=((0,1),(0,2)))
        return tFock1

    def _get_tFock1_2b (self, p, q, kappa):
        k, kH = self.k, self.k.conjugate ().T
        pH = p.conjugate ().T
        qH = q.conjugate ().T
        tFock1 = np.zeros ((p.shape[-1], q.shape[-1]))
        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            # The term is always positive as long as the "a" index is the ~second~ index of kappa
            qKa = qH @ kappa @ a
            if linalg.norm (qKa) > 1e-8:
                g_paaa = self.eris (p, a, a, a)
                l_qaaa = np.tensordot (qKa, t, axes=1) # contraction on the first index
                tFock1 += np.tensordot (g_paaa, l_qaaa, axes=((1,2,3),(1,2,3)))
            q2a = qH @ a
            kap_ka = kH @ kappa @ a
            if linalg.norm (q2a) > 1e-8 and linalg.norm (kap_ka) > 1e-8:
                l_qaaa = np.tensordot (q2a, t, axes=1)
                g_pkaa = np.tensordot (self.eris (p, k, a, a), kap_ka, axes=((1),(0))).transpose (
                    0,3,1,2) # contraction on the second index
                g_paka = np.tensordot (self.eris (p, a, a, k), kap_ka, axes=1) # contract 4th idx
                g_paka += g_paka.transpose (0, 1, 3, 2) # contract on the 3rd index
                tFock1 += np.tensordot (g_pkaa, l_qaaa, axes=((1,2,3),(1,2,3)))
                tFock1 += np.tensordot (g_paka, l_qaaa, axes=((1,2,3),(1,2,3)))
        return tFock1

    def __call__(self, p, q, x_rs):
        if 0 in p.shape or 0 in q.shape: return np.zeros ((p.shape[-1], q.shape[-1]))
        is_diag = (x_rs.size == self.r.shape[-1]) and (x_rs.size == self.s.shape[-1])
        kappa = self._unpack_x (x_rs, diag=is_diag)
        pH = p.conjugate ().T
        tFock1 = self._get_tFock1_1b (kappa)
        cgrad = pH @ (tFock1 - tFock1.T) @ q
        cgrad += self._get_tFock1_2b (p, q, kappa) 
        cgrad -= self._get_tFock1_2b (q, p, kappa).T
        return cgrad

class LASSCFHessianOperator (LASSCFHessianCalculator, HessianOperator):
    __init__ = HessianOperator.__init__
    __call__ = HessianOperator.__call__

class DFLASSCFHessianOperator (LASSCFHessianOperator):
    def __init__(self, calculator, r, s):
        self.__dict__.update (calculator.__dict__)
        self.r = r
        self.s = s
        self.rs = self._get_collective_basis (r, s)[0]
        self.m2rs = self.ints.ao2loc @ self.rs

    def _get_tFock1_1b (self, kappa):
        # Do only strictly 1-body part; no veff here
        tdm1s = np.stack ([kappa @ dm - dm @ kappa for dm in self.oneRDMs], axis=0)
        return np.tensordot (self.fock, tdm1s, axes=((0,1),(0,2)))
        
    def _get_tFock1_2b (self, p, q, kappa):
        ''' This assumes that p and q are unentangled subspaces, that p has no active orbitals,
        that r,s contains only full active subspaces, and that r,s are contained entirely in
        exactly one of p,q!'''
        # Include veff terms here by reusing intermediates
        # Include both F_pq and F_qp!
        m2p = self.ints.ao2loc @ p
        m2q = self.ints.ao2loc @ q
        m2r = self.m2rs
        p2m = m2p.conjugate ().T
        q2m = m2q.conjugate ().T
        r2m = m2r.conjugate ().T
        qH = q.conjugate ().T
        rsH = self.rs.conjugate ().T
        rs2q = r2q = rsH @ q
        q2rs = q2r = rs2q.conjugate ().T
        svals_rinq = linalg.svd (rs2q)[1].sum ()
        rs2p = rsH @ p
        svals_rinp = linalg.svd (rs2p)[1].sum ()
        rinq = abs (svals_rinq - self.rs.shape[-1]) < 1e-5
        rinp = abs (svals_rinp - self.rs.shape[-1]) < 1e-5
        if rinp:
            return np.zeros ((p.shape[-1], q.shape[-1]))
            # Hack to keep both F_pq and F_qp in the rinq case
        elif not rinq:
            raise RuntimeError (("Totally off-diagonal Hessian elements not supported "
                                 "({} {} {})").format (svals_rinq, svals_rinp, self.rs.shape[-1]))
        tdm1s = np.stack ([kappa @ dm - dm @ kappa for dm in self.oneRDMs], axis=0)
        tdm1s = np.dot (qH, np.dot (tdm1s, q)).transpose (1,0,2) # Put in q (impurity) basis
        pH = p.conjugate ().T
        qH = q.conjugate ().T
        dm_pp = np.dot (pH, np.dot (self.oneRDMs, p)).transpose (1,0,2)
        dm_qq = np.dot (qH, np.dot (self.oneRDMs, q)).transpose (1,0,2)
        b_mqP = sparsedf_array (self.ints.with_df._cderi).contract1 (m2q)
        b_rqP = np.tensordot (m2r, b_mqP, axes=((0),(0)))
        g_mqrq = np.tensordot (b_mqP, b_rqP, axes=((2),(2)))
        tdm_qr = np.dot (tdm1s, r2q.conjugate ().T)
        # Coulomb
        vj_pq = p2m @ (np.tensordot (g_mqrq, tdm_qr.sum (0).T, axes=2))
        # Exchange
        vk_pq = np.dot (p2m, np.tensordot (tdm_qr, g_mqrq, axes=((1,2),(1,2)))).transpose (1,0,2)
        # Veff 
        veff_pq = vj_pq[None,:,:] - vk_pq
        tFock1 = np.tensordot (veff_pq, dm_qq, axes=((0,2),(0,1)))
        tFock1 -= np.tensordot (dm_pp, veff_pq, axes=((0,1),(0,1)))
        # Cumulant
        g_mrrr = np.tensordot (r2q, np.dot (g_mqrq, q2r), axes=((1),(1))).transpose (1,0,2,3)
        for t, a, n in zip (self.twoCDM, self.mo2amo, self.ncas):
            rs2a = rsH @ a
            if linalg.norm (rs2a) < 1e-8: continue
            l_rrrr = np.tensordot (rs2a, t,      axes=((1),(1))) # (ar|aa) (idxs 0,1 switched)
            l_rrrr = np.tensordot (rs2a, l_rrrr, axes=((1),(1))) # (rr|aa) (calling idx 1 gets 0)
            l_rrrr = np.tensordot (l_rrrr, rs2a, axes=((2),(1))) # (rr|ra) (idxs 2,3 switched)
            l_rrrr = np.tensordot (l_rrrr, rs2a, axes=((2),(1))) # (rr|rr) (calling idx 2 gets 3)
            rKr = rsH @ kappa @ self.rs
            # This is a sum, so I can't do the index-order switching thing I did above
            # It's always positive as long as I always contract the ~second~ axis of rKr
            lk_rrrr  = np.tensordot (rKr, l_rrrr, axes=((1),(0))) 
            lk_rrrr += np.tensordot (rKr, l_rrrr, axes=((1),(1))).transpose (1,0,2,3) 
            lk_rrrr += np.tensordot (l_rrrr, rKr, axes=((3),(1))) 
            lk_rrrr += np.tensordot (l_rrrr, rKr, axes=((2),(1))).transpose (0,1,3,2)
            tF_mr = np.tensordot (g_mrrr, lk_rrrr, axes=((1,2,3),(1,2,3)))
            tFock1 += (p2m @ tF_mr @ rs2q) - (q2m @ tF_mr @ rs2p).T
        return tFock1
        
class DFLASSCFHessianOperator_noxc (DFLASSCFHessianOperator):
    ''' Approximate the Hessian by omitting the response of the exchange potential and correlation
    part of the gradient, leaving only the one-body density and Coulomb responses. '''

    def _get_tFock1_2b (self, p, q, kappa):
        rsH = self.rs.conjugate ().T
        rs2q = r2q = rsH @ q
        q2rs = q2r = rs2q.conjugate ().T
        svals_rinq = linalg.svd (rs2q)[1].sum ()
        rs2p = rsH @ p
        svals_rinp = linalg.svd (rs2p)[1].sum ()
        rinq = abs (svals_rinq - self.rs.shape[-1]) < 1e-5
        rinp = abs (svals_rinp - self.rs.shape[-1]) < 1e-5
        if rinp:
            return np.zeros ((p.shape[-1], q.shape[-1]))
            # Hack to keep both F_pq and F_qp in the rinq case
        elif not rinq:
            raise RuntimeError (("Totally off-diagonal Hessian elements not supported "
                                 "({} {} {})").format (svals_rinq, svals_rinp, self.rs.shape[-1]))
        m2p = self.ints.ao2loc @ p
        m2q = self.ints.ao2loc @ q
        p2m = m2p.conjugate ().T
        ao2loc = self.ints.ao2loc
        loc2ao = ao2loc.conjugate ().T
        dm1 = self.oneRDMs[0] + self.oneRDMs[1]
        tdm1 = ao2loc @ (kappa @ dm1 - dm1 @ kappa) @ loc2ao
        tdm1 = numpy_helper.pack_tril (tdm1 + tdm1.T - np.diag (np.diag (tdm1)))      
        vj_pq = np.zeros ((p.shape[-1], q.shape[-1]), dtype=tdm1.dtype)
        for cderi in self.ints.with_df.loop ():
            rho = np.dot (cderi, tdm1)
            vj_pq += p2m @ numpy_helper.unpack_tril (np.dot (rho, cderi)) @ m2q
        pH = p.conjugate ().T
        qH = q.conjugate ().T
        dm1_pp = pH @ dm1 @ p
        dm1_qq = qH @ dm1 @ q
        tFock1 = vj_pq @ dm1_qq - dm1_pp @ vj_pq
        return tFock1

