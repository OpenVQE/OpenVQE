import time
import numpy as np
from scipy import linalg
from pyscf import gto, lib, ao2mo
from mrh.my_pyscf.mcscf import lasci, lasci_sync, lasscf_sync_o0, _DFLASCI
from functools import partial

# Let's finally implement, in the more pure LASSCF rewrite, the ERI-
# related cost savings that I made such a big deal about in JCTC 2020,
# 16, 4923

def make_schmidt_spaces (h_op):
    ''' Build the spaces which active orbitals will explore in this
    macrocycle, based on gradient and Hessian SVDs and Schmidt
    decompositions. Must be called after __init__ is complete

    Args:
        h_op: LASSCF Hessian operator instance

    Returns:
        uschmidt : nfrag ndarrays of shape (nmo, *)
            The number of subspaces built is equal to the
            product of the number of irreps in the molecule
            and the number of fragments, minus the number
            of null spaces.

    '''
    las = h_op.las
    ugg = h_op.ugg
    mo_coeff = h_op.mo_coeff
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    dm1 = h_op.dm1s.sum (0)
    g_vec = h_op.get_grad ()
    prec_op = h_op.get_prec ()
    hx_vec = prec_op (-g_vec)
    gorb1 = ugg.unpack (g_vec)[0]
    gorb2 = gorb1 + ugg.unpack (hx_vec)[0]

    def _svd (metric, q, p):
        m, n = q.shape[1], p.shape[1]
        k, l = min (m, n), max (m, n)
        svals = np.zeros (l)
        qh = q.conj ().T
        u, svals[:k], vh = linalg.svd (qh @ metric @ p)
        idx_sort = np.argsort (-np.abs (svals))
        svals = svals[idx_sort]
        u[:,:k] = u[:,idx_sort[:k]]
        return q @ u, svals

    def _check (tag, umat_p, umat_q):
        np, nq = umat_p.shape[1], umat_q.shape[1]
        k = min (np, nq)
        lib.logger.debug (las, '%s size of pspace = %d, qspace = %d', tag, np, nq)
        return k

    def _grad_svd (tag, geff, umat_p, umat_q, ncoup=0):
        umat_q, svals = _svd (geff, umat_q, umat_p)
        umat_p = np.append (umat_p, umat_q[:,:ncoup], axis=1)
        umat_q = umat_q[:,ncoup:]
        k = _check ('after {} SVD'.format (tag), umat_p, umat_q)
        lib.logger.debug (las, '%s SVD lowest eigenvalue = %e', tag, svals[ncoup-1])
        return umat_p, umat_q, k

    def _schmidt (tag, umat_p, umat_q, thresh=1e-8):
        umat_q, svals = _svd (dm1, umat_q, umat_p)
        ncoup = np.count_nonzero (np.abs (svals) > thresh)
        umat_p = np.append (umat_p, umat_q[:,:ncoup], axis=1)
        umat_q = umat_q[:,ncoup:]
        dm_pp = umat_p.conj ().T @ dm1 @ umat_p
        lib.logger.debug (las, 'number of electrons in p-space after %s Schmidt = %e', tag, np.trace (dm_pp))
        k = _check ('after {} Schmidt'.format (tag), umat_p, umat_q)
        return umat_p, umat_q, k

    def _make_single_space (p_mask, q_mask):
        # The pmin stuff below probably isn't as useful as I'd like. Because of
        # the way svd works there is no sorting among the null space that
        # makes any sense. It's really only useful for testing the limit
        # against implementations without SVD pspace selection.
        # I also badly need to rename this because 'Schmidt decompositon'
        # stopped being a useful way to think about this a long time ago.
        nlas = np.count_nonzero (p_mask)
        umat_p = np.diag (p_mask.astype (mo_coeff.dtype))[:,p_mask]
        umat_q = np.diag (q_mask.astype (mo_coeff.dtype))[:,q_mask]
        if h_op._debug_full_pspace: return np.append (umat_p, umat_q, axis=1)
        # At any of these steps we might run out of orbitals...
        # The Schmidt steps might turn out to be completely unnecessary
        k = _check ('initial', umat_p, umat_q)
        if k == 0: return umat_p
        umat_p, umat_q, k = _grad_svd ('g', gorb1, umat_p, umat_q, ncoup=k)
        #if k == 0: return umat_p
        #umat_p, umat_q, k = _schmidt ('first', umat_p, umat_q) 
        if k == 0: return umat_p
        pcurr = umat_p.shape[1] - nlas
        ncoup = min (k, 2*nlas)
        umat_p, umat_q, k = _grad_svd ('g+hx', gorb2, umat_p, umat_q, ncoup=ncoup)
        #if k == 0: return umat_p
        #umat_p, umat_q, k = _schmidt ('second', umat_p, umat_q)
        return umat_p

    orbsym = getattr (mo_coeff, 'orbsym', np.zeros (nmo))
    uschmidt = []
    for ilas in range (len (las.ncas_sub)):
        i = sum (las.ncas_sub[:ilas]) + ncore
        j = i + las.ncas_sub[ilas]
        irreps, idx_irrep = np.unique (orbsym[i:j], return_inverse=True)
        ulist = []
        for ix in range (len (irreps)):
            idx = np.squeeze (np.where (idx_irrep==ix)) + i
            p_mask = np.zeros (nmo, dtype=np.bool_)
            p_mask[idx] = True
            q_mask = ~p_mask
            # I am assuming that we still have paaa eris elsewhere
            # So I don't want to do the ua_aa this way
            q_mask[ncore:nocc] = False 
            ulist.append (_make_single_space (p_mask, q_mask))
        uschmidt.append (np.concatenate (ulist, axis=1))

    return uschmidt

class LASSCF_HessianOperator (lasscf_sync_o0.LASSCF_HessianOperator):

    def __init__(self, las, ugg, **kwargs):
        self._debug_full_pspace = kwargs.pop ('_debug_full_pspace', getattr (las, '_debug_full_pspace', False))
        self._debug_o0 = kwargs.pop ('_debug_o0', getattr (las, '_debug_o0', False))
        lasscf_sync_o0.LASSCF_HessianOperator.__init__(self, las, ugg, **kwargs)
        self.h1_bare = self.mo_coeff.conj ().T @ self.las.get_hcore () @ self.mo_coeff

    def make_schmidt_spaces (self):
        if self._debug_o0:
            return [np.eye (self.nmo)]
        return make_schmidt_spaces (self)

    def _init_eri_(self):
        lasci_sync._init_df_(self)
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        self.uschmidt = uschmidt = self.make_schmidt_spaces ()
        t1 = lib.logger.timer (self.las, 'build schmidt spaces', *t0)
        if isinstance (self.las, _DFLASCI):
            eri = self.las.with_df.ao2mo
        elif getattr (self.las._scf, '_eri', None) is not None:
            eri = partial (ao2mo.full, self.las._scf._eri)
        else:
            eri = partial (ao2mo.full, self.las.mol)
        self.eri_imp = []
        for ix, umat in enumerate (uschmidt):
            nimp = umat.shape[1]
            mo = self.mo_coeff @ umat
            self.eri_imp.append (ao2mo.restore (1, eri (mo), nimp))
            t1 = lib.logger.timer (self.las, 'schmidt-space {} eri array'.format (ix), *t1)
        # eri_cas is taken from h2eff_sub
        lib.logger.timer (self.las, '_init_eri', *t0)

    def get_veff_Heff (self, odm1rs, tdm1frs):
        raise NotImplementedError ("Matt, you need to update this for odm1rs->odm1s!")
        # Separating out ua degrees of freedom and treating them differently for reasons
        ncore, nocc = self.ncore, self.nocc
        odm1rs_frz = odm1rs.copy ()
        odm1rs_frz[:,:,ncore:nocc,:ncore] = 0.0
        odm1rs_frz[:,:,ncore:nocc,nocc:]  = 0.0
        veff_mo, h1frs = lasscf_sync_o0.LASSCF_HessianOperator.get_veff_Heff (self, odm1rs_frz, tdm1frs)

        # Now I need to specifically add back the (H.x_ua)_aa degrees of freedom to both of these
        odm1rs_ua = odm1rs[:,:,ncore:nocc,:].copy ()
        odm1rs_ua[:,:,:,ncore:nocc] = 0.0
        odm1s_ua = np.einsum ('r,rsap->sap', self.weights, odm1rs_ua)
        veff_aa = np.tensordot (odm1s_ua, self.eri_paaa, axes=((2,1),(0,1)))
        veff_aa += veff_aa[::-1] # vj(a) + vj(b)
        veff_aa -= np.tensordot (odm1s_ua, self.eri_paaa, axes=((2,1),(0,3)))
        veff_aa += veff_aa.transpose (0,2,1)
        veff_mo[:,ncore:nocc,ncore:nocc] += veff_aa
        for isub, h1rs in enumerate (h1frs):
            i = sum (self.ncas_sub[:isub])
            j = i + self.ncas_sub[isub]
            dm1rs = odm1rs_ua.copy ()
            dm1rs[:,:,i:j,:] = 0.0
            v  = np.tensordot (dm1rs, self.eri_paaa, axes=((3,2),(0,1)))
            v += v[:,::-1,:,:]
            v -= np.tensordot (dm1rs, self.eri_paaa, axes=((3,2),(0,3)))
            v += v.transpose (0,1,3,2)
            h1rs[:,:,:,:] += v

        return veff_mo, h1frs

    def get_veff (self, dm1s_mo=None):
        # I can't do better than O(N^4), but maybe I can do better than O(M^4)
        # Neither dm1_vv nor veff_vv elements are needed here
        # Nothing in this function should distinguish between core and active orbitals!
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        if not isinstance (self.las, _DFLASCI):
            return lasscf_sync_o0.LASSCF_HessianOperator.get_veff (self, dm1s_mo=dm1s_mo)
        nocc, mo, bPpj = self.nocc, self.mo_coeff, self.bPpj
        moH = mo.conj ().T
        dm1_mo = dm1s_mo.sum (0)
        # vj
        # [(P|qj) + (P|jq)] * D_qj = 2 * (P|qj) * D_qj
        # D_jj elements within D_qj multiplied by 1/2 to cancel double-counting
        # Avoid the other double-counting explicitly: (pi|**) -> (ia|**)
        dm1_rect = dm1_mo[:,:nocc].copy ()
        dm1_rect[:nocc,:nocc] *= 0.5 # subtract (P|jj) D_jj
        rho = np.tensordot (bPpj, dm1_rect, axes=2) * 2 # add transpose
        vj_pj = np.tensordot (rho, bPpj, axes=1)
        vj_pp = np.zeros_like (dm1_mo)
        vj_pp[:,:nocc] = vj_pj
        vj_pp[:nocc,nocc:] = vj_pj[nocc:,:nocc].T
        # vk
        # (pq|ji), (iq|ja), (pj|qi), (ij|qa) * D_qj 
        # D_jj elements within D_qj multiplied by 1/2 to cancel double-counting
        # Avoid the other double-counting explicitly: (p*|*i) -> (i*|*a)
        vPpj = np.ascontiguousarray (self.las.cderi_ao2mo (mo, mo @ dm1_rect, compact=False))
        vPij, bPij, bPaj = vPpj[:,:nocc,:], bPpj[:,:nocc,:], bPpj[:,nocc:,:]
        vk_pp = np.zeros_like (dm1_mo)
        vk_pp[:,:nocc]     = np.tensordot (vPpj, bPij, axes=((0,2),(0,2))) # (pq|ji) x D_qj
        vk_pp[:nocc,nocc:] = np.tensordot (vPij, bPaj, axes=((0,2),(0,2))) # (iq|ja) x D_qj
        vk_pp += vk_pp.T # Index transpose -> (pj|qi), (ij|qa) x D_bj
        t1 = lib.logger.timer (self.las, 'h_op.get_veff', *t0)
        return vj_pp - 0.5 * vk_pp 

    def split_veff (self, veff_mo, dm1s_mo):
        veff_c = veff_mo.copy ()
        ncore = self.ncore
        nocc = self.nocc
        sdm = dm1s_mo[0] - dm1s_mo[1]
        veff_s = np.zeros_like (veff_c)
        sdm_cas = sdm[ncore:nocc,ncore:nocc]
        veff_s[:,ncore:nocc] = np.tensordot (self.eri_paaa, sdm_cas, axes=((1,2),(0,1)))
        veff_s[ncore:nocc,:] = veff_s[:,ncore:nocc].T
        veff_s[:,:] *= -0.5
        veffa = veff_c + veff_s
        veffb = veff_c - veff_s
        return np.stack ([veffa, veffb], axis=0)

    def orbital_response (self, kappa, odm1rs, ocm2, tdm1frs, tcm2, veff_prime):
        ''' Parent class does everything except va/ac degrees of freedom
        (c: closed; a: active; v: virtual; p: any) '''
        raise NotImplementedError ("Matt, you need to update this for odm1rs->odm1s!")
        ncore, nocc, nmo = self.ncore, self.nocc, self.nmo
        # Separate out active-unactive degrees of freedom
        odm1rs_frz = odm1rs.copy ()
        odm1rs_frz[:,:,ncore:nocc,:ncore] = 0.0
        odm1rs_frz[:,:,ncore:nocc,nocc:]  = 0.0
        odm1rs_ua = odm1rs - odm1rs_frz
        gorb = lasci_sync.LASCI_HessianOperator.orbital_response (self, kappa, odm1rs_frz,
            ocm2, tdm1frs, tcm2, veff_prime)
        # Add back terms omitted for (H.x_ua)_aa
        odm1s_ua = np.einsum ('r,rspq->spq', self.weights, odm1rs_ua)
        odm1s_ua += odm1s_ua.transpose (0,2,1)
        f1_prime = self.h1_bare @ odm1s_ua.sum (0)
        f1_prime[ncore:nocc,ncore:nocc] = sum ([h @ d for h, d in zip (self.h1s, odm1s_ua)])[ncore:nocc,ncore:nocc]
        ocm2_ua = ocm2.copy ()
        ocm2_ua[:,:,:,ncore:nocc] = 0.0
        # (H.x_ua)_ua, (H.x_ua)_vc
        # This doesn't currently work correctly for state-average, because the cumulant
        # decomposition of the state-average density matrices doesn't work the same way.
        # Straightforward fix is to do the cumulant decomposition root-by-root, but that 
        # will require having f1 be done root-by-root. What to do?
        for uimp, eri in zip (self.uschmidt, self.eri_imp):
            uimp_cas = uimp[ncore:nocc,:]
            edm1s = np.dot (uimp.conj ().T, np.dot (odm1s_ua, uimp)).transpose (1,0,2)
            edm1 = edm1s.sum (0)
            dm1s = np.dot (uimp.conj ().T, np.dot (self.dm1s, uimp)).transpose (1,0,2)
            dm1 = dm1s.sum (0)
            cm2 = np.tensordot (ocm2_ua, uimp_cas, axes=((2),(0))) # pqrs -> pqsr
            cm2 = np.tensordot (cm2, uimp, axes=((2),(0))) # pqsr -> pqrs
            cm2 = np.tensordot (uimp_cas.conj (), cm2, axes=((0),(1))) # pqrs -> qprs
            cm2 = np.tensordot (uimp_cas.conj (), cm2, axes=((0),(1))) # qprs -> pqrs
            cm2 += cm2.transpose (1,0,3,2)
            dm2 = cm2 + np.multiply.outer (edm1, dm1)
            dm2 -= sum ([np.multiply.outer (e, d).transpose (0,3,2,1) for e, d in zip (edm1s, dm1s)])
            dm2 += dm2.transpose (2,3,0,1)
            f1 = np.tensordot (eri, dm2, axes=((1,2,3),(1,2,3)))
            f1 = uimp @ f1 @ uimp.conj ().T
            f1[ncore:nocc,ncore:nocc] = 0.0
            f1_prime += f1
        # (H.x_aa)_ua
        ecm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)
        ecm2 += ecm2.transpose (2,3,0,1) + tcm2
        f1_prime[:ncore,ncore:nocc] += np.tensordot (self.eri_paaa[:ncore], ecm2, axes=((1,2,3),(1,2,3)))
        f1_prime[nocc:,ncore:nocc] += np.tensordot (self.eri_paaa[nocc:], ecm2, axes=((1,2,3),(1,2,3)))
        # (H.x_ua)_aa
        ecm2 = ocm2.copy ()
        f1_aa = f1_prime[ncore:nocc,ncore:nocc]
        f1_aa[:,:] += (np.tensordot (self.eri_paaa[:ncore], ocm2[:,:,:,:ncore], axes=((0,2,3),(3,0,1)))
                     + np.tensordot (self.eri_paaa[nocc:],  ocm2[:,:,:,nocc:],  axes=((0,2,3),(3,0,1))))
        f1_aa[:,:] += (np.tensordot (self.eri_paaa[:ncore], ocm2[:,:,:,:ncore], axes=((0,1,3),(3,2,1)))
                     + np.tensordot (self.eri_paaa[nocc:],  ocm2[:,:,:,nocc:],  axes=((0,1,3),(3,2,1))))
        f1_aa[:,:] += (np.tensordot (self.eri_paaa[:ncore], ocm2[:,:,:,:ncore], axes=((0,1,2),(3,2,0)))
                     + np.tensordot (self.eri_paaa[nocc:],  ocm2[:,:,:,nocc:],  axes=((0,1,2),(3,2,0))))
        return gorb + (f1_prime - f1_prime.T)

class LASSCFNoSymm (lasscf_sync_o0.LASSCFNoSymm):
    _hop = LASSCF_HessianOperator
    def __init__(self, *args, **kwargs):
        self._debug_full_pspace = kwargs.pop ('_debug_full_pspace', False)
        self._debug_o0 = kwargs.pop ('_debug_o0', False)
        lasscf_sync_o0.LASSCFNoSymm.__init__(self, *args, **kwargs)

class LASSCFSymm (lasscf_sync_o0.LASSCFSymm):
    _hop = LASSCF_HessianOperator
    def __init__(self, *args, **kwargs):
        self._debug_full_pspace = kwargs.pop ('_debug_full_pspace', False)
        self._debug_o0 = kwargs.pop ('_debug_o0', False)
        lasscf_sync_o0.LASSCFSymm.__init__(self, *args, **kwargs)

def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = lasci.density_fit (las, with_df = mf.with_df) 
    return las

        

