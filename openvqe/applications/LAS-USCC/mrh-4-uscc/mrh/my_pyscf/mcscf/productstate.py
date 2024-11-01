import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring
from pyscf.mcscf.addons import state_average as state_average_mcscf
from mrh.my_pyscf.fci.csf import CSFFCISolver
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from mrh.my_pyscf.mcscf.addons import StateAverageNMixFCISolver
from itertools import combinations

# TODO: linkstr support
class ProductStateFCISolver (StateAverageNMixFCISolver, lib.StreamObject):

    def __init__(self, fcisolvers, stdout=None, verbose=0, **kwargs):
        self.fcisolvers = fcisolvers
        self.verbose = verbose
        self.stdout = stdout
        self.log = lib.logger.new_logger (self, verbose)

    def kernel (self, h1, h2, norb_f, nelec_f, ecore=0, ci0=None, orbsym=None,
            conv_tol_grad=1e-4, conv_tol_self=1e-10, max_cycle_macro=50,
            **kwargs):
        log = self.log
        converged = False
        e_sigma = conv_tol_self + 1
        ci1 = self._check_init_guess (ci0, norb_f, nelec_f) # TODO: get_init_guess
        log.info ('Entering product-state fixed-point CI iteration')
        for it in range (max_cycle_macro):
            h1eff, h0eff = self.project_hfrag (h1, h2, ci1, norb_f, nelec_f,
                ecore=ecore, **kwargs)
            grad = self._get_grad (h1eff, h2, ci1, norb_f, nelec_f, **kwargs)
            grad_max = np.amax (np.abs (grad))
            log.info ('Cycle %d: max grad = %e ; sigma = %e', it, grad_max,
                e_sigma)
            if ((grad_max < conv_tol_grad) and (e_sigma < conv_tol_self)):
                converged = True
                break
            e, ci1 = self._1shot (h0eff, h1eff, h2, ci1, norb_f, nelec_f,
                orbsym=orbsym, **kwargs)
            e_sigma = np.amax (e) - np.amin (e)
        conv_str = ['NOT converged','converged'][int (converged)]
        log.info (('Product_state fixed-point CI iteration {} after {} '
                   'cycles').format (conv_str, it))
        if not converged: self._debug_csfs (log, ci1, norb_f, nelec_f, grad)
        energy_elec = self.energy_elec (h1, h2, ci1, norb_f, nelec_f,
            ecore=ecore, **kwargs)
        return converged, energy_elec, ci1

    def _check_init_guess (self, ci0, norb_f, nelec_f):
        ci1 = []
        for ix, (no, ne, solver) in enumerate (zip (norb_f, nelec_f, self.fcisolvers)):
            neleca, nelecb = self._get_nelec (solver, ne)
            na = cistring.num_strings (no, neleca)
            nb = cistring.num_strings (no, nelecb)
            zguess = np.zeros ((solver.nroots,na,nb))
            cguess = np.asarray (ci0[ix]).reshape (-1,na,nb)
            ngroots = min (zguess.shape[0], cguess.shape[0])
            zguess[:ngroots,:,:] = cguess[:ngroots,:,:]
            ci1.append (zguess)
            if solver.nroots>na*nb:
                raise RuntimeError ("{} roots > {} determinants in fragment {}".format (
                    solver.nroots, na*nb, ix))
            if isinstance (solver, CSFFCISolver):
                solver.check_transformer_cache ()
                if solver.nroots>solver.transformer.ncsf:
                    raise RuntimeError ("{} roots > {} CSFs in fragment {}".format (
                        solver.nroots, solver.transformer.ncsf, ix))
        return ci1
                
    def _debug_csfs (self, log, ci1, norb_f, nelec_f, grad):
        if not all ([isinstance (s, CSFFCISolver) for s in self.fcisolvers]):
            return
        if log.verbose < lib.logger.INFO: return
        transformers = [s.transformer for s in self.fcisolvers]
        grad_f = []
        for s,t in zip (self.fcisolvers, transformers):
            grad_f.append (grad[:t.ncsf*s.nroots].reshape (s.nroots, t.ncsf))
            offs = (t.ncsf*s.nroots) + (s.nroots*(s.nroots-1)//2)
            grad = grad[offs:]
        assert (len (grad) == 0)
        log.info ('Debugging CI and gradient vectors...')
        for ix, (grad, ci, s, t) in enumerate (zip (grad_f, ci1, self.fcisolvers, transformers)):
            log.info ('Fragment %d', ix)
            ci_csf, ci_norm = t.vec_det2csf (ci, normalize=True, return_norm=True)
            log.info ('CI vector norm = %s', str(ci_norm))
            grad_norm = linalg.norm (grad)
            log.info ('Gradient norm = %e', grad_norm)
            log.info ('CI vector leading components:')
            lbls, coeffs = t.printable_largest_csf (ci_csf, 10)
            for l, c in zip (lbls[0], coeffs[0]):
                log.info ('%s : %e', l, c)
            log.info ('Grad vector leading components:')
            lbls, coeffs = t.printable_largest_csf (grad, 10, normalize=False)
            for l, c in zip (lbls[0], coeffs[0]):
                log.info ('%s : %e', l, c)

    def _1shot (self, h0eff, h1eff, h2, ci, norb_f, nelec_f, orbsym=None,
            **kwargs):
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h0eff, h1eff, ci, norb_f, nelec_f, self.fcisolvers, ni, nj]
        e1 = []
        ci1 = []
        for h0e, h1e, c, no, ne, solver, i, j in zip (*zipper):
            h2e = h2[i:j,i:j,i:j,i:j]
            osym = getattr (solver, 'orbsym', None)
            if orbsym is not None: osym=orbsym[i:j]
            nelec = self._get_nelec (solver, ne)
            e, c1 = solver.kernel (h1e, h2e, no, nelec, ci0=c, ecore=h0e,
                orbsym=osym, **kwargs)
            e1.append (e)
            ci1.append (c1)
        return e1, ci1

    def _get_grad (self, h1eff, h2, ci, norb_f, nelec_f, orbsym=None,
            **kwargs):
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h1eff, ci, norb_f, nelec_f, self.fcisolvers, ni, nj]
        grad = []
        for h1e, c, no, ne, solver, i, j in zip (*zipper):
            nelec = self._get_nelec (solver, ne)
            nroots = solver.nroots
            h2e = h2[i:j,i:j,i:j,i:j]
            h2e = solver.absorb_h1e (h1e, h2e, no, nelec, 0.5)
            if nroots==1: c=c[None,:]
            hc = [solver.contract_2e (h2e, col, no, nelec) for col in c]
            c, hc = np.asarray (c), np.asarray (hc)
            chc = np.dot (np.asarray (c).reshape (nroots,-1).conj (),
                          np.asarray (hc).reshape (nroots,-1).T)
            hc = hc - np.tensordot (chc, c, axes=1)
            if isinstance (solver, CSFFCISolver):
                hc = solver.transformer.vec_det2csf (hc, normalize=False)
            # External degrees of freedom: not weighted, because I want
            # to converge all of the roots even if they don't contribute
            # to the mean field
            assert (hc.size == nroots*solver.transformer.ncsf)
            grad.append (hc.ravel ())
            # Internal degrees of freedom: weighted and lower-triangular
            # TODO: confirm the sign choice below before using this gradient
            # for something more advanced than convergence checking
            if nroots>1 and getattr (solver, 'weights', None) is not None:
                chc *= np.asarray (solver.weights)[:,None]
                chc -= chc.T
                grad.append (chc[np.tril_indices (nroots,k=-1)])
        return np.concatenate (grad)

    def energy_elec (self, h1, h2, ci, norb_f, nelec_f, ecore=0, **kwargs):
        dm1 = np.stack (self.make_rdm1 (ci, norb_f, nelec_f), axis=0)
        dm2 = self.make_rdm2 (ci, norb_f, nelec_f)
        energy_tot = (ecore + np.tensordot (h1, dm1, axes=2)
                        + 0.5*np.tensordot (h2, dm2, axes=4))
        return energy_tot

    def project_hfrag (self, h1, h2, ci, norb_f, nelec_f, ecore=0, **kwargs):
        dm1s = np.stack (self.make_rdm1s (ci, norb_f, nelec_f), axis=0)
        dm1 = dm1s.sum (0)
        dm2 = self.make_rdm2 (ci, norb_f, nelec_f)
        energy_tot = (ecore + np.tensordot (h1, dm1, axes=2)
                        + 0.5*np.tensordot (h2, dm2, axes=4))
        v1  = np.tensordot (dm1s, h2, axes=2)
        v1 += v1[::-1] # ja + jb
        v1 -= np.tensordot (dm1s, h2, axes=((1,2),(2,1)))
        f1 = h1[None,:,:] + v1
        h1eff = []
        h0eff = []
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        for i, j in zip (ni, nj):
            dm1s_i = dm1s[:,i:j,i:j]
            dm2_i = dm2[i:j,i:j,i:j,i:j]
            # v1 self-interaction
            h2_i = h2[i:j,i:j,:,:]
            v1_i = np.tensordot (dm1s_i, h2_i, axes=2)
            v1_i += v1_i[::-1] # ja + jb
            h2_i = h2[:,i:j,i:j,:]
            v1_i -= np.tensordot (dm1s_i, h2_i, axes=((1,2),(2,1)))
            # cancel off-diagonal energy double-counting
            e_i = energy_tot - np.tensordot (dm1s, v1_i, axes=3) # overcorrects
            # cancel h1eff double-counting
            v1_i = v1_i[:,i:j,i:j] 
            h1eff.append (f1[:,i:j,i:j]-v1_i)
            # cancel diagonal energy double-counting
            h1_i = h1[None,i:j,i:j] - v1_i # v1_i fixes overcorrect
            h2_i = h2[i:j,i:j,i:j,i:j]
            e_i -= (np.tensordot (h1_i, dm1s_i, axes=3)
              + 0.5*np.tensordot (h2_i, dm2_i, axes=4))
            h0eff.append (e_i)
        return h1eff, h0eff

    def make_rdm1s (self, ci, norb_f, nelec_f, **kwargs):
        norb = sum (norb_f)
        dm1a = np.zeros ((norb, norb))
        dm1b = np.zeros ((norb, norb))
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        for ix, (i, j, c, no, ne, s) in enumerate (zip (ni, nj, ci, norb_f, nelec_f, self.fcisolvers)):
            nelec = self._get_nelec (s, ne)
            try:
                a, b = s.make_rdm1s (c, no, nelec)
            except AssertionError as e:
                print (type (c), np.asarray (c).shape)
                raise (e)
            except ValueError as e:
                print ("frag=",ix,"nroots=",s.nroots,"no=",no,"ne=",nelec,'c.shape=',np.asarray(c).shape)
                if isinstance (s, CSFFCISolver):
                    print ("smult=",s.smult,"ncsf=",s.transformer.ncsf)
                raise (e)
            dm1a[i:j,i:j] = a[:,:]
            dm1b[i:j,i:j] = b[:,:]
        return dm1a, dm1b

    def make_rdm1 (self, ci, norb_f, nelec_f, **kwargs):
        dm1a, dm1b = self.make_rdm1s (ci, norb_f, nelec_f, **kwargs)
        return dm1a + dm1b

    def make_rdm2 (self, ci, norb_f, nelec_f, **kwargs):
        norb = sum (norb_f)
        dm2 = np.zeros ([norb,]*4)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        dm1a, dm1b = self.make_rdm1s (ci, norb_f, nelec_f, **kwargs)
        for i, j, c, no, ne, s in zip (ni, nj, ci, norb_f, nelec_f, self.fcisolvers):
            nelec = self._get_nelec (s, ne)
            dm2[i:j,i:j,i:j,i:j] = s.make_rdm2 (c, no, nelec)
        dm1 = dm1a + dm1b
        for (i,j), (k,l) in combinations (zip (ni, nj), 2):
            d1_ij, d1a_ij, d1b_ij = dm1[i:j,i:j], dm1a[i:j,i:j], dm1b[i:j,i:j]
            d1_kl, d1a_kl, d1b_kl = dm1[k:l,k:l], dm1a[k:l,k:l], dm1b[k:l,k:l]
            d2 = np.multiply.outer (d1_ij, d1_kl)
            dm2[i:j,i:j,k:l,k:l] = d2
            dm2[k:l,k:l,i:j,i:j] = d2.transpose (2,3,0,1)
            d2  = np.multiply.outer (d1a_ij, d1a_kl)
            d2 += np.multiply.outer (d1b_ij, d1b_kl)
            dm2[i:j,k:l,k:l,i:j] = -d2.transpose (0,2,3,1)
            dm2[k:l,i:j,i:j,k:l] = -d2.transpose (2,0,1,3)
        return dm2

def state_average_fcisolver (solver, weights=(.5,.5), wfnsym=None):
    # hoo boy this is real dumb
    dummy = type ("dummy",(object,),{"_keys":set()}) ()
    dummy.fcisolver = solver
    return state_average_mcscf (dummy, weights=weights, wfnsym=wfnsym).fcisolver

class ImpureProductStateFCISolver (ProductStateFCISolver):
    def __init__(self, fcisolvers, stdout=None, verbose=0, lweights=None, **kwargs):
        ProductStateFCISolver.__init__(self, fcisolvers, stdout=stdout, verbose=verbose, **kwargs)
        if lweights is None: lweights = [[.5,.5],]*len(fcisolvers)
        for ix, (fcisolver, weights) in enumerate (zip (self.fcisolvers, lweights)):
            if len (weights) > 1:
                self.fcisolvers[ix] = state_average_fcisolver (fcisolver, weights=weights)



