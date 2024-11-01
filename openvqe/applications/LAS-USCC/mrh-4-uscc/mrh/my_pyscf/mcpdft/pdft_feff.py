import time, math
import numpy as np
from scipy import linalg
from itertools import combinations_with_replacement, product
from pyscf import lib, ao2mo
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.dft.numint import _contract_rho
from pyscf.mcscf import mc1step
from pyscf.scf import hf
from pyscf.mcpdft.otpd import *
from pyscf.mcpdft.otpd import _grid_ao2mo
from pyscf.mcpdft.tfnal_derivs import contract_fot, _unpack_sigma_vector
from pyscf.mcpdft.pdft_veff import _contract_vot_ao, _contract_vot_rho
from pyscf.mcpdft.pdft_veff import _dot_ao_mo

def _contract_rho_all (bra, ket):
    # Apply the product rule when computing density & derivs on a grid
    if bra.ndim == 2: bra = bra[None,:,:]
    if ket.ndim == 2: ket = ket[None,:,:]
    nderiv, ngrids, norb = bra.shape
    rho = np.empty ((nderiv, ngrids), dtype=bra.dtype)
    if norb==0:
        rho[:] = 0.0
        return rho
    rho[0] = _contract_rho (bra[0], ket[0])
    for ideriv in range (1,min(nderiv,4)):
        rho[ideriv]  = _contract_rho (bra[ideriv], ket[0])
        rho[ideriv] += _contract_rho (bra[0], ket[ideriv])
    if nderiv > 4: raise NotImplementedError (('ftGGA or Colle-Salvetti type '
        'functionals'))
    return rho

from mrh.util import la
def vector_error (test, ref): return la.vector_error (test, ref, 'rel')

# PySCF's overall sign convention is
#   de = h.D - D.h
#   dD = x.D - D.x
# which is consistent with
#   |trial> = exp(k_pq E_pq)|0>
# where k_pq is an antihermitian matrix
# However, in various individual terms the two conventions in 
# h_op are simultaneously reversed:
#   de = D.h - h.D
#   dD = D.x - x.D
# Specifically, update_jk_in_ah has this return signature. So I have
# to return -hx.T specifically.
# I'm also really confused by factors of 2. It looks like PySCF does
# the full commutator
#   dD = x.D - D.x
# in update_jk_in_ah, but only does 
#   dD = x.D
# everywhere else? Why would the update_jk_in_ah terms be
# multiplied by 2?

# TODO: better docstring, private/public member distinction?
# Reposition class-member kwargs as calling kwargs where appropriate and
# possible.
class EotOrbitalHessianOperator (object):
    '''Callable object for computing

    (f.x)_pq = (int fot * drho/dk_pq drho/dk_rs x_rs dr)

    where fot is the second functional derivative of an on-top
    functional, and "rho" is any of the density, the on-top pair
    density or their derivatives, in the context of an MC-PDFT
    calculation the ncore, ncas, mo_coeff.

    Does not compute any contribution of the type
    (int vot * d^2 rho / dk_pq dk_rs x_rs dr)
    except optionally for that related to the Pi = (1/4) rho^2
    cumulant approximation. All other vot terms are more efficiently
    computed using cached effective Hamiltonian tensors that map
    straightforwardly to those involved in CASSCF orbital
    optimization.
    '''
    
    def __init__(self, mc, ot=None, mo_coeff=None, ncore=None, ncas=None,
            casdm1=None, casdm2=None, max_memory=None, do_cumulant=True,
            incl_d2rho=False):
        if ot is None: ot = mc.otfnal
        if mo_coeff is None: mo_coeff = mc.mo_coeff
        if ncore is None: ncore = mc.ncore
        if ncas is None: ncas = mc.ncas
        if max_memory is None: max_memory = mc.max_memory
        if (casdm1 is None) or (casdm2 is None):
            dm1, dm2 = mc.fcisolver.make_rdm12 (mc.ci, ncas, mc.nelecas)
            if casdm1 is None: casdm1 = dm1
            if casdm2 is None: casdm2 = dm2

        self.ot = ot
        omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc)
        hyb_x, hyb_c = hyb
        if abs (omega) > 1e-11:
            raise NotImplementedError ("range-separated on-top functionals")
        if abs (hyb_x) > 1e-11 or abs (hyb_c) > 1e-11:
            raise NotImplementedError ("2nd fnal derivatives for hybrid functionals")

        self.verbose, self.stdout = ot.verbose, ot.stdout
        self.log = lib.logger.new_logger (self, self.verbose)
        self.ni, self.xctype = ni, xctype = ot._numint, ot.xctype
        self.rho_deriv, self.Pi_deriv = ot.dens_deriv, ot.Pi_deriv
        deriv = ot.dens_deriv
        self.nderiv_rho = (1,4,10)[int (ot.dens_deriv)]
        self.nderiv_Pi = (1,4)[int (ot.Pi_deriv)]
        self.nderiv_ao = (deriv+1)*(deriv+2)*(deriv+3)//6
        self.mo_coeff = mo_coeff
        self.nao, self.nmo = nao, nmo = mo_coeff.shape
        self.ncore = ncore
        self.ncas = ncas
        self.nocc = nocc = ncore + ncas
        self.casdm2 = casdm2
        self.casdm1s = casdm1s = np.stack ([casdm1, casdm1], axis=0)/2
        self.cascm2 = cascm2 = dm2_cumulant (casdm2, casdm1)
        self.max_memory = max_memory        
        self.do_cumulant = do_cumulant
        self.incl_d2rho = incl_d2rho

        dm1 = 2 * np.eye (nocc, dtype=casdm1.dtype)
        dm1[ncore:,ncore:] = casdm1
        occ_coeff = mo_coeff[:,:nocc]
        no_occ, uno = linalg.eigh (dm1)
        no_coeff = occ_coeff @ uno
        dm1 = occ_coeff @ dm1 @ occ_coeff.conj ().T
        self.dm1 = dm1 = lib.tag_array (dm1, mo_coeff=no_coeff, mo_occ=no_occ)

        self.make_rho = ni._gen_rho_evaluator (ot.mol, dm1, 1)[0]
        self.pack_uniq_var = mc.pack_uniq_var
        self.unpack_uniq_var = mc.unpack_uniq_var        

        self.shls_slice = (0, ot.mol.nbas)
        self.ao_loc = ot.mol.ao_loc_nr()

        if incl_d2rho: # Include d^2rho/dk^2 type derivatives
            # Also include a full E_OT value and gradient recalculator
            # for debugging purposes
            veff1, veff2 = self.veff1, self.veff2 = mc.get_pdft_veff (
                mo=mo_coeff, casdm1s=casdm1s, casdm2=casdm2, incl_coul=False)
            get_hcore = lambda * args: self.veff1
            with lib.temporary_env (mc, get_hcore=get_hcore):
                g_orb, _, h_op, h_diag = mc1step.gen_g_hop (mc, mo_coeff,
                    np.eye (nmo), casdm1, casdm2, self.veff2)
            # dressed gen_g_hop objects
            from mrh.my_pyscf.mcpdft.orb_scf import get_gorb_update
            gorb_update_u = get_gorb_update (mc, mo_coeff, ncore=ncore,
                ncas=ncas, eot_only=True)
            # Must correct for different orbital bases in seminumerical
            # calculation. Have to completely redo the gradient matrix
            # because "redundant" d.o.f. aren't
            v1 = mo_coeff.conj ().T @ veff1 @ mo_coeff + veff2.vhf_c
            v2 = np.zeros ((nmo,ncas,ncas,ncas), dtype=v1.dtype)
            for i in range (nmo):
                jbuf = veff2.ppaa[i]
                v2[i,:,:,:] = jbuf[ncore:nocc,:,:]
                v1[i,:] += np.tensordot (jbuf, casdm1, axes=2) * 0.5
            f1 = np.zeros_like (v1)
            f1[:,:ncore] = 2 * v1[:,:ncore]
            f1[:,ncore:nocc] += v1[:,ncore:nocc] @ casdm1
            f1[:,ncore:nocc] += np.tensordot (v2, cascm2,
                axes=((1,2,3),(1,2,3)))
            self._f1_test = f1
            dE = f1 - f1.T # gradient, but the full matrix
            def delta_gorb (x):
                u = mc.update_rotate_matrix (x)
                g1 = gorb_update_u (u, mc.ci)
                x = self.unpack_uniq_var (x)
                g1 += self.pack_uniq_var (x @ dE - dE @ x)/2
                return g1 - g_orb
            jk_null = np.zeros ((ncas,nmo)), np.zeros ((ncore,nmo-ncore))
            update_jk = lambda * args: jk_null
            def d2rho_h_op (x):
                with lib.temporary_env (mc, update_jk_in_ah=update_jk):
                    return h_op (x)
            self.g_orb = g_orb
            self.delta_gorb = delta_gorb
            self.d2rho_h_op = d2rho_h_op
            self.h_diag = h_diag
            # on-top energy and calculator
            mo_cas = mo_coeff[:,ncore:nocc]
            dm1s = np.dot (mo_cas, casdm1s).transpose (1,0,2)
            dm1s = np.dot (dm1s, mo_cas.conj ().T)
            mo_core = mo_coeff[:,:ncore]
            dm1s += (mo_core @ mo_core.conj ().T)[None,:,:]
            e_ot = mc.energy_dft (ot=ot, mo_coeff=mo_coeff, casdm1s=casdm1s,
                casdm2=casdm2)
            def delta_eot (x):
                u = mc.update_rotate_matrix (x)
                mo1 = mo_coeff @ u
                mo_cas = mo1[:,ncore:nocc]
                dm1s = np.dot (mo_cas, casdm1s).transpose (1,0,2)
                dm1s = np.dot (dm1s, mo_cas.conj ().T)
                mo_core = mo1[:,:ncore]
                dm1s += (mo_core @ mo_core.conj ().T)[None,:,:]
                e1 = mc.energy_dft (ot=ot, mo_coeff=mo1, casdm1s=casdm1s,
                    casdm2=casdm2)
                return e1 - e_ot
            self.e_ot = e_ot
            self.delta_eot = delta_eot

        if self.verbose > lib.logger.DEBUG:
            from pyscf.mcpdft.pdft_veff import lazy_kernel
            v1, v2 = lazy_kernel (ot, dm1s, cascm2, mo_coeff[:,ncore:nocc])
            self._v1 = mo_coeff.conj ().T @ v1 @ mo_coeff
            self._v2 = ao2mo.full (v2, mo_coeff)

    def get_blocksize (self):
        nderiv_ao, nao = self.nderiv_ao, self.nao
        nderiv_rho, nderiv_Pi = self.nderiv_rho, self.nderiv_Pi
        ncas, nocc = self.ncas, self.nocc
        # Ignore everything that doesn't scale with the size of the molecule
        # or the active space
        nvar = 2 + int (self.rho_deriv) + 2*int (self.Pi_deriv)
        ncol = (nderiv_ao*(2*nao+ncas)        # ao + copy + mo_cas
             + (2+nderiv_Pi)*(ncas**2)        # tensor-product intermediate
             + nocc*(2*nderiv_rho+nderiv_Pi)) # drho_a, drho_b, dPi
        ncol *= 1.1 # fudge factor
        ngrids = self.ot.grids.coords.shape[0]
        remaining_floats = (self.max_memory-lib.current_memory()[0]) * 1e6 / 8
        blksize = int (remaining_floats/(ncol*BLKSIZE))*BLKSIZE
        ngrids_blk = int (ngrids / BLKSIZE) * BLKSIZE
        return max(BLKSIZE,min(blksize,ngrids_blk,BLKSIZE*1200))

    def make_dens0 (self, ao, mask, make_rho=None, casdm1s=None, cascm2=None,
            mo_cas=None):
        if make_rho is None: make_rho = self.make_rho
        if casdm1s is None: casdm1s = self.casdm1s
        if cascm2 is None: cascm2 = self.cascm2
        if mo_cas is None: mo_cas = self.mo_coeff[:,self.ncore:self.nocc]
        if ao.shape[0] == 1 and ao.ndim == 3: ao = ao[0]
        rho = make_rho (0, ao, mask, self.xctype)
        if ao.ndim == 2: ao = ao[None,:,:]
        rhos = np.stack ([rho, rho], axis=0)/2
        Pi = get_ontop_pair_density (self.ot, rhos, ao, cascm2,
            mo_cas, deriv=self.Pi_deriv, non0tab=mask)
        return rho, Pi
        # volatile memory footprint:
        #   nderiv_ao * nao * ngrids            (copying ao in make_rho)
        # + nderiv_ao * ncas * ngrids           (mo_cas on a grid)
        # + (2+nderiv_Pi) * ncas**2 * ngrids    (tensor-product intermediates)

    def make_ddens (self, ao, rho, mask):
        occ_coeff = self.mo_coeff[:,:self.nocc]
        rhos = np.stack ([rho, rho], axis=0)/2
        mo = _grid_ao2mo (self.ot.mol, ao[:self.nderiv_rho], occ_coeff,
            non0tab=mask)
        drhos, dPi = density_orbital_derivative (self.ot, self.ncore,
            self.ncas, self.casdm1s, self.cascm2, rhos, mo,
            deriv=self.Pi_deriv, non0tab=mask)
        return drhos.sum (0), dPi
        # persistent memory footprint:
        #   nderiv_rho * nocc * ngrids          (drho)
        # + nderiv_Pi * nocc * ngrids           (dPi)
        # volatile memory footprint:
        #   nderiv_rho * 2 * ngrids             (copying rho)
        # + 2 * nderiv_rho * nocc * ngrids      (mo & copied drho)
        # + (2+nderiv_Pi) * ncas**2 * ngrids    (tensor-product intermediates)

    def make_dens1 (self, ao, drho, dPi, mask, x):
        # In mc1step.update_jk_in_ah:
        #   hx = ddm1 @ h - h @ ddm1
        #   ddm1 = dm1 @ x - x @ dm1
        # the PROPER convention for consistent sign is
        #   hx = h @ ddm1 - ddm1 @ h
        #   ddm1 = x @ dm1 - dm1 @ x
        # The dm1 index is hidden in drho and dPi
        # Therefore,
        # 1) the SECOND index of x contracts with drho
        # 2) we MULTIPLY BY TWO to to account for + transpose
        
        ngrids = drho.shape[-1]
        ncore, nocc = self.ncore, self.nocc
        occ_coeff_1 = self.mo_coeff @ x[:,:self.nocc] * 2
        mo1 = _grid_ao2mo (self.ot.mol, ao, occ_coeff_1, non0tab=mask)
        Pi1 = _contract_rho_all (mo1[:self.nderiv_Pi], dPi)
        mo1 = mo1[:self.nderiv_rho]
        drho_c, mo1_c = drho[:,:,:ncore],     mo1[:,:,:ncore]
        drho_a, mo1_a = drho[:,:,ncore:nocc], mo1[:,:,ncore:nocc]
        rho1_c = _contract_rho_all (mo1_c, drho_c)
        rho1_a = _contract_rho_all (mo1_a, drho_a)
        return rho1_c, rho1_a, Pi1

    def debug_dens1 (self, ao, mask, x, weights, rho0, Pi0, rho1_test,
            Pi1_test):
        # This requires the full-space 2RDM
        ncore, nocc, nmo = self.ncore, self.nocc, self.nmo
        casdm1 = self.casdm1s.sum (0)
        dm1 = 2 * np.eye (nmo, dtype=self.dm1.dtype)
        dm1[ncore:nocc,ncore:nocc] = casdm1
        dm1[nocc:,nocc:] = 0
        dm2 = np.multiply.outer (dm1, dm1)
        dm2 -= dm2.transpose (0,3,2,1)/2
        dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] += self.cascm2
        dm1 = x @ dm1 - dm1 @ x
        dm2 = np.dot (dm2, x.T)
        dm2 += dm2.transpose (1,0,3,2)
        dm2 += dm2.transpose (2,3,0,1)
        cm2 = dm2_cumulant (dm2, dm1)
        dm1s = dm1/2
        dm1s = np.stack ([dm1s, dm1s], axis=0)
        dm1_ao = self.mo_coeff @ dm1 @ self.mo_coeff.conj ().T
        make_rho = self.ni._gen_rho_evaluator (self.ot.mol, dm1_ao, 1)[0]
        rho1_ref, Pi1_ref = self.make_dens0 (ao, mask, make_rho=make_rho,
            casdm1s=dm1s, cascm2=cm2, mo_cas=self.mo_coeff)
        if rho0.ndim == 1: rho0 = rho0[None,:]
        if Pi0.ndim == 1: Pi0 = Pi0[None,:]
        if rho1_ref.ndim == 1: rho1_ref = rho1_ref[None,:]
        if Pi1_ref.ndim == 1: Pi1_ref = Pi1_ref[None,:]
        nderiv_Pi = self.nderiv_Pi
        Pi0 = Pi0[:nderiv_Pi]
        Pi1_test = Pi1_test[:nderiv_Pi]
        Pi1_ref = Pi1_ref[:nderiv_Pi]
        rho1_err = linalg.norm (rho1_test - rho1_ref)
        Pi1_err = linalg.norm (Pi1_test - Pi1_ref)
        x_norm = linalg.norm (x)
        self.log.debug ("shifted dens: |x|, |rho1_err|, |Pi1_err| = %e, %e, "
            "%e", x_norm, rho1_err, Pi1_err)
        return rho1_err, Pi1_err

    def get_fot (self, rho, Pi, weights):
        rho = np.stack ([rho,rho], axis=0)/2
        eot, vot, fot = self.ot.eval_ot (rho, Pi, dderiv=2, weights=weights,
            _unpack_vot=False)
        return vot, fot

    def get_vot (self, rho, Pi, weights):
        rho = np.stack ([rho,rho], axis=0)/2
        eot, vot, _ = self.ot.eval_ot (rho, Pi, dderiv=1, weights=weights)
        vrho, vPi = vot
        return vrho, vPi

    def get_fxot (self, ao, rho0, Pi0, drho, dPi, x, weights, mask,
            return_num=False):
        vot, fot = self.get_fot (rho0, Pi0, weights)
        rho1_c, rho1_a, Pi1 = self.make_dens1 (ao, drho, dPi, mask, x)
        rho1 = rho1_c + rho1_a
        if self.verbose > lib.logger.DEBUG:
            self.debug_dens1 (ao, mask, x, weights, rho0, Pi0, rho1, Pi1)
        fxrho, fxPi = contract_fot (self.ot, fot, rho0, Pi0, rho1, Pi1,
            unpack=True, vot_packed=vot)
        rho0_deriv = rho0[1:4,:] if self.rho_deriv else None
        Pi0_deriv = Pi0[1:4,:] if self.Pi_deriv else None
        vrho, vPi = _unpack_sigma_vector (vot, rho0_deriv, Pi0_deriv)
        if return_num:
            dvrho, dvPi = self.get_vot (rho0+rho1, Pi0+Pi1, weights)
            dvot = [dvrho - vrho, dvPi - vPi, rho1, Pi1]
        de = (np.dot (rho1.ravel (), (vrho * weights[None,:]).ravel ())
            + np.dot (Pi1.ravel (), (vPi * weights[None,:]).ravel ()))
        Pi1 = fxrho_a = fxrho_c = rho1 = None
        if self.do_cumulant and self.ncore: # fxrho gets the D_all D_c part
                                            # D_c D_a part has to be separate
            if vPi.ndim == 1: vPi = vPi[None,:]
            fxrho_c = _contract_vot_rho (vPi, rho1_c)
            fxrho_a = _contract_vot_rho (vPi, rho1_a)
        if return_num: return de, fxrho, fxPi, fxrho_c, fxrho_a, dvot
        return de, fxrho, fxPi, fxrho_c, fxrho_a

    def contract_v_ddens (self, v, ddens, ao, weights, mask):
        vw = v * weights[None,:]
        vao = _contract_vot_ao (vw, ao)
        return sum ([_dot_ao_mo (self.ot.mol, v, d, non0tab=mask,
            shls_slice=self.shls_slice, ao_loc=self.ao_loc,
            hermi=0) for v, d in zip (vao, ddens)])

    def debug_cumulant (self, x, dg_cum):
        norm_x = linalg.norm (x)
        ncore, nocc, nmo = self.ncore, self.nocc, self.nmo
        casdm1 = self.casdm1s.sum (0)
        dm1 = 2 * np.eye (nmo, dtype=self.dm1.dtype)
        dm1[ncore:nocc,ncore:nocc] = casdm1
        dm1[nocc:,nocc:] = 0
        dm_core = dm1.copy ()
        dm_core[ncore:,ncore:] = 0.0
        dm_cas = dm1.copy ()
        dm_cas[:ncore,:ncore] = 0.0
        x_core = x @ dm_core - dm_core @ x
        x_cas = x @ dm_cas - dm_cas @ x
        v_c = np.tensordot (self._v2, x_core) / 2
        v_a = np.tensordot (self._v2, x_cas) / 2
        dm_core, dm_cas = dm_core[:nocc], dm_cas[:nocc]
        dg_cum_ref = -dm_core @ (v_c + v_a) - dm_cas @ v_c
        norm_ref = linalg.norm (dg_cum_ref)
        norm_err = linalg.norm (dg_cum-dg_cum_ref)
        self.log.debug ("cumulant debug: |x|, |ref|, |err| = %e, %e, %e",
            norm_x, norm_ref, norm_err)

    def __call__(self, x, packed=False, algorithm='analytic'):
        ''' Compute Hessian-vector and gradient-vector product of on-top
            energy wrt orbital rotation.

            Args:
                x : ndarray
                    Orbital-rotation step vector. See kwarg "packed" for
                    shape
    
            Kwargs:
                packed : logical
                    If true, x has shape given by mc.pack_uniq_var
                algorithm : string
                    Select Hessian-vector product calculation type:
                    'analytic': Analytical Hessian-vector product
                    'seminum': gradient @ x - gradient @ x=0.
    
            Returns:
                dg : ndarray of shape (x.shape)
                    Hessian-vector product
                de : float
                    gradient-vector product 
        '''
        if algorithm.lower () == 'analytic': return self.kernel (x, 
            packed=packed)
        elif 'seminum' in algorithm.lower (): return self.seminum_orb (x)
        else: raise RuntimeError ("Unknown algorithm '{}'".format (algorithm))

    def kernel (self, x, packed=False):
        ncore, nocc = self.ncore, self.nocc
        if self.incl_d2rho:
            packed = True
            dg_d2rho = self.d2rho_h_op (x)
        if packed: 
            x_packed = x.copy ()
            x = self.unpack_uniq_var (x)
        else:
            x_packed = self.pack_uniq_var (x)
        dg = np.zeros ((self.nocc, self.nao), dtype=x.dtype)
        dg_cum = np.zeros_like (dg)
        de = 0
        for ao, mask, weights, coords in self.ni.block_loop (self.ot.mol,
                self.ot.grids, self.nao, self.rho_deriv, self.max_memory,
                blksize=self.get_blocksize ()):
            rho0, Pi0 = self.make_dens0 (ao, mask)
            if ao.ndim == 2: ao = ao[None,:,:]
            drho, dPi = self.make_ddens (ao, rho0, mask)
            dde, fxrho, fxPi, fxrho_c, fxrho_a = self.get_fxot (ao, rho0, Pi0,
                drho, dPi, x, weights, mask)
            de += dde
            dg -= self.contract_v_ddens (fxrho, drho, ao, weights, mask).T
            dg -= self.contract_v_ddens (fxPi, dPi, ao, weights, mask).T
            # Transpose because update_jk_in_ah requires this shape
            # Minus because I want to use 1 consistent sign rule here
            if self.do_cumulant and ncore: # The D_c D_a part
                drho_c = drho[:self.nderiv_Pi,:,:ncore]
                drho_a = drho[:self.nderiv_Pi,:,ncore:nocc]
                dg_cum[:ncore] -= self.contract_v_ddens (fxrho_c, drho_c,
                    ao, weights, mask).T
                dg_cum[:ncore] -= self.contract_v_ddens (fxrho_a, drho_c,
                    ao, weights, mask).T
                dg_cum[ncore:nocc] -= self.contract_v_ddens (fxrho_c, drho_a,
                    ao, weights, mask).T
        dg = np.dot (dg, self.mo_coeff) 
        dg_cum = np.dot (dg_cum, self.mo_coeff) 
        if self.incl_d2rho:
            de_test = 2 * np.dot (x_packed, self.g_orb)
            # The factor of 2 is because g_orb is evaluated in terms of square
            # antihermitian arrays, but only the lower-triangular parts are
            # stored in x and g_orb.
            self.log.debug (('E from integration: %e; from stored grad: %e; '
                'diff: %e'), de, de_test, de-de_test)
            if self.verbose > lib.logger.DEBUG: 
                self.debug_d2rho (x, dg_d2rho, dg_cum)
        dg += dg_cum
        if self.verbose > lib.logger.DEBUG and self.do_cumulant and ncore:
            self.debug_cumulant (x, dg_cum)
        if packed:
            dg_full = np.zeros ((self.nmo, self.nmo), dtype=dg.dtype)
            dg_full[:self.nocc,:] = dg[:,:]
            dg_full -= dg_full.T
            dg = self.pack_uniq_var (dg_full)
            if self.incl_d2rho: dg += dg_d2rho
        return dg, de

    def seminum_orb (self, x):
        ''' Calculate energy and gradient change seminumerically using
            updated-orbital recalculation of everything '''
        assert (self.incl_d2rho)
        return self.delta_gorb (x), self.delta_eot (x)

    def debug_d2rho (self, x, dg_test=None, dg_cum=None):
        ncore, nocc, nmo, nao = self.ncore, self.nocc, self.nmo, self.nao
        dg = np.zeros ((self.nocc, self.nao), dtype=x.dtype)
        if dg_test is None:
            dg_test = self.d2rho_h_op (self.pack_uniq_var (x))
        casdm1 = self.casdm1s.sum (0)
        dm1 = 2 * np.eye (nmo, dtype=self.dm1.dtype)
        dm1[ncore:nocc,ncore:nocc] = casdm1
        dm1[nocc:,nocc:] = 0
        dm2 = np.zeros ([nmo,]*4, dtype=self.casdm2.dtype)
        dm2 = np.multiply.outer (dm1, dm1)
        dm2 -= 0.5 * dm2.transpose (0,3,2,1)
        dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = self.casdm2
        f = self._v1 @ dm1
        f += np.tensordot (self._v2, dm2, axes=((1,2,3),(1,2,3)))
        # ^ This f is 100% validated
        dg = (f @ x - x @ f) 
        dm1 = x @ dm1 - dm1 @ x
        dm2 = np.dot (dm2, x.T)
        dm2 += dm2.transpose (1,0,3,2)
        dm2 += dm2.transpose (2,3,0,1)
        dg += self._v1 @ dm1
        dg += np.tensordot (self._v2, dm2, axes=((1,2,3),(1,2,3)))
        if dg_cum is not None: dg[:nocc,:] -= dg_cum
        dg -= dg.T
        dg = self.pack_uniq_var (dg)
        norm_err, theta = vector_error (dg_test, dg)
        norm_test = linalg.norm (dg_test)
        norm_ref = linalg.norm (dg)
        self.log.debug (('d2rho debug: |PySCF code| = %e; '
            '|reimplementation| = %e; |error| = %e; theta = %e'),
            norm_test, norm_ref, norm_err, theta)
        return dg

    def debug_hessian_blocks (self, x, packed=False, mask_dcon=False):
        log = self.log
        nao, nmo, nocc = self.nao, self.nmo, self.nocc
        make_dens1 = self.make_dens1
        norm_x = linalg.norm (x)
        def mask_rho_(arr, iel, idx0=None):
            if iel != 0: arr[0:1,:] = 0.0
            if iel != 2: arr[1:4,:] = 0.0
            if idx0 is not None:
                arr[:,idx0] = 0.0
            return arr
        def mask_Pi_(arr, iel, idx0=None): 
            return mask_rho_(arr,iel-1,idx0)
        def mask_dens1 (icol):
            def make_masked_dens1 (ao, drho, dPi, mask, my_x):
                rho1_c, rho1_a, Pi1 = make_dens1 (ao, drho, dPi, mask, my_x)
                rho1_c = mask_rho_(rho1_c, icol)
                rho1_a = mask_rho_(rho1_a, icol)
                Pi1 = mask_Pi_(Pi1, icol)
                return rho1_c, rho1_a, Pi1
            return make_masked_dens1
        get_fxot = self.get_fxot
        def mask_fxot (irow, my_dg):
            def get_masked_fxot (ao, rho0, Pi0, drho, dPi, my_x, weights,
                    mask):
                de, fxrho, fxPi, fxrho_c, fxrho_a, dvot = get_fxot (ao, rho0,
                    Pi0, drho, dPi, my_x, weights, mask, return_num=True)
                norm_x = linalg.norm (my_x)
                if rho0.ndim == 1: rho0 = rho0[None,:]
                dvrho, dvPi, rho1, Pi1 = dvot
                idx = rho0[0]>0
                R0 = self.ot.get_ratio (np.atleast_2d (Pi0), rho0/2)[0]
                idx_dcon = np.abs (1.0-R0) < 1e-3 if mask_dcon else None
                fxrho = mask_rho_(fxrho, irow, idx_dcon)
                dvrho = mask_rho_(dvrho, irow, idx_dcon)
                fxPi = mask_Pi_(fxPi, irow, idx_dcon)
                dvPi = mask_Pi_(dvPi, irow, idx_dcon)
                my_dg[:,:] -= self.contract_v_ddens (dvrho, drho, ao, weights,
                    mask).T
                my_dg[:,:] -= self.contract_v_ddens (dvPi, dPi, ao, weights,
                    mask).T
                return de, fxrho, fxPi, fxrho_c, fxrho_a
            return get_masked_fxot
        ndim = 2 + int (self.rho_deriv) + int (self.Pi_deriv)
        lbls = ('rho','Pi',"rho'","Pi'")
        for irow, icol in product (range (ndim), repeat=2):
            sector = 'f_' + lbls[irow] + ',' + lbls[icol]
            dg_num = np.zeros ((nmo if packed else nocc, nao), dtype=x.dtype)
            with lib.temporary_env (self, incl_d2rho=False, do_cumulant=False,
                    get_fxot = mask_fxot (irow, dg_num[:nocc,:]),
                    make_dens1 = mask_dens1 (icol)):
                dg_an = self (x, packed=packed)[0]
            dg_num = np.dot (dg_num, self.mo_coeff) 
            if packed: dg_num = self.pack_uniq_var (dg_num-dg_num.T)
            norm_an = linalg.norm (dg_an)
            norm_num = linalg.norm (dg_num)
            norm_err_rel, theta = vector_error (dg_an, dg_num)
            assert (not (np.isnan (theta))), '{} {} {} {}'.format (numer,
                denom, norm_an, norm_num)
            log.debug (('Debugging %s: |x| = %8.2e, |num| = %8.2e, '
                '|an-num|/|num| = %8.2e, theta(an,num) = %8.2e'), sector,
                norm_x, norm_num, norm_err_rel, theta)

class ExcOrbitalHessianOperator (object):
    ''' for comparison '''

    def __init__(self, ks, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = ks.mo_coeff
        if mo_occ is None: mo_occ = ks.mo_occ
        self.nao, self.nmo = nao, nmo = mo_coeff.shape[-2:]
        self.ks = ks = ks.newton ()
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

        dm = ks.make_rdm1 (mo_coeff=mo_coeff, mo_occ=mo_occ)
        vxc = ks.get_veff (dm=dm)
        self.exc = exc = vxc.exc
        vxc -= vxc.vj
        self.vxc = vxc
        def no_j (*args, **kwargs): return 0
        def no_jk (*args, **kwargs): return 0, 0
        with lib.temporary_env (ks, get_j=no_j, get_jk=no_jk):
            g_orb, h_op, h_diag = ks.gen_g_hop (mo_coeff, mo_occ, fock_ao=vxc)
        self.g_orb = g_orb
        self.h_op = h_op
        self.h_diag = h_diag

    def __call__(self, x):
        ''' return dg, de; always packed '''
        def no_j (*args, **kwargs): return 0
        def no_jk (*args, **kwargs): return 0, 0
        with lib.temporary_env (self.ks, get_j=no_j, get_jk=no_jk):
            dg = self.h_op (x)
        de = 2*np.dot (self.g_orb.ravel (), x.ravel ())
        return dg, de

    def seminum_orb (self, x):
        ks = self.ks
        mo_occ = self.mo_occ
        u = self.ks.update_rotate_matrix (x, mo_occ)
        mo1 = ks.rotate_mo (self.mo_coeff, u)
        dm1 = self.ks.make_rdm1 (mo_coeff=mo1, mo_occ=mo_occ)
        vxc1 = self.ks.get_veff (dm=dm1)
        exc1 = vxc1.exc
        vxc1 -= vxc1.vj
        de = exc1 - self.exc
        g1 = ks.gen_g_hop (mo1, mo_occ, fock_ao=vxc1)[0]
        dg = g1 - self.g_orb 
        return dg, de

    def pack_uniq_var (self, x):
        return hf.pack_uniq_var (x, self.mo_occ)

    def unpack_uniq_var (self, x):
        return hf.unpack_uniq_var (x, self.mo_occ)

if __name__ == '__main__':
    from pyscf import gto, scf, dft, mcpdft
    mol = gto.M (atom = 'Li 0 0 0; H 1.2 0 0', basis = 'sto-3g',
        verbose=lib.logger.DEBUG, output='pdft_feff.log')
    mf = scf.RHF (mol).run ()
    def debug_hess (hop):
        print ("g_orb:", linalg.norm (hop.g_orb))
        print ("h_diag:", linalg.norm (hop.h_diag))
        x0 = -hop.g_orb / hop.h_diag
        x0[hop.g_orb==0] = 0
        print ("x0 = g_orb/h_diag:", linalg.norm (x0))
        print (" n " + ' '.join (['{:>10s}',]*7).format ('x_norm','de_test',
            'de_ref','de_relerr','dg_test','dg_ref','dg_relerr'))
        for p in range (20):
            fac = 1/(2**p)
            x1 = x0 * fac
            x_norm = linalg.norm (x1)
            dg_test, de_test = hop (x1)
            dg_ref,  de_ref  = hop.seminum_orb (x1)
            e_err = abs ((de_test-de_ref)/de_ref)
            dg_err_norm, dg_theta = vector_error (dg_test, dg_ref) 
            dg_test_norm = linalg.norm (dg_test)
            dg_ref_norm = linalg.norm (dg_ref)
            row = [p, x_norm, abs(de_test), abs(de_ref), e_err, dg_test_norm,
                dg_ref_norm, dg_err_norm]
            if callable (getattr (hop, 'debug_hessian_blocks', None)):
                hop.debug_hessian_blocks (x1, packed=True,
                mask_dcon=False)#(hop.ot.otxc[0]=='t'))
            print ((" {:2d} " + ' '.join (['{:10.3e}',]*7)).format (*row))
        dg_err = dg_test - dg_ref
        denom = dg_ref.copy ()
        denom[np.abs(dg_ref)<1e-8] = 1.0
        dg_err /= denom
        fmt_str = ' '.join (['{:10.3e}',]*hop.nmo)
        print ("dg_test:")
        for row in hop.unpack_uniq_var (dg_test): print (fmt_str.format (*row))
        print ("dg_ref:")
        for row in hop.unpack_uniq_var (dg_ref): print (fmt_str.format (*row))
        fmt_str = ' '.join (['{:6.2f}',]*hop.nmo)
        print ("dg_err (relative):")
        for row in hop.unpack_uniq_var (dg_err): print (fmt_str.format (*row))
        print ("")
    from mrh.my_pyscf.tools import molden
    mini_grid = {'atom_grid': (1,1)}
    for nelecas, lbl in zip ((2, (2,0)), ('Singlet','Triplet')):
        #if nelecas is not 2: continue
        print (lbl,'case\n')
        #for fnal in 'LDA,VWN3', 'PBE':
        #    ks = dft.RKS (mol).set (xc=fnal).run ()
        #    print ("LiH {} energy:".format (fnal),ks.e_tot)
        #    exc_hop = ExcOrbitalHessianOperator (ks)
        #    debug_hess (exc_hop)
        for fnal in 'tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE':
            if fnal[:3] != 'ftP': continue
            mc = mcpdft.CASSCF (mf, fnal, 2, nelecas, grids_level=1).run ()
            mc.canonicalize_(cas_natorb=True)
            molden.from_mcscf (mc, lbl + '.molden')
            print ("LiH {} energy:".format (fnal),mc.e_tot)
            eot_hop = EotOrbitalHessianOperator (mc, incl_d2rho=True)
            debug_hess (eot_hop)
        print ("")

