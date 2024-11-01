from pyscf import lib, symm
from mrh.my_pyscf.fci.csfstring import ImpossibleCIvecError
from mrh.my_pyscf.mcscf import _DFLASCI
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg 
import numpy as np

# This must be locked to CSF solver for the forseeable future, because I know of no other way to
# handle spin-breaking potentials while retaining spin constraint

class MicroIterInstabilityException (Exception):
    pass

def kernel (las, mo_coeff=None, ci0=None, casdm0_fr=None, conv_tol_grad=1e-4, 
        assert_no_dupes=False, verbose=lib.logger.NOTE):
    from mrh.my_pyscf.mcscf.lasci import _eig_inactive_virtual
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if assert_no_dupes: las.assert_no_duplicates ()
    log = lib.logger.new_logger(las, verbose)
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    log.debug('Start LASCI')

    h2eff_sub = las.get_h2eff (mo_coeff)
    t1 = log.timer('integral transformation to LAS space', *t0)

    # In the first cycle, I may pass casdm0_fr instead of ci0.
    # Therefore, I need to work out this get_veff call separately.
    # This is only for compatibility with the "old" algorithm
    if ci0 is None and casdm0_fr is not None:
        casdm0_sub = [np.einsum ('rsij,r->sij', dm, las.weights) for dm in casdm0_fr]
        dm1_core = mo_coeff[:,:las.ncore] @ mo_coeff[:,:las.ncore].conjugate ().T
        dm1s_sub = [np.stack ([dm1_core, dm1_core], axis=0)]
        for idx, casdm1s in enumerate (casdm0_sub):
            mo = las.get_mo_slice (idx, mo_coeff=mo_coeff)
            moH = mo.conjugate ().T
            dm1s_sub.append (np.tensordot (mo, np.dot (casdm1s, moH), 
                                           axes=((1),(1))).transpose (1,0,2))
        dm1s_sub = np.stack (dm1s_sub, axis=0)
        dm1s = dm1s_sub.sum (0)
        veff = las.get_veff (dm1s=dm1s.sum (0))
        veff = las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, casdm1s_sub=casdm0_sub)
        casdm1s_sub = casdm0_sub
        casdm1frs = casdm0_fr
    else:
        if (ci0 is None or any ([c is None for c in ci0]) or
          any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
            ci0 = las.get_init_guess_ci (mo_coeff, h2eff_sub, ci0)
        if (ci0 is None or any ([c is None for c in ci0]) or
          any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
            raise RuntimeError ("failed to populate get_init_guess")
        veff = las.get_veff (dm1s = las.make_rdm1 (mo_coeff=mo_coeff, ci=ci0))
        casdm1s_sub = las.make_casdm1s_sub (ci=ci0)
        casdm1frs = las.states_make_casdm1s_sub (ci=ci0)
        veff = las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, ci=ci0, casdm1s_sub=casdm1s_sub)
    t1 = log.timer('LASCI initial get_veff', *t1)

    ugg = None
    converged = False
    ci1 = ci0
    t2 = (t1[0], t1[1])
    it = 0
    for it in range (las.max_cycle_macro):
        e_cas, ci1 = ci_cycle (las, mo_coeff, ci1, veff, h2eff_sub, casdm1frs, log)
        if ugg is None: ugg = las.get_ugg (mo_coeff, ci1)
        log.info ('LASCI subspace CI energies: {}'.format (e_cas))
        t1 = log.timer ('LASCI ci_cycle', *t1)

        veff = veff.sum (0)/2
        # Canonicalize inactive and virtual spaces to set many off-diagonal elements of the
        # orbital-rotation Hessian to zero, which should improve the microiteration below
        fock = las.get_hcore () + veff
        fock = mo_coeff.conj ().T @ fock @ mo_coeff
        orbsym = getattr (mo_coeff, 'orbsym', None)
        ene, umat = _eig_inactive_virtual (las, fock, orbsym=orbsym)
        mo_coeff = mo_coeff @ umat
        if orbsym is not None:
            mo_coeff = lib.tag_array (mo_coeff, orbsym=orbsym)
        h2eff_sub[:,:] = umat.conj ().T @ h2eff_sub

        casdm1s_new = las.make_casdm1s_sub (ci=ci1)
        if not isinstance (las, _DFLASCI) or las.verbose > lib.logger.DEBUG:
            #veff = las.get_veff (mo_coeff=mo_coeff, ci=ci1)
            veff_new = las.get_veff (dm1s = las.make_rdm1 (mo_coeff=mo_coeff, ci=ci1))
            if not isinstance (las, _DFLASCI): veff = veff_new
        if isinstance (las, _DFLASCI):
            dcasdm1s = [dm_new - dm_old for dm_new, dm_old in zip (casdm1s_new, casdm1s_sub)]
            veff += las.fast_veffa (dcasdm1s, h2eff_sub, mo_coeff=mo_coeff, ci=ci1) 
            if las.verbose > lib.logger.DEBUG:
                errmat = veff - veff_new
                lib.logger.debug (las, 'fast_veffa error: {}'.format (linalg.norm (errmat)))
        veff = las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, ci=ci1)
        casdm1s_sub = casdm1s_new

        t1 = log.timer ('LASCI get_veff after ci', *t1)
        H_op = las.get_hop (ugg=ugg, mo_coeff=mo_coeff, ci=ci1, h2eff_sub=h2eff_sub, veff=veff,
                            do_init_eri=False)
        g_vec = H_op.get_grad ()
        if las.verbose > lib.logger.INFO:
            g_orb_test, g_ci_test = las.get_grad (ugg=ugg, mo_coeff=mo_coeff, ci=ci1,
                                                  h2eff_sub=h2eff_sub, veff=veff)[:2]
            if ugg.nvar_orb:
                err = linalg.norm (g_orb_test - g_vec[:ugg.nvar_orb])
                log.debug ('GRADIENT IMPLEMENTATION TEST: |D g_orb| = %.15g', err)
                assert (err < 1e-5), '{}'.format (err)
            for isub in range (len (ci1)): # TODO: double-check that this code works in SA-LASSCF
                i = ugg.ncsf_sub[:isub].sum ()
                j = i + ugg.ncsf_sub[isub].sum ()
                k = i + ugg.nvar_orb
                l = j + ugg.nvar_orb
                log.debug ('GRADIENT IMPLEMENTATION TEST: |D g_ci({})| = %.15g'.format (isub), 
                           linalg.norm (g_ci_test[i:j] - g_vec[k:l]))
            err = linalg.norm (g_ci_test - g_vec[ugg.nvar_orb:])
            assert (err < 1e-5), '{}'.format (err)
        gx = H_op.get_gx ()
        prec_op = H_op.get_prec ()
        prec = prec_op (np.ones_like (g_vec)) # Check for divergences
        norm_gorb = linalg.norm (g_vec[:ugg.nvar_orb]) if ugg.nvar_orb else 0.0
        norm_gci = linalg.norm (g_vec[ugg.nvar_orb:]) if ugg.ncsf_sub.sum () else 0.0
        norm_gx = linalg.norm (gx) if gx.size else 0.0
        x0 = prec_op._matvec (-g_vec)
        norm_xorb = linalg.norm (x0[:ugg.nvar_orb]) if ugg.nvar_orb else 0.0
        norm_xci = linalg.norm (x0[ugg.nvar_orb:]) if ugg.ncsf_sub.sum () else 0.0
        lib.logger.info (
            las, 'LASCI macro %d : E = %.15g ; |g_int| = %.15g ; |g_ci| = %.15g ; |g_x| = %.15g',
            it, H_op.e_tot, norm_gorb, norm_gci, norm_gx)
        #log.info (
        #    ('LASCI micro init : E = %.15g ; |g_orb| = %.15g ; |g_ci| = %.15g ; |x0_orb| = %.15g '
        #    '; |x0_ci| = %.15g'), H_op.e_tot, norm_gorb, norm_gci, norm_xorb, norm_xci)
        las.dump_chk (mo_coeff=mo_coeff, ci=ci1)
        if (norm_gorb<conv_tol_grad and norm_gci<conv_tol_grad)or((norm_gorb+norm_gci)<norm_gx/10):
            converged = True
            break
        H_op._init_eri_() 
        # ^ This is down here to save time in case I am already converged at initialization
        t1 = log.timer ('LASCI Hessian constructor', *t1)
        microit = [0]
        last_x = [0]
        first_norm_x = [None]
        def my_callback (x):
            microit[0] += 1
            norm_xorb = linalg.norm (x[:ugg.nvar_orb]) if ugg.nvar_orb else 0.0
            norm_xci = linalg.norm (x[ugg.nvar_orb:]) if ugg.ncsf_sub.sum () else 0.0
            addr_max = np.argmax (np.abs (x))
            id_max = ugg.addr2idstr (addr_max)
            x_max = x[addr_max]/np.pi
            log.debug ('Maximum step vector element x[{}] = {}*pi ({})'.format (addr_max, x_max, id_max))
            if las.verbose > lib.logger.INFO:
                Hx = H_op._matvec (x) # This doubles the price of each iteration!!
                resid = g_vec + Hx
                norm_gorb = linalg.norm (resid[:ugg.nvar_orb]) if ugg.nvar_orb else 0.0
                norm_gci = linalg.norm (resid[ugg.nvar_orb:]) if ugg.ncsf_sub.sum () else 0.0
                xorb, xci = ugg.unpack (x)
                xci = [[x_s * las.weights[iroot] for iroot, x_s in enumerate (x_rs)]
                       for x_rs in xci]
                xscale = ugg.pack (xorb, xci)
                Ecall = H_op.e_tot + xscale.dot (g_vec + (Hx/2))
                log.info (('LASCI micro %d : E = %.15g ; |g_orb| = %.15g ; |g_ci| = %.15g ;'
                          '|x_orb| = %.15g ; |x_ci| = %.15g'), microit[0], Ecall, norm_gorb,
                          norm_gci, norm_xorb, norm_xci)
            else:
                log.info ('LASCI micro %d : |x_orb| = %.15g ; |x_ci| = %.15g', microit[0],
                          norm_xorb, norm_xci)
            if abs(x_max)>.5: # Nonphysical step vector element
                if last_x[0] is 0:
                    x[np.abs (x)>.5*np.pi] = 0
                    last_x[0] = x
                raise MicroIterInstabilityException ("|x[i]| > pi/2")
            norm_x = linalg.norm (x)
            if first_norm_x[0] is None:
                first_norm_x[0] = norm_x
            elif norm_x > 10*first_norm_x[0]:
                raise MicroIterInstabilityException ("||x(n)|| > 10*||x(0)||")
            last_x[0] = x.copy ()

        my_tol = max (conv_tol_grad, norm_gx/10)
        try:
            x = sparse_linalg.cg (H_op, -g_vec, x0=x0, atol=my_tol,
                                  maxiter=las.max_cycle_micro, callback=my_callback,
                                  M=prec_op)[0]
            t1 = log.timer ('LASCI {} microcycles'.format (microit[0]), *t1)
            mo_coeff, ci1, h2eff_sub = H_op.update_mo_ci_eri (x, h2eff_sub)
            t1 = log.timer ('LASCI Hessian update', *t1)

            #veff = las.get_veff (mo_coeff=mo_coeff, ci=ci1)
            veff = las.get_veff (dm1s = las.make_rdm1 (mo_coeff=mo_coeff, ci=ci1))
            veff = las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, ci=ci1)
            t1 = log.timer ('LASCI get_veff after secondorder', *t1)
        except MicroIterInstabilityException as e:
            log.info ('Unstable microiteration aborted: %s', str (e))
            t1 = log.timer ('LASCI {} microcycles'.format (microit[0]), *t1)
            x = last_x[0]
            for i in range (3): # Make up to 3 attempts to scale-down x if necessary
                mo2, ci2, h2eff_sub2 = H_op.update_mo_ci_eri (x, h2eff_sub)
                t1 = log.timer ('LASCI Hessian update', *t1)
                veff2 = las.get_veff (dm1s = las.make_rdm1 (mo_coeff=mo2, ci=ci2))
                veff2 = las.split_veff (veff2, h2eff_sub2, mo_coeff=mo2, ci=ci2)
                t1 = log.timer ('LASCI get_veff after secondorder', *t1)
                e2 = las.energy_nuc () + las.energy_elec (mo_coeff=mo2, ci=ci2, h2eff=h2eff_sub2,
                                                          veff=veff2)
                if e2 < H_op.e_tot:
                    break
                log.info ('New energy ({}) is higher than keyframe energy ({})'.format (
                    e2, H_op.e_tot))
                log.info ('Attempt {} of 3 to scale down trial step vector'.format (i+1))
                x *= .5
            mo_coeff, ci1, h2eff_sub, veff = mo2, ci2, h2eff_sub2, veff2


        casdm1frs = las.states_make_casdm1s_sub (ci=ci1)
        casdm1s_sub = las.make_casdm1s_sub (ci=ci1)

    t2 = log.timer ('LASCI {} macrocycles'.format (it), *t2)

    e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, ci=ci1, h2eff=h2eff_sub,
                                                 veff=veff)
    if log.verbose > lib.logger.INFO:
        e_tot_test = las.get_hop (ugg=ugg, mo_coeff=mo_coeff, ci=ci1, h2eff_sub=h2eff_sub,
                                  veff=veff, do_init_eri=False).e_tot
    dm_core = 2 * mo_coeff[:,:las.ncore] @ mo_coeff[:,:las.ncore].conj ().T
    veff_a = np.stack ([las.fast_veffa ([d[state] for d in casdm1frs], h2eff_sub,
                                        mo_coeff=mo_coeff, ci=ci1, _full=True)
                        for state in range (las.nroots)], axis=0)
    veff_c = las.get_veff (dm1s=dm_core)
    # veff's spin-summed component should be correct because I called get_veff with spin-summed rdm
    veff = veff_c[None,None,:,:] + veff_a 
    veff = lib.tag_array (veff, c=veff_c, sa=np.einsum ('rsij,r->sij', veff, las.weights))
    e_states = las.energy_nuc () + np.array (las.states_energy_elec (mo_coeff=mo_coeff, ci=ci1,
                                                                     h2eff=h2eff_sub, veff=veff))
    if log.verbose > lib.logger.INFO:
        e_tot_test = np.dot (las.weights, e_states)
        if abs (e_tot_test-e_tot) > 1e-8:
            log.warn ('order-of-operations disagreement of %e in state-averaged energy (%e)',
                      e_tot_test-e_tot, e_tot)

    # I need the true veff, with f^a_a and f^i_i spin-separated, in order to use the Hessian
    # Better to do it here with bmPu than in localintegrals

    log.info ('LASCI %s after %d cycles', ('not converged', 'converged')[converged], it+1)
    log.info ('LASCI E = %.15g ; |g_int| = %.15g ; |g_ci| = %.15g ; |g_ext| = %.15g', e_tot,
              norm_gorb, norm_gci, norm_gx)
    t1 = log.timer ('LASCI wrap-up', *t1)
        
    mo_coeff, mo_energy, mo_occ, ci1, h2eff_sub = las.canonicalize (mo_coeff, ci1, veff=veff.sa,
                                                                    h2eff_sub=h2eff_sub)
    t1 = log.timer ('LASCI canonicalization', *t1)

    t0 = log.timer ('LASCI kernel function', *t0)

    las.dump_chk (mo_coeff=mo_coeff, ci=ci1)

    return converged, e_tot, e_states, mo_energy, mo_coeff, e_cas, ci1, h2eff_sub, veff

def ci_cycle (las, mo, ci0, veff, h2eff_sub, casdm1frs, log):
    if ci0 is None: ci0 = [None for idx in range (las.nfrags)]
    # CI problems
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    h1eff_sub = las.get_h1eff (mo, veff=veff, h2eff_sub=h2eff_sub, casdm1frs=casdm1frs)
    ncas_cum = np.cumsum ([0] + las.ncas_sub.tolist ()) + las.ncore
    e_cas = []
    ci1 = []
    e0 = 0.0 
    for isub, (fcibox, ncas, nelecas, h1e, fcivec) in enumerate (zip (las.fciboxes, las.ncas_sub,
                                                                      las.nelecas_sub, h1eff_sub,
                                                                      ci0)):
        eri_cas = las.get_h2eff_slice (h2eff_sub, isub, compact=8)
        max_memory = max(400, las.max_memory-lib.current_memory()[0])
        orbsym = getattr (mo, 'orbsym', None)
        if orbsym is not None:
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            orbsym = orbsym[i:j]
            orbsym_io = orbsym.copy ()
            if np.issubsctype (orbsym.dtype, np.integer):
                orbsym_io = np.asarray ([symm.irrep_id2name (las.mol.groupname, x)
                                         for x in orbsym])
            log.info ("LASCI subspace {} with orbsyms {}".format (isub, orbsym_io))
        else:
            log.info ("LASCI subspace {} with no orbsym information".format (isub))
        if log.verbose > lib.logger.DEBUG: 
         for state, solver in enumerate (fcibox.fcisolvers):
            wfnsym = getattr (solver, 'wfnsym', None)
            if (wfnsym is not None) and (orbsym is not None):
                if isinstance (wfnsym, str):
                    wfnsym_str = wfnsym
                else:
                    wfnsym_str = symm.irrep_id2name (las.mol.groupname, wfnsym)
                log.debug1 ("LASCI subspace {} state {} with wfnsym {}".format (isub, state,
                                                                                wfnsym_str))

        e_sub, fcivec = fcibox.kernel(h1e, eri_cas, ncas, nelecas,
                                      ci0=fcivec, verbose=log,
                                      max_memory=max_memory,
                                      ecore=e0, orbsym=orbsym)
        e_cas.append (e_sub)
        ci1.append (fcivec)
        t1 = log.timer ('FCI box for subspace {}'.format (isub), *t1)
    return e_cas, ci1

def all_nonredundant_idx (nmo, ncore, ncas_sub):
    ''' Generate a index mask array addressing all nonredundant, lower-triangular elements of an
    nmo-by-nmo orbital-rotation unitary generator amplitude matrix for a LASSCF or LASCI problem
    with ncore inactive orbitals and len (ncas_sub) fragments with ncas_sub[i] active orbitals in
    the ith fragment:

        <--------------nmo--------------->
        <-ncore->|<-sum(ncas_sub)->|
        __________________________________
        | False  |False|False| ... |False|
        |  True  |False|False| ... |False|
        |  True  | True|False| ... |False|
        |  ...   | ... | ... | ... |False|
        |  True  | True| True| ....|False|
        ----------------------------------
    '''
    nocc = ncore + sum (ncas_sub)
    idx = np.zeros ((nmo, nmo), dtype=np.bool_)
    idx[ncore:,:ncore] = True # inactive -> everything
    idx[nocc:,ncore:nocc] = True # active -> virtual
    sub_slice = np.cumsum ([0] + ncas_sub.tolist ()) + ncore
    idx[sub_slice[-1]:,:sub_slice[0]] = True
    for ix1, i in enumerate (sub_slice[:-1]):
        j = sub_slice[ix1+1]
        for ix2, k in enumerate (sub_slice[:ix1]):
            l = sub_slice[ix2+1]
            idx[i:j,k:l] = True
    # active -> active
    return idx

class LASCI_UnitaryGroupGenerators (object):
    ''' Object for `pack'ing (for root-finding algorithms) and `unpack'ing (for direct
    manipulation) the nonredundant variables ('unitary generator amplitudes') of a `LASCI' problem.
    `LASCI' here means that the CAS is frozen relative to inactive or external orbitals, but active
    orbitals from different fragments may rotate into one another, and inactive orbitals may rotate
    into virtual orbitals, and CI vectors may also evolve. Transforms between the nonredundant
    lower-triangular part ('x') of a skew-symmetric orbital rotation matrix ('kappa')
    and transforms CI transfer vectors between the determinant and configuration state function
    bases. Subclass me to apply point-group symmetry or to do a full LASSCF calculation.

    Attributes:
        nmo : int
            Number of molecular orbitals
        frozen : sequence of int or index mask array
            Identify orbitals which are frozen.
        nfrz_orb_idx : index mask array
            Identifies all nonredundant orbital rotation amplitudes for non-frozen orbitals
        uniq_orb_idx : index mask array
            The same as nfrz_orb_idx, but omitting active<->(inactive,virtual) degrees of freedom.
            (In the LASSCF child class uniq_orb_idx == nfrz_orb_idx.)
        ci_transformer : sequence of shape (nfrags,nroots) of :class:`CSFTransformer`
            Element [i][j] transforms between single determinants and CSFs for the ith fragment in
            the jth state
        nvar_orb : int
            Total number of nonredundant orbital-rotation degrees of freedom
        ncsf_sub : ndarray of shape (nfrags,nroots)
            Number of CSF vector elements in each fragment and state.
        nvar_tot : int
            Total length of the packed vector - approximately the number of nonredundant degrees
            of freedom (the CSF vector representation of the CI part of the problem still contains
            some redundancy even in `packed' form; fixing this is more trouble than it's worth).
    '''

    def __init__(self, las, mo_coeff, ci):
        self.nmo = mo_coeff.shape[-1]
        self.frozen = las.frozen
        self._init_orb (las, mo_coeff, ci)
        self._init_ci (las, mo_coeff, ci)

    def _init_nonfrozen_orb (self, las):
        nmo, ncore, ncas_sub = self.nmo, las.ncore, las.ncas_sub
        idx = all_nonredundant_idx (nmo, ncore, ncas_sub)
        if self.frozen is not None:
            idx[self.frozen,:] = idx[:,self.frozen] = False
        self.nfrz_orb_idx = idx

    def _init_orb (self, las, mo_coeff, ci):
        self._init_nonfrozen_orb (las)
        ncore, nocc = las.ncore, las.ncore + las.ncas
        idx = self.nfrz_orb_idx.copy ()
        idx[ncore:nocc,:ncore] = False # no inactive -> active
        idx[nocc:,ncore:nocc] = False # no active -> virtual
        # No external rotations of active orbitals
        self.uniq_orb_idx = idx

    def get_gx_idx (self):
        ''' Returns an index mask array identifying all nonredundant, nonfrozen orbital rotations
        which are not considered in the current phase of the phase of the problem:
        active<->inactive and active<->virtual for the LASCI parent class; nothing (all elements
        False) in the LASSCF child class. '''
        return np.logical_and (self.nfrz_orb_idx, np.logical_not (self.uniq_orb_idx))

    def _init_ci (self, las, mo_coeff, ci):
        self.ci_transformers = []
        for i, fcibox in enumerate (las.fciboxes):
            norb, nelec = las.ncas_sub[i], las.nelecas_sub[i]
            tf_list = []
            for j, solver in enumerate (fcibox.fcisolvers):
                solver.norb = norb
                solver.nelec = fcibox._get_nelec (solver, nelec)
                try:
                    solver.check_transformer_cache ()
                except ImpossibleCIvecError as e:
                    lib.logger.error (las, 'impossible CI vector in LAS frag %d, state %d', i, j)
                    raise (e)
                tf_list.append (solver.transformer)
            self.ci_transformers.append (tf_list)

    def pack (self, kappa, ci_sub):
        x = kappa[self.uniq_orb_idx]
        for trans_frag, ci_frag in zip (self.ci_transformers, ci_sub):
            for transformer, ci in zip (trans_frag, ci_frag):
                x = np.append (x, transformer.vec_det2csf (ci, normalize=False))
        assert (x.shape[0] == self.nvar_tot)
        return x

    def unpack (self, x):
        kappa = np.zeros ((self.nmo, self.nmo), dtype=x.dtype)
        kappa[self.uniq_orb_idx] = x[:self.nvar_orb]
        kappa = kappa - kappa.T

        y = x[self.nvar_orb:]
        ci_sub = []
        for trans_frag in self.ci_transformers:
            ci_frag = []
            for transformer in trans_frag:
                ncsf = transformer.ncsf
                ci_frag.append (transformer.vec_csf2det (y[:ncsf], normalize=False))
                y = y[ncsf:]
            ci_sub.append (ci_frag)

        return kappa, ci_sub

    def addr2idstr (self, addr):
        if addr<self.nvar_orb:
            probe_orb = np.argwhere (self.uniq_orb_idx)[addr]
            idstr = 'orb: {},{}'.format (*probe_orb)
        else:
            addr -= self.nvar_orb
            ncsf_frag = self.ncsf_sub.sum (1)
            for i, trans_frag in enumerate (self.ci_transformers):
                if addr >= ncsf_frag[i]:
                    addr -= ncsf_frag[i]
                    continue
                for j, trans in enumerate (trans_frag):
                    if addr >= trans.ncsf:
                        addr -= trans.ncsf
                        continue
                    idstr = 'CI({}): <{}|{}>'.format (
                        i, j, trans.printable_csfstring (addr))
                    break
                break
        return idstr

    @property
    def nvar_orb (self):
        return np.count_nonzero (self.uniq_orb_idx)

    @property
    def ncsf_sub (self):
        return np.asarray ([[transformer.ncsf for transformer in trans_frag]
                            for trans_frag in self.ci_transformers])

    @property
    def nvar_tot (self):
        return self.nvar_orb + self.ncsf_sub.sum ()

class LASCISymm_UnitaryGroupGenerators (LASCI_UnitaryGroupGenerators):
    __doc__ = LASCI_UnitaryGroupGenerators.__doc__ + '''

    Symmetry subclass forbids rotations between orbitals of different point groups or CSFs of
    other-than-specified point group -> sets many additional elements of nfrz_orb_idx and
    uniq_orb_idx to False and reduces the values of nvar_orb, ncsf_sub, and nvar_tot.
    '''

    def __init__(self, las, mo_coeff, ci): 
        self.nmo = mo_coeff.shape[-1]
        self.frozen = las.frozen
        if getattr (mo_coeff, 'orbsym', None) is None:
            mo_coeff = las.label_symmetry_(mo_coeff)
        orbsym = mo_coeff.orbsym
        self._init_orb (las, mo_coeff, ci, orbsym)
        self._init_ci (las, mo_coeff, ci, orbsym)
    
    def _init_orb (self, las, mo_coeff, ci, orbsym):
        LASCI_UnitaryGroupGenerators._init_orb (self, las, mo_coeff, ci)
        self.symm_forbid = (orbsym[:,None] ^ orbsym[None,:]).astype (np.bool_)
        self.uniq_orb_idx[self.symm_forbid] = False
        self.nfrz_orb_idx[self.symm_forbid] = False

    def _init_ci (self, las, mo_coeff, ci, orbsym):
        sub_slice = np.cumsum ([0] + las.ncas_sub.tolist ()) + las.ncore
        orbsym_sub = [orbsym[i:sub_slice[isub+1]] for isub, i in enumerate (sub_slice[:-1])]
        self.ci_transformers = []
        for norb, nelec, orbsym, fcibox in zip (las.ncas_sub, las.nelecas_sub, orbsym_sub,
                                                las.fciboxes):
            tf_list = []
            fcibox.orbsym = orbsym
            for solver in fcibox.fcisolvers:
                solver.norb = norb
                solver.nelec = fcibox._get_nelec (solver, nelec)
                solver.orbsym = orbsym
                solver.check_transformer_cache ()
                tf_list.append (solver.transformer)
            self.ci_transformers.append (tf_list)

def _init_df_(h_op):
    from mrh.my_pyscf.mcscf.lasci import _DFLASCI
    if isinstance (h_op.las, _DFLASCI):
        h_op.with_df = h_op.las.with_df
        if h_op.bPpj is None: h_op.bPpj = np.ascontiguousarray (
            h_op.las.cderi_ao2mo (h_op.mo_coeff, h_op.mo_coeff[:,:h_op.nocc],
            compact=False))

class LASCI_HessianOperator (sparse_linalg.LinearOperator):
    ''' The Hessian-vector product for a `LASCI' energy minimization, implemented as a linear
    operator from the scipy.sparse.linalg module. `LASCI' here means that the CAS is frozen
    relative to inactive or external orbitals, but active orbitals from different fragments may
    rotate into one another, and inactive orbitals may rotate into virtual orbitals, and CI vectors
    may also evolve. Implements the get_grad (gradient of the energy), get_prec (preconditioner for
    conjugate-gradient iteration), get_gx (gradient along non-`LASCI' degrees of freedom), and
    update_mo_ci_eri (apply a shift vector `x' to MO coefficients and CI vectors) in addition to
    _matvec and _rmatvec. For a shift vector `x', in terms of attributes and methods of this class,
    the second-order power series for the total (state-averaged) electronic energy is

    e = self.e_tot + np.dot (self.get_grad (), x) + (.5 * np.dot (self._matvec (x), x))

    Args:
        las : instance of :class:`LASCINoSymm`
        ugg : instance of :class:`LASCI_UnitaryGroupGenerators`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Molecular orbitals for trial state(s)
        ci : list (length = nfrags) of lists (length = nroots) of ndarrays
            CI vectors of the trial state(s); element [i][j] describes the ith fragment in the jth
            state
        casdm1frs : list of length (nfrags) of ndarrays
            ith element has shape (nroots,2,ncas_sub[i],ncas_sub[i])
            Contains spin-separated 1-RDMs for the active orbitals of each fragment in each state.
        casdm2fr : list of length (nfrags) of ndarrays
            ith element has shape [nroots,] + [ncas_sub[i],]*4
            Contains spin-summed 2-RDMs for the active orbitals of each fragment in each state.
        ncore : int
            Number of doubly-occupied inactive orbitals
        ncas_sub : list of length (nfrags)
            Number of active orbitals in each fragment
        nelecas_sub : list of list of length (2) of length (nfrags)
            Number of active electrons in each fragment
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices, where p1 is any MO
            and an is any active MO (in any fragment).
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        do_init_eri : logical
            If False, the bPpj attribute is not initialized until the _init_eri_ method is
            separately called.

    Attributes:
        ah_level_shift : float
            Shift added to the diagonal of the Hessian to improve convergence. Default = 1e-8.
        ncas : int
            Total number of active orbitals
        nao : int
            Total number of atomic orbitals
        nmo : int
            Total number of molecular orbitals
        nocc : int
            Total number of inactive plus active orbitals
        nroots : int
            Total number of states whose energies are averaged
        weights : list of length (nroots)
            Weights of the different states in the state average
        fciboxes : list of length (nfrags) of instances of :class:`H1EZipFCISolver`
            Contains the FCISolver objects for each fragment which implement the CI vector
            manipulation methods
        bPpj : ndarray of shape (naux,nmo,nocc)
            MO-basis CDERI array; only used in combination with density fitting. If
            do_init_eri=False is passed to the constructor
        casdm(N=1,2)[f][r][s] : ndarray or list of ndarrays
            Various 1RDMs (if N==1) or 2RDMs (if N==2) of active orbitals, obtained by summing or
            averaging over the casdm1frs and casdm2fr kwargs.
            If `f' is present, it is a list of ndarrays of length nfrags, and the last 2*N
            dimensions of the ith element are ncas_sub[i]. Otherwise, it is a single ndarray, and
            the last 2*N dimensions are ncas.
            If `r' is present, density matrices are separated by state and the first dimension of
            the ndarray(s) is nroots. Otherwise, density matrices are state-averaged.
            If 's' is present, density matrices are spin-separated and the first dimension of
            the ndarray(s) is 1+N. Otherwise, density matrices are spin-summed.
        cascm2 : ndarray of shape (ncas,ncas,ncas,ncas)
            The cumulant of the state-averaged, spin-summed 2-RDM of the active orbitals.
        dm1s : ndarray of shape (2,nmo,nmo)
            State-averaged, spin-separated 1-RDM of the whole molecule in the MO basis.
        eri_paaa : ndarray of shape (nmo, ncas, ncas, ncas)
            Same as kwarg h2eff_sub, be reshaped to be more accessible
        eri_cas : ndarray of shape [ncas,]*4
            ERIs (a1a2|a3a4)
        h1s : ndarray of shape (2,nmo,nmo)
            Spin-separated, state-averaged effective 1-electron Hamiltonian elements in MO basis
        h1s_cas : ndarray of shape (2,nmo,ncas)
            Spin-separated effective 1-electron Hamiltonian experience by the CAS, including the
            mean-field potential generated by the inactive electrons but not by any active space
        h1frs : list of length nroots of ndarray
            ith element has shape (nroots,2,ncas_sub[i],ncas_sub[i])
            Spin-separated effective 1-electron Hamiltonian experienced by each fragment in each
            state
        e_tot : float
            Total (state-averaged) electronic energy for the trial state(s) at x=0
        fock1 : ndarray of shape (nmo,nmo)
            State-averaged first-order effective Fock matrix
        hci0 : list (length = nfrags) of lists (length = nroots) of ndarrays
            (H(i,j) - e0[i][j]) |ci[i][j]>, where H(i,j) is the effective Hamiltonian experienced
            by the ith fragment in the jth state, stored as a CI vector
        e0 : list (length = nfrags) of lists (length = nroots) of floats
            <ci[i][j]|H(i,j)|ci[i][j]>, where H(i,j) is the effective Hamiltonian experienced by
            the ith fragment in the jth state
        linkstr[l] : list (length = nfrags) of lists (length = nroots)
            PySCF FCI module linkstr and linkstrl arrays, for accelerating CI manipulation
    '''

    def __init__(self, las, ugg, mo_coeff=None, ci=None, casdm1frs=None,
            casdm2fr=None, ncore=None, ncas_sub=None, nelecas_sub=None,
            h2eff_sub=None, veff=None, do_init_eri=True):
        if mo_coeff is None: mo_coeff = las.mo_coeff
        if ci is None: ci = las.ci
        if ncore is None: ncore = las.ncore
        if ncas_sub is None: ncas_sub = las.ncas_sub
        if nelecas_sub is None: nelecas_sub = las.nelecas_sub
        if casdm1frs is None: casdm1frs = las.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm2fr is None: casdm2fr = las.states_make_casdm2_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
        self.las = las
        self.ah_level_shift = las.ah_level_shift
        self.ugg = ugg
        self.mo_coeff = mo_coeff
        self.ci = ci = [[c.ravel () for c in cr] for cr in ci] 
        self.ncore = ncore
        self.ncas_sub = ncas_sub
        self.nelecas_sub = nelecas_sub
        self.ncas = ncas = sum (ncas_sub)
        self.nao = nao = mo_coeff.shape[0]
        self.nmo = nmo = mo_coeff.shape[-1]
        self.nocc = nocc = ncore + ncas
        self.fciboxes = las.fciboxes
        self.nroots = las.nroots
        self.weights = las.weights
        self.bPpj = None

        self._init_dms_(casdm1frs, casdm2fr)
        self._init_ham_(h2eff_sub, veff)
        self._init_orb_()
        self._init_ci_()
        # turn this off for extra optimization in kernel
        if do_init_eri: self._init_eri_()

    def _init_dms_(self, casdm1frs, casdm2fr):
        las, ncore, nocc = self.las, self.ncore, self.nocc
        self.casdm1frs = casdm1frs 
        self.casdm1fs = las.make_casdm1s_sub (casdm1frs=self.casdm1frs)
        self.casdm1rs = las.states_make_casdm1s (casdm1frs=self.casdm1frs)
        self.casdm2fr = casdm2fr
        casdm1a = linalg.block_diag (*[dm[0] for dm in self.casdm1fs])
        casdm1b = linalg.block_diag (*[dm[1] for dm in self.casdm1fs])
        self.casdm1s = np.stack ([casdm1a, casdm1b], axis=0)
        casdm1 = self.casdm1s.sum (0)
        self.casdm2 = las.make_casdm2 (casdm1frs=casdm1frs, casdm2fr=casdm2fr)
        self.cascm2 = self.casdm2 - np.multiply.outer (casdm1, casdm1)
        self.cascm2 += np.multiply.outer (casdm1a, casdm1a).transpose (0,3,2,1)
        self.cascm2 += np.multiply.outer (casdm1b, casdm1b).transpose (0,3,2,1)
        self.dm1s = np.stack ([np.eye (self.nmo, dtype=self.dtype),
                               np.eye (self.nmo, dtype=self.dtype)], axis=0)
        self.dm1s[0,ncore:nocc,ncore:nocc] = casdm1a
        self.dm1s[1,ncore:nocc,ncore:nocc] = casdm1b
        self.dm1s[:,nocc:,nocc:] = 0
        
    def _init_ham_(self, h2eff_sub, veff):
        las, mo_coeff, ncas_sub = self.las, self.mo_coeff, self.ncas_sub
        ncore, ncas, nocc = self.ncore, self.ncas, self.nocc
        nao, nmo, nocc = self.nao, self.nmo, ncore+ncas
        casdm1a, casdm1b = tuple (self.casdm1s)
        casdm1 = casdm1a + casdm1b
        moH_coeff = mo_coeff.conjugate ().T
        if veff is None:
            from mrh.my_pyscf.mcscf.lasci import _DFLASCI 
            if isinstance (las, _DFLASCI):
                _init_df_(self)
                # Can't use this module's get_veff because here I need to have f_aa and f_ii
                # On the other hand, I know that dm1s spans only the occupied orbitals
                rho = np.tensordot (self.bPpj[:,:nocc,:], self.dm1s[:,:nocc,:nocc].sum (0))
                vj_ao = np.zeros (nao*(nao+1)//2, dtype=rho.dtype)
                b0 = 0
                for eri1 in self.with_df.loop ():
                    b1 = b0 + eri1.shape[0]
                    vj_ao += np.dot (rho[b0:b1], eri1)
                    b0 = b1
                vj_mo = moH_coeff @ lib.unpack_tril (vj_ao) @ mo_coeff
                vPpi = self.bPpj[:,:,:ncore] * np.sqrt (2.0)
                no_occ, no_coeff = linalg.eigh (casdm1)
                no_occ[no_occ<0] = 0.0
                no_coeff *= np.sqrt (no_occ)[None,:]
                vPpu = np.dot (self.bPpj[:,:,ncore:nocc], no_coeff)
                vPpj = np.append (vPpi, vPpu, axis=2)
                vk_mo = np.tensordot (vPpj, vPpj, axes=((0,2),(0,2)))
                smo = las._scf.get_ovlp () @ mo_coeff
                smoH = smo.conjugate ().T
                veff = smo @ (vj_mo - vk_mo/2) @ smoH
            else:
                veff = las.get_veff (dm1s = np.dot (mo_coeff, 
                                                    np.dot (self.dm1s.sum (0), moH_coeff)))
            veff = las.split_veff (veff, h2eff_sub, mo_coeff=mo_coeff, casdm1s_sub=self.casdm1fs)
        self.eri_paaa = eri_paaa = lib.numpy_helper.unpack_tril (
            h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)).reshape (nmo, ncas,
            ncas, ncas)
        self.eri_cas = eri_cas = eri_paaa[ncore:nocc,:,:,:]
        h1s = las.get_hcore ()[None,:,:] + veff
        h1s = np.dot (h1s, mo_coeff)
        self.h1s = np.dot (moH_coeff, h1s).transpose (1,0,2)
        self.h1s_cas = self.h1s[:,:,ncore:nocc].copy ()
        self.h1s_cas -= np.tensordot (eri_paaa, casdm1, axes=2)[None,:,:]
        self.h1s_cas += np.tensordot (self.casdm1s, eri_paaa, axes=((1,2),(2,1)))

        self.h1frs = [np.zeros ((self.nroots, 2, nlas, nlas)) for nlas in ncas_sub]
        for ix, h1rs in enumerate (self.h1frs):
            i = sum (ncas_sub[:ix])
            j = i + ncas_sub[ix]
            k, l = i + ncore, j + ncore
            for h1s_sub, casdm1s in zip (h1rs, self.casdm1rs):
                h1s_sub[:,:,:] = self.h1s[:,k:l,k:l].copy ()
                dm1s = casdm1s.copy ()
                dm1s[:,i:j,i:j] = 0.0 # No double-counting
                dm1s[0] -= casdm1a # No state-averaging
                dm1s[1] -= casdm1b # No state-averaging
                dm1 = dm1s[0] + dm1s[1]
                h1s_sub[:,:,:] += np.tensordot (dm1, eri_cas, axes=((0,1),(2,3)))[None,i:j,i:j]
                h1s_sub[:,:,:] -= np.tensordot (dm1s, eri_cas, axes=((1,2),(2,1)))[:,i:j,i:j]

        # Total energy (for callback)
        h1 = (self.h1s + (moH_coeff @ las.get_hcore () @ mo_coeff)[None,:,:]) / 2
        self.e_tot = (las.energy_nuc ()
            + np.dot (h1.ravel (), self.dm1s.ravel ())
            + np.tensordot (self.eri_cas, self.cascm2, axes=4) / 2)

    def _init_orb_(self):
        eri_paaa, ncore, nocc = self.eri_paaa, self.ncore, self.nocc
        self.fock1 = sum ([f @ d for f,d in zip (list (self.h1s), list (self.dm1s))])
        self.fock1[:,ncore:nocc] += np.tensordot (eri_paaa, self.cascm2, axes=((1,2,3),(1,2,3)))

    def _init_ci_(self):
        ci, ncas_sub, nelecas_sub = self.ci, self.ncas_sub, self.nelecas_sub
        self.linkstrl = []
        self.linkstr = []
        for fcibox, no, ne in zip (self.fciboxes, ncas_sub, nelecas_sub):
            self.linkstrl.append (fcibox.states_gen_linkstr (no, ne, True)) 
            self.linkstr.append (fcibox.states_gen_linkstr (no, ne, False))
        self.hci0 = self.Hci_all (None, self.h1frs, self.eri_cas, ci)
        self.e0 = [[hc.dot (c) for hc, c in zip (hcr, cr)] for hcr, cr in zip (self.hci0, ci)]
        self.hci0 = [[hc - c*e for hc, c, e in zip (hcr, cr, er)]
                     for hcr, cr, er in zip (self.hci0, ci, self.e0)]

    _init_eri_ = _init_df_

    @property
    def dtype (self):
        return self.mo_coeff.dtype

    @property
    def shape (self):
        return ((self.ugg.nvar_tot, self.ugg.nvar_tot))

    def Hci (self, fcibox, no, ne, h0r, h1rs, h2, ci, linkstrl=None):
        ''' For a single fragment, evaluate the FCI operation H(i)|ci[i]>, where H(i) is the
        effective Hamiltonian experienced by the fragment in the ith state

        Args:
            fcibox : instance of :class:`H1EZipFCISolver`
                The FCI solver method for the fragment
            no : integer
                Number of active orbitals in the fragment
            ne : list of length (2) of integers
                Number of spin-up and spin-down electrons in the fragment
            h0r : list of length nroots
                Constant part of the effective Hamiltonian for each state
            h1rs : ndarray of shape (nroots,2,no,no)
                Spin-separated 1-electron part of the effective Hamiltonian for each state
            h2 : ndarray of shape (no,no,no,no)
                Two-electron integrals
            ci : list of length nroots of ndarray
                CI vectors

        Kwargs:
            linkstrl : see pyscf.fci module documentation

        Returns:
            hcr : list of length nroots of ndarray
        '''
        hr = fcibox.states_absorb_h1e (h1rs, h2, no, ne, 0.5)
        hcr = fcibox.states_contract_2e (hr, ci, no, ne, link_index=linkstrl)
        hcr = [hc + (h0 * c) for hc, h0, c in zip (hcr, h0r, ci)]
        return hcr

    def Hci_all (self, h0fr, h1frs, h2, ci_sub):
        ''' For all fragments, evaluate the FCI operations H(i,j)|ci_sub[i][j]>, where H(i,j) is
        the effective Hamiltonian experienced by the ith fragment in the jth state.

        Args:
            h0fr : list of length nfrags of lists of length nroots
                Constant part of the effective Hamiltonian for each fragment and state
            h1frs : list of length nfrags of ndarrays
                Spin-separated 1-electron parts of the effective Hamiltonian for each fragment and
                state
            h2 : ndarray of shape (ncas,ncas,ncas,ncas)
                Two-electron integrals spanning the entire active space
            ci_sub : list of length nfrags of list of length nroots of ndarray
                CI vectors

        Returns:
            hcfr : list of length nfrags of list of length nroots of ndarray
        '''
        if h0fr is None: h0fr = [[0.0 for h1r in h1rs] for h1rs in h1frs]
        hcfr = []
        for isub, (fcibox, h0, h1rs, ci) in enumerate (zip (self.fciboxes, h0fr, h1frs, ci_sub)):
            if self.linkstrl is not None: linkstrl = self.linkstrl[isub] 
            ncas = self.ncas_sub[isub]
            nelecas = self.nelecas_sub[isub]
            i = sum (self.ncas_sub[:isub])
            j = i + ncas
            h2_i = h2[i:j,i:j,i:j,i:j]
            h1rs_i = h1rs
            hcfr.append (self.Hci (fcibox, ncas, nelecas, h0, h1rs_i, h2_i, ci, linkstrl=linkstrl))
        return hcfr

    def make_tdm1s2c_sub (self, ci1):
        ''' Make effective 1-body and 2-body cumulant density matrices to first order
        in a CI rotation vector. 

        Args:
            ci : list (length = nfrags) of lists (length = nroots) of ndarrays
                CI shift vectors

        Returns:
            tdm1s : ndarray of shape (nroots,2,ncas,ncas)
                Spin-separated effective 1-body density matrix
            tcm2 : ndarray of shape (ncas,ncas,ncas,ncas)
                Spin-summed state-averaged cumulant effective 2-body density matrix
        '''
        tdm1rs = np.zeros ((self.nroots, 2, self.ncas, self.ncas), dtype=self.dtype)
        tcm2 = np.zeros ([self.ncas,]*4, dtype=self.dtype)
        for isub, (fcibox, ncas, nelecas, c1, c0, casdm1rs, casdm1s, casdm2r) in enumerate (
          zip (self.fciboxes, self.ncas_sub, self.nelecas_sub, ci1, self.ci,
          self.casdm1frs, self.casdm1fs, self.casdm2fr)):
            s01 = [c1i.dot (c0i) for c1i, c0i in zip (c1, c0)]
            i = sum (self.ncas_sub[:isub])
            j = i + ncas
            linkstr = None if self.linkstr is None else self.linkstr[isub]
            dm1, dm2 = fcibox.states_trans_rdm12s (c1, c0, ncas, nelecas, link_index=linkstr)
            # Subtrahend: super important, otherwise the veff part of CI response is even worse
            # With this in place, I don't have to worry about subtracting an overlap times gradient
            tdm1rs[:,:,i:j,i:j] = np.stack ([np.stack (t, axis=0) - c * s
                                             for t, c, s in zip (dm1, casdm1rs, s01)], axis=0)
            dm2 = np.stack ([(sum (t) - (c*s)) / 2
                             for t, c, s, in zip (dm2, casdm2r, s01)], axis=0)
            dm2 = np.einsum ('rijkl,r->ijkl', dm2, fcibox.weights)
            #tdm1frs[isub,:,:,i:j,i:j] = tdm1rs 
            tcm2[i:j,i:j,i:j,i:j] = dm2

        # Cumulant decomposition so I only have to do one jk call for orbrot response
        # The only rules are 1) the sectors that you think are zero must really be zero, and
        #                    2) you subtract here what you add later
        tdm1s = np.einsum ('r,rspq->spq', self.weights, tdm1rs)
        cdm1s = np.einsum ('r,rsqp->spq', self.weights, self.casdm1rs)
        tcm2 -= np.multiply.outer (tdm1s[0] + tdm1s[1], cdm1s[0] + cdm1s[1])
        tcm2 += np.multiply.outer (tdm1s[0], cdm1s[0]).transpose (0,3,2,1)
        tcm2 += np.multiply.outer (tdm1s[1], cdm1s[1]).transpose (0,3,2,1)

        # Two transposes 
        tdm1rs += tdm1rs.transpose (0,1,3,2) 
        tcm2 += tcm2.transpose (1,0,3,2)        
        tcm2 += tcm2.transpose (2,3,0,1)        

        return tdm1rs, tcm2    

    def get_veff_Heff (self, odm1s, tdm1rs):
        ''' Compute first-order effective potential (relevant to the orbital-rotation sector of the
        Hessian-vector product) and first-order effective 1-body Hamiltonian operator (relevant to
        the CI-rotation sector of the Hessian-vector product) from first-order effective density
        matrices. "First-order" means proportional to one power of the orbital/CI-rotation step
        vector.

        Args:
            odm1s : ndarray of shape (2,nmo,nmo)
                Pre-symmetrization effective spin-separated state-averaged 1-RDM from the orbital
                rotation part of the step vector
            tdm1rs : ndarray of shape (nroots,2,ncas,ncas)
                Effective spin-separated 1-RDMs from the CI rotation part of the step vector,
                separated by root

        Returns:
            veff_mo : ndarray of shape (nmo,nmo)
                Spin-symmetric effective 1-body potential, including the effects of both the
                orbital and CI parts of the step vector
            h1frs : list of length nfrags of ndarrays
                ith element has shape (nroots,2,ncas_sub[i],ncas_sub[i]).
                Effective one-electron Hamiltonian amplitudes for each state and fragment + h.c.
                Includes the effects of orbital rotation and CI rotation of other fragments (i.e.,
                omits double-counting).
        '''

        ncore, nocc, nroots = self.ncore, self.nocc, self.nroots
        tdm1s_sa = np.einsum ('rspq,r->spq', tdm1rs, self.weights)
        dm1s_mo = odm1s + odm1s.transpose (0,2,1)
        dm1s_mo[:,ncore:nocc,ncore:nocc] += tdm1s_sa
        mo = self.mo_coeff
        moH = mo.conjugate ().T

        # Overall veff for gradient: the one and only jk call per microcycle that I will allow.
        veff_mo = self.get_veff (dm1s_mo=dm1s_mo)
        veff_mo = self.split_veff (veff_mo, dm1s_mo)

        # Core-orbital-effect only for individual CI problems
        odm1s_core = np.copy (odm1s)
        odm1s_core[:,ncore:nocc,:] = 0.0
        odm1s_core += odm1s_core.transpose (0,2,1)
        err_dm1s = odm1s_core - dm1s_mo
        # Deal with nonsymmetric eri: Coulomb part
        err_dm1s = err_dm1s[:,:,ncore:nocc] * 2.0
        err_dm1s[:,ncore:nocc,:] /= 2.0
        veff_ci = np.tensordot (err_dm1s, self.eri_paaa, axes=2)
        veff_ci += veff_ci[::-1,:,:]
        veff_ci -= np.tensordot (err_dm1s, self.eri_paaa, axes=((1,2),(0,3)))
        # Deal with nonsymmetric eri: exchange part
        veff_ci += veff_ci.transpose (0,2,1)
        veff_ci /= 2.0
        veff_ci += veff_mo[:,ncore:nocc,ncore:nocc]
        
        # SO, individual CI problems!
        # 1) There is NO constant term. Constant terms immediately drop out via the ugg defs!
        # 2) veff_ci is correctfor the orbrots, so long as I don't explicitly add h.c. at the end
        # 3) If I don't add h.c., then the (non-self) mf effect of the 1-tdms needs to be 2x strong
        # 4) Of course, self-interaction (from both 1-odms and 1-tdms) needs to be eliminated
        # 5) I do the latter by copying the eris, rather than the tdms, in case nroots is large
        h1frs = [np.zeros ((nroots, 2, nlas, nlas), dtype=self.dtype) for nlas in self.ncas_sub]
        eri_tmp = self.eri_cas.copy ()
        for isub, nlas in enumerate (self.ncas_sub):
            i = sum (self.ncas_sub[:isub])
            j = i + nlas
            h1frs[isub][:,:,:,:] = veff_ci[None,:,i:j,i:j]
            eri_tmp[:,:,:,:] = self.eri_cas[:,:,:,:]
            eri_tmp[i:j,i:j,:,:] = 0.0
            err_h1rs = 2.0 * np.tensordot (tdm1rs, eri_tmp, axes=2) 
            err_h1rs += err_h1rs[:,::-1] # ja + jb
            eri_tmp[:,:,:,:] = self.eri_cas[:,:,:,:]
            eri_tmp[i:j,:,:,i:j] = 0.0
            err_h1rs -= 2.0 * np.tensordot (tdm1rs, eri_tmp, axes=((2,3),(0,3)))
            #err_dm1rs = 2 * (tdm1frs.sum (0) - tdm1rs)
            #err_h1rs = np.tensordot (err_dm1rs, self.eri_cas, axes=2)
            #err_h1rs += err_h1rs[:,::-1] # ja + jb
            #err_h1rs -= np.tensordot (err_dm1rs, self.eri_cas, axes=((2,3),(0,3)))
            h1frs[isub][:,:,:,:] += err_h1rs[:,:,i:j,i:j]

        return veff_mo, h1frs

    def get_veff (self, dm1s_mo=None):
        '''THIS FUNCTION IS OVERWRITTEN WITH A CALL TO LAS.GET_VEFF IN THE LASSCF_O0 CLASS. IT IS
        ONLY RELEVANT TO THE "LASCI" STEP OF THE OLDER, DEPRECATED, DMET-BASED ALGORITHM.

        Compute the effective potential from a 1-RDM in the MO basis (presumptively the first-order
        effective 1-RDM which is proportional to a step vector in MO and CI rotation coordinates).
        If density fitting is used, the effective potential is approximate: it omits the
        unoccupied-unoccupied lower-diagonal block.

        Kwargs:
            dm1s_mo : ndarray of shape (2,nmo,nmo)
                Contains spin-separated 1-RDM

        Returns:
            veff_mo : ndarray of shape (nmo,nmo)
                Spin-symmetric effective potential in the MO basis
        '''
        mo = self.mo_coeff
        moH = mo.conjugate ().T
        nmo = mo.shape[-1]
        dm1_mo = dm1s_mo.sum (0)
        if getattr (self, 'bPpj', None) is None:
            dm1_ao = np.dot (mo, np.dot (dm1_mo, moH))
            veff_ao = np.squeeze (self.las.get_veff (dm1s=dm1_ao))
            return np.dot (moH, np.dot (veff_ao, mo)) 
        ncore, nocc, ncas = self.ncore, self.nocc, self.ncas
        # vj
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        veff_mo = np.zeros_like (dm1_mo)
        dm1_rect = dm1_mo + dm1_mo.T
        dm1_rect[ncore:nocc,ncore:nocc] /= 2
        dm1_rect = dm1_rect[:,:nocc]
        rho = np.tensordot (self.bPpj, dm1_rect, axes=2)
        vj_pj = np.tensordot (rho, self.bPpj, axes=((0),(0)))
        t1 = lib.logger.timer (self.las, 'vj_mo in microcycle', *t0)
        dm_bj = dm1_mo[ncore:,:nocc]
        vPpj = np.ascontiguousarray (self.las.cderi_ao2mo (mo, mo[:,ncore:]@dm_bj, compact=False))
        # Don't ask my why this is faster than doing the two degrees of freedom separately...
        t1 = lib.logger.timer (self.las, 'vk_mo vPpj in microcycle', *t1)
        # vk (aa|ii), (uv|xy), (ua|iv), (au|vi)
        vPbj = vPpj[:,ncore:,:] #np.dot (self.bPpq[:,ncore:,ncore:], dm_ai)
        vk_bj = np.tensordot (vPbj, self.bPpj[:,:nocc,:], axes=((0,2),(0,1)))
        t1 = lib.logger.timer (self.las, 'vk_mo (bb|jj) in microcycle', *t1)
        # vk (ai|ai), (ui|av)
        dm_ai = dm1_mo[nocc:,:ncore]
        vPji = vPpj[:,:nocc,:ncore] #np.dot (self.bPpq[:,:nocc, nocc:], dm_ai)
        # I think this works only because there is no dm_ui in this case, so I've eliminated all
        # the dm_uv by choosing this range
        bPbi = self.bPpj[:,ncore:,:ncore]
        vk_bj += np.tensordot (bPbi, vPji, axes=((0,2),(0,2)))
        t1 = lib.logger.timer (self.las, 'vk_mo (bi|aj) in microcycle', *t1)
        # veff
        vj_bj = vj_pj[ncore:,:]
        veff_mo[ncore:,:nocc] = vj_bj - 0.5*vk_bj
        veff_mo[:nocc,ncore:] = veff_mo[ncore:,:nocc].T
        #vj_ai = vj_bj[ncas:,:ncore]
        #vk_ai = vk_bj[ncas:,:ncore]
        #veff_mo[ncore:,:nocc] = vj_bj
        #veff_mo[:ncore,nocc:] = vj_ai.T
        #veff_mo[ncore:,:nocc] -= vk_bj/2
        #veff_mo[:ncore,nocc:] -= vk_ai.T/2
        return veff_mo

    def split_veff (self, veff_mo, dm1s_mo):
        # This function seems orphaned? Is it used anywhere?
        veff_c = veff_mo.copy ()
        ncore = self.ncore
        nocc = self.nocc
        dm1s_cas = dm1s_mo[:,ncore:nocc,ncore:nocc]
        sdm = dm1s_cas[0] - dm1s_cas[1]
        vk_aa = -np.tensordot (self.eri_cas, sdm, axes=((1,2),(0,1))) / 2
        veff_s = np.zeros_like (veff_c)
        veff_s[ncore:nocc, ncore:nocc] = vk_aa
        veffa = veff_c + veff_s
        veffb = veff_c - veff_s
        return np.stack ([veffa, veffb], axis=0)

    def _matvec (self, x):
        log = lib.logger.new_logger (self.las, self.las.verbose)
        extra_timing = getattr (self.las, '_extra_hessian_timing', False)
        extra_timer = log.timer if extra_timing else log.timer_debug1
        t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        kappa1, ci1 = self.ugg.unpack (x)
        t1 = extra_timer ('LASCI sync Hessian operator 1: unpack', *t0)

        # Effective density matrices, veffs, and overlaps from linear response
        odm1s = -np.dot (self.dm1s, kappa1)
        ocm2 = -np.dot (self.cascm2, kappa1[self.ncore:self.nocc])
        tdm1rs, tcm2 = self.make_tdm1s2c_sub (ci1)
        t1 = extra_timer ('LASCI sync Hessian operator 2: effective density matrices', *t1)
        veff_prime, h1s_prime = self.get_veff_Heff (odm1s, tdm1rs)
        t1 = extra_timer ('LASCI sync Hessian operator 3: effective potentials', *t1)

        # Responses!
        kappa2 = self.orbital_response (kappa1, odm1s, ocm2, tdm1rs, tcm2, veff_prime)
        t1 = extra_timer ('LASCI sync Hessian operator 4: (Hx)_orb', *t1)
        ci2 = self.ci_response_offdiag (kappa1, h1s_prime)
        t1 = extra_timer ('LASCI sync Hessian operator 5: (Hx)_CI offdiag', *t1)
        ci2 = [[x+y for x,y in zip (xr, yr)] for xr, yr in zip (ci2, self.ci_response_diag (ci1))]
        t1 = extra_timer ('LASCI sync Hessian operator 6: (Hx)_CI diag', *t1)

        # LEVEL SHIFT!!
        kappa3, ci3 = self.ugg.unpack (self.ah_level_shift * np.abs (x))
        kappa2 += kappa3
        ci2 = [[x+y for x,y in zip (xr, yr)] for xr, yr in zip (ci2, ci3)]
        t1 = extra_timer ('LASCI sync Hessian operator 7: level shift', *t1)

        Hx = self.ugg.pack (kappa2, ci2)
        t1 = extra_timer ('LASCI sync Hessian operator 8: pack', *t1)
        t0 = log.timer ('LASCI sync Hessian operator total', *t0)
        return Hx

    _rmatvec = _matvec # Hessian is Hermitian in this context!

    def orbital_response (self, kappa, odm1s, ocm2, tdm1rs, tcm2, veff_prime):
        '''Compute the orbital-response sector of the Hessian-vector product. It's conceptually
        pretty simple:

        Hx_pq = F'_pq - F'_qp + .5*(F_pr k_rq - k_pr F_rq)
        F'_pq = h_pr D'_qr + g_prst d'_qrst

        Since we use the cumulant decomposition:

        d'_pqrs = l'_pqrs + D'_pq D_rs + D_pq D'_rs
                  - .5*(D[s]'_ps D[s]_qr + D[s]_ps D[s]'_qr)

        where [s] means spin index, we find that

        F'_pq = h_pr D_qr + veff[s]_pr D'[s]_qr + veff'[s]_pr D[s]_qr + g_prst l'_qrst

        where veff is the effective potential from the zeroth-order 1-RDMs and veff' is that from
        the first-order 1-RDMs.

        Args:
            kappa : ndarray of shape (nmo,nmo)
                Unpacked orbital-rotation step vector
            odm1s : ndarray of shape (2,nmo,nmo)
                Pre-symmetrization effective spin-separated state-averaged 1-RDM from the orbital
                rotation part of the step vector
            ocm2 : ndarray of shape (ncas,ncas,ncas,nmo)
                Pre-symmetrization spin-summed state-averaged effective cumulant of the 2-RDM
                from the orbital-rotation part of the step vector
            tdm1rs : ndarray of shape (nroots,2,ncas,ncas)
                Effective spin-separated 1-RDMs from the CI rotation part of the step vector,
                separated by root
            tcm2 : ndarray of shape (ncas,ncas,ncas,nncas)
                Spin-summed state-averaged effective cumulant of the 2-RDM from the CI rotation
                part of the step vector
            veff_prime : ndarray of shape (2,nmo,nmo)
                Spin-separated state-averaged effective potential proportional to the first power
                of the step vector (all sectors)

        Returns:
            kappa2 : ndarray of shape (nmo,nmo)
                Contains the unpacked orbital-rotation sector of the Hessian-vector product.
        '''
        ncore, nocc = self.ncore, self.nocc
        # I put off + h.c. until now in order to make other things more natural
        odm1s += odm1s.transpose (0,2,1)
        ocm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)
        ocm2 += ocm2.transpose (2,3,0,1)
        # Effective density matrices
        edm1s = odm1s
        edm1s[:,ncore:nocc,ncore:nocc] += np.einsum ('rspq,r->spq', tdm1rs, self.weights)
        ecm2 = ocm2 + tcm2
        # Evaluate hx = (F2..x) - (F2..x).T + (F1.x) - (F1.x).T
        fock1  = self.h1s[0] @ edm1s[0] + self.h1s[1] @ edm1s[1]
        fock1 += veff_prime[0] @ self.dm1s[0] + veff_prime[1] @ self.dm1s[1]
        fock1[ncore:nocc,ncore:nocc] += np.tensordot (self.eri_cas, ecm2, axes=((1,2,3),(1,2,3)))
        fock1 += (np.dot (self.fock1, kappa) - np.dot (kappa, self.fock1)) / 2
        return fock1 - fock1.T

    def ci_response_offdiag (self, kappa1, h1frs_prime):
        '''Compute part of the CI rotation sector of the Hessian-vector product corresponding
        to off-diagonal blocks of the Hessian matrix; i.e., for a given fragment block of the
        Hessian-vector product, the input step vector omits CI degrees of freedom of that fragment,
        but includes CI degrees of freedom for all other fragments as well as orbital-rotation
        degrees of freedom.

        Args:
            kappa1 : ndarray of shape (nmo,nmo)
                Unpacked orbital-rotation step vector
            h1frs : list of length nfrags of ndarrays
                ith element has shape (nroots,2,ncas_sub[i],ncas_sub[i]).
                Effective one-electron Hamiltonian amplitudes for each state and fragment + h.c.
                Includes the effects of orbital rotation and CI rotation of other fragments (i.e.,
                omits double-counting).

        Returns:
            Kci0 : list (length = nfrags) of lists (length = nroots) of ndarrays
                Contains unpacked CI sector of partial Hessian-vector product
        '''
        # Since h1frs contains + h.c., I do NOT explicitly add + h.c. in this function
        ncore, nocc, ncas_sub, nroots = self.ncore, self.nocc, self.ncas_sub, self.nroots
        kappa1_cas = kappa1[ncore:nocc,:]
        h1frs = [np.zeros_like (h1) for h1 in h1frs_prime]
        h1_core = -np.tensordot (kappa1_cas, self.h1s_cas, axes=((1),(1))).transpose (1,0,2)
        h1_core += h1_core.transpose (0,2,1)
        h2 = -np.tensordot (kappa1_cas, self.eri_paaa, axes=1)
        h2 += h2.transpose (2,3,0,1)
        h2 += h2.transpose (1,0,3,2)
        # ^ h2 should also include + h.c.
        for j, casdm1s in enumerate (self.casdm1rs):
            for i, (h1rs, h1rs_prime) in enumerate (zip (h1frs, h1frs_prime)):
                k = sum (ncas_sub[:i])
                l = k + ncas_sub[i]
                h1s, h1s_prime = h1rs[j], h1rs_prime[j]
                dm1s = casdm1s.copy ()
                dm1s[:,k:l,k:l] = 0.0 # no double-counting
                dm1 = dm1s.sum (0)
                h1s[:,:,:] = h1_core[:,k:l,k:l].copy ()
                h1s[:,:,:] += np.tensordot (h2, dm1, axes=2)[None,k:l,k:l]
                h1s[:,:,:] -= np.tensordot (dm1s, h2, axes=((1,2),(2,1)))[:,k:l,k:l]
                #h1s[:,:,:] += h1s.transpose (0,2,1)
                h1s[:,:,:] += h1s_prime[:,:,:]
        Kci0 = self.Hci_all (None, h1frs, h2, self.ci)
        Kci0 = [[Kc - c*(c.dot (Kc)) for Kc, c in zip (Kcr, cr)]
                for Kcr, cr in zip (Kci0, self.ci)]
        # ^ The definition of the unitary group generator compels you to do this always!!!
        return Kci0

    def ci_response_diag (self, ci1):
        '''Compute part of the CI response sector of the Hessian-vector product corresponding
        to diagonal blocks of the Hessian matrix; i.e., for a given fragment block of the Hessian-
        vector product, the input step vector includes ONLY CI degrees of freedom for THAT
        FRAGMENT.

        Args:
            ci1 : list (length = nfrags) of lists (length = nroots) of ndarrays
                Contains unpacked CI sector of input step vector

        Returns:
            ci2 : list (length = nfrags) of lists (length = nroots) of ndarrays
                Contains unpacked CI sector of partial Hessian-vector product
        '''
        # IMPORTANT: this disagrees with PySCF, but I still think it's right and PySCF is wrong
        ci1HmEci0 = [[c.dot (Hci) for c, Hci in zip (cr, Hcir)] 
                     for cr, Hcir in zip (ci1, self.hci0)]
        s01 = [[c1.dot (c0) for c1,c0 in zip (c1r, c0r)] for c1r, c0r in zip (ci1, self.ci)]
        ci2 = self.Hci_all ([[-e for e in er] for er in self.e0], self.h1frs, self.eri_cas, ci1)
        ci2 = [[x-(y*z) for x,y,z in zip (xr,yr,zr)] for xr,yr,zr in zip (ci2, self.ci, ci1HmEci0)]
        ci2 = [[x-(y*z) for x,y,z in zip (xr,yr,zr)] for xr,yr,zr in zip (ci2, self.hci0, s01)]
        return [[x*2 for x in xr] for xr in ci2]

    def get_prec (self):
        '''Obtain the preconditioner for conjugate-gradient descent using a second-order power
        series of the energy from a given LAS-state keyframe (a single "macrocycle"). In general,
        the preconditioner should approximate multiplication by the matrix-inverse of the Hessian.
        Here, however, we also use it to identify and mask degrees of freedom along which the
        quadratically-approximated energy is numerically unstable.

        N.B. to future developers: an "exact" inverted-Hessian preconditioner is actually not
        desirable, because a failure of optimization is more likely due to the unsuitability of a
        quadratic power series in fundamentally periodic variables. I.O.W., we can't get too hung
        up on solving Ax=b, because Ax=b is an approximate equation in the first place. The actual
        goal is to minimize successive keyframe (aka "macrocycle" aka "trial") energies.

        Returns:
            prec_op : LinearOperator
                Approximately the inverse of the Hessian
        '''
        log = lib.logger.new_logger (self.las, self.las.verbose)
        Hdiag = self._get_Hdiag () + self.ah_level_shift
        Hdiag[np.abs (Hdiag)<1e-8] = 1e-8
        # The quadratic power series is a bad approximation if the magnitude of the gradient in
        # the current keyframe is such that we will tend to predict steps with magnitude greater
        # than .5*pi (a step of exactly .5*pi transposes two states). This preconditioner should
        # mask out the corresponding degrees of freedom
        g_vec = self.get_grad ()
        b = linalg.norm (g_vec)
        probe_x0 = b/Hdiag
        log.debug ('|probe_x0| / ndeg = %g', linalg.norm (probe_x0) / len (probe_x0))
        ndeg = len (probe_x0)
        idx_unstable = np.abs (probe_x0) > np.pi*.5
        # We can't mask everything, because that behavior would obfuscate the problem
        # If NO stable D.O.F. exist, then keyframe is just bad and it has to be handled upstream
        ndeg_unstable = np.count_nonzero (idx_unstable)
        ndeg_stable = np.count_nonzero (~idx_unstable)
        g_unst = linalg.norm (g_vec[idx_unstable]) if ndeg_unstable else 0
        if ndeg_stable and (round (g_unst/b, 2) < 1):
            Hdiag[idx_unstable] = np.inf
            ndeg_unstable = ndeg - ndeg_stable
            log.debug ('%d/%d d.o.f. masked in LASCI sync preconditioner (masked gradient = %g)',
                       ndeg_unstable, ndeg, g_unst)
        else:
            log.warn ('LASCI encountered an unmaskable instability; calculation may not converge')
        def prec_op (x):
            t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
            Mx = x/Hdiag
            log.timer ('LASCI sync preconditioner call', *t0)
            return Mx
        return sparse_linalg.LinearOperator (self.shape,matvec=prec_op,dtype=self.dtype)

    def _get_Horb_diag (self):
        fock = np.stack ([np.diag (h) for h in list (self.h1s)], axis=0)
        num = np.stack ([np.diag (d) for d in list (self.dm1s)], axis=0)
        Horb_diag = sum ([np.multiply.outer (f,n) for f,n in zip (fock, num)])
        Horb_diag -= np.diag (self.fock1)[None,:]
        Horb_diag += Horb_diag.T
        # This is where I stop unless I want to add the split-c and split-x terms
        # Split-c and split-x, for inactive-external rotations, requires I calculate a bunch
        # of extra eris (g^aa_ii, g^ai_ai)
        return Horb_diag[self.ugg.uniq_orb_idx]

    def _get_Hci_diag (self):
        Hci_diag = []
        for ix, (fcibox, norb, nelec, h1rs, csf_list) in enumerate (zip (self.fciboxes, 
         self.ncas_sub, self.nelecas_sub, self.h1frs, self.ugg.ci_transformers)):
            i = sum (self.ncas_sub[:ix])
            j = i + norb
            h2 = self.eri_cas[i:j,i:j,i:j,i:j]
            hdiag_csf_list = fcibox.states_make_hdiag_csf (h1rs, h2, norb, nelec)
            for csf, hdiag_csf in zip (csf_list, hdiag_csf_list):
                Hci_diag.append (csf.pack_csf (hdiag_csf))
        return Hci_diag

    def _get_Hdiag (self):
        return np.concatenate ([self._get_Horb_diag ()] + self._get_Hci_diag ())

    def update_mo_ci_eri (self, x, h2eff_sub):
        kappa, dci = self.ugg.unpack (x)
        umat = linalg.expm (kappa/2)
        # The 1/2 here is because my actual variables are just the lower-triangular
        # part of kappa, or equivalently 1/2 k^p_q (E^p_q - E^q_p). I can simplify
        # this to k^p_q E^p_q when evaluating derivatives, but not when exponentiating,
        # because the operator part has to be anti-hermitian.
        mo1 = self._update_mo (umat)
        ci1 = self._update_ci (dci)
        h2eff_sub = self._update_h2eff_sub (mo1, umat, h2eff_sub)
        return mo1, ci1, h2eff_sub

    def _update_mo (self, umat):
        mo1 = self.mo_coeff @ umat
        if hasattr (self.mo_coeff, 'orbsym'):
            mo1 = lib.tag_array (mo1, orbsym=self.mo_coeff.orbsym)
        return mo1

    def _update_ci (self, dci):
        ci1 = []
        for c_r, dc_r in zip (self.ci, dci):
            ci1_r = []
            for c, dc in zip (c_r, dc_r):
                dc[:] -= c * c.dot (dc)
                phi = linalg.norm (dc)
                cosp = np.cos (phi)
                if np.abs (phi) > 1e-8: sinp = np.sin (phi) / phi
                else: sinp = 1 # as precise as it can be w/ 64 bits
                c1 = cosp*c + sinp*dc
                assert (np.isclose (linalg.norm (c1), 1))
                ci1_r.append (c1)
            ci1.append (ci1_r)
        return ci1

    def _update_h2eff_sub (self, mo1, umat, h2eff_sub):
        ncore, ncas, nocc, nmo = self.ncore, self.ncas, self.nocc, self.nmo
        ucas = umat[ncore:nocc, ncore:nocc]
        bmPu = None
        if hasattr (h2eff_sub, 'bmPu'): bmPu = h2eff_sub.bmPu
        h2eff_sub = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
        h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub)
        h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas, ncas)
        h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(1))) # bpaa
        h2eff_sub = np.tensordot (umat, h2eff_sub, axes=((0),(1))) # qbaa
        h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbab
        h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbbb
        ix_i, ix_j = np.tril_indices (ncas)
        h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas*ncas)
        h2eff_sub = h2eff_sub[:,:,(ix_i*ncas)+ix_j]
        h2eff_sub = h2eff_sub.reshape (nmo, -1)
        if bmPu is not None:
            bmPu = np.dot (bmPu, ucas)
            h2eff_sub = lib.tag_array (h2eff_sub, bmPu = bmPu)
        return h2eff_sub

    def get_grad (self):
        gorb = self.fock1 - self.fock1.T
        gci = [[2*hci0 for hci0 in hci0r] for hci0r in self.hci0]
        return self.ugg.pack (gorb, gci)

    def get_gx (self):
        gorb = self.fock1 - self.fock1.T
        gx = gorb[self.ugg.get_gx_idx ()]
        return gx


