import numpy as np
import scipy, time
from pyscf import fci, lib, mcscf
from pyscf.mcscf import casci
from pyscf.mcscf.mc1step import _fake_h_for_fast_casci
from pyscf.mcpdft import mcpdft
logger = lib.logger

def get_heff_cas (mc, mo_coeff, ci, link_index=None):
    ncore, ncas, nelec = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    veff1, veff2 = mc.get_pdft_veff (mo=mo_coeff, ci=ci, incl_coul=False,
        aaaa_only=True)
    h1_ao = (mc.get_hcore () + veff1
           + mc._scf.get_j (dm=mc.make_rdm1 (mo_coeff=mo_coeff, ci=ci)))

    h0 = ((mc._scf.energy_nuc () 
        + (h1_ao @ mo_core).ravel ().dot (mo_core.conj ().ravel ()))/2
        + veff2.energy_core)
    h1 = (mo_cas.conj ().T @ (h1_ao) @ mo_cas
        + veff2.vhf_c[ncore:nocc,ncore:nocc])
    h2 = np.zeros ([ncas,]*4) # Forward-compatibility for outcore pdft veff...
    for i in range (ncore, nocc): 
        h2[i-ncore] = veff2.ppaa[i][ncore:nocc].copy ()

    return h0, h1, h2

def _ci_min_epdft_fp (mc, mo_coeff, ci0, hcas=None, verbose=None):
    '''Minimize the PDFT energy of a single state by repeated
    diagonalizations of the effective PDFT Hamiltonian
    hpdft = Pcas (vnuc + dE/drdm1 op1 + dE/drdm2 op2) Pcas
    (as if that makes sense...) 

    Args:
        mc : mcscf object
        mo_coeff : ndarray of shape (nao,nmo)
        ci0 : ndarray of size (ndeta*ndetb)
            Initial guess CI vector; required!

    Kwargs:
        hcas : (float, [ncas,]*2 ndarray, [ncas,]*4 ndarray) or None
            The true Hamiltonian projected into the active space
        verbose : integer
            logger verbosity of function output; defaults to mc.verbose

    Returns:
        epdft : float
            Minimized MC-PDFT energy
        h0_pdft : float
            At convergence, the constant term of hpdft
            You might need this because ????
        ci1 : ndarray of size (ndeta*ndetb)
            Optimized CI vector
        emcscf : float or None
            <ci1|hcas|ci1>
    '''
    t0 = (logger.process_clock (), logger.perf_counter ())
    ncas, nelecas = mc.ncas, mc.nelecas
    if verbose is None: verbose = mc.verbose
    log = logger.new_logger (mc, verbose)
    if hasattr (mc.fcisolver, 'gen_linkstr'):
        linkstrl = mc.fcisolver.gen_linkstr(ncas, nelecas, True)
    else:
        linkstrl = None 
    h0_pdft, h1_pdft, h2_pdft = get_heff_cas (mc, mo_coeff, ci0)
    max_memory = max(400, mc.max_memory-lib.current_memory()[0])

    epdft = 0
    chc_last = 0
    emcscf = None
    ci1 = ci0.copy ()
    for it in range (mc.max_cycle_fp):
        h2eff = mc.fcisolver.absorb_h1e (h1_pdft, h2_pdft, ncas, nelecas, 0.5)
        hc = mc.fcisolver.contract_2e (h2eff, ci1, ncas, nelecas, link_index=linkstrl).ravel ()
        chc = ci1.conj ().ravel ().dot (hc)
        ci_grad = hc - (chc * ci1.ravel ())
        ci_grad_norm = ci_grad.dot (ci_grad)
        epdft_last = epdft
        epdft = mcpdft.energy_tot (mc, ot=mc.otfnal, mo_coeff=mo_coeff, ci=ci1)[0]

        dchc = chc + h0_pdft - chc_last # careful; don't mess up ci_grad
        depdft = epdft - epdft_last
        if hcas is None:
            log.info ('MC-PDFT CI fp iter %d EPDFT = %e, |grad| = %e, dEPDFT ='
                ' %e, d<c.Hpdft.c> = %e', it, epdft, ci_grad_norm, depdft,
                dchc)
        else:
            h2eff = mc.fcisolver.absorb_h1e (hcas[1], hcas[2], ncas, nelecas,
                0.5)
            hc = mc.fcisolver.contract_2e (h2eff, ci1, ncas, nelecas, link_index=linkstrl).ravel ()
            emcscf = ci1.conj ().ravel ().dot (hc) + hcas[0]
            log.info ('MC-PDFT CI fp iter %d ECAS = %e, EPDFT = %e, |grad| = '
                '%e, dEPDFT = %e, d<c.Hpdft.c> = %e', it, emcscf, epdft,
                ci_grad_norm, depdft, dchc)
         

        if (ci_grad_norm < mc.conv_tol_ci_fp 
            and np.abs (dchc) < mc.conv_tol_ci_fp): break
       
        chc_last, ci1 = mc.fcisolver.kernel (h1_pdft, h2_pdft, ncas, nelecas,
                                               ci0=ci1, verbose=log,
                                               max_memory=max_memory,
                                               ecore=h0_pdft)
        h0_pdft, h1_pdft, h2_pdft = get_heff_cas (mc, mo_coeff, ci1)
        # putting this at the bottom to 
        # 1) get a good max_memory outside the loop with
        # 2) as few integrations as possible

    log.timer ('MC-PDFT CI fp iteration', *t0)
    return epdft, h0_pdft, ci1, emcscf
    
def mc1step_casci(mc, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
    ''' Wrapper for _ci_min_epdft_fp to mcscf.mc1step.casci function '''
    if ci0 is None: ci0 = mc.ci
    if verbose is None: verbose = mc.verbose
    t0 = (logger.process_clock (), logger.perf_counter ())
    ncas, nelecas = mc.ncas, mc.nelecas
    linkstrl = mc.fcisolver.gen_linkstr(ncas, nelecas, True)
    linkstr  = mc.fcisolver.gen_linkstr(ncas, nelecas, False)
    log = lib.logger.new_logger (mc, verbose)
    if eris is None:
        h0_cas, h1_cas = mcscf.casci.h1e_for_las (mc, mo_coeff, mc.ncas,
            mc.ncore)
        h2_cas = mcscf.casci.CASCI.ao2mo (mc, mo_coeff)
    else:
        fcasci = _fake_h_for_fast_casci (mc, mo_coeff, eris)
        h1_cas, h0_cas = fcasci.get_h1eff ()
        h2_cas = fcasci.get_h2eff ()

    if ci0 is None: 
        # Use real Hamiltonian? Or use HF?
        hdiag = mc.fcisolver.make_hdiag (h1_cas, h2_cas, ncas, nelecas)
        ci0 = mc.fcisolver.get_init_guess (ncas, nelecas, 1, hdiag)[0]

    epdft, h0_pdft, ci1, emcscf = _ci_min_epdft_fp (mc, mo_coeff, ci0, 
        hcas=(h0_cas,h1_cas,h2_cas), verbose=verbose)
    eci = epdft - h0_cas

    if envs is not None and log.verbose >= lib.logger.INFO:
        log.debug('CAS space CI energy = %.15g', eci)

        if getattr(mc.fcisolver, 'spin_square', None):
            ss = mc.fcisolver.spin_square(ci1, mc.ncas, mc.nelecas)
        else:
            ss = None

        if 'imicro' in envs:  # Within CASSCF iteration
            if ss is None:
                log.info('macro iter %d (%d JK  %d micro), '
                         'MC-PDFT E = %.15g  dE = %.8g  CASCI E = %.15g',
                         envs['imacro'], envs['njk'], envs['imicro'],
                         epdft, epdft-envs['elast'], emcscf)
            else:
                log.info('macro iter %d (%d JK  %d micro), '
                         'MC-PDFT E = %.15g  dE = %.8g  S^2 = %.7f  CASCI E = '
                         '%.15g', envs['imacro'], envs['njk'], envs['imicro'],
                         epdft, epdft-envs['elast'], ss[0], emcscf)
            if 'norm_gci' in envs:
                log.info('               |grad[o]|=%5.3g  '
                         '|grad[c]|= %s  |ddm|=%5.3g',
                         envs['norm_gorb0'],
                         envs['norm_gci'], envs['norm_ddm'])
            else:
                log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g',
                         envs['norm_gorb0'], envs['norm_ddm'])
        else:  # Initialization step
            if ss is None:
                log.info('MC-PDFT E = %.15g  CASCI E = %.15g', epdft, emcscf)
            else:
                log.info('MC-PDFT E = %.15g  S^2 = %.7f  CASCI E = %.15g',
                    epdft, ss[0], emcscf)

    # For correct reporting, this function should return <CAS|H|CAS>
    return emcscf, emcscf-h0_cas, ci1

def mc1step_update_casdm(mc, mo, u, fcivec, e_cas, eris, envs={}):
    ''' Wrapper for mc1step.update_casdm to use the PDFT effective
    Hamiltonian, rather than the physical Hamiltonian, for approximate
    micro-step CI vector updates. '''
    mo1 = np.dot (mo, u)
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas

    h0_pdft, h1_pdft, h2_pdft = get_heff_cas (mc, mo1, fcivec)
    h2eff = mc.fcisolver.absorb_h1e (h1_pdft, h2_pdft, ncas, nelecas, 0.5)
    hc = mc.fcisolver.contract_2e (h2eff, fcivec, ncas, nelecas).ravel ()
    chc = fcivec.conj ().ravel ().dot (hc) + h0_pdft
    ci1, g = mc.solve_approx_ci(h1_pdft, h2_pdft, fcivec, h0_pdft, chc, envs)
    if g is not None:  # So state average CI, DMRG etc will not be applied
        ovlp = np.dot(fcivec.ravel(), ci1.ravel())
        norm_g = np.linalg.norm(g)
        if 1-abs(ovlp) > norm_g * mc.ci_grad_trust_region:
            logger.debug(mc, '<ci1|ci0>=%5.3g |g|=%5.3g, ci1 out of trust '
                         'region', ovlp, norm_g)
            ci1 = fcivec.ravel() + g
            ci1 *= 1/np.linalg.norm(ci1)
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci1, mc.ncas, mc.nelecas)

    return casdm1, casdm2, g, ci1

def casci_kernel(mc, mo_coeff=None, ci0=None, verbose=None):
    ''' Wrapper for _ci_min_epdft_fp to mcscf.casci.kernel (and
        mcscf.casci.CASCI.kernel) function '''

    # casci.CASCI.kernel throatclearing

    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    else:
        mc.mo_coeff = mo_coeff
    if ci0 is None:
        ci0 = mc.ci
    log = logger.new_logger(mc, verbose)

    if mc.verbose >= logger.WARN:
        mc.check_sanity()
    mc.dump_flags(log)

    # enter casci.kernel part

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start CASCI')

    ncas = mc.ncas
    nelecas = mc.nelecas

    # 2e
    eri_cas = mc.get_h2eff(mo_coeff)
    t1 = log.timer('integral transformation to CAS space', *t0)

    # 1e
    h1eff, energy_core = mc.get_h1eff(mo_coeff)
    log.debug('core energy = %.15g', energy_core)
    t1 = log.timer('effective h1e in CAS space', *t1)

    if h1eff.shape[0] != ncas:
        raise RuntimeError('Active space size error. nmo=%d ncore=%d ncas=%d' %
                           (mo_coeff.shape[1], mc.ncore, ncas))

    # ci0 can't be none in PDFT-SCF
    if ci0 is None:
        hdiag = mc.fcisolver.make_hdiag (h1eff, eri_cas, ncas, nelecas)
        ci0 = mc.fcisolver.get_init_guess (ncas, nelecas, 1, hdiag)[0]

    # the actually important part
    mc.e_tot, h0_pdft, mc.ci, mc.e_mcscf = _ci_min_epdft_fp (mc, mo_coeff, ci0,
        hcas=(energy_core,h1eff,eri_cas), verbose=verbose)
    mc.e_cas = mc.e_mcscf - energy_core

    # exit back to casci.CASCI.kernel throatclearing

    if mc.canonicalization:
        mc.canonicalize_(mo_coeff, mc.ci,
                           sort=mc.sorting_mo_energy,
                           cas_natorb=mc.natorb, verbose=log)
    elif mc.natorb:
        # FIXME (pyscf-2.0): Whether to transform natural orbitals in
        # active space when this flag is enabled?
        log.warn('The attribute .natorb of mcscf object affects only the '
                 'orbital canonicalization.\n'
                 'If you would like to get natural orbitals in active space '
                 'without touching core and external orbitals, an explicit '
                 'call to mc.cas_natorb_() is required')

    if getattr(mc.fcisolver, 'converged', None) is not None:
        mc.converged = np.all(mc.fcisolver.converged)
        if mc.converged:
            log.info('CASCI converged')
        else:
            log.info('CASCI not converged')
    else:
        mc.converged = True
    mc._finalize()
    return mc.e_tot, mc.e_cas, mc.ci, mc.mo_coeff, mc.mo_energy

def casci_finalize(mc):
    ''' This function is for I/O clarity only '''
    log = logger.Logger(mc.stdout, mc.verbose)
    if log.verbose >= logger.NOTE and getattr(mc.fcisolver, 'spin_square',
            None):
        if isinstance(mc.e_cas, (float, np.number)):
            ss = mc.fcisolver.spin_square(mc.ci, mc.ncas, mc.nelecas)
            log.note('MC-PDFT E = %.15g  E(CASCI) = %.15g  S^2 = %.7f',
                     mc.e_tot, mc.e_cas, ss[0])
        else:
            for i, e in enumerate(mc.e_cas):
                ss = mc.fcisolver.spin_square(mc.ci[i], mc.ncas, mc.nelecas)
                log.note('MC-PDFT state %d  E = %.15g  E(CASCI) = %.15g  S^2 ='
                         ' %.7f', i, mc.e_tot[i], e, ss[0])
    else:
        if isinstance(mc.e_cas, (float, np.number)):
            log.note('MC-PDFT E = %.15g  E(CASCI) = %.15g', mc.e_tot, mc.e_cas)
        else:
            for i, e in enumerate(mc.e_cas):
                log.note('MC-PDFT state %d  E = %.15g  E(CASCI) = %.15g',
                         i, mc.e_tot[i], e)
    return mc

    


