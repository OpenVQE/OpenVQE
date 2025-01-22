import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import mc1step, mc1step_symm
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.my_pyscf.fci import csf, csf_symm
from mrh.my_pyscf.fci.csfstring import transform_civec_det2csf, transform_civec_csf2det
''' The approximate CI response carried out within microiterations in the 1-step MCSCF algorithm has hitherto been
carried out in the determinant basis, even if the full CASCI problem is solved in the CSF basis. 
Usually this wouldn't be a huge problem, but if you know for a fact that S^2 spontaneously breaks
and you really need to enforce it then fixing this will be mandatory. '''


def solve_approx_ci_csf (mc, h1, h2, ci0, ecore, e_cas, envs):
    ''' This is identical to pyscf.mcscf.mc1step.CASSCF.solve_approx_ci
    (with %s/self/mc/g) as of 03/24/2019 for the first 48 lines '''
    ncas = mc.ncas
    nelecas = mc.nelecas
    ncore = mc.ncore
    nocc = ncore + ncas
    if 'norm_gorb' in envs:
        tol = max(mc.conv_tol, envs['norm_gorb']**2*.1)
    else:
        tol = None
    if getattr(mc.fcisolver, 'approx_kernel', None):
        fn = mc.fcisolver.approx_kernel
        e, ci1 = fn(h1, h2, ncas, nelecas, ecore=ecore, ci0=ci0,
                    tol=tol, max_memory=mc.max_memory)
        return ci1, None
    elif not (getattr(mc.fcisolver, 'contract_2e', None) and
              getattr(mc.fcisolver, 'absorb_h1e', None)):
        fn = mc.fcisolver.kernel
        e, ci1 = fn(h1, h2, ncas, nelecas, ecore=ecore, ci0=ci0,
                    tol=tol, max_memory=mc.max_memory,
                    max_cycle=mc.ci_response_space)
        return ci1, None

    h2eff = mc.fcisolver.absorb_h1e(h1, h2, ncas, nelecas, .5)

    # Be careful with the symmetry adapted contract_2e function. When the
    # symmetry adapted FCI solver is used, the symmetry of ci0 may be
    # different to fcisolver.wfnsym. This function may output 0.
    if getattr(mc.fcisolver, 'guess_wfnsym', None):
        wfnsym = mc.fcisolver.guess_wfnsym(mc.ncas, mc.nelecas, ci0)
    else:
        wfnsym = None
    def contract_2e(c):
        if wfnsym is None:
            hc = mc.fcisolver.contract_2e(h2eff, c, ncas, nelecas)
        else:
            with lib.temporary_env(mc.fcisolver, wfnsym=wfnsym):
                hc = mc.fcisolver.contract_2e(h2eff, c, ncas, nelecas)
        return hc.ravel()

    hc = contract_2e(ci0)
    g = hc - (e_cas-ecore) * ci0.ravel()

    if mc.ci_response_space > 7:
        logger.debug(mc, 'CI step by full response')
        # full response
        max_memory = max(400, mc.max_memory-lib.current_memory()[0])
        e, ci1 = mc.fcisolver.kernel(h1, h2, ncas, nelecas, ecore=ecore,
                                       ci0=ci0, tol=tol, max_memory=max_memory)
    else:
        # MRH 03/24/2019: this is where I intervene to enforce CSFs
        fci = mc.fcisolver
        smult = fci.smult
        neleca, nelecb = _unpack_nelec (nelecas)
        norb = np.asarray (h1).shape[-1]
        if hasattr (fci, 'wfnsym') and hasattr (fci, 'confsym'):
            idx_sym = fci.confsym[fci.econf_csf_mask] == fci.wfnsym
        else:
            idx_sym = None
        xs = [csf.pack_sym_ci (transform_civec_det2csf (ci0, norb, neleca, nelecb, smult, csd_mask=fci.csd_mask, do_normalize=True)[0], idx_sym)]
        nd = min(max(mc.ci_response_space, 2), xs[0].size)
        logger.debug(mc, 'CI step by %dD subspace response', nd)
        def contract_2e_csf (x):
            x_det = transform_civec_csf2det (csf.unpack_sym_ci (x, idx_sym), norb, neleca, nelecb, smult, csd_mask=fci.csd_mask)[0]
            hx = contract_2e(x_det)
            hx = transform_civec_det2csf (hx, norb, neleca, nelecb, smult, csd_mask=fci.csd_mask, do_normalize=False)[0]
            return csf.pack_sym_ci (hx, idx_sym).ravel ()
        ax = [contract_2e_csf (xs[0])]
        heff = np.empty((nd,nd))
        seff = np.empty((nd,nd))
        heff[0,0] = np.dot(xs[0], ax[0])
        seff[0,0] = 1
        for i in range(1, nd):
            xs.append(ax[i-1] - xs[i-1] * e_cas)
            ax.append(contract_2e_csf(xs[i]))
            for j in range(i+1):
                heff[i,j] = heff[j,i] = (xs[i] * ax[j]).sum ()
                seff[i,j] = seff[j,i] = (xs[i] * xs[j]).sum ()
        e, v = lib.safe_eigh(heff, seff)[:2]
        ci1 = xs[0] * v[0,0]
        for i in range(1,nd):
            ci1 += xs[i] * v[i,0]
        ci1 = transform_civec_csf2det (csf.unpack_sym_ci (ci1, idx_sym), norb, neleca, nelecb, smult, csd_mask=fci.csd_mask, do_normalize=True)[0]
    return ci1, g

def fix_ci_response_csf (mc):
    ''' MRH, 03/24/2019: The approximate CI response carried out within microiterations in the 1-step MCSCF algorithm has hitherto been
    carried out in the determinant basis, even if the full CASCI problem is solved in the CSF basis. 
    Usually this wouldn't be a huge problem, but if you know for a fact that S^2 spontaneously breaks
    and you really need to enforce it then fixing this will be mandatory.'''
    class ci_response_fixed(mc.__class__):
        ''' MRH, 03/24/2019: Patching solve_approx_ci
     
        ''' + str (mc.__class__.__doc__)
        def __init__(self, my_mc):
            self.__dict__.update (my_mc.__dict__)
    
        def solve_approx_ci (self, h1, h2, ci0, ecore, e_cas, envs):
            if not isinstance (self.fcisolver, (csf.FCISolver, csf_symm.FCISolver,)):
                return super().solve_approx_ci (h1, h2, ci0, ecore, e_cas, envs)
            return solve_approx_ci_csf (self, h1, h2, ci0, ecore, e_cas, envs)

    return ci_response_fixed (mc)


