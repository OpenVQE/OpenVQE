import numpy as np
from scipy import linalg, special
from scipy.sparse.linalg import LinearOperator
from pyscf.lib import logger, temporary_env
from pyscf.mcscf.addons import StateAverageMCSCFSolver, StateAverageMixFCISolver, state_average_mix
from pyscf.mcscf.addons import StateAverageMixFCISolver_state_args as _state_arg
from pyscf.mcscf.addons import StateAverageMixFCISolver_solver_args as _solver_arg
from pyscf.fci.direct_spin1 import _unpack_nelec

class StateAverageNMixFCISolver (StateAverageMixFCISolver):
    def _get_nelec (self, solver, nelec):
        n = np.sum (nelec)
        m = solver.spin if solver.spin is not None else n%2
        c = getattr (solver, 'charge', 0) or 0
        n -= c
        nelec = (n+m)//2, (n-m)//2
        return nelec

def get_sanmix_fcisolver (samix_fcisolver):

    # Recursion protection
    if isinstance (samix_fcisolver, StateAverageNMixFCISolver):
        return samix_fcisolver

    class FCISolver (samix_fcisolver.__class__, StateAverageNMixFCISolver):
        _get_nelec = StateAverageNMixFCISolver._get_nelec

    sanmix_fcisolver = FCISolver (samix_fcisolver.mol)
    sanmix_fcisolver.__dict__.update (samix_fcisolver.__dict__)
    return sanmix_fcisolver

def state_average_n_mix (casscf, fcisolvers, weights=(0.5,0.5)):
    sacasscf = state_average_mix (casscf, fcisolvers, weights=weights)
    sacasscf.fcisolver = get_sanmix_fcisolver (sacasscf.fcisolver)
    # Inject "charge" into "_keys" to suppress annoying warning msg
    keys = set(('charge',))
    for solver in sacasscf.fcisolver.fcisolvers:
        solver._keys = set (solver._keys).union (keys)
    return sacasscf

def state_average_n_mix_(casscf, fcisolvers, weights=(0.5,0.5)):
    sacasscf = state_average_n_mix (casscf, fcisolvers, weights)
    casscf.__class__ = sacasscf.__class__
    casscf.__dict__.update(sacasscf.__dict__)
    return casscf

class H1EZipFCISolver (object):
    pass

def get_h1e_zipped_fcisolver (fcisolver):
    ''' Wrap a state-average-mix FCI solver to take a list of h1es to apply to each solver. '''

    # Recursion protection
    if isinstance (fcisolver, H1EZipFCISolver):
        return fcisolver

    assert isinstance (fcisolver, StateAverageMixFCISolver), 'requires StateAverageMixFCISolver'
    has_spin_square = getattr(fcisolver, 'states_spin_square', None)

    class FCISolver (fcisolver.__class__, H1EZipFCISolver):

        def kernel(self, h1, h2, norb, nelec, ci0=None, verbose=0, ecore=0, orbsym=None, **kwargs):
            # Note self.orbsym is initialized lazily in mc1step_symm.kernel function
            log = logger.new_logger(self, verbose)
            es = []
            cs = []
            if isinstance (ecore, (int, float, np.integer, np.floating)):
                ecore = [ecore,] * len (h1)
            if orbsym is None: orbsym=self.orbsym
            for solver, my_args, my_kwargs in self._loop_solver(_state_arg (ci0), _solver_arg (h1), _state_arg (ecore)):
                c0 = my_args[0]
                h1e = my_args[1]
                e0 = my_args[2]
                e, c = solver.kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                     orbsym=orbsym, verbose=log, ecore=e0, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            self.e_states = es
            self.converged = np.all(getattr(sol, 'converged', True)
                                       for sol in self.fcisolvers)

            if log.verbose >= logger.DEBUG:
                if has_spin_square:
                    ss = self.states_spin_square(cs, norb, nelec)[0]
                    for i, ei in enumerate(es):
                        if i>10 and log.verbose < logger.DEBUG1:
                            log.debug ('printout for %d more states truncated', len(es)-11)
                            break
                        log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
                else:
                    for i, ei in enumerate(es):
                        if i>10 and log.verbose < logger.DEBUG1:
                            log.debug ('printout for %d more states truncated', len(es)-11)
                            break
                        log.debug('state %d  E = %.15g', i, ei)
            return np.einsum('i,i', np.array(es), self.weights), cs

        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, orbsym=None, **kwargs):
            es = []
            cs = []
            if orbsym is None: orbsym=self.orbsym
            for solver, my_args, _ in self._loop_solver (_state_arg (ci0), _solver_arg (h1)):
                c0 = my_args[0]
                h1e = my_args[1]
                try:
                    e, c = solver.approx_kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                                orbsym=orbsym, **kwargs)
                except AttributeError:
                    e, c = solver.kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                         orbsym=orbsym, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            return np.einsum('i,i->', es, self.weights), cs

        def states_absorb_h1e (self, h1, h2, norb, nelec, fac):
            op = []
            for solver, my_args, _ in self._loop_solver (_solver_arg (h1)):
                h1e = my_args[0]
                op.append (solver.absorb_h1e (h1e, h2, norb, self._get_nelec (solver, nelec), fac) if h1 is not None else h2)
            return op

        def states_contract_2e (self, h2, ci, norb, nelec, link_index=None):
            hc = []
            for solver, my_args, _ in self._loop_solver (_state_arg (ci), _state_arg (h2), _solver_arg (link_index)):
                c0 = my_args[0]
                h2e = my_args[1]
                linkstr = my_args[2]
                hc.append (solver.contract_2e (h2e, c0, norb, self._get_nelec (solver, nelec), link_index=linkstr))
            return hc

        def states_make_hdiag (self, h1, h2, norb, nelec):
            hdiag = []
            for solver, my_args, _ in self._loop_solver (_solver_arg (h1)):
                h1e = my_args[0]
                hdiag.append (solver.make_hdiag (h1e, h2, norb, self._get_nelec (solver, nelec)))
            return hdiag

        def states_make_hdiag_csf (self, h1, h2, norb, nelec):
            hdiag = []
            for solver, my_args, _ in self._loop_solver (_solver_arg (h1)):
                h1e = my_args[0]
                with temporary_env (solver, orbsym=self.orbsym):
                    hdiag.append (solver.make_hdiag_csf (h1e, h2, norb, self._get_nelec (solver, nelec)))
            return hdiag

        # The below can conceivably be added to pyscf.mcscf.addons.StateAverageMixFCISolver in future

        def states_gen_linkstr (self, norb, nelec, tril=True):
            linkstr = []
            for solver in self.fcisolvers:
                with temporary_env (solver, orbsym=self.orbsym):
                    linkstr.append (solver.gen_linkstr (norb, self._get_nelec (solver, nelec), tril=tril)
                        if getattr (solver, 'gen_linkstr', None) else None)
            return linkstr
                    
        def states_transform_ci_for_orbital_rotation (self, ci0, norb, nelec, umat):
            ci1 = []
            for solver, my_args, _ in self._loop_solver (_state_arg (ci0)):
                ne = self._get_nelec (solver, nelec)
                ndet = [special.comb (norb, n, exact=True) for n in ne]
                try:
                    ci0_i = my_args[0].reshape (ndet)
                    ci1.append (solver.transform_ci_for_orbital_rotation (ci0_i, norb, ne, umat))
                except ValueError as err:
                    ci0_i = my_args[0].reshape ([-1,]+ndet)
                    ci1.append (np.stack ([solver.transform_ci_for_orbital_rotation (c, norb, ne, umat)
                                           for c in ci0_i], axis=0))
            return ci1

        def states_trans_rdm12s (self, ci1, ci0, norb, nelec, link_index=None, **kwargs):
            ci1 = _state_arg (ci1)
            ci0 = _state_arg (ci0)
            link_index = _solver_arg (link_index)
            nelec = _solver_arg ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
            tdm1 = []
            tdm2 = []
            for dm1, dm2 in self._collect ('trans_rdm12s', ci1, ci0, norb, nelec, link_index=link_index, **kwargs):
                tdm1.append (dm1)
                tdm2.append (dm2)
            return tdm1, tdm2

        # TODO: remove this?
        absorb_h1e = states_absorb_h1e
        contract_2e = states_contract_2e
        make_hdiag = states_make_hdiag

    h1ezipped_fcisolver = FCISolver (fcisolver.mol)
    h1ezipped_fcisolver.__dict__.update (fcisolver.__dict__)
    return h1ezipped_fcisolver

def las2cas_civec (las):
    from mrh.my_pyscf.lassi.op_o0 import ci_outer_product
    norb_f = las.ncas_sub
    nelec_fr = [[_unpack_nelec (fcibox._get_nelec (solver, nelecas)) for solver in fcibox.fcisolvers] for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
    ci, nelec = ci_outer_product (las.ci, norb_f, nelec_fr)
    return ci, nelec

def debug_lasscf_hessian_(las, check_horb_matvec=False, perfect_orbital_preconditioner=False):
    ''' Monkeypatch function to debug the LASSCF hessian operator for the synchronous algorithm.
    Computes the full dense orbital-rotation Hessian and outputs its smallest eigenvalues and
    condition number at the beginning of every macrocycle of the kernel. Unless the
    perfect_orbital_preconditioner kwarg is True, this function only adds information to the
    las.stdout stream (at substantial additional computational cost); it does not change the
    execution of the calculation. Output is added at the "info" level of verbosity, so if
    las.verbose is smaller than lib.logger.INFO, no additional output is produced. This is too
    expensive to be the default for verbose = lib.logger.DEBUG, but the higher verbosities do way
    too much.

    Args:
        las : instance of :class:`LASCINoSymm`
            The method object to debug. Modified in-place!

    Kwargs:
        check_horb_matvec : logical
            If True, outputs the difference between a direct matrix-vector product with the dense
            Horb and the sparse Horb in every microcycle, in addition to the analysis of the dense
            Horb in every macrocycle. Note the CI sector is ignored on both the internal and
            external indices.
        perfect_orbital_preconditioner : logical
            If True, the orbital sector of the preconditioner function is replaced with direct
            solution for z of Horb.z = x. Note that this leaves the CI sector unchanged, so the
            inner CG iteration will still take multiple cycles to converge in general.

    Returns:
        las : instance of :class:`LASCINoSymm`
            Same as arg, after in-place modification.
        parent_hop : class
            The original value of the overwritten las._hop. Reassign it (las._hop = parent_hop) to
            undo the effects of this function.
    '''
    from mrh.my_dmet.orbital_hessian import HessianCalculator
    from mrh.util.la import vector_error
    from pyscf.lib import current_memory
    import os, sys
    parent_hop = las.__class__._hop

    nao = las.mol.nao_nr ()
    max_memory = (las.max_memory - current_memory ()[0])
    reqd_memory = (nao**4)*8*2.5/1e6 # 2.5 is safety margin
    if reqd_memory > max_memory:
        logger.warn (las, ('Insufficient memory (%f required; %f available) to debug Hessian! '
                           'Aborting debug monkeypatch...'), reqd_memory, max_memory)
        return las, parent_hop

    # HessianCalculator uses "print" statements, which sucks. Make them go away.
    class SuppressPrint ():
        def __enter__(self):
            self._true_stdout = sys.stdout
            sys.stdout = open (os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close ()
            sys.stdout = self._true_stdout

    class _hop (las._hop):
        def __init__(self, *args, **kwargs):
            parent_hop.__init__(self, *args, **kwargs)
            self.Horb_full = Horb_full = self._get_dense_Horb ()
            Horb_evals, Horb_evecs = linalg.eigh (Horb_full)
            Horb_mineval = Horb_evals[np.argmin (Horb_evals)]
            Horb_amineval = Horb_evals[np.argmin (np.abs (Horb_evals))]
            log = logger.new_logger (self.las, self.las.verbose)
            log.info ('Orbital Hessian smallest eigenvalue: %.15g', Horb_mineval)
            log.info ('Orbital Hessian smallest-magnitude eigenvalue: %.15g', Horb_amineval)
            log.info ('Orbital Hessian condition number: %.15e', np.linalg.cond (Horb_full))

        def _get_dense_Horb (self):
            ncore, nocc = self.ncore, self.nocc
            dm1s, mo = self.dm1s, self.mo_coeff
            dm1s_ao = np.dot (mo.conj (), np.dot (dm1s, mo.T)).transpose (1,0,2)
            mo_cas = mo[:,ncore:nocc]
            with SuppressPrint ():
                Horb_full = HessianCalculator (self.las._scf, dm1s_ao, self.cascm2, mo_cas)
                idx = self.ugg.uniq_orb_idx.ravel ()
                Horb_full = Horb_full (mo).reshape (self.nmo**2, self.nmo**2)
            Horb_full = Horb_full[np.ix_(idx,idx)]
            return Horb_full

        if check_horb_matvec:
            def _matvec (self, x):
                # Double-check dense Hessian
                # Only checks the orbital-orbital sector
                nvar_orb = self.ugg.nvar_orb
                Hx_orb_test = self._get_dense_Horb () @ x[:nvar_orb]
                xp = x.copy ()
                xp[nvar_orb:] = 0.0
                Hx_orb_ref = parent_hop._matvec (self, xp)[:nvar_orb]
                log = logger.new_logger (self.las, self.las.verbose)
                err_norm, err_angle = vector_error (Hx_orb_test, Hx_orb_ref)
                log.info ('|Horbx - Horbx_ref| = %.5g, %.5g', err_norm, err_angle)
                return parent_hop._matvec (self, x)

        if perfect_orbital_preconditioner:
            def get_prec (self):
                Hci_diag = np.concatenate (self._get_Hci_diag ())
                Hci_diag += self.ah_level_shift
                Hci_diag[np.abs (Hci_diag)<1e-8] = 1e-8
                def prec_op (x):
                    xorb, xci = x[:self.ugg.nvar_orb], x[self.ugg.nvar_orb:]
                    Mxorb = linalg.solve (self.Horb_full, xorb)
                    Mxci = xci / Hci_diag
                    return np.append (Mxorb, Mxci)
                return LinearOperator (self.shape, matvec=prec_op, dtype=self.dtype)
    las._hop = _hop
    return las, parent_hop

class lasscf_hessian_debugger (object):
    ''' Context-manager version of debug_lasscf_hessian_.

    debug_lasscf_hessian_.__doc__:

    ''' + debug_lasscf_hessian_.__doc__
    def __init__(self, las, **kwargs):
        self.las = las
        self.old_hop = None
        self.kwargs = kwargs
    def __enter__(self):
        self.las, self.old_hop = debug_lasscf_hessian_(self.las, **self.kwargs)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.las._hop = self.old_hop

