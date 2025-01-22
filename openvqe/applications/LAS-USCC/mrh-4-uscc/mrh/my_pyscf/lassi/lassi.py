import numpy as np
import time
from scipy import linalg
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi import op_o1
from pyscf import lib, symm
from pyscf.lib.numpy_helper import tag_array
from pyscf.fci.direct_spin1 import _unpack_nelec
from itertools import combinations, product
from mrh.my_pyscf.mcscf import soc_int as soc_int

# TODO: fix stdm1 index convention in both o0 and o1

# TODO: adopt consistent nomenclature viz "states", "spaces", "roots"

# TODO: remove the dependence of lassi_op_o1 on las.fciboxes in some way
# The fcisolvers contain linkstr and symmetry information, but probably
# only the former is necessary. Once the connection to the parent LAS
# instance is severed, remove the dangerous "_LASSI_subspace_env"
# temporary environment.

LINDEP_THRESHOLD = 1.0e-5

op = (op_o0, op_o1)

def ham_2q (las, mo_coeff, veff_c=None, h2eff_sub=None, soc=0):
    '''Construct second-quantization Hamiltonian in CAS, using intermediates from
    a LASSCF calculation.

    Args:
        las : instance of :class:`LASCINoSymm`
        mo_coeff: ndarray of shape (nao,nmo)
            Contains MO coefficients

    Kwargs:
        veff_c : ndarray of shape (nao,nao)
            Effective potential of inactive electrons
        h2eff_sub : ndarray of shape (nmo,(ncas**2)*(ncas+1)/2)
            Two-electron integrals
        soc : integer
            Order of spin-orbit coupling to include. Currently only 0 or 1 supported.
            Including spin-orbit coupling increases the size of return arrays to
            account for additional spin-symmetry-breaking sectors of the Hamiltonian.

    Returns:
        h0 : float
            Constant part of the CAS Hamiltonian
        h1 : ndarray of shape (ncas,ncas) or (2*ncas,2*ncas)
            One-electron part of the CAS Hamiltonian. If soc==True, h1 is returned in
            the spinorbital representation: ncas spin-up states followed by ncas
            spin-down states
        h2 : ndarray of shape [ncas,]*4
            Two-electron part of the CAS Hamiltonian.
    '''
    # a'b = sx + i*sy
    # b'a = sx - i*sy
    # a'a = 1/2 n + sz
    # b'b = 1/2 n - sz
    #   ->
    # sx =  (1/2) (a'b + b'a)
    # sy = (-i/2) (a'b - b'a)
    # sz =  (1/2) (a'a - b'b)
    # 
    # l.s = lx.sx + ly.sy + lz.sz
    #     = (1/2) (lx.(a'b + b'a) - i*ly.(a'b - b'a) + lz.(a'a - b'b))
    #     = (1/2) (  (lx - i*ly).a'b 
    #              + (lx + i*ly).b'a
    #              + lz.(a'a - b'b)
    #             )
    if soc>1: raise NotImplementedError ("Two-electron spin-orbit coupling")
    ncore, ncas, nocc = las.ncore, las.ncas, las.ncore + las.ncas
    norb = sum(las.ncas_sub)
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    hcore = las._scf.get_hcore ()
    if veff_c is None: 
        dm_core = 2 * mo_core @ mo_core.conj ().T
        veff_c = las.get_veff (dm1s=dm_core)
    if h2eff_sub is None:
        h2eff_sub = las.ao2mo (mo_coeff)

    h0 = las._scf.energy_nuc () + 2 * (((hcore + veff_c/2) @ mo_core) * mo_core).sum ()

    h1 = mo_cas.conj ().T @ (hcore + veff_c) @ mo_cas
    if soc:
        dm0 = soc_int.amfi_dm (las.mol)
        hsoao = soc_int.compute_hso(las.mol, dm0, amfi=True)
        hso = .5*lib.einsum ('ip,rij,jq->rpq', mo_cas.conj (), hsoao, mo_cas)

        h1 = linalg.block_diag (h1, h1).astype (complex)
        h1[ncas:2*ncas,0:ncas] = (hso[0] + 1j * hso[1]) # b'a
        h1[0:ncas,ncas:2*ncas] = (hso[0] - 1j * hso[1]) # a'b
        h1[0:ncas,0:ncas] += hso[2] # a'a
        h1[ncas:2*ncas,ncas:2*ncas] -= hso[2] # b'b

    h2 = h2eff_sub[ncore:nocc].reshape (ncas*ncas, ncas * (ncas+1) // 2)
    h2 = lib.numpy_helper.unpack_tril (h2).reshape (ncas, ncas, ncas, ncas)

    return h0, h1, h2

def las_symm_tuple (las, break_spin=False, break_symmetry=False, verbose=None):
    '''Identify the symmetries/quantum numbers of of each LAS excitation space within a LASSI
    model space which are to be preserved by projecting the Hamiltonian into the corresponding
    diagonal symmetry blocks.

    Args:
        las : instance of :class:`LASCINoSymm`

    Kwargs:
        break_spin : logical
            Whether to mix states of different neleca-nelecb (necessary for spin-orbit coupling).
            If True, the first item of each element in statesym is the total number of electrons;
            otherwise, the first two items are the total number of spin-up and spin-down
            electrons.
        break_symmetry : logical
            Whether to mix states of different point-group irreps (may also be necessary for
            spin-orbit coupling). If True, the point-group irrep of each state is omitted from
            the elements of statesym; otherwise, this datum is the last item of each element.

    Returns:
        statesym : list of length nroots
            Each element is a tuple describing all enforced symmetries of a LAS state. 
            The length of each tuple varies between 1 and 4 based on the kwargs break_spin and
            break_symmetry.
        s2_states : list of length nroots
            The expectation values of the <S**2> operator for each state, for convenience
    '''
    # kwarg logic setup
    qn_lbls = ['Neleca', 'Nelecb', 'Nelec', 'Wfnsym']
    incl_spin = not (bool (break_spin))
    incl_symmetry = not (bool (break_symmetry))
    qn_filter = [incl_spin, incl_spin, not incl_spin, incl_symmetry]
    # end kwarg logic setup
    full_statesym = [] # keep everything for i/o...
    statesym = [] # ...but return only this
    s2_states = []
    log = lib.logger.new_logger (las, verbose)
    for iroot in range (las.nroots):
        neleca = 0
        nelecb = 0
        wfnsym = 0
        s = 0
        m = []
        for fcibox, nelec in zip (las.fciboxes, las.nelecas_sub):
            solver = fcibox.fcisolvers[iroot]
            na, nb = _unpack_nelec (fcibox._get_nelec (solver, nelec))
            neleca += na
            nelecb += nb
            s_frag = (solver.smult - 1) / 2
            s += s_frag * (s_frag + 1)
            m.append ((na-nb)/2)
            fragsym = getattr (solver, 'wfnsym', 0) or 0 # in case getattr returns "None"
            if isinstance (fragsym, str):
                fragsym = symm.irrep_name2id (solver.mol.groupname, fragsym)
            assert isinstance (fragsym, (int, np.integer)), '{} {}'.format (type (fragsym),
                                                                            fragsym)
            wfnsym ^= fragsym
        s += sum ([2*m1*m2 for m1, m2 in combinations (m, 2)])
        s2_states.append (s)
        all_qns = [neleca, nelecb, neleca+nelecb, wfnsym]
        full_statesym.append (tuple (all_qns))
        statesym.append (tuple (qn for qn, incl in zip (all_qns, qn_filter) if incl))
    log.info ('Symmetry analysis of %d LAS rootspaces:', las.nroots)
    qn_lbls = ['Neleca', 'Nelecb', 'Nelec', 'Wfnsym']
    qn_fmts = ['{:6d}', '{:6d}', '{:6d}', '{:>6s}']
    lbls = ['ix', 'Energy', '<S**2>'] + qn_lbls
    fmt_str = ' {:2s}  {:>16s}  {:6s}  ' + '  '.join (['{:6s}',]*len(qn_lbls))
    log.info (fmt_str.format (*lbls))
    for ix, (e, sy, s2) in enumerate (zip (las.e_states, full_statesym, s2_states)):
        data = [ix, e, s2] + list (sy)
        data[-1] = symm.irrep_id2name (las.mol.groupname, data[-1])
        fmts = ['{:2d}','{:16.10f}','{:6.3f}'] + qn_fmts
        fmt_str = ' ' + '  '.join (fmts)
        log.info (fmt_str.format (*data))
    if break_spin:
        log.info ("States with different neleca-nelecb can be mixed by LASSI")
    if break_symmetry:
        log.info ("States with different point-group symmetry can be mixed by LASSI")

    return statesym, np.asarray (s2_states)

class _LASSI_subspace_env (object):
    def __init__(self, las, idx):
        self.las = las
        self.idx = np.where (idx)[0]
        self.fcisolvers = [f.fcisolvers for f in las.fciboxes]
        self.e_states = las.e_states
    def __enter__(self):
        for f, g in zip (self.las.fciboxes, self.fcisolvers):
            f.fcisolvers = [g[ix] for ix in self.idx]
        self.las.e_states = self.e_states[self.idx]
    def __exit__(self, type, value, traceback):
        for ix, f in enumerate (self.las.fciboxes):
            f.fcisolvers = self.fcisolvers[ix]
        self.las.e_states = self.e_states

def iterate_subspace_blocks (las, ci, spacesym, subset=None):
    if subset is None: subset = set (spacesym)
    lroots = np.array ([[1 if c.ndim<3 else c.shape[0]
                         for c in ci_r]
                        for ci_r in ci])
    nprods_r = np.product (lroots, axis=0)
    prod_off = np.cumsum (nprods_r) - nprods_r
    nprods = nprods_r.sum ()
    for sym in subset:
        idx_space = np.all (np.array (spacesym) == sym, axis=1)
        idx = np.where (idx_space)[0]
        ci_blk = [[c[i] for i in idx] for c in ci]
        idx_prod = np.zeros (nprods, dtype=bool)
        for i in idx:
            idx_prod[prod_off[i]:prod_off[i]+nprods_r[i]] = True
        with _LASSI_subspace_env (las, idx_space):
            nelec_blk = np.array (
                [[_unpack_nelec (fcibox._get_nelec (solver, nelecas))
                  for solver in fcibox.fcisolvers]
                 for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
            )
            yield las, sym, (idx_space, idx_prod), (ci_blk, nelec_blk)

class LASSIOop01DisagreementError (RuntimeError):
    def __init__(self, message, errvec):
        self.message = message + ("\n"
            "max abs errvec = {}; ||errvec|| = {}").format (
                np.amax (np.abs (errvec)), linalg.norm (errvec))
        self.errvec = errvec
    def __str__(self):
        return self.message

def lassi (las, mo_coeff=None, ci=None, veff_c=None, h2eff_sub=None, orbsym=None, soc=False,
           break_symmetry=False, opt=1):
    ''' Diagonalize the state-interaction matrix of LASSCF '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    o0_memcheck = op_o0.memcheck (las, ci, soc=soc)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    # Construct second-quantization Hamiltonian
    e0, h1, h2 = ham_2q (las, mo_coeff, veff_c=veff_c, h2eff_sub=h2eff_sub, soc=soc)

    # Symmetry tuple: neleca, nelecb, irrep
    statesym, s2_states = las_symm_tuple (las, break_spin=soc, break_symmetry=break_symmetry)

    # Initialize matrices
    e_roots = []
    s2_roots = []
    rootsym = []
    si = []
    s2_mat = []
    idx_allprods = []
    dtype = complex if soc else np.float64

    # Loop over symmetry blocks
    qn_lbls = ['nelec',] if soc else ['neleca','nelecb',]
    if not break_symmetry: qn_lbls.append ('irrep')
    for it, (las1,sym,indices,indexed) in enumerate (iterate_subspace_blocks(las,ci,statesym)):
        idx_space, idx_prod = indices
        ci_blk, nelec_blk = indexed
        idx_allprods.extend (list(np.where(idx_prod)[0]))
        lib.logger.info (las, 'Diagonalizing LASSI symmetry block %d\n'
                         + '{} = {}\n'.format (qn_lbls, sym)
                         + '(%d rootspaces; %d states)', it,
                         np.count_nonzero (idx_space), 
                         np.count_nonzero (idx_prod))
        if np.count_nonzero (idx_prod) == 1:
            lib.logger.debug (las, 'Only one state in this symmetry block')
            e_roots.extend (las1.e_states - e0)
            si.append (np.ones ((1,1), dtype=dtype))
            s2_mat.append (s2_states[idx_prod]*np.ones((1,1)))
            s2_roots.extend (s2_states[idx_prod])
            rootsym.extend ([sym,])
            continue
        wfnsym = None if break_symmetry else sym[-1]
        e, c, s2_blk = _eig_block (las1, e0, h1, h2, ci_blk, nelec_blk, sym, soc,
                                   orbsym, wfnsym, o0_memcheck, opt)
        s2_mat.append (s2_blk)
        si.append (c)
        s2_blk = c.conj ().T @ s2_blk @ c
        lib.logger.debug2 (las, 'Block S**2 in adiabat basis:')
        lib.logger.debug2 (las, '{}'.format (s2_blk))
        e_roots.extend (list(e))
        s2_roots.extend (list (np.diag (s2_blk)))
        rootsym.extend ([sym,]*c.shape[1])

    # Sort results by energy
    si = linalg.block_diag (*si)[idx_allprods,:]
    s2_mat = linalg.block_diag (*s2_mat)[np.ix_(idx_allprods,idx_allprods)]
    idx = np.argsort (e_roots)
    rootsym = np.asarray (rootsym)[idx]
    e_roots = np.asarray (e_roots)[idx] + e0
    s2_roots = np.asarray (s2_roots)[idx]
    if soc == False:
        nelec_roots = [tuple(rs[0:2]) for rs in rootsym]
    else:
        nelec_roots = [rs[0] for rs in rootsym]
    if break_symmetry:
        wfnsym_roots = [None for rs in rootsym]
    else:
        wfnsym_roots = [rs[-1] for rs in rootsym]

    # Results tagged on si array....
    si = si[:,idx]
    si = tag_array (si, s2=s2_roots, s2_mat=s2_mat, nelec=nelec_roots, wfnsym=wfnsym_roots,
                    rootsym=rootsym, break_symmetry=break_symmetry, soc=soc)

    # I/O
    lib.logger.info (las, 'LASSI eigenvalues (%d total):', len (e_roots))
    fmt_str = ' {:2s}  {:>16s}  {:6s}  '
    col_lbls = ['Nelec'] if soc else ['Neleca','Nelecb']
    if not break_symmetry: col_lbls.append ('Wfnsym')
    fmt_str += '  '.join (['{:6s}',]*len(col_lbls))
    col_lbls = ['ix','Energy','<S**2>'] + col_lbls
    lib.logger.info (las, fmt_str.format (*col_lbls))
    fmt_str = ' {:2d}  {:16.10f}  {:6.3f}  '
    col_fmts = ['{:6d}',]*(2-int(soc))
    if not break_symmetry: col_fmts.append ('{:>6s}')
    fmt_str += '  '.join (col_fmts)
    for ix, (er, s2r, rsym) in enumerate (zip (e_roots, s2_roots, rootsym)):
        if np.iscomplexobj (s2r):
            assert (abs (s2r.imag) < 1e-8)
            s2r = s2r.real
        nelec = rsym[0:1] if soc else rsym[:2]
        row = [ix,er,s2r] + list (nelec)
        if not break_symmetry: row.append (symm.irrep_id2name (las.mol.groupname, rsym[-1]))
        lib.logger.info (las, fmt_str.format (*row))
    return e_roots, si

def _eig_block (las, e0, h1, h2, ci_blk, nelec_blk, rootsym, soc, orbsym, wfnsym, o0_memcheck, opt):
    # TODO: simplify
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    if (las.verbose > lib.logger.INFO) and (o0_memcheck):
        ham_ref, s2_ref, ovlp_ref = op_o0.ham (las, h1, h2, ci_blk, nelec_blk, soc=soc,
                                               orbsym=orbsym, wfnsym=wfnsym)
        t0 = lib.logger.timer (las, 'LASSI diagonalizer rootsym {} CI algorithm'.format (
            rootsym), *t0)

        h1_sf = h1
        if soc:
            h1_sf = (h1[0:las.ncas,0:las.ncas]
                     - h1[las.ncas:2*las.ncas,las.ncas:2*las.ncas]).real/2
        ham_blk, s2_blk, ovlp_blk = op_o1.ham (las, h1_sf, h2, ci_blk, nelec_blk, orbsym=orbsym,
                                               wfnsym=wfnsym)
        t0 = lib.logger.timer (las, 'LASSI diagonalizer rootsym {} TDM algorithm'.format (
            rootsym), *t0)
        lib.logger.debug (las,
            'LASSI diagonalizer rootsym {}: ham o0-o1 algorithm disagreement = {}'.format (
                rootsym, linalg.norm (ham_blk - ham_ref))) 
        lib.logger.debug (las,
            'LASSI diagonalizer rootsym {}: S2 o0-o1 algorithm disagreement = {}'.format (
                rootsym, linalg.norm (s2_blk - s2_ref))) 
        lib.logger.debug (las,
            'LASSI diagonalizer rootsym {}: ovlp o0-o1 algorithm disagreement = {}'.format (
                rootsym, linalg.norm (ovlp_blk - ovlp_ref))) 
        errvec = np.concatenate ([(ham_blk-ham_ref).ravel (), (s2_blk-s2_ref).ravel (),
                                  (ovlp_blk-ovlp_ref).ravel ()])
        if np.amax (np.abs (errvec)) > 1e-8 and soc == False: # tmp until SOC in op_o1
            raise LASSIOop01DisagreementError ("Hamiltonian + S2 + Ovlp", errvec)
        if opt == 0:
            ham_blk = ham_ref
            s2_blk = s2_ref
            ovlp_blk = ovlp_ref
    else:
        if (las.verbose > lib.logger.INFO): lib.logger.debug (
            las, 'Insufficient memory to test against o0 LASSI algorithm')
        ham_blk, s2_blk, ovlp_blk = op[opt].ham (las, h1, h2, ci_blk, nelec_blk, soc=soc,
                                                 orbsym=orbsym, wfnsym=wfnsym)
        t0 = lib.logger.timer (las, 'LASSI H build rootsym {}'.format (rootsym), *t0)
    log_debug = lib.logger.debug2 if las.nroots>10 else lib.logger.debug
    if np.iscomplexobj (ham_blk):
        log_debug (las, 'Block Hamiltonian - ecore (real):')
        log_debug (las, '{}'.format (ham_blk.real.round (8)))
        log_debug (las, 'Block Hamiltonian - ecore (imag):')
        log_debug (las, '{}'.format (ham_blk.imag.round (8)))
    else:
        log_debug (las, 'Block Hamiltonian - ecore:')
        log_debug (las, '{}'.format (ham_blk.round (8)))
    log_debug (las, 'Block S**2:')
    log_debug (las, '{}'.format (s2_blk.round (8)))
    log_debug (las, 'Block overlap matrix:')
    log_debug (las, '{}'.format (ovlp_blk.round (8)))
    # Error catch: diagonal Hamiltonian elements
    # This diagnostic is simply not valid for local excitations;
    # the energies aren't supposed to be additive
    lroots = np.array ([[1 if ci.ndim<3 else ci.shape[0]
                         for ci in ci_r]
                        for ci_r in ci_blk])
    if np.all (lroots==1) and soc==False: # tmp?
        diag_test = np.diag (ham_blk)
        diag_ref = las.e_states - e0
        maxerr = np.max (np.abs (diag_test-diag_ref))
        if maxerr>1e-5:
            lib.logger.debug (las, '{:>13s} {:>13s} {:>13s}'.format ('Diagonal', 'Reference',
                                                                     'Error'))
            for ix, (test, ref) in enumerate (zip (diag_test, diag_ref)):
                lib.logger.debug (las, '{:13.6e} {:13.6e} {:13.6e}'.format (test, ref, test-ref))
            lib.logger.warn (las, 'LAS states in basis may not be converged (%s = %e)',
                             'max(|Hdiag-e_states|)', maxerr)
    # Error catch: linear dependencies in basis
    try:
        e, c = linalg.eigh (ham_blk, b=ovlp_blk)
    except linalg.LinAlgError as e:
        ovlp_det = linalg.det (ovlp_blk)
        lc = 'checking if LASSI basis has lindeps: |ovlp| = {:.6e}'.format (ovlp_det)
        lib.logger.info (las, 'Caught error %s, %s', str (e), lc)
        if ovlp_det < LINDEP_THRESHOLD:
            err_str = ('LASSI basis appears to have linear dependencies; '
                       'double-check your state list.\n'
                       '|ovlp| = {:.6e}').format (ovlp_det)
            raise RuntimeError (err_str) from e
        else: raise (e) from None
    return e, c, s2_blk

def make_stdm12s (las, ci=None, orbsym=None, soc=False, break_symmetry=False, opt=1):
    ''' Evaluate <I|p'q|J> and <I|p'r'sq|J> where |I>, |J> are LAS states.

        Args:
            las: LASCI object

        Kwargs:
            ci: list of list of ci vectors
            orbsym: None or list of orbital symmetries spanning the whole orbital space
            soc: logical
                Whether to include the effects of spin-orbit coupling (in the 1-RDMs only)
            break_symmetry: logical
                Whether to allow coupling between states of different point-group irreps
            opt: Optimization level, i.e.,  take outer product of
                0: CI vectors
                1: TDMs

        Returns:
            stdm1s: ndarray of shape (nroots,2,ncas,ncas,nroots) if soc==False;
                or of shape (nroots,2*ncas,2*ncas,nroots) if soc==True.
            stdm2s: ndarray of shape (nroots,2,ncas,ncas,2,ncas,ncas,nroots)
    '''
    # NOTE: A spin-pure dm1s is two ncas-by-ncas matrices,
    #    _______    _______
    #    |     |    |     |
    #  [ | a'a | ,  | b'b | ]
    #    |     |    |     |
    #    -------    -------  
    # Spin-orbit coupling generates the a'b and b'a sectors, which are
    # in the missing off-diagonal blocks,
    # _____________
    # |     |     |  
    # | a'a | a'b |  
    # |     |     |  
    # -------------
    # |     |     |  
    # | b'a | b'b |  
    # |     |     |  
    # -------------

    if ci is None: ci = las.ci
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    o0_memcheck = op_o0.memcheck (las, ci, soc=soc)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    # Loop over symmetry blocks
    statesym = las_symm_tuple (las, break_spin=soc, break_symmetry=break_symmetry, verbose=0)[0]
    idx_allprods = []
    d1s_all = []
    d2s_all = []
    nprods = 0
    for las1, sym, indices, indexed in iterate_subspace_blocks (las, ci, statesym):
        idx_sp, idx_prod = indices
        ci_blk, nelec_blk = indexed
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        wfnsym = None if break_symmetry else sym[-1]
        # TODO: implement SOC in op_o1 and then re-enable the debugging block below
        if (las.verbose > lib.logger.INFO) and (o0_memcheck) and (soc==False):
            d1s, d2s = op_o0.make_stdm12s (las1, ci_blk, nelec_blk, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_stdm12s rootsym {} CI algorithm'.format (
                sym), *t0)
            d1s_test, d2s_test = op_o1.make_stdm12s (las1, ci_blk, nelec_blk)
            t0 = lib.logger.timer (las, 'LASSI make_stdm12s rootsym {} TDM algorithm'.format (
                sym), *t0)
            lib.logger.debug (las,
                'LASSI make_stdm12s rootsym {}: D1 o0-o1 algorithm disagreement = {}'.format (
                    sym, linalg.norm (d1s_test - d1s))) 
            lib.logger.debug (las,
                'LASSI make_stdm12s rootsym {}: D2 o0-o1 algorithm disagreement = {}'.format (
                    sym, linalg.norm (d2s_test - d2s))) 
            errvec = np.concatenate ([(d1s-d1s_test).ravel (), (d2s-d2s_test).ravel ()])
            if np.amax (np.abs (errvec)) > 1e-8:#
                raise LASSIOop01DisagreementError ("State-transition density matrices", errvec)
            if opt == 1:
                d1s = d1s_test
                d2s = d2s_test
        else:
            if not o0_memcheck: lib.logger.debug (
                las, 'Insufficient memory to test against o0 LASSI algorithm')
            d1s, d2s = op[opt].make_stdm12s (las1, ci_blk, nelec_blk, orbsym=orbsym, wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_stdm12s rootsym {}'.format (sym), *t0)
        idx_allprods.append (list(np.where(idx_prod)[0]))
        nprods += len (idx_allprods[-1])
        d1s_all.append (d1s)
        d2s_all.append (d2s)

    # Sort block-diagonal matrices
    norb = las.ncas
    if soc:
        stdm1s = np.zeros ((nprods, nprods, 2*norb, 2*norb),
            dtype=ci[0][0].dtype).transpose (0,2,3,1)
    else:
        stdm1s = np.zeros ((nprods, nprods, 2, norb, norb),
            dtype=ci[0][0].dtype).transpose (0,2,3,4,1)
    # TODO: 2e- SOC
    stdm2s = np.zeros ((nprods, nprods, 2, norb, norb, 2, norb, norb),
        dtype=ci[0][0].dtype).transpose (0,2,3,4,5,6,7,1)
    for idx_prod, d1s, d2s in zip (idx_allprods, d1s_all, d2s_all):
        for (i,a), (j,b) in product (enumerate (idx_prod), repeat=2):
            stdm1s[a,...,b] = d1s[i,...,j]
            stdm2s[a,...,b] = d2s[i,...,j]
    return stdm1s, stdm2s

def roots_make_rdm12s (las, ci, si, orbsym=None, soc=None, break_symmetry=None, opt=1):
    '''Evaluate 1- and 2-electron reduced density matrices of LASSI states

        Args:
            las: LASCI object
            ci: list of list of ci vectors
            si: tagged ndarray of shape (nroots,nroots)
               Linear combination vectors defining LASSI states.
               Requires tag "rootsym"

        Kwargs:
            orbsym: None or list of orbital symmetries spanning the whole orbital space
            soc: logical
                Whether to include the effects of spin-orbit coupling (in the 1-RDMs only)
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            break_symmetry: logical
                Whether to allow coupling between states of different point-group irreps
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            opt: Optimization level, i.e.,  take outer product of
                0: CI vectors
                1: TDMs

        Returns:
            rdm1s: ndarray of shape (nroots,2,ncas,ncas) if soc==False;
                or of shape (nroots,2*ncas,2*ncas) if soc==True.
            rdm2s: ndarray of shape (nroots,2,ncas,ncas,2,ncas,ncas)
    '''
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    if soc is None: soc = si.soc
    if break_symmetry is None: break_symmetry = si.break_symmetry
    o0_memcheck = op_o0.memcheck (las, ci, soc=soc)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    # Initialize matrices
    norb = las.ncas
    nroots = si.shape[1]
    if soc:
        rdm1s = np.zeros ((nroots, 2*norb, 2*norb),
            dtype=si.dtype)
    else:
        rdm1s = np.zeros ((nroots, 2, norb, norb),
            dtype=si.dtype)
    # TODO: 2e- SOC
    rdm2s = np.zeros ((nroots, 2, norb, norb, 2, norb, norb),
        dtype=si.dtype)

    # Loop over symmetry blocks
    statesym = las_symm_tuple (las, break_spin=soc, break_symmetry=break_symmetry, verbose=0)[0]
    rootsym = [tuple (x) for x in si.rootsym]
    for las1, sym, indcs, indxd in iterate_subspace_blocks(las,ci,statesym,subset=set(rootsym)):
        idx_ci, idx_prod = indcs
        ci_blk, nelec_blk = indxd
        idx_si = np.all (np.array (rootsym) == sym, axis=1)
        wfnsym = None if break_symmetry else sym[-1]
        si_blk = si[np.ix_(idx_prod,idx_si)]
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        # TODO: implement SOC in op_o1 and then re-enable the debugging block below
        if (las.verbose > lib.logger.INFO) and (o0_memcheck) and (soc==False):
            d1s, d2s = op_o0.roots_make_rdm12s (las1, ci_blk, nelec_blk, si_blk, orbsym=orbsym,
                                                wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_rdm12s rootsym {} CI algorithm'.format (sym),
                                   *t0)
            d1s_test, d2s_test = op_o1.roots_make_rdm12s (las1, ci_blk, nelec_blk, si_blk)
            t0 = lib.logger.timer (las, 'LASSI make_rdm12s rootsym {} TDM algorithm'.format (sym),
                                   *t0)
            lib.logger.debug (las,
                'LASSI make_rdm12s rootsym {}: D1 o0-o1 algorithm disagreement = {}'.format (
                    sym, linalg.norm (d1s_test - d1s))) 
            lib.logger.debug (las,
                'LASSI make_rdm12s rootsym {}: D2 o0-o1 algorithm disagreement = {}'.format (
                    sym, linalg.norm (d2s_test - d2s))) 
            errvec = np.concatenate ([(d1s-d1s_test).ravel (), (d2s-d2s_test).ravel ()])
            if np.amax (np.abs (errvec)) > 1e-8 and soc == False: # tmp until SOC in for op_o1
                raise LASSIOop01DisagreementError ("LASSI mixed-state RDMs", errvec)
            if opt == 1:
                d1s = d1s_test
                d2s = d2s_test
        else:
            if not o0_memcheck: lib.logger.debug (las,
                'Insufficient memory to test against o0 LASSI algorithm')
            d1s, d2s = op[opt].roots_make_rdm12s (las1, ci_blk, nelec_blk, si_blk, orbsym=orbsym,
                                                  wfnsym=wfnsym)
            t0 = lib.logger.timer (las, 'LASSI make_rdm12s rootsym {}'.format (sym), *t0)
        idx_int = np.where (idx_si)[0]
        for (i,a) in enumerate (idx_int):
            rdm1s[a] = d1s[i]
            rdm2s[a] = d2s[i]
    return rdm1s, rdm2s

def root_make_rdm12s (las, ci, si, state=0, orbsym=None, soc=None, break_symmetry=None, opt=1):
    '''Evaluate 1- and 2-electron reduced density matrices of one single LASSI state

        Args:
            las: LASCI object
            ci: list of list of ci vectors
            si: tagged ndarray of shape (nroots,nroots)
               Linear combination vectors defining LASSI states.
               Requires tag "rootsym"

        Kwargs:
            orbsym: None or list of orbital symmetries spanning the whole orbital space
            soc: logical
                Whether to include the effects of spin-orbit coupling (in the 1-RDMs only)
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            break_symmetry: logical
                Whether to allow coupling between states of different point-group irreps
                Overrides tag of si if provided by caller. I have no idea what will happen
                if they contradict. This should probably be removed.
            opt: Optimization level, i.e.,  take outer product of
                0: CI vectors
                1: TDMs

        Returns:
            rdm1s: ndarray of shape (2,ncas,ncas) if soc==False;
                or of shape (2*ncas,2*ncas) if soc==True.
            rdm2s: ndarray of shape (2,ncas,ncas,2,ncas,ncas)
    '''
    si_column = si[:,state:state+1]
    if soc is None: soc = si.soc
    if break_symmetry is None: break_symmetry = si.break_symmetry
    rootsym = si.rootsym[state]
    si_column = tag_array (si_column, rootsym=[rootsym])
    rdm1s, rdm2s = roots_make_rdm12s (las, ci, si_column, orbsym=orbsym, soc=soc,
                                      break_symmetry=break_symmetry, opt=opt)
    return rdm1s[0], rdm2s[0]

class LASSI(lib.StreamObject):
    '''
    LASSI Method class
    '''
    def __init__(self, las, mo_coeff=None, ci=None, veff_c=None, h2eff_sub=None, orbsym=None, soc=False, break_symmetry=False, opt=1,  **kwargs):
        from mrh.my_pyscf.mcscf.lasci import LASCINoSymm
        if isinstance(las, LASCINoSymm): self._las = las
        else: raise RuntimeError("LASSI requires las instance")
        self.__dict__.update(las.__dict__)
        keys = set(('e_roots', 'si', 's2', 's2_mat', 'nelec', 'wfnsym', 'rootsym', 'break_symmetry', 'soc', 'opt'))
        self.e_roots = None
        self.si = None
        self.s2 = None
        self.s2_mat = None
        self.nelec = None
        self.wfnsym = None
        self.rootsym = None
        self.break_symmetry = break_symmetry
        self.soc = soc
        self.opt = opt
        self._keys = set((self.__dict__.keys())).union(keys)

    def kernel(self, mo_coeff=None, ci=None, veff_c=None, h2eff_sub=None, orbsym=None, soc=False,\
               break_symmetry=False, opt=1,  **kwargs):
        e_roots, si = lassi(self._las, mo_coeff=mo_coeff, ci=ci, veff_c=veff_c, h2eff_sub=h2eff_sub, orbsym=orbsym, \
                            soc=soc, break_symmetry=break_symmetry, opt=opt)
        self.e_roots = e_roots
        self.si, self.s2, self.s2_mat, self.nelec, self.wfnsym, self.rootsym, self.break_symmetry, self.soc  = \
            si, si.s2, si.s2_mat, si.nelec, si.wfnsym, si.rootsym, si.break_symmetry, si.soc
        return self.e_roots, self.si
