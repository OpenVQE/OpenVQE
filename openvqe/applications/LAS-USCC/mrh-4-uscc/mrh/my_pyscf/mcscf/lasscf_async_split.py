import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.df.df_jk import _DFHF
from mrh.util.la import safe_svd_warner

class FragSizeError (RuntimeError):
    pass

class LASImpurityOrbitalCallable (object):
    '''Construct an impurity subspace for a specific "fragment" of a LASSCF calculation defined
    as the union of a set of trial orbitals for a particular localized active subspace and the
    AO basis of a specified collection of atoms.

    Constructor args:
        las : object of :class:`LASCINoSymm`
            Mined for basic configuration info about the problem: `mole` object, _scf, ncore,
            ncas_sub, with_df, mo_coeff. The last is copied at construction and should be any
            othornormal basis (las.mo_coeff.conj ().T @ mol.get_ovlp () @ las.mo_coeff = a unit
            matrix)
        frag_id : integer or None
            identifies which active subspace is associated with this fragment
        frag_orbs : list of integer
            identifies which AOs are identified with this fragment

    Calling args:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains MO coefficients
        dm1s : ndarray of shape (2,nao,nao)
            State-averaged spin-separated 1-RDM in the AO basis
        veff : ndarray of shape (2,nao,nao)
            State-averaged spin-separated effective potential in the AO basis
        fock1 : ndarray of shape (nmo,nmo)
            First-order effective Fock matrix in the MO basis
        
    Returns:
        fo_coeff : ndarray of shape (nao, *)
            Orbitals defining an unentangled subspace containing the frag_idth set of active
            orbitals and frag_orbs
        nelec_fo : 2-tuple of integers
            Number of electrons (spin-up, spin-down) in the impurity subspace
    '''

    def __init__(self, las, frag_id, frag_orbs, schmidt_thresh=1e-8, nelec_int_thresh=1e-4):
        self.mol = las.mol
        self._scf = las._scf
        self.ncore, self.ncas_sub, self.ncas = las.ncore, las.ncas_sub, las.ncas
        self.with_df = getattr (las, 'with_df', None)
        self.frag_id = frag_id
        self.schmidt_thresh = schmidt_thresh
        self.nelec_int_thresh = nelec_int_thresh
        self.do_gradorbs = True

        # Convenience
        self.las0 = self.nlas = self.las1 = 0
        if frag_id is not None:
            self.las0 = self.ncore + sum(self.ncas_sub[:frag_id])
            self.nlas = self.ncas_sub[frag_id]
            self.las1 = self.las0 + self.nlas
        self.ncas = sum (self.ncas_sub)
        self.nocc = self.ncore + self.ncas
        self.s0 = self._scf.get_ovlp ()
        self.log = lib.logger.new_logger (las, las.verbose)
        self.svd = safe_svd_warner (self.log.warn)

        # For now, I WANT these to overlap for different atoms. That is why I am pretending that
        # self.s0 is block-diagonal (so that <*|f>.(<f|f>^-1).<f|*> ~> P_f <f|f> P_f).
        self.frag_orbs = frag_orbs
        self.set_utility_basis_(las.mo_coeff)

    def set_utility_basis_(self, oo_coeff):
        self.oo_coeff = oo_coeff.copy ()
        frag_orbs = self.frag_orbs
        self.hcore = self.oo_coeff.conj ().T @ self._scf.get_hcore () @ self.oo_coeff
        self.soo_coeff = self.s0 @ self.oo_coeff
        self.nao_frag = len (frag_orbs)
        ovlp_frag = self.s0[frag_orbs,:][:,frag_orbs] # <f|f> in the comment above
        proj_oo = self.oo_coeff[frag_orbs,:] # P_f . oo_coeff in the comment above
        s1 = proj_oo.conj ().T @ ovlp_frag @ proj_oo
        w, u = linalg.eigh (-s1) # negative: sort from largest to smallest
        self.frag_umat = u[:,:self.nao_frag]

    def __call__(self, mo_coeff, dm1s, veff, fock1, max_size='mid'):
        # TODO: how to handle active/active rotations
        self.log.info ("nmo = %d", mo_coeff.shape[1])

        # Everything in the utility basis
        oo, soo = self.oo_coeff, self.soo_coeff
        mo = soo.conj ().T @ mo_coeff
        dm1s_ = [0,0]
        veff_ = [0,0]
        dm1s_[0] = soo.conj ().T @ dm1s[0] @ soo
        dm1s_[1] = soo.conj ().T @ dm1s[1] @ soo
        veff_[0] = oo.conj ().T @ veff[0] @ oo
        veff_[1] = oo.conj ().T @ veff[1] @ oo
        dm1s = dm1s_
        veff = veff_
        fock1 = mo @ fock1 @ mo.conj ().T
        self._test_orthonormality (mo_coeff, mo)
        _ = self._get_nelec_fo (mo, dm1s, tag='whole molecule')
        # ^ This is a sanity test of the density matrix. If dm1s or mo somehow
        # diverged or were transformed improperly, it might show a non-integer
        # total number of electrons.

        def build (max_size, fock1):
            fo, eo = self._get_orthnorm_frag (mo)
            _ = self._get_nelec_fo (np.append (fo[:,self.nlas:], eo, axis=1),
                                    dm1s, tag='whole-molecule non-active (first check)')
            # ^ If the MOs are somehow not orthonormal, this sanity test might fail
            # even if the one above passes
            self.log.info ("nfrag before gradorbs = %d", fo.shape[1])
            if isinstance (max_size, str) and "small" in max_size.lower():
                max_size = 2*fo.shape[1]
            fo, eo, fock1 = self._a2i_gradorbs (fo, eo, fock1, veff, dm1s)
            _ = self._get_nelec_fo (np.append (fo[:,self.nlas:], eo, axis=1),
                                    dm1s, tag='whole-molecule non-active (second check)')
            # ^ If a2i_gradorbs is broken somehow
            if self.do_gradorbs: self.log.info ("nfrag after gradorbs 1 = %d", fo.shape[1])
            if isinstance (max_size, str) and "mid" in max_size.lower():
                max_size = 2*fo.shape[1]
            fo, eo = self._schmidt (fo, eo, mo)
            self.log.info ("nfrag after schmidt = %d", fo.shape[1])
            nelec_fo = self._get_nelec_fo (fo, dm1s, tag='pre-ia2x_gradorbs impurity')
            self.log.info ("nelec in fragment = %d", nelec_fo)
            if isinstance (max_size, str) and "large" in max_size.lower():
                max_size = 2*fo.shape[1]
            if max_size < fo.shape[1]:
                raise FragSizeError ("max_size of {} is too small".format (max_size))
            fo, eo = self._ia2x_gradorbs (fo, eo, mo, fock1, max_size)
            self.log.info ("nfrag after gradorbs 2 = %d", fo.shape[1])
            nelec_fo = self._get_nelec_fo (fo, dm1s, tag='final impurity')
            self.log.info ("nelec in fragment = %d", nelec_fo)
            nelec_eo = self._get_nelec_fo (eo, dm1s, tag='final environment')
            self.log.info ("%d occupied and %d unoccupied inactive unentangled env orbs",
                           nelec_eo//2, eo.shape[1] - (nelec_eo//2))
            fo_coeff = self.oo_coeff @ fo
            return fo_coeff, nelec_fo

        try:
            fo_coeff, nelec_fo = build (max_size, fock1)
        except FragSizeError as e:
            self.log.warn (("Attempting to satisfy request for a smaller fragment by "
                            "discarding gradient orbitals"))
            with lib.temporary_env (self, do_gradorbs=False):
                fo_coeff, nelec_fo = build (max_size, fock1)
        return fo_coeff, nelec_fo

    def _test_orthonormality (self, mo_coeff, mo):
        def _test (s1, tag=None):
            errmax = np.amax (np.abs (s1-np.eye(s1.shape[0])))
            if errmax>1e-4:
                self.log.warn ('MO coeffs in %s basis not orthonormal: %e',
                               tag, errmax)
        _test (mo_coeff.conj ().T @ self.s0 @ mo_coeff, 'AO')
        _test (mo.conj ().T @ mo, 'OO')

    def _get_orthnorm_frag (self, mo, ovlp_tol=1e-4):
        '''Get an orthonormal basis spanning the union of the frag_idth active space and
        ao_coeff, projected orthogonally to all other active subspaces.

        Args:
            mo : ndarray of shape (nmo,nmo)
                Contains MO coefficients in self.oo_coeff basis

        Kwargs:
            ovlp_tol : float
                Minimimum singular value for a proposed fragment orbital to be included. Tighter
                values of this tolerance lead to larger impurities with less-localized orbitals.
                On the other hand, looser values may lead to impurity orbital spaces that don't
                entirely span the requested AOs.

        Returns:
            fo : ndarray of shape (nmo,*)
                Contains frag_idth active orbitals plus frag_orbs approximately projected onto the
                inactive/external space in self.oo_coeff basis
            eo : ndarray of shape (nmo,*)
                Contains complementary part of the inactive/external space in self.oo_coeff basis
        '''
        # TODO: edge case for no active orbitals
        fo = mo[:,self.las0:self.las1]
        idx = np.ones (mo.shape[1], dtype=np.bool_)
        idx[self.ncore:self.nocc] = False
        uo = mo[:,idx]

        s1 = uo.conj ().T @ self.frag_umat
        u, svals, vh = self.svd (s1, full_matrices=True)
        idx = np.zeros (u.shape[1], dtype=np.bool_)
        idx[:len(svals)][np.abs (svals)>=ovlp_tol] = True
        fo = np.append (fo, uo @ u[:,idx], axis=-1)

        eo = uo @ u[:,~idx]

        return fo, eo

    def _a2i_gradorbs (self, fo, eo, fock1, veff, dm1s):
        '''Augment fragment-orbitals with environment orbitals coupled by the gradient to the
        active space

        Args:
            fo : ndarray of shape (nmo,*)
                Contains fragment-orbital coefficients in self.oo_coeff basis
            eo : ndarray of shape (nmo,*)
                Contains environment-orbital coefficients in self.oo_coeff basis
            fock1 : ndarray of shape (nmo,nmo)
                First-order effective Fock matrix in self.oo_coeff basis
            veff : ndarray of shape (2,nmo,nmo)
                State-averaged spin-separated effective potential in self.oo_coeff basis
            dm1s : ndarray of shape (2,nmo,nmo)
                State-averaged spin-separated 1-RDM in self.oo_coeff basis

        Returns:
            fo : ndarray of shape (nmo,*)
                Same as input, except with self.nlas additional gradient-coupled env orbs
            eo : ndarray of shape (nmo,*)
                Same as input, less the orbitals added to fo
            fock1 : ndarray of shape (nmo,nmo)
                Same as input, after an approximate step towards optimizing the active orbitals
        '''
        iGa = eo.conj ().T @ (fock1-fock1.T) @ fo[:,:self.nlas]
        if not (iGa.size and self.do_gradorbs): return fo, eo, fock1
        u, svals, vh = self.svd (iGa, full_matrices=True)
        ngrad = min (self.nlas, u.shape[1])
        fo = np.append (fo, eo @ u[:,:ngrad], axis=1)
        eo = eo @ u[:,ngrad:]
        mo = np.append (fo, eo, axis=1)

        # Get an estimated active-orbital relaxation step size
        ao, uo = fo[:,:self.nlas], fo[:,self.nlas:]
        uGa = uo.conj ().T @ (fock1-fock1.T) @ ao
        u, uGa, vh = self.svd (uGa, full_matrices=False)
        uo = uo @ u[:,:self.nlas]
        ao = ao @ vh[:self.nlas,:].conj ().T
        f0 = self.hcore[None,:,:] + veff
        f0_aa = (np.dot (f0, ao) * ao[None,:,:]).sum (1)
        f0_uu = (np.dot (f0, uo) * uo[None,:,:]).sum (1)
        f0_ua = (np.dot (f0, ao) * uo[None,:,:]).sum (1)
        dm1s_aa = (np.dot (dm1s, ao) * ao[None,:,:]).sum (1)
        dm1s_uu = (np.dot (dm1s, uo) * uo[None,:,:]).sum (1)
        dm1s_ua = (np.dot (dm1s, ao) * uo[None,:,:]).sum (1)
        uHa = ((f0_aa*dm1s_uu) + (f0_uu*dm1s_aa) - (2*f0_ua*dm1s_ua)).sum (0)
        uXa = (u * ((-uGa/uHa)[None,:])) @ vh # x = -b/A
        kappa1 = np.zeros ((mo.shape[1], mo.shape[1]), dtype=mo.dtype)
        kappa1[self.nlas:fo.shape[1],:self.nlas] = uXa
        kappa1 -= kappa1.T 
        kappa1 = mo @ kappa1 @ mo.conj ().T

        # approximate update to fock1
        tdm1 = -np.dot (dm1s, kappa1)
        tdm1 += tdm1.transpose (0,2,1)
        v1 = self._get_veff (tdm1)
        fock1 += (fock1@kappa1 - kappa1@fock1) / 2
        fock1 += f0[0]@tdm1[0] + f0[1]@tdm1[1]
        fock1 += v1[0]@dm1s[0] + v1[1]@dm1s[1]

        return fo, eo, fock1

    def _schmidt (self, fo, eo, mo):
        '''Do the Schmidt decomposition of the inactive determinant

        Args:
            fo : ndarray of shape (nao,*)
                Contains fragment-orbital coefficients in self.oo_coeff basis
            eo : ndarray of shape (nao,*)
                Contains environment-orbital coefficients in self.oo_coeff basis
            mo : ndarray of shape (nao,nmo)
                Contains MO coefficients in self.oo_coeff basis

        Returns:
            fbo : ndarray of shape (nao,*)
                Contains fragment and bath orbital coefficients
            ueo : ndarray of shape (nao,*)
                Contains unentangled inactive/external environment orbital coefficients
        '''
        nf = fo.shape[1] - self.nlas
        # TODO: edge case for eo.shape[1] < fo.shape[1]
        mo_core = mo[:,:self.ncore]
        dm_core = mo_core @ mo_core.conj ().T
        self._schmidt_idempotency_check (dm_core)
        s1 = eo.conj ().T @ dm_core @ fo[:,self.nlas:]
        if not s1.size: return fo, eo
        u, svals, vh = self.svd (s1)
        self._schmidt_svd_check (s1, u, svals, vh)
        idx = np.zeros (u.shape[1], dtype=np.bool_)
        idx[:len(svals)][np.abs(svals)>self.schmidt_thresh] = True
        eo = eo @ u
        fbo = np.append (fo, eo[:,idx], axis=-1)
        ueo = eo[:,~idx]
        return fbo, ueo

    def _schmidt_idempotency_check (self, dm):
        errmax = np.amax (np.abs (dm @ dm - dm))
        if errmax>1e-4:
            self.log.warn ("Schmidt of density matrix with idempotency error = %e", errmax)
        else:
            self.log.info ("Schmidt of density matrix with idempotency error = %e", errmax)
        return

    def _schmidt_svd_check (self, a, u, svals, vh):
        K = len (svals)
        u, vh = u[:,:K], vh[:K]
        a1 = (u * svals[None,:]) @ vh
        errmax = np.amax (np.abs (a1 - a))
        if errmax>1e-4:
            self.log.warn ("Schmidt decomp SVD error = %e", errmax)
        else:
            self.log.info ("Schmidt decomp SVD error = %e", errmax)
        return

    def _ia2x_gradorbs (self, fo, eo, mo, fock1, ntarget):
        '''Augment fragment space with gradient/Hessian orbs

        Args:
            fo : ndarray of shape (nao,*)
                Contains fragment-orbital coefficients in self.oo_coeff basis
            eo : ndarray of shape (nao,*)
                Contains environment-orbital coefficients in self.oo_coeff basis
            mo : ndarray of shape (nao,nmo)
                Contains MO coefficients in self.oo_coeff basis
            fock1 : ndarray of shape (nmo,nmo)
                First-order effective Fock matrix in self.oo_coeff basis
            ntarget : integer
                Desired number of fragment orbitals when all is said and done

        Returns:
            fo : ndarray of shape (nmo,*)
                Same as input, except with additional gradient/Hessian-coupled env orbs
            eo : ndarray of shape (nmo,*)
                Same as input, less the orbitals added to fo
        '''

        # Split environment orbitals into inactive and external
        eSi = eo.conj ().T @ mo[:,:self.ncore]
        if not eSi.size: return fo, eo
        u, svals, vH = self.svd (eSi, full_matrices=True)
        ni = np.count_nonzero (svals>0.5)
        idx = ~np.isclose (svals, np.around (svals))
        if np.count_nonzero(idx):
            self.log.warn ((
                "Can't separate env occ from env virt in async_split ia2x_gradorbs; "
                "Schmidt failure? svals={}".format (svals[idx])))
        eo = eo @ u
        eio = eo[:,:ni]
        exo = eo[:,ni:]
 
        # Separate SVDs to avoid re-entangling fragment to environment
        svals_i = svals_x = np.zeros (0)
        if eio.shape[1]:
            eGf = eio.conj ().T @ (fock1-fock1.T) @ fo
            u_i, svals_i, vh = self.svd (eGf, full_matrices=True)
            eio = eio @ u_i
        if exo.shape[1]:
            eGf = exo.conj ().T @ (fock1-fock1.T) @ fo
            u_x, svals_x, vh = self.svd (eGf, full_matrices=True)
            exo = exo @ u_x
        eo = np.append (eio, exo, axis=1)
        svals = np.append (svals_i, svals_x)
        idx = np.argsort (-np.abs (svals))
        eo = eo[:,idx]
        
        # Augment fo
        nadd = min (u.shape[1], ntarget-fo.shape[1])
        assert (nadd>=0)
        fo = np.append (fo, eo[:,:nadd], axis=1)
        eo = eo[:,nadd:]

        return fo, eo

    def _get_nelec_fo (self, fo, dm1s, tag='this'):
        neleca = (dm1s[0] @ fo).ravel ().dot (fo.conj ().ravel ())
        nelecb = (dm1s[1] @ fo).ravel ().dot (fo.conj ().ravel ())
        nelec = neleca + nelecb
        nelec_err = nelec - int (round (nelec))
        if abs(nelec_err)>self.nelec_int_thresh:
            self.log.warn ("Non-integer number of electrons in %s subspace! (neleca,nelecb)=%f,%f",
                           tag,neleca,nelecb)
        nelec = int (round (nelec))
        return nelec

    def _get_veff (self, dm1s):
        dm1s = lib.einsum ('ip,spq,jq->sij', self.oo_coeff, dm1s, self.oo_coeff.conj ())
        if isinstance (self._scf, _DFHF): # only J is cheaper
            veff = self._scf.get_j (dm=dm1s.sum (0))
            veff = np.stack ([veff, veff], axis=0)
        else: # J and K are equally expensive
            vj, vk = self._scf.get_jk (dm=dm1s)
            veff = vj.sum (0)[None,:,:] - vk
        veff = lib.einsum ('ip,sij,jq->spq', self.oo_coeff.conj (), veff, self.oo_coeff)
        return veff

def get_impurity_space_constructor (las, frag_id, frag_atoms=None, frag_orbs=None):
    '''Construct an impurity subspace for a specific "fragment" of a LASSCF calculation defined
    as the union of a set of trial orbitals for a particular localized active subspace and the
    AO basis of a specified collection of atoms.

    Args:
        las : object of :class:`LASCINoSymm`
            Mined for basic configuration info about the problem: `mole` object, _scf, ncore,
            ncas_sub, with_df, mo_coeff. The last is copied at construction and should be any
            othornormal basis (las.mo_coeff.conj ().T @ mol.get_ovlp () @ las.mo_coeff = a unit
            matrix)
        frag_id : integer or None
            identifies which active subspace is associated with this fragment

    Kwargs:
        frag_atoms : list of integer
            Atoms considered part of the fragment. All AOs associated with these atoms are appended
            to frag_orbs.
        frag_orbs : list of integer
            Individual AOs considered part of this fragment. Combined with all AOs of frag_atoms.

    Returns:
        get_imporbs : callable
            Args:
                mo_coeff : ndarray of shape (nao,nmo)
                    Contains MO coefficients
                dm1s : ndarray of shape (2,nao,nao)
                    State-averaged spin-separated 1-RDM in the AO basis
                veff : ndarray of shape (2,nao,nao)
                    State-averaged spin-separated effective potential in the AO basis
                fock1 : ndarray of shape (nmo,nmo)
                    First-order effective Fock matrix in the MO basis

            Returns:
                fo_coeff : ndarray of shape (nao, *)
                    Orbitals defining an unentangled subspace containing the frag_idth set of
                    active orbitals and frag_orbs
                nelec_fo : 2-tuple of integers
                    Number of electrons (spin-up, spin-down) in the impurity subspace
    '''
    if frag_orbs is None: frag_orbs = []
    if frag_atoms is None: frag_atoms = []
    if len (frag_atoms):
        ao_offset = las.mol.offset_ao_by_atom ()
        frag_orbs += [orb for atom in frag_atoms
                      for orb in list (range (ao_offset[atom,2], ao_offset[atom,3]))]
        frag_orbs = list (np.unique (frag_orbs))
    assert (len (frag_orbs)), 'Must specify fragment orbitals'
    return LASImpurityOrbitalCallable (las, frag_id, frag_orbs)

if __name__=='__main__':
    from mrh.tests.lasscf.c2h4n4_struct import structure as struct
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
    from pyscf import scf

    mol = struct (3.0, 3.0, 'cc-pvdz', symmetry=False)
    mol.verbose = lib.logger.INFO
    mol.output = __file__+'.log'
    mol.build ()

    mf = scf.RHF (mol).run ()
    mc = LASSCF (mf, (4,4), ((3,1),(1,3)), spin_sub=(3,3))
    frag_atom_list = (list (range (3)), list (range (7,10)))
    mo_coeff = mc.localize_init_guess (frag_atom_list, mf.mo_coeff)
    mc.max_cycle_macro = 1
    mc.kernel (mo_coeff)

    print ("Kernel done")
    ###########################
    from mrh.my_pyscf.mcscf.lasci import get_grad_orb
    dm1s = mc.make_rdm1s ()
    veff = mc.get_veff (dm1s=dm1s)
    fock1 = get_grad_orb (mc, hermi=0)
    ###########################
    get_imporbs_0 = get_impurity_space_constructor (mc, 0, frag_atoms=frag_atom_list[0])
    fo_coeff, nelec_fo = get_imporbs_0 (mc.mo_coeff, dm1s, veff, fock1)

    from pyscf.tools import molden
    molden.from_mo (mol, __file__+'.molden', fo_coeff)
