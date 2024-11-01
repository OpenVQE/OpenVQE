import re, sys, time
import numpy as np
import scipy as sp 
from math import floor, ceil
from pyscf import gto, scf, ao2mo, fci
from pyscf.scf.hf import dot_eri_dm
from pyscf.scf.addons import project_mo_nr2nr
from pyscf.symm.addons import symmetrize_space, label_orb_symm
from pyscf.lib import logger
from pyscf.tools import molden
from pyscf.lib.numpy_helper import unpack_tril, pack_tril
from mrh.my_dmet import pyscf_rhf, pyscf_mp2, pyscf_cc, pyscf_casscf, qcdmethelper, pyscf_fci #, chemps2
from mrh.util import params
from mrh.util.basis import *
from mrh.util.io import prettyprint_ndarray as prettyprint
from mrh.util.io import warnings
from mrh.util.rdm import Schmidt_decomposition_idempotent_wrapper, idempotize_1RDM, get_1RDM_from_OEI, get_2RDM_from_2CDM, get_2CDM_from_2RDM, Schmidt_decompose_1RDM
from mrh.util.tensors import symmetrize_tensor
from mrh.util.my_math import is_close_to_integer
from mrh.my_pyscf.tools.jmol import cas_mo_energy_shift_4_jmol
from mrh.my_dmet.orbital_hessian import LASSCFHessianCalculator
from functools import reduce
from itertools import product
import traceback
import sys
import copy

#def make_fragment_atom_list (ints, frag_atom_list, solver_name, active_orb_list = np.empty (0, dtype=int), name="NONE", norbs_bath_max=None, idempotize_thresh=0.0, mf_attr={}, corr_attr={}):
def make_fragment_atom_list (ints, frag_atom_list, solver_name, **kwargs):
    assert (len (frag_atom_list) < ints.mol.natm)
    assert (np.amax (frag_atom_list) < ints.mol.natm)
    ao_offset = ints.mol.offset_ao_by_atom ()
    frag_orb_list = [orb for atom in frag_atom_list for orb in list (range (ao_offset[atom,2], ao_offset[atom,3]))]
    '''
    for atom in range (ints.mol.natm):
        print ("atom_shell_ids({}) = {}".format (atom, ints.mol.atom_shell_ids (atom)))
        print ("angular momentum = {}".format ([ints.mol.bas_angular (shell) for shell in ints.mol.atom_shell_ids (atom)]))
    norbs_in_atom = [int (np.sum ([2 * ints.mol.bas_angular (shell) + 1 for shell in ints.mol.atom_shell_ids (atom)])) for atom in range (ints.mol.natm)]
    print ("norbs_in_atom = {}".format (norbs_in_atom))
    norbs_to_atom = [int (np.sum (norbs_in_atom[:atom])) for atom in range (ints.mol.natm)]
    print ("norbs_to_atom = {}".format (norbs_to_atom))
    frag_orb_list = [i + norbs_to_atom[atom] for atom in frag_atom_list for i in range (norbs_in_atom[atom])]
    print ("frag_orb_list = {}".format (frag_orb_list))
    '''
    print ("Fragment atom list\n{0}\nproduces a fragment orbital list as {1}".format ([ints.mol.atom_symbol (atom) for atom in frag_atom_list], frag_orb_list))
    #return fragment_object (ints, np.asarray (frag_orb_list), solver_name, active_orb_list=np.asarray (active_orb_list), name=name, mf_attr=mf_attr, corr_attr=corr_attr)
    return fragment_object (ints, np.asarray (frag_orb_list), solver_name, **kwargs)

#def make_fragment_orb_list (ints, frag_orb_list, solver_name, active_orb_list = np.empty (0, dtype=int), name="NONE", norbs_bath_max=None, idempotize_thresh=0.0, mf_attr={}, corr_attr={}):
#    return fragment_object (ints, frag_orb_list, solver_name, np.asarray (active_orb_list), name=name, mf_attr=mf_attr, corr_attr=corr_attr)
def make_fragment_orb_list (ints, frag_orb_list, solver_name, **kwargs):
    return fragment_object (ints, frag_orb_list, solver_name, **kwargs)

def dummy_rhf (frag, oneRDM_imp, chempot_imp):
    ''' Skip solving the impurity problem and assume the trial and impurity wave functions are the same. '''
    assert (np.amax (np.abs (chempot_imp)) < 1e-10)
    frag.oneRDM_loc = frag.ints.oneRDM_loc
    frag.twoCDM_imp = None
    frag.E_imp = frag.ints.e_tot
    frag.loc2mo = np.dot (frag.loc2imp, 
        matrix_eigen_control_options (represent_operator_in_basis (frag.ints.activeFOCK, frag.loc2imp),
            sort_vecs=1, only_nonzero_vals=False)[1])

class fragment_object:

    def __init__ (self, ints, frag_orb_list, solver_name, **kwargs): #active_orb_list, name, norbs_bath_max=None, idempotize_thresh=0.0, mf_attr={}, corr_attr={}):

        # Kwargs
        self.active_orb_list = []
        self.frozen_orb_list = [] 
        self.norbs_frag = len (frag_orb_list)
        self.nelec_imp = 0
        self.norbs_bath_max = len (frag_orb_list) 
        self.solve_time = 0.0
        self.frag_name = 'None'
        self.active_space = None
        self.idempotize_thresh = 0.0
        self.bath_tol = 1e-8
        self.num_mf_stab_checks = 0
        self.target_S = 0
        self.target_MS = 0
        self.mol_output = None
        self.mol_stdout = None
        self.debug_energy = False
        self.imp_maxiter = None # Currently only does anything for casscf solver
        self.quasidirect = True # Currently only does anything for rhf (in development)
        self.project_cderi = False
        self.mf_attr = {}
        self.corr_attr = {}
        self.cas_guess_callback = None
        self.molden_missing_aos = False
        self.add_virtual_bath = True
        self.virtual_bath_gradient_svd = False
        self.enforce_symmetry = False
        self.wfnsym = None
        self.quasifrag_ovlp = False
        self.quasifrag_gradient = True
        self.approx_hessbath = True
        self.conv_tol_grad = 1e-4
        for key in kwargs:
            if key in self.__dict__:
                self.__dict__[key] = kwargs[key]
        if 'name' in kwargs:
            self.frag_name = kwargs['name']

        # Args
        self.ints = ints
        self.norbs_tot = self.ints.mol.nao_nr ()
        self.frag_orb_list = frag_orb_list

        # To be assigned by the DMET main object
        self.filehead = None

        # Assign solver function
        solver_function_map = {
            "dummy RHF" : dummy_rhf ,
            "FCI"       : pyscf_fci.solve ,
            "RHF"       : pyscf_rhf.solve ,
            "MP2"       : pyscf_mp2.solve ,
            "CC"        : pyscf_cc.solve ,
            "CASSCF"    : pyscf_casscf.solve
            }
        solver_longname_map = {
            "dummy RHF" : "dummy restricted Hartree-Fock",
            "FCI"       : "full configuration interaction",
            "RHF"       : "restricted Hartree-Fock",
            "MP2"       : "MP2 perturbation theory",
            "CC"        : "coupled-cluster with singles and doubles",
            "CASSCF"    : "complete active space SCF"
            }
        imp_solver_name = re.sub ("\([0-9,]+\)", "", solver_name)
        self.imp_solver_name = imp_solver_name
        self.imp_solver_longname = solver_longname_map[imp_solver_name]
        self.imp_solver_function = solver_function_map[imp_solver_name].__get__(self)
        active_space = re.compile ("\([0-9,]+\)").search (solver_name)
        if active_space:
            self.active_space = eval (active_space.group (0))
            if not (len (self.active_space) == 2):
                raise RuntimeError ("Active space {0} not usable; only CASSCF currently implemented".format (solver.active_space))
            self.imp_solver_longname += " with {0} electrons in {1} active-space orbitals".format (self.active_space[0], self.active_space[1])

                 
        # Set up the main basis functions. Before any Schmidt decomposition all environment states are treated as "core"
        # self.loc2emb is always defined to have the norbs_frag fragment states, the norbs_bath bath states, and the norbs_core core states in that order
        self.restore_default_embedding_basis ()
        self.oneRDMfroz_loc = np.zeros ((self.norbs_tot, self.norbs_tot))
        self.oneSDMfroz_loc = np.zeros ((self.norbs_tot, self.norbs_tot))
        self.twoCDMfroz_tbc = []
        self.loc2tbc        = []
        self.E2froz_tbc     = []
        self.imp_cache      = []
        
        # Impurity Hamiltonian
        self.Ecore_frag   = 0.0  # In case this exists
        self.impham_CONST = None # Does not include nuclear potential
        self.impham_OEI_C = None
        self.impham_OEI_S = None
        self.impham_TEI   = None
        self.impham_CDERI = None

        # Point-group symmetry information
        self.groupname = 'C1'
        self.loc2symm  = [np.eye (self.norbs_tot)]
        self.ir_names  = ['A']
        self.ir_ids    = [0]

        # Basic outputs of solving the impurity problem
        self.E_frag = 0.0
        self.E_imp  = 0.0
        self.oneRDM_loc = self.ints.oneRDM_loc
        self.oneSDM_loc = self.ints.oneSDM_loc
        self.twoCDM_imp = None
        self.loc2mo     = np.zeros((self.norbs_tot,0))
        self.loc2fno    = np.zeros((self.norbs_tot,0))
        self.fno_evals  = None
        self.E2_cum     = 0

        # Outputs of CAS calculations use to fix CAS-DMET
        self.loc2amo       = np.zeros((self.norbs_tot,0))
        self.loc2amo_guess = None
        self.oneRDMas_loc  = np.zeros((self.norbs_tot,self.norbs_tot))
        self.oneSDMas_loc  = np.zeros((self.norbs_tot,self.norbs_tot))
        self.twoCDMimp_amo = np.zeros((0,0,0,0))
        self.eri_gradient  = np.zeros((self.norbs_tot,0,0,0))
        self.ci_as         = None
        self.ci_as_orb     = None
        self.mfmo_printed  = False
        self.impo_printed  = False

        # Initialize some runtime warning bools
        self.Schmidt_done = False
        self.impham_built = False
        self.imp_solved   = False

        # Report
        print ("Constructed a fragment of {0} orbitals for a system with {1} total orbitals".format (self.norbs_frag, self.norbs_tot))
        print ("Using a {0} [{1}] calculation to solve the impurity problem".format (self.imp_solver_longname, self.imp_solver_method))
        print ("Fragment orbitals: {0}".format (self.frag_orb_list))



    # Common runtime warning checks
    ###########################################################################################################################
    def warn_check_Schmidt (self, cstr="NONE"):
        wstr = "Schmidt decomposition not performed at call to {0}. Undefined behavior likely!".format (cstr)
        return warnings.warn (wstr, RuntimeWarning) if (not self.Schmidt_done) else None

    def warn_check_impham (self, cstr="NONE"):
        wstr =  "Impurity Hamiltonian not built at call to {0} (did you redo the Schmidt decomposition".format (cstr)
        wstr += " without rebuilding impham?). Undefined behavior likely!"
        return warnings.warn (wstr, RuntimeWarning) if not self.impham_built else None

    def warn_check_imp_solve (self, cstr="NONE"):
        wstr =  "Impurity problem not solved at call to {0} (did you redo the Schmidt decomposition or".format (cstr)
        wstr += " rebuild the impurity Hamiltonian without re-solving?). Undefined behavior likely!"
        return warnings.warn (wstr, RuntimeWarning) if not self.imp_solved else None



    # Dependent attributes, never to be modified directly
    ###########################################################################################################################
    @property
    def loc2imp (self):
        return self.loc2emb[:,:self.norbs_imp]

    @property
    def loc2core (self):
        return self.loc2emb[:,self.norbs_imp:]

    @property
    def frag2loc (self):
        return self.loc2frag.conjugate ().T

    @property
    def amo2loc (self):
        return self.loc2amo.conjugate ().T

    @property
    def emb2loc (self):
        return self.loc2emb.conjugate ().T

    @property
    def imp2loc (self):
        return self.loc2imp.conjugate ().T

    @property
    def core2loc (self):
        return self.loc2core.conjugate ().T

    @property
    def mo2loc (self):
        return self.loc2mo.conjugate ().T

    @property
    def imp2frag (self):
        return np.dot (self.imp2loc, self.loc2frag)

    @property
    def frag2imp (self):
        return np.dot (self.frag2loc, self.loc2imp)

    @property
    def amo2imp (self):
        return np.dot (self.amo2loc, self.loc2imp)

    @property
    def imp2amo (self):
        return np.dot (self.imp2loc, self.loc2amo)

    @property
    def imp2unac (self):
        return get_complementary_states (self.imp2amo, symmetry=self.imp2symm, enforce_symmetry=self.enforce_symmetry)

    @property
    def mo2imp (self):
        return np.dot (self.mo2loc, self.loc2imp)

    @property
    def imp2mo (self):
        return np.dot (self.imp2loc, self.loc2mo)

    @property
    def amo2frag (self):
        return np.dot (self.amo2loc, self.loc2frag)

    @property
    def frag2amo (self):
        return np.dot (self.frag2loc, self.loc2amo)

    @property
    def symm2loc (self):
        return self.loc2symm.conjugate ().T

    @property
    def imp2symm (self):
        return [orthonormalize_a_basis (np.dot (self.imp2loc, loc2ir)) for loc2ir in self.loc2symm]

    @property
    def is_frag_orb (self):
        r = np.zeros (self.norbs_tot, dtype=bool)
        r[self.frag_orb_list] = True
        return r

    @property
    def is_env_orb (self):
        return np.logical_not (self.is_frag_orb)

    @property
    def env_orb_list (self):
        return np.flatnonzero (self.is_env_orb)

    @property
    def is_active_orb (self):
        r = np.zeros (self.norbs_tot, dtype=bool)
        r[self.active_orb_list] = True
        return r

    @property
    def is_inactive_orb (self):
        return np.logical_not (self.is_active_orb)

    @property
    def inactive_orb_list (self):
        return np.flatnonzero (self.is_inactive_orb)

    @property
    def norbs_as (self):
        return self.loc2amo.shape[1]

    @property
    def nelec_as (self):
        result = np.trace (self.oneRDMas_loc)
        #if is_close_to_integer (result, params.num_zero_atol) == False:
        #    raise RuntimeError ("Somehow you got a non-integer number of electrons in your active space! ({})".format (result))
        return int (round (result))

    @property
    def norbs_tbc (self):
        return loc2tbc.shape[1]

    @property
    def norbs_core (self):
        self.warn_check_Schmidt ("norbs_core")
        return self.norbs_tot - self.norbs_imp

    @property
    def imp_solver_method (self):
        if self.active_space:
            return self.imp_solver_name + str (self.active_space)
        else:
            return self.imp_solver_name

    @property
    def loc2tb_all (self):
        if self.imp_solver_name != 'RHF':
            if self.norbs_as > 0:
                return [self.loc2amo] + self.loc2tbc
            return [self.loc2imp] + self.loc2tbc
        return self.loc2tbc

    @property
    def twoCDM_all (self):
        if self.imp_solver_name != 'RHF':
            if self.norbs_as > 0:
                return [self.twoCDMimp_amo] + self.twoCDMfroz_tbc
            return [self.twoCDM_imp] + self.twoCDMfroz_tbc
        return self.twoCDMfroz_tbc

    @property
    def symmetry (self):
        if len (self.loc2symm) == 1: return False
        return self.groupname

    @symmetry.setter
    def symmetry (self, x):
        if not x:
            self.groupname = 'C1'
            self.loc2symm  = [np.eye (self.norbs_tot)]
            self.ir_names  = ['A']
            self.ir_ids    = [0]
        else: self.groupname = x

    def get_loc2bath (self):
        ''' Don't use this too much... I don't know how it's gonna behave under ofc_emb'''
        loc2nonbath = orthonormalize_a_basis (np.append (self.loc2frag, self.loc2core, axis=1))
        loc2bath = get_complementary_states (loc2nonbath)
        return loc2bath

    def get_true_loc2frag (self):
        return np.eye (self.norbs_tot)[:,self.frag_orb_list]

    def get_loc2canon_imp (self, oneRDM_loc=None, fock_loc=None):
        ''' Be careful: if you don't provide oneRDM_loc and fock_loc both, they might not be consistent with each other '''
        if oneRDM_loc is None: oneRDM_loc = self.oneRDM_loc.copy ()
        if fock_loc is None: fock_loc = self.ints.activeFOCK
        loc2unac_imp = self.loc2imp @ self.imp2unac
        norbs_inac = int (round (compute_nelec_in_subspace (oneRDM_loc, loc2unac_imp))) // 2
        norbs_occ = norbs_inac + self.norbs_as
        unac_ene, loc2unac_imp, unac_lbls = matrix_eigen_control_options (fock_loc, subspace=loc2unac_imp, symmetry=self.loc2symm,
            strong_symm=self.enforce_symmetry, sort_vecs=1, only_nonzero_vals=False)
        amo_occ, loc2amo_imp, amo_lbls = matrix_eigen_control_options (oneRDM_loc, subspace=self.loc2amo, symmetry=self.loc2symm,
            strong_symm=self.enforce_symmetry, sort_vecs=-1, only_nonzero_vals=False)
        occ_canon = np.zeros (self.norbs_imp)
        occ_canon[:norbs_inac] = 2
        occ_canon[norbs_inac:][:self.norbs_as] = amo_occ[:]
        ene_canon = np.zeros (self.norbs_imp)
        ene_canon[:norbs_inac] = unac_ene[:norbs_inac]
        ene_canon[norbs_occ:] = unac_ene[norbs_inac:]
        loc2canon_imp = np.concatenate ([loc2unac_imp[:,:norbs_inac], loc2amo_imp, loc2unac_imp[:,norbs_inac:]], axis=1)
        return loc2canon_imp, occ_canon, ene_canon, norbs_inac

    def get_loc2canon_core (self, all_frags, oneRDM_loc=None, fock_loc=None):
        ''' Be careful: if you don't provide oneRDM_loc and fock_loc both, they might not be consistent with each other '''
        if oneRDM_loc is None: oneRDM_loc = self.oneRDM_loc.copy ()
        if fock_loc is None: fock_loc = self.ints.activeFOCK
        active_frags = [f for f in all_frags if f.norbs_as]
        active_frags = [f for f in all_frags if f is not self]
        loc2amo_core = np.concatenate ([f.loc2amo for f in active_frags], axis=1)
        core2amo = self.core2loc @ loc2amo_core
        core2unac = get_complementary_states (core2amo)
        loc2unac_core = self.loc2core @ core2unac
        norbs_inac = int (round (compute_nelec_in_subspace (oneRDM_loc, loc2unac_core))) // 2
        norbs_as = loc2amo_core.shape[-1]
        norbs_occ = norbs_inac + norbs_as
        unac_ene, loc2unac_core, unac_lbls = matrix_eigen_control_options (fock_loc, subspace=loc2unac_core, symmetry=self.loc2symm,
            strong_symm=self.enforce_symmetry, sort_vecs=1, only_nonzero_vals=False)
        err = measure_subspace_blockbreaking (loc2unac_core, self.loc2symm, self.ir_names)
        labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[unac_lbls], return_counts=True)))
        print ("Core unactive irreps: {}, err = {}".format (labeldict, err))
        loc2inac_core = loc2unac_core[:,:norbs_inac]
        inac_lbls = unac_lbls[:norbs_inac]
        err = measure_subspace_blockbreaking (loc2inac_core, self.loc2symm, self.ir_names)
        labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[inac_lbls], return_counts=True)))
        print ("Core inactive irreps: {}, err = {}".format (labeldict, err))
        loc2virt_core = loc2unac_core[:,norbs_inac:]
        virt_lbls = unac_lbls[norbs_inac:]
        err = measure_subspace_blockbreaking (loc2virt_core, self.loc2symm, self.ir_names)
        labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[virt_lbls], return_counts=True)))
        print ("Core virtual irreps: {}, err = {}".format (labeldict, err))
        amo_occ, loc2amo_core, amo_lbls = matrix_eigen_control_options (oneRDM_loc, subspace=loc2amo_core, symmetry=self.loc2symm,
            strong_symm=self.enforce_symmetry, sort_vecs=-1, only_nonzero_vals=False)
        err = measure_subspace_blockbreaking (loc2amo_core, self.loc2symm, self.ir_names)
        labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[amo_lbls], return_counts=True)))
        print ("Core active irreps: {}, err = {}".format (labeldict, err))
        occ_canon = np.zeros (self.norbs_core)
        occ_canon[:norbs_inac] = 2
        occ_canon[norbs_inac:][:norbs_as] = amo_occ[:]
        ene_canon = np.zeros (self.norbs_core)
        ene_canon[:norbs_inac] = unac_ene[:norbs_inac]
        ene_canon[norbs_occ:] = unac_ene[norbs_inac:]
        loc2canon_core = np.concatenate ([loc2inac_core, loc2amo_core, loc2virt_core], axis=1)
        return loc2canon_core, occ_canon, ene_canon, norbs_inac, norbs_as




    ############################################################################################################################




    # The Schmidt decomposition
    ############################################################################################################################
    def restore_default_embedding_basis (self):
        idx                  = np.append (self.frag_orb_list, self.env_orb_list)
        self.norbs_frag      = len (self.frag_orb_list)
        self.norbs_imp       = self.norbs_frag
        self.loc2frag        = self.get_true_loc2frag ()
        self.loc2emb         = np.eye (self.norbs_tot)[:,idx]
        self.E2_frag_core    = 0
        self.twoCDMfroz_tbc  = []
        self.loc2tbc         = []

    def set_new_fragment_basis (self, loc2frag):
        self.loc2frag = orthonormalize_a_basis (loc2frag)
        self.loc2emb = np.eye (self.norbs_tot)
        self.norbs_frag = loc2frag.shape[1]
        self.norbs_imp = self.norbs_frag
        self.E2_frag_core = 0
        self.twoCDM_froz_tbc = []
        self.loc2tbc = []

    def do_Schmidt (self, oneRDM_loc, all_frags, loc2wmcs, doLASSCF):
        self.imp_cache = []
        if doLASSCF:
            try:
                self.do_Schmidt_LASSCF (oneRDM_loc, all_frags, loc2wmcs)
            except np.linalg.linalg.LinAlgError as e:
                if self.imp_solver_name == 'dummy RHF':
                    print ("Linear algebra error in Schmidt decomposition of {}".format (self.frag_name))
                    print ("Ignoring for now, because this is a dummy fragment")
                    self.Schmidt_done = True
                    self.impham_built = False
                else:
                    raise (e)
        else:
            print ("DMET Schmidt decomposition of {0} fragment".format (self.frag_name))
            self.loc2emb, norbs_bath, self.nelec_imp, self.oneRDMfroz_loc, emb_labels = Schmidt_decomposition_idempotent_wrapper (oneRDM_loc,
                self.loc2frag, self.norbs_bath_max, symmetry=self.loc2symm, fock_helper=self.ints.activeFOCK, enforce_symmetry=self.enforce_symmetry,
                idempotize_thresh=self.idempotize_thresh, bath_tol=self.bath_tol, num_zero_atol=params.num_zero_atol)
            self.norbs_imp = self.norbs_frag + norbs_bath
            self.Schmidt_done = True
            self.impham_built = False
            self.imp_solved = False
            print ("Final impurity for {0}: {1} electrons in {2} orbitals".format (self.frag_name, self.nelec_imp, self.norbs_imp))
        #if self.impo_printed == False:
        #    self.impurity_molden ('imporb_begin')
        #    self.impo_printed = True
        sys.stdout.flush ()

    def do_Schmidt_LASSCF (self, oneRDM_loc, all_frags, loc2wmcs):
        print ("LASSCF Schmidt decomposition of {0} fragment".format (self.frag_name))
        # First, I should add as many "quasi-fragment" states as there are last-iteration active orbitals, just so I don't
        # lose bath states.
        # How many do I need?
        '''
        if self.imp_solver_name == 'dummy RHF':
            self.Schmidt_done = True
            self.impham_built = False
            print ("Skipping Schmidt decomposition for dummy fragment")
            sys.stdout.flush ()
            return
        '''
        if not ('dummy' in self.imp_solver_name):
            w0, t0 = time.time (), time.process_time ()
            self.hesscalc = LASSCFHessianCalculator (self.ints, oneRDM_loc, all_frags, self.ints.activeFOCK,
                self.ints.activeVSPIN, Hop_noxc=self.approx_hessbath) 
            print ("Time in building Hessian constructor: {:.8f} wall, {:.8f} clock".format (time.time () - w0, time.process_time () - t0))
        frag2wmcs = np.dot (self.frag2loc, loc2wmcs)
        proj = np.dot (frag2wmcs.conjugate ().T, frag2wmcs)
        norbs_wmcsf = np.trace (proj)
        norbs_xtra = int (round (self.norbs_frag - norbs_wmcsf))
        assert (norbs_xtra == self.norbs_as), "{} active orbitals, {} extra fragment orbitals".format (self.norbs_as, norbs_xtra)
        err = measure_basis_nonorthonormality (self.loc2frag)
        print ("Fragment orbital overlap error = {}".format (err))


        def _analyze_intermediates (loc2int, tag):
            loc2int = align_states (loc2int, self.loc2symm)
            int_labels = assign_blocks_weakly (loc2int, self.loc2symm)
            err = measure_subspace_blockbreaking (loc2int, self.loc2symm, self.ir_names)
            labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[int_labels], return_counts=True)))
            print ("{} irreps: {}, err = {}".format (tag, labeldict, err))


        # Now get them. (Make sure I don't add active-space orbitals by mistake!)
        if norbs_xtra:
            norbs_qfrag = 0
            loc2wfrag = self.loc2frag
            if self.quasifrag_ovlp:
                nq, loc2wfrag = self.get_quasifrag_ovlp (loc2wfrag, loc2wmcs, norbs_xtra)
                norbs_qfrag += nq
            if self.quasifrag_gradient:
                nq, loc2wfrag = self.get_quasifrag_gradient (loc2wfrag, loc2wmcs, oneRDM_loc, norbs_xtra)
                norbs_qfrag += nq
        else:
            norbs_qfrag = 0
            loc2wfrag = self.loc2frag
            print ("NO quasi-fragment orbitals constructed")

        # This will RuntimeError on me if I don't have integer.
        # For safety's sake, I'll project into wmcs subspace and add wmas part back to self.oneRDMfroz_loc afterwards.
        oneRDMi_loc = project_operator_into_subspace (oneRDM_loc, loc2wmcs)
        oneRDMa_loc = oneRDM_loc - oneRDMi_loc
        self.loc2emb, norbs_bath, self.nelec_imp, self.oneRDMfroz_loc, emb_labels = Schmidt_decomposition_idempotent_wrapper (oneRDMi_loc, loc2wfrag,
            self.norbs_bath_max, symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry, bath_tol=self.bath_tol, fock_helper=self.ints.activeFOCK,
            idempotize_thresh=self.idempotize_thresh, num_zero_atol=params.num_zero_atol)
        self.norbs_imp = self.norbs_frag + norbs_qfrag + norbs_bath
        dm = project_operator_into_subspace (self.oneRDM_loc, self.loc2core)
        #if not ('dummy' in self.imp_solver_name): print ("CHECK ME: initial projection error = {:.6e}".format (linalg.norm (dm - self.oneRDMfroz_loc)))
        q = self.loc2core @ self.core2loc
        dm = q @ self.oneRDM_loc @ q
        #if not ('dummy' in self.imp_solver_name): print ("CHECK ME: initial projection error again = {:.6e}".format (linalg.norm (dm - self.oneRDMfroz_loc)))
        self.Schmidt_done = True
        oneRDMacore_loc = project_operator_into_subspace (oneRDMa_loc, self.loc2core)
        nelec_impa = compute_nelec_in_subspace (oneRDMa_loc, self.loc2imp)
        nelec_impa_target = 0 if self.active_space is None else self.active_space[0]
        print ("Adding {} active-space electrons to impurity and {} active-space electrons to core".format (nelec_impa, np.trace (oneRDMacore_loc)))
        self.oneRDMfroz_loc += oneRDMacore_loc
        self.nelec_imp += int (round (nelec_impa))
        #if not ('dummy' in self.imp_solver_name): print ("CHECK ME: projection error after adding 'oneRDMacore_loc' = {:.6e}".format (linalg.norm (dm - self.oneRDMfroz_loc)))

        # Weak symmetry alignment, for the sake of moldening. (Extended) fragment, then (extended) bath
        if self.symmetry:
            norbs_frag = loc2wfrag.shape[1]
            loc2frag = self.loc2emb[:,:norbs_frag]
            loc2bath = self.loc2emb[:,norbs_frag:self.norbs_imp]
            frag_labels = emb_labels[:norbs_frag].astype (int)
            bath_labels = emb_labels[norbs_frag:self.norbs_imp].astype (int)
            #evals, loc2frag[:,:], frag_labels = matrix_eigen_control_options (oneRDM_loc, symmetry=self.loc2symm,
            #    subspace=loc2frag, sort_vecs=-1, only_nonzero_vals=False, strong_symm=False)
            labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[frag_labels], return_counts=True)))
            err = measure_subspace_blockbreaking (loc2frag, self.loc2symm, self.ir_names)
            print ("Fragment-orbital irreps: {}, err = {}".format (labeldict, err))
            #evals, loc2bath[:,:], bath_labels = matrix_eigen_control_options (oneRDM_loc, symmetry=self.loc2symm,
            #    subspace=loc2bath, sort_vecs=1, only_nonzero_vals=False, strong_symm=False)
            labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[bath_labels], return_counts=True)))
            err = measure_subspace_blockbreaking (loc2bath, self.loc2symm, self.ir_names)
            print ("Bath-orbital irreps: {}, err = {}".format (labeldict, err))

        # I need to work symmetry handling into this as well
        norbs_hessbath = max (0, min (2 * (self.norbs_frag+self.norbs_as), self.norbs_tot) - self.norbs_imp)
        print ("Fragment {} basis set instability virtual orbital loss: 2 * ({} + {}) - {} = {} missing bath orbitals".format (self.frag_name,
            self.norbs_frag, self.norbs_as, self.norbs_imp, norbs_hessbath))
        if norbs_hessbath and self.add_virtual_bath and self.imp_solver_name != 'dummy RHF' and self.norbs_as:
            loc2canon_core, _, _, norbs_inac_core, norbs_as_core = self.get_loc2canon_core (all_frags, oneRDM_loc=oneRDM_loc, fock_loc=self.ints.activeFOCK)
            norbs_occ_core = norbs_inac_core + norbs_as_core
            norbs_virt_core = self.norbs_core - norbs_occ_core
            norbs_unac_core = norbs_inac_core + norbs_virt_core
            loc2canon_imp, _, _, norbs_inac_imp = self.get_loc2canon_imp (oneRDM_loc=oneRDM_loc, fock_loc=self.ints.activeFOCK)
            norbs_occ_imp = norbs_inac_imp + self.norbs_as
            norbs_ninac_imp = self.norbs_imp - norbs_inac_imp
            print (("Searching for {} extra bath orbitals among a set of {}/{} virtuals/inactives "
                " accessible by {}/{} occupied/noninactive orbitals").format (norbs_hessbath, norbs_virt_core, norbs_inac_core, norbs_occ_imp, norbs_ninac_imp))
            # Occupied orbitals in the impurity
            loc2occ_imp = loc2canon_imp[:,:norbs_occ_imp]
            loc2ninac_imp = loc2canon_imp[:,norbs_inac_imp:]
            # Virtual orbitals in the core
            loc2inac_core = loc2canon_core[:,:norbs_inac_core]
            loc2virt_core = loc2canon_core[:,norbs_occ_core:]
            loc2amo_core = loc2canon_core[:,norbs_inac_core:norbs_occ_core]
            loc2unac_core = np.append (loc2inac_core, loc2virt_core, axis=1)
            # Unactive orbitals in the impurity
            loc2unac_imp = np.append (loc2canon_imp[:,:norbs_inac_imp], loc2canon_imp[:,norbs_occ_imp:], axis=1)
            # Sort the core orbitals of loc2emb to get the active orbitals out of the way
            loc2sort_core = np.concatenate ([loc2virt_core, loc2inac_core, loc2amo_core], axis=1)
            self.loc2emb[:,self.norbs_imp:] = loc2sort_core[:,:]
            # Get the conjugate gradient. Push into the loc basis so I can use the weak inner symmetry capability of the svd function
            w0, t0 = time.time (), time.process_time ()
            pq_pairs = ((loc2virt_core, loc2occ_imp), (loc2inac_core, loc2ninac_imp))
            #grad = self.hesscalc.get_conjugate_gradient (pq_pairs, loc2unac_imp, self.loc2amo)
            #grad_test = self.hesscalc.get_impurity_conjugate_gradient (loc2canon_imp, loc2canon_core, self.loc2amo)
            #grad_test = ((loc2virt_core @ grad_test[norbs_occ_core:,:norbs_occ_imp] @ loc2occ_imp.conjugate ().T)
            #            + loc2inac_core @ grad_test[:norbs_inac_core,norbs_inac_imp:] @ loc2ninac_imp.conjugate ().T)
            #print ("Error in grad_test: {}".format (linalg.norm (grad_test-grad)))
            print (norbs_inac_imp, " inactive in impurity; ", norbs_inac_core, " inactive in core")
            grad = self.hesscalc.get_impurity_conjugate_gradient (loc2canon_imp, loc2unac_core, self.loc2amo)
            print ("Time in Hessian module: {:.8f} wall, {:.8f} clock".format (time.time () - w0, time.process_time () - t0))
            # Zero gradient escape
            if np.count_nonzero (np.abs (grad) > 1e-8): 
                # SVD and add to the bath
                self.loc2emb[:,self.norbs_imp:][:,:norbs_virt_core], _, svals_vhb = get_overlapping_states (loc2virt_core, loc2occ_imp,
                    inner_symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry, across_operator=grad,
                    full_matrices=True, only_nonzero_vals=False)[:3]
                offs = self.norbs_imp+norbs_virt_core
                self.loc2emb[:,offs:][:,:norbs_inac_core], _, svals_ohb = get_overlapping_states (loc2inac_core, loc2ninac_imp,
                    inner_symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry, across_operator=grad,
                    full_matrices=True, only_nonzero_vals=False)[:3]
                if norbs_virt_core - len (svals_vhb): svals_vhb = np.append (svals_vhb, np.zeros (norbs_virt_core - len (svals_vhb)))
                if norbs_inac_core - len (svals_ohb): svals_ohb = np.append (svals_ohb, np.zeros (norbs_inac_core - len (svals_ohb)))
                svals = np.append (svals_vhb, svals_ohb)
                core_occ = np.zeros (len (svals))
                core_occ[norbs_virt_core:] = 2
                idx = np.argsort (np.abs (svals))[::-1]
                self.loc2emb[:,self.norbs_imp:][:,:norbs_unac_core] = self.loc2emb[:,self.norbs_imp:][:,:norbs_unac_core][:,idx]
                svals = svals[idx]
                core_occ = core_occ[idx]
                print ("Adding {} virtual orbitals to bath from Hessian".format (np.count_nonzero (core_occ[:norbs_hessbath]==0)))
                print ("Conjugate-gradient svals: " + " ".join (["{:9.2e}".format (sval) for sval in svals[:norbs_hessbath]]))
                print ("Adding {} electrons to impurity from Hessian bath orbitals".format (np.sum (core_occ[:norbs_hessbath])))
                self.nelec_imp += int (round (np.sum (core_occ[:norbs_hessbath])))
                self.norbs_imp += norbs_hessbath
                nelec_err = self.nelec_imp - compute_nelec_in_subspace (oneRDM_loc, self.loc2imp)
                print ("Impurity orbital nelec err after adding Hessian baths = {}".format (nelec_err))

            else:
                print ("Gradient is zero; can't make hessbath using gradient")
        self.oneRDMfroz_loc = project_operator_into_subspace (self.oneRDM_loc, self.loc2core)
        #self.oneRDMfroz_loc -= project_operator_into_subspace (self.oneRDMfroz_loc, self.loc2imp)
        #dm = project_operator_into_subspace (self.oneRDM_loc, self.loc2core)
        #if not ('dummy' in self.imp_solver_name): print ("CHECK ME: projection error after hessbath = {:.6e}".format (linalg.norm (dm - self.oneRDMfroz_loc)))

        # This whole block below me is an old attempt at this that doesn't really work
        '''
        try: 
            norbs_bath_xtra = self.norbs_bath_max - norbs_bath
            loc2virtbath = self.analyze_ao_imp (oneRDM_loc, loc2wmcs, norbs_bath_xtra)
            norbs_virtbath = min (norbs_bath_xtra, loc2virtbath.shape[1])
            if self.add_virtual_bath and norbs_virtbath:
                print ("Adding {} virtual bath orbitals".format (norbs_virtbath))
                self.loc2emb[:,self.norbs_imp:][:,:norbs_virtbath] = loc2virtbath[:,:norbs_virtbath]
                self.norbs_imp += norbs_virtbath
                self.loc2emb = get_complete_basis (self.loc2imp, symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)
                emb_labels = assign_blocks_weakly (self.loc2emb, self.loc2symm)
        except np.linalg.linalg.LinAlgError as e:
            if self.imp_solver_name == 'dummy RHF':
                print ("Generating virtual core/virtual bath orbitals or calculating the gradient SVD failed with a linear algebra error for {}".format (self.frag_name))
                print ("Ignoring error for now, because this is a dummy fragment")
            else:
                raise (e)
        '''

        # Weak symmetry alignment, for the sake of moldening. (Extended) fragment, then (extended) bath
        if self.symmetry:
            norbs_frag = loc2wfrag.shape[1]
            loc2frag = self.loc2emb[:,:norbs_frag]
            loc2bath = self.loc2emb[:,norbs_frag:self.norbs_imp]
            frag_labels = emb_labels[:norbs_frag].astype (int)
            bath_labels = emb_labels[norbs_frag:self.norbs_imp].astype (int)
            #evals, loc2frag[:,:], frag_labels = matrix_eigen_control_options (oneRDM_loc, symmetry=self.loc2symm,
            #    subspace=loc2frag, sort_vecs=-1, only_nonzero_vals=False, strong_symm=False)
            labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[frag_labels], return_counts=True)))
            err = measure_subspace_blockbreaking (loc2frag, self.loc2symm, self.ir_names)
            print ("Fragment-orbital irreps: {}, err = {}".format (labeldict, err))
            #evals, loc2bath[:,:], bath_labels = matrix_eigen_control_options (oneRDM_loc, symmetry=self.loc2symm,
            #    subspace=loc2bath, sort_vecs=1, only_nonzero_vals=False, strong_symm=False)
            labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[bath_labels], return_counts=True)))
            err = measure_subspace_blockbreaking (loc2bath, self.loc2symm, self.ir_names)
            print ("Bath-orbital irreps: {}, err = {}".format (labeldict, err))

        err = measure_basis_nonorthonormality (self.loc2imp)
        print ("Impurity orbital overlap error = {}".format (err))
        err = measure_basis_nonorthonormality (self.loc2core) if self.loc2core.size else 0.0
        print ("Core orbital overlap error = {}".format (err))
        err = measure_basis_nonorthonormality (self.loc2emb)
        print ("Whole embedding basis overlap error = {}".format (err))

        # Core 2CDMs
        active_frags = [frag for frag in all_frags if frag is not self and frag.norbs_as > 0]
        self.twoCDMfroz_tbc = [np.copy (frag.twoCDMimp_amo) for frag in active_frags]
        self.loc2tbc        = [np.copy (frag.loc2amo) for frag in active_frags]
        self.E2froz_tbc     = [frag.E2_cum for frag in active_frags]
        self.oneSDMfroz_loc = sum ([frag.oneSDMas_loc for frag in all_frags if frag is not self])

        self.impham_built = False
        sys.stdout.flush ()
        #dm = project_operator_into_subspace (self.oneRDM_loc, self.loc2core)
        #if not ('dummy' in self.imp_solver_name): print ("CHECK ME: final projection error = {:.5e}".format (linalg.norm (dm - self.oneRDMfroz_loc)))

    def get_quasifrag_ovlp (self, loc2frag, loc2wmcs, norbs_xtra):
        ''' Add "quasi-fragment orbitals" (to compensate for active orbitals' inability to generate bath) to "fragment orbitals" and return "working fragment orbitals
        using overlap on "true fragment" criterion '''
        loc2qfrag, _, svals = get_overlapping_states (loc2wmcs, self.get_true_loc2frag (), inner_symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)[:3]
        loc2qenv = get_complementary_states (loc2qfrag, already_complete_warning=False, symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)
        loc2wmas = get_complementary_states (loc2wmcs, already_complete_warning=False, symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)
        loc2p = orthonormalize_a_basis (np.concatenate ([loc2frag, loc2qenv, loc2wmas], axis=1), symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)
        loc2qfrag = get_complementary_states (loc2p, symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)
        norbs_qfrag = min (loc2qfrag.shape[1], norbs_xtra)
        # Align the symmetry and make sure not to pick only part of a degenerate manifold because this will cause artificial symmetry breaking
        loc2qfrag, _, svals, qfrag_labels, _ = get_overlapping_states (loc2qfrag, self.get_true_loc2frag (), inner_symmetry=self.loc2symm,
            enforce_symmetry=self.enforce_symmetry, full_matrices=True, only_nonzero_vals=False)
        if (len (svals) > norbs_qfrag) and (norbs_qfrag > 0):
            bottom_sval = svals[norbs_qfrag-1]
            ndegen = np.count_nonzero (np.isclose (svals[norbs_qfrag:], bottom_sval))
            if ndegen > 0:
                print ("Warning: adding {} instead of {} quasi-fragment orbitals in order to avoid artificial symmetry breaking by adding only part of a degenerate manifold".format (
                    norbs_qfrag+ndegen, norbs_qfrag))
                print (svals)
            norbs_qfrag += ndegen-1
        if norbs_qfrag > 0:
            print ("Add {} of {} possible quasi-fragment orbitals ".format (
                norbs_qfrag, loc2qfrag.shape[1])
                + "to compensate for {} active orbitals which cannot generate bath states".format (self.norbs_as))
            loc2qfrag = loc2qfrag[:,:norbs_qfrag]
            qfrag_labels = np.asarray (self.ir_names)[qfrag_labels[:norbs_qfrag]]
            qfrag_labels = dict (zip (*np.unique (qfrag_labels, return_counts=True)))
            err = measure_subspace_blockbreaking (loc2qfrag, self.loc2symm, self.ir_names)
            print ("Quasi-fragment irreps = {}, err = {}".format (qfrag_labels, err))
            loc2wfrag = np.append (loc2frag, loc2qfrag, axis=1)
            loc2wfrag, wfrag_labels = matrix_eigen_control_options (self.ints.activeFOCK, subspace=loc2wfrag, symmetry=self.loc2symm,
                strong_symm=self.enforce_symmetry, only_nonzero_vals=False, sort_vecs=1)[1:]
            wfrag_labels = np.asarray (self.ir_names)[wfrag_labels]
            wfrag_labels = dict (zip (*np.unique (wfrag_labels, return_counts=True)))
            err = measure_subspace_blockbreaking (loc2wfrag, self.loc2symm, self.ir_names)
            print ("Working fragment irreps = {}, err = {}".format (wfrag_labels, err))
            err = measure_basis_nonorthonormality (loc2wfrag)
            print ("Working fragment orbital overlap error = {}".format (err))
        else:
            print ("No valid quasi-fragment orbitals found")
            loc2wfrag = loc2frag
        return norbs_qfrag, loc2wfrag

    def get_quasifrag_gradient (self, loc2frag, loc2wmcs, oneRDM_loc, norbs_xtra):
        ''' Add "quasi-fragment orbitals" (to compensate for active orbitals' inability to generate bath) to "fragment orbitals" and return "working fragment orbitals"
        using SVD of orbital rotation gradient '''
        if self.norbs_as == 0:
            return 0, loc2frag
        fock_loc = self.ints.activeFOCK
        loc2amo = self.loc2amo
        loc2env = get_complementary_states (loc2frag, already_complete_warning=False, symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)
        loc2cenv = get_overlapping_states (loc2wmcs, loc2env, inner_symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)[0]
        cenv2loc = loc2cenv.conjugate ().T
        grad = self.hesscalc._get_Fock1 (loc2cenv, loc2amo) - self.hesscalc._get_Fock1 (loc2amo, loc2cenv).T
        '''
        # 1-body part
        grad_comp = fock_loc @ oneRDM_loc
        grad_comp -= grad.T
        # 2-body part
        eri = self.ints.general_tei ([loc2cenv, self.loc2amo, self.loc2amo, self.loc2amo])
        eri_grad = np.tensordot (eri, self.twoCDMimp_amo, axes=((1,2,3),(1,2,3))) # NOTE: just saying axes=3 gives an INCORRECT result
        grad_comp += loc2cenv @ eri_grad @ self.amo2loc
        # Testing hessian calculator
        print ("************************************* TEST ****************************************")
        print ("In first iteration, active orbitals may overlap, which will cause this test to fail")
        grad_comp = cenv2loc @ grad @ loc2amo
        for i, j in product (range (loc2cenv.shape[-1]), range (loc2amo.shape[-1])):
            print ("{} {} {:.9e} {:.9e}".format (i, j, grad[i,j], grad_comp[i,j]))
        print ("*********************************** END TEST **************************************")
        #assert (False)
        '''
        if np.all (np.abs (grad) < 1e-8):
            print ("Gradient appears to be zero; defaulting to overlap criterion for quasifragment orbitals")
            return self.get_quasifrag_ovlp (loc2frag, loc2wmcs, norbs_xtra)
        grad = loc2cenv @ grad @ self.amo2loc
        # SVD
        loc2qfrag, _, svals, qfrag_labels, _ = get_overlapping_states (loc2cenv, loc2amo, inner_symmetry=self.loc2symm,
            enforce_symmetry=self.enforce_symmetry, across_operator=grad, full_matrices=True, only_nonzero_vals=False)
        print ("The gradient norm (outside of the fragment space) of the {} fragment is {}".format (self.frag_name, linalg.norm (svals)))
        print ("Gradient svals: {}".format (svals))
        norbs_qfrag = len (svals)
        if (len (svals) > norbs_qfrag) and (norbs_qfrag > 0):
            bottom_sval = svals[norbs_qfrag-1]
            ndegen = np.count_nonzero (np.isclose (svals[norbs_qfrag:], bottom_sval))
            if ndegen > 0:
                print ("Warning: adding {} instead of {} quasi-fragment orbitals in order to avoid artificial symmetry breaking by adding only part of a degenerate manifold".format (
                    norbs_qfrag+ndegen, norbs_qfrag))
                print (svals)
            norbs_qfrag += ndegen-1
        if norbs_qfrag > 0:
            print ("Add {} of {} possible quasi-fragment orbitals ".format (
                norbs_qfrag, loc2qfrag.shape[1])
                + "to compensate for {} active orbitals which cannot generate bath states".format (self.norbs_as))
            loc2qfrag = loc2qfrag[:,:norbs_qfrag]
            qfrag_labels = np.asarray (self.ir_names)[qfrag_labels[:norbs_qfrag]]
            qfrag_labels = dict (zip (*np.unique (qfrag_labels, return_counts=True)))
            err = measure_subspace_blockbreaking (loc2qfrag, self.loc2symm, self.ir_names)
            print ("Quasi-fragment irreps = {}, err = {}".format (qfrag_labels, err))
            loc2wfrag = np.append (loc2frag, loc2qfrag, axis=1)
            loc2wfrag, wfrag_labels = matrix_eigen_control_options (self.ints.activeFOCK, subspace=loc2wfrag, symmetry=self.loc2symm,
                strong_symm=self.enforce_symmetry, only_nonzero_vals=False, sort_vecs=1)[1:]
            wfrag_labels = np.asarray (self.ir_names)[wfrag_labels]
            wfrag_labels = dict (zip (*np.unique (wfrag_labels, return_counts=True)))
            err = measure_subspace_blockbreaking (loc2wfrag, self.loc2symm, self.ir_names)
            print ("Working fragment irreps = {}, err = {}".format (wfrag_labels, err))
            err = measure_basis_nonorthonormality (loc2wfrag)
            print ("Working fragment orbital overlap error = {}".format (err))
        else:
            print ("No valid quasi-fragment orbitals found")
            loc2wfrag = loc2frag
        return norbs_qfrag, loc2wfrag
        

    def analyze_ao_imp (self, oneRDM_loc, loc2wmcs, norbs_bath_xtra):
        ''' See how much of the atomic-orbitals corresponding to the true fragment ended up in the impurity and how much ended 
            up in the virtual-core space '''
        loc2ao = orthonormalize_a_basis (self.ints.ao2loc[self.frag_orb_list,:].conjugate ().T)
        ao2loc = loc2ao.conjugate ().T
        svals = get_overlapping_states (loc2ao, self.loc2core)[2]
        lost_aos = np.sum (svals)

        mo_occ, mo_evec = matrix_eigen_control_options (oneRDM_loc, sort_vecs=1, only_nonzero_vals=False, subspace=loc2wmcs, symmetry=self.loc2symm, strong_symm=self.enforce_symmetry)[:2]
        idx_virtunac = np.isclose (mo_occ, 0)
        loc2virtunac = mo_evec[:,idx_virtunac]
        ''' These orbitals have to be splittable into purely on the impurity/purely in the core for the same reason that nelec_imp has to
            be an integer, I think. '''
        loc2virtunaccore, loc2corevirtunac, svals = get_overlapping_states (loc2virtunac, self.loc2core, inner_symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)[:3]
        idx_virtunaccore = np.isclose (svals, 1)
        loc2virtunaccore = loc2virtunaccore[:,idx_virtunaccore]

        loc2virtbath, svals, _, virtbath_labels = get_overlapping_states (loc2ao, loc2virtunaccore, inner_symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)[1:]
        assert (len (svals) == loc2virtbath.shape[1])
        if loc2virtbath.shape[1] == 0: return loc2virtbath
        check_nocc = np.trace (represent_operator_in_basis (oneRDM_loc, loc2virtbath))
        check_bath = np.amax (np.abs (np.dot (self.imp2loc, loc2virtbath)))
        print ("Are my `virtual bath' orbitals unoccupied and currently in the core? check_nocc = {}, check_bath = {}".format (check_nocc, check_bath))
        aos_in_virt_core = np.count_nonzero (np.logical_not (np.isclose (svals, 0))) 
        print (("For this Schmidt decomposition, the impurity basis loses {} of {} atomic orbitals, accounted for by"
        " {} of {} virtual core orbitals").format (lost_aos, self.norbs_frag, aos_in_virt_core, loc2virtbath.shape[1]))
        norbs_xtra = min (loc2virtbath.shape[1], norbs_bath_xtra)
        labels = virtbath_labels[:norbs_xtra]
        labeldict = dict (zip (*np.unique (np.asarray (self.ir_names)[labels], return_counts=True)))
        err = measure_subspace_blockbreaking (loc2virtbath[:,:norbs_xtra], self.loc2symm)
        print (("The first {} virtual bath orbitals account for approximately {} missing fragment "
                "orbitals\n and have irreps {}, err = {}").format (norbs_xtra, sum (svals[:norbs_xtra]), labeldict, err))
        virtbathGocc, occ_labels = self.gradient_for_virtbath (loc2virtbath, oneRDM_loc, loc2wmcs, fock=self.ints.activeFOCK)
        my_ene = -svals * svals
        my_occ = virtbathGocc.sum (1)
        if self.virtual_bath_gradient_svd:
            umat, svals_fock, vmat = matrix_svd_control_options (virtbathGocc, sort_vecs=-1, only_nonzero_vals=False, lsymm=virtbath_labels, rsymm=occ_labels, full_matrices=True, strong_symm=self.enforce_symmetry)[:3]
            idx = np.argsort (np.abs (svals_fock))[::-1]
            umat[:,:len (svals_fock)] = umat[:,:len (svals_fock)][:,idx]
            vmat[:,:len (svals_fock)] = vmat[:,:len (svals_fock)][:,idx]
            svals_fock = svals_fock[idx]
            check_nocc = np.trace (represent_operator_in_basis (oneRDM_loc, loc2virtbath @ umat))
            check_bath = np.amax (np.abs (np.dot (self.imp2loc, loc2virtbath @ umat)))
            print ("Are my `virtual bath' orbitals unoccupied and currently in the core? check_nocc = {}, check_bath = {}".format (check_nocc, check_bath))
            print ("Maximum gradient singular value for virtual bath orbitals: {}".format (svals_fock[0]))
            loc2virtbath = loc2virtbath @ umat
            my_occ = (loc2ao.conjugate ().T @ loc2virtbath).sum (0)
            my_ene = -svals_fock
        if self.molden_missing_aos:
            ao2molden = np.dot (self.ints.ao2loc, loc2virtbath)
            molden.from_mo (self.ints.mol, self.filehead + self.frag_name + '_missing_AOs.molden', ao2molden, ene=my_ene, occ=my_occ)
            self.molden_missing_aos = False
        return loc2virtbath

    def gradient_for_virtbath (self, loc2virtbath, oneRDM_loc, loc2wmcs, fock=None):
        loc2imp_unac = get_overlapping_states (self.loc2imp, loc2wmcs, only_nonzero_vals=True, inner_symmetry=self.loc2symm, enforce_symmetry=self.enforce_symmetry)[0]
        evals, loc2no_unac, no_labels = matrix_eigen_control_options (oneRDM_loc, subspace=loc2imp_unac,
            symmetry=self.loc2symm, sort_vecs=-1, only_nonzero_vals=True, strong_symm=self.enforce_symmetry)
        npair_core = (self.nelec_imp - self.nelec_as) // 2
        # This becomes kind of approximate if there is more than one active space
        loc2occ = np.append (loc2no_unac, self.loc2amo, axis=1)
        occ_labels = np.append (no_labels, assign_blocks_weakly (self.loc2amo, self.loc2symm))
        if fock is None:
            fock = self.ints.loc_rhf_fock_bis (oneRDM_loc)
        virtbath2loc = loc2virtbath.conjugate ().T
        grad = virtbath2loc @ fock @ oneRDM_loc @ loc2occ
        if self.norbs_as > 0:
            eri = self.ints.general_tei ([loc2virtbath, self.loc2amo, self.loc2amo, self.loc2amo])
            lamb = np.tensordot (self.amo2loc @ loc2occ, self.twoCDMimp_amo, axes=(0,0))
            print (eri.shape, lamb.shape)
            grad += np.tensordot (eri, lamb, axes=((1,2,3),(1,2,3))) # axes=3 is INCORRECT!
        return 2 * grad, occ_labels
            
    def test_Schmidt_basis_energy (self):
        def _test (dma, dmb, label):
            dma_ao, dmb_ao = (represent_operator_in_basis (dm, self.ints.ao2loc.conjugate ().T) for dm in (dma, dmb))
            vj, vk = self.ints.get_jk_ao ([dma_ao, dmb_ao])
            va = vj[0] + vj[1] - vk[0]
            vb = vj[0] + vj[1] - vk[1]
            va, vb = (represent_operator_in_basis (v, self.ints.ao2loc) for v in (va, vb))
            etot = self.ints.activeCONST + (self.ints.activeOEI * (dma + dmb)).sum ()
            etot += (va * dma).sum () / 2
            etot += (vb * dmb).sum () / 2
            etot += self.E2_cum + sum (self.E2froz_tbc)
            print ("LASSCF energy total error in Schmidt basis using {}: {:.6e}".format (label, etot - self.ints.e_tot))
        dma, dmb = (self.oneRDM_loc + self.oneSDM_loc)/2, (self.oneRDM_loc - self.oneSDM_loc)/2
        _test (dma, dmb, 'full rdm')
        p = self.loc2imp @ self.imp2loc
        q = self.loc2core @ self.core2loc
        dma_p = p @ dma @ p + q @ dma @ q
        dmb_p = p @ dmb @ p + q @ dmb @ q
        _test (dma_p, dmb_p, 'projected rdm')
        dma_s = p @ dma @ p + self.oneRDMfroz_loc/2
        dmb_s = p @ dmb @ p + self.oneRDMfroz_loc/2
        _test (dma_s, dmb_s, 'stored rdm (VALID FOR SINGLET ENVIRONMENT ONLY)')
        print ("CHECK ME: norm of spin density matrix in this case: {}".format (linalg.norm (self.oneSDM_loc)))
        print ("CHECK ME: diff between self.ints.oneRDM_loc and self.oneRDM_loc: {}".format (linalg.norm (self.oneRDM_loc - self.ints.oneRDM_loc)))
        print ("CHECK ME: diff between self.ints.oneSDM_loc and self.oneSDM_loc: {}".format (linalg.norm (self.oneSDM_loc - self.ints.oneSDM_loc)))
        print ("CHECK ME: diff between projected rdm and self.oneRDMfroz_loc: {}".format (linalg.norm (q @ (dma + dmb) @ q - self.oneRDMfroz_loc)))
        

    ##############################################################################################################################





    # Impurity Hamiltonian
    ###############################################################################################################################
    def construct_impurity_hamiltonian (self, xtra_CONST=0.0):
        w0, t0 = time.time (), time.process_time () 
        self.warn_check_Schmidt ("construct_impurity_hamiltonian")
        if self.imp_solver_name == "dummy RHF":
            self.E2_frag_core = 0
            self.impham_built = True
            self.imp_solved   = False
            return
        if self.imp_solver_name == "RHF" and self.quasidirect:
            ao2imp = np.dot (self.ints.ao2loc, self.loc2imp)
            def my_jk (mol, dm, hermi=1):
                dm_ao        = represent_operator_in_basis (dm, ao2imp.T)
                vj_ao, vk_ao = self.ints.get_jk_ao (dm_ao, hermi)
                vj_basis     = represent_operator_in_basis (vj_ao, ao2imp)
                vk_basis     = represent_operator_in_basis (vk_ao, ao2imp)
                return vj_basis, vk_basis
            self.impham_TEI = None 
            #self.impham_TEI_fiii = None # np.empty ([self.norbs_frag] + [self.norbs_imp for i in range (3)], dtype=np.float64)
            self.impham_get_jk = my_jk
            vj, vk_c = self.impham_get_jk (self.ints.mol, self.get_oneRDM_imp ())
            vk_s = self.impham_get_jk (self.ints.mol, self.get_oneSDM_imp ())[1]
            cdm = self.get_oneRDM_imp ()
            sdm = self.get_oneSDM_imp ()
            sie = self.E2_cum
            sie += np.tensordot (vj, cdm) / 2
            sie -= np.tensordot (vk_c, cdm) / 4
            sie -= np.tensordot (vk_s, sdm) / 4
        elif self.project_cderi:
            self.impham_TEI = None
            self.impham_get_jk = None
            self.impham_CDERI = self.ints.dmet_cderi (self.loc2emb, self.norbs_imp)
            cdm = self.get_oneRDM_imp ()
            sdm = self.get_oneSDM_imp ()
            cdm_pack = cdm + cdm.T
            cdm_pack[np.diag_indices (self.norbs_imp)] /= 2
            cdm_pack = pack_tril (cdm_pack)
            rho = np.dot (self.impham_CDERI, cdm_pack)
            vj = unpack_tril (np.dot (rho, self.impham_CDERI))
            cderi = unpack_tril (self.impham_CDERI)
            vk_c = np.dot (cderi, cdm)
            vk_c = np.tensordot (cderi, vk_c, axes=((0,2),(0,2)))
            vk_s = np.dot (cderi, sdm)
            vk_s = np.tensordot (cderi, vk_s, axes=((0,2),(0,2)))
            sie = self.E2_cum
            sie += np.tensordot (vj, cdm) / 2
            sie -= np.tensordot (vk_c, cdm) / 4
            sie -= np.tensordot (vk_s, sdm) / 4
            cderi = rho = cdm_pack = cdm = sdm = None
        else:
            f = self.loc2frag
            i = self.loc2imp
            self.impham_TEI = self.ints.dmet_tei (self.loc2emb, self.norbs_imp, symmetry=8) 
            #self.impham_TEI_fiii = self.ints.general_tei ([f, i, i, i])
            self.impham_get_jk = None
            cdm = self.get_oneRDM_imp ()
            sdm = self.get_oneSDM_imp ()
            eri = ao2mo.restore (1, self.impham_TEI, self.norbs_imp)
            vj = np.tensordot (eri, cdm, axes=2)
            vk_c = np.tensordot (eri, cdm, axes=((1,2),(0,1)))
            vk_s = np.tensordot (eri, sdm, axes=((1,2),(0,1)))
            sie = self.E2_cum
            sie += np.tensordot (vj, cdm) / 2
            sie -= np.tensordot (vk_c, cdm) / 4
            sie -= np.tensordot (vk_s, sdm) / 4
            eri = cdm = sdm = None

        #OEI_C = self.ints.dmet_fock (self.loc2emb, self.norbs_imp, self.oneRDMfroz_loc)
        #OEI_S = -self.ints.dmet_k (self.loc2emb, self.norbs_imp, self.oneSDMfroz_loc) / 2
        self.impham_OEI_C = represent_operator_in_basis (self.ints.activeFOCK, self.loc2imp) - (vj - vk_c/2)
        self.impham_OEI_S = represent_operator_in_basis (self.ints.activeVSPIN, self.loc2imp) + vk_s/2
        #print ("Error in OEI_C: {}".format (linalg.norm (OEI_C - self.impham_OEI_C)))
        #print ("Error in OEI_S: {}".format (linalg.norm (OEI_S - self.impham_OEI_S)))

        # Constant contribution to energy from core 2CDMs
        cdm, sdm = self.get_oneRDM_imp (), self.get_oneSDM_imp ()
        sie += np.tensordot (self.impham_OEI_C, cdm, axes=2)
        sie += np.tensordot (self.impham_OEI_S, sdm, axes=2)
        self.impham_CONST = self.ints.e_tot - sie 
        #impham_CONST = (self.ints.dmet_const (self.loc2emb, self.norbs_imp, self.oneRDMfroz_loc, self.oneSDMfroz_loc)
        #                     + self.ints.const () + xtra_CONST + sum (self.E2froz_tbc))
        #print ("Error in impham_CONST: {}".format (self.impham_CONST - impham_CONST))

        self.E2_frag_core = 0

        self.impham_built = True
        self.imp_solved   = False
        print ("Time in impurity Hamiltonian constructor: {:.8f} wall, {:.8f} clock".format (time.time () - w0, time.process_time () - t0))
        sys.stdout.flush ()

    def test_impurity_hamiltonian_energy (self):
        h = self.impham_OEI_C.copy ()
        dm = represent_operator_in_basis (self.oneRDM_loc, self.loc2imp)
        if self.impham_TEI is not None:
            eri = ao2mo.restore (1, self.impham_TEI, self.norbs_imp).reshape ([self.norbs_imp,]*4)
            h += np.tensordot (eri, dm) / 2
            h -= np.tensordot (eri, dm, axes=((2,1),(0,1))) / 4
            v = represent_operator_in_basis (eri, self.imp2amo)
        elif self.impham_get_jk is not None:
            vj, vk = self.impham_get_jk (self.ints.mol, dm)
            h += (vj - vk/2) / 2
            v = 0.0
        elif self.impham_CDERI is not None:
            print ("Warning: am not sure about this (veff through CDERI)!")
            cderi = unpack_tril (self.impham_CDERI)
            h += np.dot (np.tensordot (cderi, dm), cderi) / 2
            h -= np.tensordot (cderi, np.dot (cderi, dm), axes=((0,1),(0,2))) / 4
            v = np.dot (self.amo2imp, np.dot (cderi, self.imp2amo))
            v = np.tensordot (v, v, axes=((1),(1)))
        e_tot = self.impham_CONST + (h*dm).sum () + (v*self.twoCDMimp_amo).sum () / 2
        print ("LASSCF energy total error in fragments.py (using stored density-matrix data, VALID FOR SINGLET ENVIRONMENT ONLY): {:.6e}".format (e_tot - self.ints.e_tot))
        print ("CHECK ME: diff between stored E2_cum and recomputed E2_cum in this case: {}".format (self.E2_cum + sum (self.E2froz_tbc) - (v*self.twoCDMimp_amo).sum () / 2))

        if self.ci_as is not None:
            h = self.impham_OEI_C.copy ()
            ci = self.ci_as.copy ()
            ci_orb = self.ci_as_orb.copy ()
            ha, hb = h + self.impham_OEI_S, h - self.impham_OEI_S
            loc2can, _, _, ncore = self.get_loc2canon_imp ()
            ncas = self.norbs_as
            nocc = ncore + ncas
            imp2can = self.imp2loc @ loc2can
            dm_core = imp2can[:,:ncore] @ imp2can[:,:ncore].conjugate ().T * 2
            imp2cas = imp2can[:,ncore:nocc]
            fcisolver = fci.solver (self.ints.mol, singlet=False, symm=None)
            abs_2MS = int (round (2 * abs (self.target_MS)))
            nelecas = ((self.nelec_as + abs_2MS) // 2, (self.nelec_as - abs_2MS) // 2)
            casdm1a, casdm1b = fcisolver.make_rdm1s (ci, ncas, nelecas)
            casdm2 = fcisolver.make_rdm2 (ci, ncas, nelecas)
            umat = ci_orb.conjugate ().T @ loc2can[:,ncore:nocc]
            casdm1a = represent_operator_in_basis (casdm1a, umat)
            casdm1b = represent_operator_in_basis (casdm1b, umat)
            casdm2 = represent_operator_in_basis (casdm2, umat)
            if self.impham_TEI is not None:
                eri = ao2mo.restore (1, self.impham_TEI, self.norbs_imp).reshape ([self.norbs_imp,]*4)
                vj = np.tensordot (eri, dm_core)
                vk = np.tensordot (eri, dm_core, axes=((2,1),(0,1))) 
                v = represent_operator_in_basis (eri, imp2cas)
            elif self.impham_CDERI is not None:
                print ("Warning: am not sure about this (veff through CDERI)!")
                cderi = unpack_tril (self.impham_CDERI)
                vj = np.dot (np.tensordot (cderi, dm_core), cderi) 
                vk = np.tensordot (cderi, np.dot (cderi, dm_core), axes=((0,1),(0,2))) 
                v = np.dot (imp2cas.conjugate ().T, np.dot (cderi, imp2cas))
                v = np.tensordot (v, v, axes=((1),(1)))
            e_core = self.impham_CONST + ((h + vj/2 - vk/4) * dm_core).sum ()
            ha = represent_operator_in_basis (ha + vj - vk/2, imp2cas)
            hb = represent_operator_in_basis (hb + vj - vk/2, imp2cas)
            e_cas = (ha * casdm1a).sum () + (hb * casdm1b).sum () + (v * casdm2).sum () / 2
            print ("Testing e_core = {:.9f}".format (e_core))
            print ("Testing e_cas = {:.9f}".format (e_cas))
            print ("LASSCF energy total error using ci vectors in fragments.py (VALID FOR SINGLET ENVIRONMENT ONLY): {:.6e}".format (e_core + e_cas - self.ints.e_tot))

    ###############################################################################################################################


    

    # Solving the impurity problem
    ###############################################################################################################################
    def get_guess_1RDM (self, chempot_imp):
        FOCK = represent_operator_in_basis (self.ints.activeFOCK, self.loc2imp) - chempot_imp
        guess_1RDM = [get_1RDM_from_OEI (FOCK,int ( round ( (self.nelec_imp // 2) + self.target_MS))),
                      get_1RDM_from_OEI (FOCK,int ( round ( (self.nelec_imp // 2) - self.target_MS)))]
        if not self.target_MS: guess_1RDM = guess_1RDM[0] + guess_1RDM[1]
        return guess_1RDM

    def solve_impurity_problem (self, chempot_frag):
        self.warn_check_impham ("solve_impurity_problem")

        # Make chemical potential matrix and guess_1RDM
        chempot_imp = represent_operator_in_basis (chempot_frag * np.eye (self.norbs_frag), self.frag2imp)
        guess_1RDM = self.get_guess_1RDM (chempot_imp)

        # Execute solver function
        #if self.imp_solver_name != 'dummy RHF': self.analyze_gradient ()
        w0, t0 = time.time (), time.process_time ()
        self.imp_solver_function (guess_1RDM, chempot_imp)
        print ("Time in solver function: {:.8f} wall, {:.8f} clock".format (time.time () - w0, time.process_time () - t0))
        self.imp_solved = True

        # Main results: oneRDM in local basis and nelec_frag
        self.nelec_frag = self.get_nelec_frag ()
        self.E_frag     = self.get_E_frag ()
        self.S2_frag    = self.get_S2_frag ()
        print ("Impurity results for {0}: E_imp = {1}, E_frag = {2}, nelec_frag = {3}, S2_frag = {4}".format (self.frag_name,
            self.E_imp, self.E_frag, self.nelec_frag, self.S2_frag))
        #if self.imp_solver_name != 'dummy RHF': self.analyze_gradient (oneRDM_loc=self.oneRDM_loc, fock_loc=self.ints.loc_rhf_fock_bis (self.oneRDM_loc))

        # In order to comply with ``NOvecs'' bs, let's get some pseudonatural orbitals
        self.fno_evals, frag2fno = sp.linalg.eigh (self.get_oneRDM_frag ())
        self.loc2fno = np.dot (self.loc2frag, frag2fno)

        # Testing
        '''
        oneRDMimp_loc = represent_operator_in_basis (self.oneRDMimp_imp, self.imp2loc)
        idx = np.ix_(self.frag_orb_list, self.frag_orb_list)
        print ("Number of electrons on {0} from the impurity model: {1}; from the core: {2}".format (
            self.frag_name, np.trace (oneRDMimp_loc[idx]), np.trace (self.oneRDMfroz_loc[idx])))
        '''

    def analyze_gradient (self, oneRDM_loc=None, fock_loc=None):
        ''' Orbitals may be messed up by symmetry enforcement! I think symmetry enforcement may accidentally siwtch some impurity external orbitals and core external orbitals? '''
        if oneRDM_loc is None: oneRDM_loc = self.ints.oneRDM_loc
        if fock_loc is None: fock_loc = self.ints.activeFOCK
        print ("Is fock matrix symmetry-adapted? {}".format (measure_operator_blockbreaking (fock_loc, self.loc2symm)))
        print ("Is 1RDM symmetry-adapted? {}".format (measure_operator_blockbreaking (oneRDM_loc, self.loc2symm)))
        imp2unac = get_complementary_states (self.imp2amo) 
        loc2iunac = self.loc2imp @ imp2unac
        ino_occ, loc2ino = matrix_eigen_control_options (oneRDM_loc, subspace=loc2iunac, symmetry=self.loc2symm,
            strong_symm=self.enforce_symmetry, sort_vecs=-1, only_nonzero_vals=False)[:2]
        norbs_iinac = (self.nelec_imp - self.nelec_as) // 2
        loc2iinac = loc2ino[:,:norbs_iinac]
        iext2loc = loc2ino[:,norbs_iinac:].conjugate ().T
        occ_err = linalg.norm (ino_occ[:norbs_iinac]-2)
        olap_err = measure_basis_olap (loc2iinac, self.loc2core)
        print ("I think I have {} impurity inactive orbitals; occupancy error = {}, overlap error = {}".format (norbs_iinac, occ_err, olap_err))
        occ_err = linalg.norm (ino_occ[norbs_iinac:])
        olap_err = measure_basis_olap (iext2loc.conjugate ().T, self.loc2core)
        print ("I think I have {} impurity external orbitals; occupancy error = {}, overlap error = {}".format (len (ino_occ) - norbs_iinac, occ_err, olap_err))
        cno_occ, loc2cno = matrix_eigen_control_options (oneRDM_loc, subspace=self.loc2core, symmetry=self.loc2symm,
            strong_symm=self.enforce_symmetry, sort_vecs=-1, only_nonzero_vals=False)[:2]
        norbs_cinac = (self.ints.nelec_tot - self.nelec_imp) // 2
        loc2cinac = loc2cno[:,:norbs_cinac]
        cext2loc = loc2cno[:,norbs_cinac:].conjugate ().T
        iunac2loc = loc2iunac.conjugate ().T
        occ_err = linalg.norm (cno_occ[:norbs_cinac]-2) if norbs_cinac>0 else 0.0
        olap_err = measure_basis_olap (loc2cinac, self.loc2imp)
        print ("I think I have {} core inactive orbitals; occupancy error = {}, overlap error = {}".format (norbs_cinac, occ_err, olap_err))
        occ_err = linalg.norm (cno_occ[norbs_cinac:])
        olap_err = measure_basis_olap (cext2loc.conjugate ().T, self.loc2imp)
        print ("I think I have {} core external orbitals; occupancy error = {}, overlap error = {}".format (len (cno_occ) - norbs_cinac, occ_err, olap_err))
        eri_faaa = self.ints.general_tei ([np.eye (self.norbs_tot), self.loc2amo, self.loc2amo, self.loc2amo])
        eri_iunac = np.tensordot (iunac2loc, eri_faaa, axes=1)
        eri_core = np.tensordot (self.core2loc, eri_faaa, axes=1)
        # Active orbital-impurity unac
        grad = fock_loc @ oneRDM_loc
        grad -= grad.T
        grad = iunac2loc @ grad @ self.loc2amo
        eri_grad = np.tensordot (eri_iunac, self.twoCDMimp_amo, axes=((1,2,3),(1,2,3))) # NOTE: just saying axes=3 gives an INCORRECT result
        grad += eri_grad
        print ("Active to imp-unac gradient norm: {}".format (linalg.norm (grad)))
        # Active orbital-core
        grad = fock_loc @ oneRDM_loc
        grad -= grad.T
        grad = self.core2loc @ grad @ self.loc2amo
        eri_grad = np.tensordot (eri_core, self.twoCDMimp_amo, axes=((1,2,3),(1,2,3))) # NOTE: just saying axes=3 gives an INCORRECT result
        grad += eri_grad
        print ("Active to core gradient norm: {}".format (linalg.norm (grad)))
        print ("Imp-inac to imp-extern gradient norm: {}".format (linalg.norm (iext2loc @ fock_loc @ loc2iinac * 2)))
        print ("Core-inac to core-extern gradient norm: {}".format (linalg.norm (cext2loc @ fock_loc @ loc2cinac * 2)))
        print ("Core-inac to imp-extern gradient norm: {}".format (linalg.norm (iext2loc @ fock_loc @ loc2cinac * 2)))
        print ("Imp-inac to core-extern gradient norm: {}".format (linalg.norm (cext2loc @ fock_loc @ loc2iinac * 2)))


    def load_amo_guess_from_casscf_molden (self, moldenfile, norbs_cmo, norbs_amo):
        ''' Use moldenfile from whole-molecule casscf calculation to guess active orbitals '''
        print ("Attempting to load guess active orbitals from {}".format (moldenfile))
        mol, _, mo_coeff, mo_occ = molden.load (moldenfile)[:4]
        print ("Difference btwn self mol coords and moldenfile mol coords: {}".format (sp.linalg.norm (mol.atom_coords () - self.ints.mol.atom_coords ())))
        norbs_occ = norbs_cmo + norbs_amo
        amo_coeff = mo_coeff[:,norbs_cmo:norbs_occ]
        amo_coeff = scf.addons.project_mo_nr2nr (mol, amo_coeff, self.ints.mol)
        self.loc2amo = reduce (np.dot, [self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, amo_coeff])
        self.loc2amo = self.retain_fragonly_guess_amo (self.loc2amo)

    def load_amo_guess_from_casscf_npy (self, npyfile, norbs_cmo, norbs_amo):
        ''' Use npy from whole-molecule casscf calculation to guess active orbitals. Must have identical geometry orientation and basis! 
        npyfile must contain an array of shape (norbs_tot+1,norbs_active), where the first row contains natural-orbital occupancies
        for the active orbitals, and subsequent rows contain active natural orbital coefficients.'''
        matrix = np.load (npyfile)
        ano_occ = matrix[0,:]
        ano_coeff = matrix[1:,:]
        loc2ano = reduce (np.dot, (self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, ano_coeff))
        oneRDMwm_ano = np.diag (ano_occ)
        frag2ano = loc2ano[self.frag_orb_list,:]
        oneRDMano_frag = represent_operator_in_basis (oneRDMwm_ano, frag2ano.conjugate ().T)
        evals, evecs = matrix_eigen_control_options (oneRDMano_frag, sort_vecs=-1, only_nonzero_vals=False)
        self.loc2amo = np.zeros ((self.norbs_tot, self.active_space[1]))
        self.loc2amo[self.frag_orb_list,:] = evecs[:,:self.active_space[1]]
        #norbs_occ = norbs_cmo + norbs_amo
        #mo_coeff = np.load (npyfile)
        #amo_coeff = mo_coeff[:,norbs_cmo:norbs_occ]
        #self.loc2amo = reduce (np.dot, [self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, amo_coeff])
        #self.loc2amo = self.retain_fragonly_guess_amo (self.loc2amo)

    def save_amo_guess_for_pes_scan (self, npyfile):
        no_occ, no_coeff = matrix_eigen_control_options (self.oneRDMas_loc, sort_vecs=-1, only_nonzero_vals=True)
        no_coeff = np.dot (self.ints.ao2loc, no_coeff)
        matrix = np.insert (no_coeff, 0, no_occ, axis=0)
        np.save (npyfile, matrix)

    def load_amo_guess_for_pes_scan (self, npyfile):
        print ("Loading amo guess from npyfile")
        matrix = np.load (npyfile)
        no_occ = matrix[0,:]
        print ("NO occupancies: {}".format (no_occ))
        no_coeff = matrix[1:,:]
        loc2ano = reduce (np.dot, (self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, no_coeff))
        ovlp = np.dot (loc2ano.conjugate ().T, loc2ano)
        print ("Active orbital overlap matrix:\n{}".format (prettyprint (ovlp, fmt='{:5.2f}')))
        evals, evecs = matrix_eigen_control_options (ovlp, sort_vecs=-1)
        print ("Overlap eigenvalues: {}".format (evals))
        oneRDM_ano = represent_operator_in_basis (np.diag (no_occ), evecs)
        print ("1RDM_ano (trace = {}):\n{}".format (np.trace (oneRDM_ano), prettyprint (oneRDM_ano, fmt='{:5.2f}')))
        loc2ano = np.dot (loc2ano, evecs) / np.sqrt (evals)
        print ("New overlap matrix:\n{}".format (np.dot (loc2ano.conjugate ().T, loc2ano)))
        m = loc2ano.shape[1]
        self.loc2amo = loc2ano
        self.oneRDMas_loc = represent_operator_in_basis (oneRDM_ano, self.loc2amo.conjugate ().T)
        self.twoCDMimp_amo = np.zeros ((m,m,m,m), dtype=self.oneRDMas_loc.dtype)

    def retain_projected_guess_amo (self, loc2amo_guess):
        print ("Diagonalizing fragment projector in guess amo basis and retaining highest {} eigenvalues".format (self.active_space[1]))
        frag2amo = loc2amo_guess[self.frag_orb_list,:]
        proj = np.dot (frag2amo.conjugate ().T, frag2amo)
        evals, evecs = matrix_eigen_control_options (proj, sort_vecs=-1, only_nonzero_vals=False)
        print ("Projector eigenvalues: {}".format (evals))
        return np.dot (loc2amo_guess, evecs[:,:self.active_space[1]])

    def retain_fragonly_guess_amo (self, loc2amo_guess):
        print ("Diagonalizing guess amo projector in fragment basis and retaining highest {} eigenvalues".format (self.active_space[1]))
        frag2amo = loc2amo_guess[self.frag_orb_list,:]
        proj = np.dot (frag2amo, frag2amo.conjugate ().T)
        evals, evecs = matrix_eigen_control_options (proj, sort_vecs=-1, only_nonzero_vals=False)
        print ("Projector eigenvalues: {}".format (evals))
        return np.dot (self.get_true_loc2frag (), evecs[:,:self.active_space[1]])
        
    def load_amo_guess_from_dmet_molden (self, moldenfile):
        ''' Use moldenfile of impurity from another DMET calculation with the same active space (i.e., at a different geometry) to guess active orbitals '''
        print ("Attempting to load guess active orbitals from {}".format (moldenfile))
        mol, _, mo_coeff, mo_occ = molden.load (moldenfile)[:4]
        nelec_amo, norbs_amo = self.active_space
        nelec_tot = int (round (np.sum (mo_occ)))
        norbs_cmo = (nelec_tot - nelec_amo) // 2
        norbs_occ = norbs_cmo + norbs_amo
        amo_coeff = mo_coeff[:,norbs_cmo:norbs_occ]
        #amo_coeff = scf.addons.project_mo_nr2nr (mol, amo_coeff, self.ints.mol)
        self.loc2amo = reduce (np.dot, [self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, amo_coeff])
        self.loc2amo = self.retain_fragonly_guess_amo (self.loc2amo)

    def load_amo_from_aobasis (self, ao2amo, dm, twoRDM=None, twoCDM=None):
        print ("Attempting to load the provided active orbitals to fragment {}".format (self.frag_name))
        self.loc2amo = reduce (np.dot, (self.ints.ao2loc.conjugate ().T, self.ints.ao_ovlp, ao2amo))
        self.oneRDMas_loc = represent_operator_in_basis (dm, self.loc2amo.conjugate ().T)
        if twoRDM is not None:
            self.twoCDMimp_amo = get_2CDM_from_2RDM (twoRDM, dm)
        elif twoCDM is not None:
            self.twoCDMimp_amo = twoCDM
        else:
            self.twoCDMimp_amo = np.zeros ([self.norbs_as for i in range (4)])

    def align_imporb_basis (self, oneRDM_loc=None):
        if not self.symmetry:
            return imp2mo, np.eye (imp2mo.shape[1])
        if oneRDM_loc is None: oneRDM_loc=self.oneRDM_loc        
        
    def align_imporbs_symm (self, imp2mo, mol=None, sort_vecs=1, sorting_metric=None, orbital_type=""):
        if len (orbital_type) > 0: orbital_type += ' '
        mo2imp = imp2mo.conjugate ().T
        if not self.symmetry:
            return imp2mo, np.eye (imp2mo.shape[1])
        if sorting_metric is None:
            sorting_metric = np.diag (np.arange (imp2mo.shape[1]))
        if sorting_metric.shape[0] == imp2mo.shape[1]:
            sorting_metric = represent_operator_in_basis (sorting_metric, mo2imp)
        if self.enforce_symmetry and mol is not None:
            evals, new_imp2mo, labels = matrix_eigen_control_options (sorting_metric, subspace=imp2mo, symmetry=mol.symm_orb,
                sort_vecs=sort_vecs, only_nonzero_vals=False, strong_symm=self.enforce_symmetry)
            err = measure_subspace_blockbreaking (new_imp2mo, mol.symm_orb)
            #if self.enforce_symmetry:
            #    assert (all ([e < params.num_zero_atol for e in err[2:]])), "Strong symmetry enforcement required, but subspace not symmetry aligned; err = {}".format (err)
        else:
            sorting_metric = represent_operator_in_basis (sorting_metric, self.imp2loc)
            loc2mo = self.loc2imp @ imp2mo
            evals, loc2mo, labels = matrix_eigen_control_options (sorting_metric, subspace=loc2mo, symmetry=self.loc2symm,
                sort_vecs=sort_vecs, only_nonzero_vals=False, strong_symm=self.enforce_symmetry)
            err = measure_subspace_blockbreaking (loc2mo, self.loc2symm)
            new_imp2mo = self.imp2loc @ loc2mo
        labels_dict = {lbl: np.count_nonzero (labels==idx) for idx, lbl in enumerate (self.ir_names) if np.count_nonzero (labels==idx)>0}
        symm_umat = mo2imp @ new_imp2mo
        print ("Irreps of {}orbitals: {}, err = {}".format (orbital_type, labels_dict, err))
        return new_imp2mo, symm_umat


    ###############################################################################################################################





    # Convenience functions and properties for results
    ###############################################################################################################################
    def get_nelec_frag (self):
        self.warn_check_imp_solve ("get_nelec_frag")
        return np.trace (self.get_oneRDM_frag ())

    def get_E_frag (self):
        self.warn_check_imp_solve ("get_E_frag")

        # E_f = H_fi G_fi + 1/2 V_fiii G_fiii                            + E2_frag_core
        #     = H_fi G_fi + 1/2 V_fijkl (G_fi*G_jk - G_fk*G_ji + L_fijk) + E2_frag_core
        #     = (H_fi + 1/2 JK_fi[G]) G_fi + 1/2 V_fiii L_fiii           + E2_frag_core
        #     = (H_fi + JK_fi[1/2 G]) G_fi + 1/2 V_fiii L_fiii           + E2_frag_core
        #     = F[1/2 G]_fi G_fi           + 1/2 V_fiii L_fiii           + E2_frag_core

        if self.imp_solver_name == 'dummy RHF':
            fock = (self.ints.activeFOCK + self.ints.activeOEI)/2
            F_fi = np.dot (self.frag2loc, fock)
            G_fi = np.dot (self.frag2loc, self.oneRDM_loc)
        elif isinstance (self.impham_TEI, np.ndarray):
            vj, vk = dot_eri_dm (self.impham_TEI, self.get_oneRDM_imp (), hermi=1)
            fock = (self.ints.dmet_oei (self.loc2emb, self.norbs_imp) + (self.impham_OEI_C + vj - vk/2))/2
            F_fi = np.dot (self.frag2imp, fock)
            G_fi = np.dot (self.frag2imp, self.get_oneRDM_imp ())
        else:
            fock = self.ints.loc_rhf_fock_bis (self.oneRDM_loc/2)        
            F_fi = np.dot (self.frag2loc, fock)
            G_fi = np.dot (self.frag2loc, self.oneRDM_loc)

        E1 = np.tensordot (F_fi, G_fi, axes=2)
        if self.debug_energy:
            print ("get_E_frag {0} :: E1 = {1:.5f}".format (self.frag_name, float (E1)))

        E2 = 0
        # Remember that non-overlapping fragments are now, by necessity, ~contained~ within the impurity!
        if self.norbs_as > 0:
            L_fiii = np.tensordot (self.frag2amo, self.twoCDMimp_amo, axes=1)
            if isinstance (self.impham_TEI, np.ndarray):
                mo_coeffs = [self.imp2frag, self.imp2amo, self.imp2amo, self.imp2amo]
                norbs = [self.norbs_frag, self.norbs_as, self.norbs_as, self.norbs_as]
                V_fiii = ao2mo.incore.general (self.impham_TEI, mo_coeffs, compact=False).reshape (*norbs)
            elif isinstance (self.impham_CDERI, np.ndarray):
                with_df = copy.copy (self.ints.with_df)
                with_df._cderi = self.impham_CDERI
                mo_coeffs = [self.imp2frag, self.imp2amo, self.imp2amo, self.imp2amo]
                norbs = [self.norbs_frag, self.norbs_as, self.norbs_as, self.norbs_as]
                V_fiii = with_df.ao2mo (mo_coeffs, compact=False).reshape (*norbs)
            else:
                V_fiii = self.ints.general_tei ([self.loc2frag, self.loc2amo, self.loc2amo, self.loc2amo])
            E2 = 0.5 * np.tensordot (V_fiii, L_fiii, axes=4)
        elif isinstance (self.twoCDM_imp, np.ndarray):
            L_iiif = np.tensordot (self.twoCDM_imp, self.imp2frag, axes=1)
            if isinstance (self.impham_TEI, np.ndarray):
                V_iiif = ao2mo.restore (1, self.impham_TEI, self.norbs_imp)
                V_iiif = np.tensordot (V_iiif, self.imp2frag, axes=1) 
                E2 = 0.5 * np.tensordot (V_iiif, L_iiif, axes=4)
            elif isinstance (self.impham_CDERI, np.ndarray):
                raise NotImplementedError ("No opportunity to test this yet.")
                # It'll be something like:
                # (P|ii) * L_iiif -> R^P_if
                # (P|ii) * u^i_f -> (P|if)
                # (P|if) * R^P_if -> E2
                # But factors of 2 abound especially with the orbital-pair compacting of CDERI
            else:
                V_iiif = self.ints.general_tei ([self.loc2frag, self.loc2imp, self.loc2imp, self.loc2imp])
                E2 = 0.5 * np.tensordot (V_iiif, L_iiif, axes=4)
        if self.debug_energy:
            print ("get_E_frag {0} :: E2 = {1:.5f}".format (self.frag_name, float (E2)))

        if self.debug_energy:
            print ("get_E_frag {0} :: E2_frag_core = {1:.5f}".format (self.frag_name, float (self.E2_frag_core)))

        return float (E1 + E2 + self.E2_frag_core)

    def get_S2_frag (self):
        self.warn_check_imp_solve ("get_S2_frag")
        # S2_f = Tr_f [G - (G**2)/2] - 1/2 sum_fi L_fiif

        dm = self.get_oneRDM_imp ()
        exc_mat = dm - np.dot (dm, dm)/2
        if self.norbs_as > 0:
            exc_mat -= represent_operator_in_basis (np.einsum ('prrq->pq', self.twoCDMimp_amo)/2, self.amo2imp)
        elif isinstance (self.twoCDM_imp, np.ndarray):
            exc_mat -= np.einsum ('prrq->pq', self.twoCDM_imp)/2
        return np.einsum ('fp,pq,qf->', self.frag2imp, exc_mat, self.imp2frag) 

    def get_twoRDM (self, *bases):
        bases = bases if len (bases) == 4 else (basis[0] for i in range[4])
        oneRDM_pq = represent_operator_in_basis (self.oneRDM_loc, bases[0], bases[1])
        oneRDM_rs = represent_operator_in_basis (self.oneRDM_loc, bases[2], bases[3])
        oneRDM_ps = represent_operator_in_basis (self.oneRDM_loc, bases[0], bases[3])
        oneRDM_rq = represent_operator_in_basis (self.oneRDM_loc, bases[2], bases[1])
        twoRDM  =       np.einsum ('pq,rs->pqrs', oneRDM_pq, oneRDM_rs)
        twoRDM -= 0.5 * np.einsum ('ps,rq->pqrs', oneRDM_ps, oneRDM_rq)
        return twoRDM + self.get_twoCDM (*bases)

    def get_twoCDM (self, *bases):
        bases = bases if len (bases) == 4 else (bases[0] for i in range[4])
        bra1_basis, ket1_basis, bra2_basis, ket2_basis = bases
        twoCDM = np.zeros (tuple(basis.shape[1] for basis in bases))
        for loc2tb, twoCDM_tb in zip (self.loc2tb_all, self.twoCDM_all):
            tb2loc = np.conj (loc2tb.T)
            tb2bs = (np.dot (tb2loc, basis) for basis in bases)
            twoCDM += represent_operator_in_basis (twoCDM_tb, *tb2bs)
        return twoCDM

    def get_oneRDM_frag (self):
        return represent_operator_in_basis (self.oneRDM_loc, self.loc2frag)

    def get_oneRDM_imp (self):
        self.warn_check_Schmidt ("oneRDM_imp")
        return represent_operator_in_basis (self.oneRDM_loc, self.loc2imp)

    def get_oneSDM_imp (self):
        self.warn_check_Schmidt ("oneSDM_imp")
        return represent_operator_in_basis (self.oneSDM_loc, self.loc2imp)

    def impurity_molden (self, tag=None, canonicalize=False, natorb=False, molorb=False, ene=None, occ=None):
        tag = '.' if tag == None else '_' + str (tag) + '.'
        filename = self.filehead + self.frag_name + tag + 'molden'
        mol = self.ints.mol.copy ()
        mol.nelectron = self.nelec_imp
        mol.spin = int (round (2 * self.target_MS))

        oneRDM = self.oneRDM_loc.copy ()
        FOCK = self.ints.loc_rhf_fock_bis (oneRDM)

        idx_unac = np.ones (self.norbs_imp, dtype=np.bool_)
        norbs_inac = int (round (self.nelec_imp - self.nelec_as) // 2)
        norbs_occ = norbs_inac + self.norbs_as
        if self.norbs_as > 0: idx_unac[norbs_inac:norbs_occ] = False
        idx_inac = idx_unac.copy ()
        idx_virt = idx_unac.copy ()
        idx_inac[norbs_occ:] = False
        idx_virt[:norbs_inac] = False
        idx_actv = ~idx_unac
        loc2molden = self.loc2imp.copy ()
        if molorb:
            assert (not natorb)
            assert (not canonicalize)
        elif natorb or canonicalize:
            if self.norbs_as > 0: 
                try:
                    loc2molden[:,idx_unac] = self.loc2imp @ get_complementary_states (self.imp2amo)
                    loc2molden[:,idx_actv] = self.loc2amo
                except Exception as e:
                    print (idx_unac)
                    print (idx_actv)
                    print (self.norbs_imp, self.nelec_imp, self.nelec_as, loc2molden.shape, self.loc2imp.shape, get_complementary_states (self.imp2amo).shape, self.loc2amo.shape)
                    raise (e) 
            if canonicalize or (natorb and any ([x in self.imp_solver_name for x in ('CASSCF', 'RHF')])):
                loc2molden[:,idx_unac] = matrix_eigen_control_options (FOCK, strong_symm=False,
                    subspace=loc2molden[:,idx_unac], symmetry=self.loc2symm, sort_vecs=1,
                    only_nonzero_vals=False)[1]
                if self.norbs_as > 0:
                    metric = FOCK if canonicalize else oneRDM
                    order = 1 if canonicalize else -1
                    loc2molden[:,idx_actv] = matrix_eigen_control_options (metric, strong_symm=False,
                        subspace=loc2molden[:,idx_actv], symmetry=self.loc2symm, sort_vecs=order,
                        only_nonzero_vals=False)[1]
            elif natorb:
                loc2molden = matrix_eigen_control_options (oneRDM, sort_vecs=-1, only_nonzero_vals=False, symmetry=self.loc2symm,
                    strong_symm=False)[1]
        occ = ((oneRDM @ loc2molden) * loc2molden).sum (0)        
        ene = ((FOCK @ loc2molden) * loc2molden).sum (0)
        ene[idx_actv] = 0
        if self.norbs_as > 0: ene = cas_mo_energy_shift_4_jmol (ene, self.norbs_imp, self.nelec_imp, self.norbs_as, self.nelec_as)
        ao2molden = self.ints.ao2loc @ loc2molden

        molden.from_mo (mol, filename, ao2molden, ene=ene, occ=occ)


    ###############################################################################################################################




    # For interface with DMET
    ###############################################################################################################################
    def get_errvec (self, dmet, mf_1RDM_loc):
        self.warn_check_imp_solve ("get_errvec")
        # Fragment natural-orbital basis matrix elements needed
        if dmet.doDET_NO:
            mf_1RDM_fno = represent_operator_in_basis (mf_1RDM_loc, self.loc2fno)
            return np.diag (mf_1RDM_fno) - self.fno_evals
        # Bath-orbital matrix elements needed
        if dmet.incl_bath_errvec:
            mf_err1RDM_imp = represent_operator_in_basis (mf_1RDM_loc, self.loc2imp) - self.get_oneRDM_imp ()
            return mf_err1RDM_imp.flatten (order='F')
        # Only fragment-orbital matrix elements needed
        mf_err1RDM_frag = represent_operator_in_basis (mf_1RDM_loc, self.loc2frag) - self.get_oneRDM_frag ()
        if dmet.doDET:
            return np.diag (mf_err1RDM_frag)
        elif dmet.altcostfunc:
            return mf_err1RDM_frag[np.triu_indices(self.norbs_frag)]
        else:
            return self.get_oneRDM_frag ().flatten (order='F')


    def get_rsp_1RDM_elements (self, dmet, rsp_1RDM):
        self.warn_check_imp_solve ("get_rsp_1RDM_elements")
        if dmet.altcostfunc:
            raise RuntimeError("You shouldn't have gotten in to get_rsp_1RDM_elements if you're using the constrained-optimization cost function!")
        # If the error function is working in the fragment NO basis, then rsp_1RDM will already be in that basis. Otherwise it will be in the local basis
        if dmet.doDET_NO:
            return np.diag (rsp_1RDM)[self.frag_orb_list]
        # Bath-orbital matrix elements needed
        if dmet.incl_bath_errvec:
            rsp_1RDM_imp = represent_operator_in_basis (rsp_1RDM, self.loc2imp)
            return rsp_1RDM_imp.flatten (order='F')
        # Only fragment-orbital matrix elements needed
        rsp_1RDM_frag = represent_operator_in_basis (rsp_1RDM, self.loc2frag)
        if dmet.doDET:
            return np.diag (rsp_1RDM_frag)
        else:
            return rsp_1RDM_frag.flatten (order='F')






