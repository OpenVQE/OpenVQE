import sys, os
import time
from pyscf import gto, dft, scf, mcscf, df, symm
from pyscf.tools import molden
from pyscf.mcscf import avas
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list
from mrh.util.rdm import get_2RDM_from_2CDM, get_2CDM_from_2RDM
from pyscf.mcscf.addons import project_init_guess, spin_square, sort_mo_by_irrep
from pyscf.lib.parameters import BOHR
from functools import reduce
import pyscf.__config__
import numpy as np
import re
import tracemalloc
from scipy import linalg
from mrh.my_pyscf.gto.ano_contractions import contract_ano_basis

# Usage: python lasscf_calc.py S
assert (sys.argv[1] in ('0', '2'))
spinS = int (sys.argv[1])
MAX_MEMORY = int (os.environ.get ('MAX_MEMORY', 16000))
cderiname = 'fench_{}_anodz_cderi.npy'.format (('ls','hs')[spinS//2])
mfdmname = 'fench_{}_hf_anodz_mfdm.npy'.format (('ls','hs')[spinS//2])

def grab_ao (mol, mo_coeff, aolabels, sorting=-1):
    idx_ao = mol.search_ao_label (aolabels)
    ovlp = mol.intor ('int1e_ovlp_sph')
    ovlp_inv = linalg.inv (ovlp[np.ix_(idx_ao,idx_ao)])
    smo = np.dot (ovlp[idx_ao,:], mo_coeff)
    metric = reduce (np.dot, (smo.conjugate ().T, ovlp_inv, smo))
    evals, evecs = linalg.eigh (metric)
    return np.dot (mo_coeff, evecs[:,np.argsort (evals)[::sorting]])

def grab_3d_ls (mf):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mc = mcscf.CASCI (mf, 5, 6)
    ncore = mol.nelectron // 2
    symms = symm.label_orb_symm (mol, mol.irrep_name, mol.symm_orb, mo_coeff)
    idx_virt_ag = symms == 'Ag'
    idx_virt_ag[:ncore] = False
    mo_coeff[:,idx_virt_ag] = grab_ao (mol, mo_coeff[:,idx_virt_ag], 'Fe 3d', sorting=-1)
    for ir in ('B1g', 'B2g', 'B3g'):
        idx_occ_ir = symms == ir
        idx_occ_ir[ncore:] = False
        mo_coeff[:,idx_occ_ir] = grab_ao (mol, mo_coeff[:,idx_occ_ir], 'Fe 3d', sorting=1)
    irrep_cmo = {}
    for irrep in np.unique (symms[:ncore]):
        irrep_cmo[irrep] = np.count_nonzero (symms[:ncore] == irrep)
    irrep_cmo['B1g'] -= 1
    irrep_cmo['B2g'] -= 1
    irrep_cmo['B3g'] -= 1
    mo_coeff = sort_mo_by_irrep (mc, mo_coeff, {'Ag': 2, 'B1g': 1, 'B2g': 1, 'B3g': 1}, cas_irrep_ncore = irrep_cmo)
    symms = symm.label_orb_symm (mol, mol.irrep_name, mol.symm_orb, mo_coeff)
    return mo_coeff

def grab_3d_hs (mf):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mc = mcscf.CASCI (mf, 5, 6)
    ncore = (mol.nelectron // 2) - 2
    nocc = ncore + 4
    symms = symm.label_orb_symm (mol, mol.irrep_name, mol.symm_orb, mo_coeff)
    irrep_cmo = {}
    for irrep in np.unique (symms[:ncore]):
        irrep_cmo[irrep] = np.count_nonzero (symms[:ncore] == irrep)
    cas_irreps = {'Ag': 2, 'B1g': 1, 'B2g': 1, 'B3g': 1}
    for iorb in range (ncore, nocc):
        cas_irreps[symms[iorb]] -= 1
        print ("Found {} singly-occupied orbital".format (symms[iorb]))
    for key in cas_irreps:
        if cas_irreps[key] == 1: missing_3d = key
    irrep_cmo[missing_3d] -= 1
    cas_irreps = {'Ag': 2, 'B1g': 1, 'B2g': 1, 'B3g': 1}
    print ("Missing D orbital from SOMOs is apparently {}".format (missing_3d))
    print ("SOMOs: {}".format (symms[ncore:nocc]))
    idx_3d = symms == missing_3d
    idx_3d[ncore:] = False
    mo_coeff[:,idx_3d] = grab_ao (mol, mo_coeff[:,idx_3d], 'Fe 3d', sorting=1)
    mo_coeff = sort_mo_by_irrep (mc, mo_coeff, cas_irreps, cas_irrep_ncore = irrep_cmo)
    symms = symm.label_orb_symm (mol, mol.irrep_name, mol.symm_orb, mo_coeff)
    return mo_coeff

# I/O
# --------------------------------------------------------------------------------------------------------------------
fname = ('FeNCH_LS.xyz', 'FeNCH_HS.xyz')[spinS//2]
with open (fname, 'r') as f:
    carts = f.read ()
print ("[Fe(NCH)6]2+; S = {}".format (spinS))
CASlist = [] #np.empty (0, dtype=int)
ints = re.compile ("[0-9]+")
my_kwargs = {'calcname':           'fench_{}_lasscf65_anodz'.format (('ls','hs')[spinS//2]),
             'doLASSCF':           True,
             'debug_energy':       False,
             'debug_reloc':        False,
             'nelec_int_thresh':   1e-2,
             'num_mf_stab_checks': 0}
load_casscf_guess = False
load_lasscf_chk = False
load_lasscf_sto3g_chk = False
density_fitting = False
etb_beta = getattr(pyscf.__config__, 'df_addons_aug_dfbasis', 2.0)
project_cderi = False
bath_tol = 1e-8
active_orb_lists = [[31,32,33,37,38],[26,35,36,34,37]]
eri_name = my_kwargs['calcname'][:8] + my_kwargs['calcname'][-6:] + '_eri.npy'
for iargv in sys.argv[2:]:
    if iargv in my_kwargs and isinstance (my_kwargs[iargv], bool) and my_kwargs[iargv] == False:
        my_kwargs[iargv] = True
    elif iargv.split ('=')[0] in my_kwargs:
        key, val = iargv.split ('=')
        my_kwargs[key] = type (my_kwargs[key]) (val)
    elif iargv == 'load_CASSCF_guess':
        load_casscf_guess = True
    elif iargv == 'project_cderi':
        project_cderi = True
    elif iargv[:15] == 'density_fitting':
        density_fitting = True
        if len (iargv) > 16:
            etb_beta = float (iargv[16:])
    elif iargv[:9] == 'bath_tol=':
        bath_tol = float (iargv[9:])
    elif iargv[:16] == 'load_lasscf_chk':
        load_lasscf_chk = True
    elif iargv[:22] == 'load_lasscf_sto3g_chk':
        load_lasscf_sto3g_chk = True
    else: 
        CASlist = np.array ([int (i) for i in ints.findall (iargv)])
        print ("CASlist (note 1-indexing): {}".format (CASlist))

# Hartree--Fock calculation
# --------------------------------------------------------------------------------------------------------------------
mol = contract_ano_basis (gto.M (atom = carts, basis = 'ano', symmetry = True, charge = 2, spin = 2*spinS, verbose = 4, output = my_kwargs['calcname'] + '_hf.log', max_memory=MAX_MEMORY), 'VDZP')
mol_sto3g = gto.M (atom = carts, basis = 'sto-3g', symmetry = True, charge = 2, spin = 2*spinS, verbose = 0, max_memory=MAX_MEMORY)
mf = scf.RHF (mol).sfx2c1e ()
do_save_eri = False
if density_fitting:
    mf = mf.density_fit (auxbasis = df.aug_etb (mol, beta=etb_beta))
else:
    try:
        mf._eri = np.load (eri_name)
    except OSError:
        do_save_eri = True
if spinS > 0: mf.irrep_nelec = {'Ag': (20,18), 'B2g': (3,2), 'B3g': (3,2)}
try:
    dm0 = np.load (mfdmname)
except OSError:
    dm0 = None
mf.kernel (dm0)
if do_save_eri: np.save (eri_name, mf._eri)
np.save (mfdmname, np.asarray (mf.make_rdm1 ()))
if not mf.converged:
    mf = mf.newton ()
    mf.kernel ()
for i in range (my_kwargs['num_mf_stab_checks']):
    new_mo = mf.stability ()[0]
    dm0 = reduce (np.dot, (new_mo, np.diag (mf.mo_occ), new_mo.conjugate ().T))
    mf = scf.RHF (mol).sfx2c1e ().density_fit ()
    mf._eri = np.load (eri_name)
    mf.verbose = 4
    mf.kernel (dm0)
    if not mf.converged:
        mf = mf.newton ()
        mf.kernel ()
molden.from_scf (mf, my_kwargs['calcname'] + '_hf.molden')
'''
norbs_amo, nelec_amo, mo = avas.kernel (mf, 'Fe 3d')
print ("Active space according to AVAS: {} electrons in {} orbitals".format (nelec_amo, norbs_amo))
norbs_cmo = (mol.nelectron - nelec_amo) // 2
norbs_tot = mo.shape[1]
norbs_omo = norbs_amo + norbs_cmo
fake_occ = np.zeros ((norbs_tot))
fake_occ[:norbs_cmo] = 2
fake_occ[norbs_cmo:norbs_omo] = 1
#molden.from_mo (mol, my_kwargs['calcname'] + '_avas.molden', mo, occ=fake_occ)
#if spinS == 0:
#    mf.mo_coeff = mo
'''

# Set up the localized AO basis
# --------------------------------------------------------------------------------------------------------------------
myInts = localintegrals.localintegrals(mf, range(mol.nao_nr ()), 'meta_lowdin')
#myInts.molden( my_kwargs['calcname'] + '_locints.molden' )

# Build fragments from atom list
# --------------------------------------------------------------------------------------------------------------------
Fe = make_fragment_atom_list (myInts, [0], 'CASSCF(6,5)', name="Fe", project_cderi=project_cderi)
NCHa = make_fragment_atom_list (myInts, [1, 7,   8], 'dummy RHF', name='NCHa')
NCHb = make_fragment_atom_list (myInts, [2, 13, 14], 'dummy RHF', name='NCHb')
NCHc = make_fragment_atom_list (myInts, [3, 9,  10], 'dummy RHF', name='NCHc')
NCHd = make_fragment_atom_list (myInts, [4, 11, 12], 'dummy RHF', name='NCHd')
NCHe = make_fragment_atom_list (myInts, [5, 17, 18], 'dummy RHF', name='NCHe')
NCHf = make_fragment_atom_list (myInts, [6, 15, 16], 'dummy RHF', name='NCHf')
Fe.target_S = spinS
Fe.target_MS = spinS
Fe.mol_output = my_kwargs['calcname'] + '_Fe.log'
Fe.imp_maxiter = 200
Fe.bath_tol = NCHa.bath_tol = NCHb.bath_tol = NCHc.bath_tol = NCHd.bath_tol = NCHe.bath_tol = NCHf.bath_tol = bath_tol
Fe.debug_energy = NCHa.debug_energy = True
fraglist = [Fe, NCHa, NCHb, NCHc, NCHd, NCHe, NCHf] 
for l, x in  zip (['a','b','c','d','e','f'], [NCHa, NCHb, NCHc, NCHd, NCHe, NCHf]):
    x.mol_output = my_kwargs['calcname'] + '_NCH' + l + '.log'
for x in [NCHa, NCHb, NCHc, NCHd, NCHe, NCHf]:
    x.quasidirect = True

# Load or generate active orbital guess 
# --------------------------------------------------------------------------------------------------------------------
print ("Building DMET object")
fench_dmet = dmet (myInts, fraglist, **my_kwargs)
print ("Going into checkpoint loading")
if load_casscf_guess:
    npyfile = 'fench_{}_casscf65_anodz_no.npy'.format (('ls','hs')[spinS//2])
    norbs_cmo = (mol.nelectron - 6) // 2
    norbs_amo = 5
    Fe.load_amo_guess_from_casscf_npy (npyfile, norbs_cmo, norbs_amo)
elif load_lasscf_chk:
    fench_dmet.load_checkpoint (my_kwargs['calcname'] + '.chk.npy')
elif load_lasscf_sto3g_chk:
    fench_dmet.load_checkpoint (my_kwargs['calcname'][:-5] + 'sto3g.chk.npy', prev_mol=mol_sto3g)
else:
    fn = (grab_3d_ls, grab_3d_hs)[spinS//2]
    fench_dmet.generate_frag_cas_guess (fn (mf), force_imp=True, confine_guess=False)

# Calculation
# --------------------------------------------------------------------------------------------------------------------
print ("Going into calculation")
energy_result = fench_dmet.doselfconsistent ()
fench_dmet.save_checkpoint (my_kwargs['calcname'] + '.chk.npy')
print ("----S = {} energy: {:.8f}".format (spinS, energy_result))

# Save natural-orbital moldens
# --------------------------------------------------------------------------------------------------------------------



