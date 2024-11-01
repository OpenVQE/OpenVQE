import sys
sys.path.append ('../../../..')
from pyscf import gto, dft, scf, mcscf
from pyscf.tools import molden
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list
from pyscf.mcscf.addons import project_init_guess, spin_square, sort_mo_by_irrep
from pyscf.lib.parameters import BOHR
from functools import reduce
import numpy as np
import re
import c2h6n4_struct
import tracemalloc

assert (len (sys.argv) > 1), "Usage: python lasscf_calc.py dr_nn [options]"
# dr_nn is the shift in the N=N internuclear distance in Angstrom compared to the equilibrium geometry (1.236816)
# most important options:
#   load_scan_guess=dr_nn_guess
#       Load active orbitals and 1-RDM from a different point on the potential energy curve
#       dr_nn_guess is shift of N=N internuclear distance from another LASSCF calculation,
#       which checkpoint file exists in the current directory.
#   load_CASSCF_guess
#       Load active orbitals from a comparable CASSCF calculation.
#       Requires file named "c2h6n4_casano.${dr_nn}.npy" in current directory.
#       File must contain 2d-ndarray of shape (norb+1, norb_active) where first
#       row contains active-space natural orbital occupancies, and remaining rows
#       contain the corresponding natural orbital coefficients in terms of atomic orbitals

# I/O
# --------------------------------------------------------------------------------------------------------------------
basis = '6-31g'
dr_nn = float (sys.argv[1])
print ("c2h6n4 at dr_nn = {}".format (dr_nn))
CASlist = None #np.empty (0, dtype=int)
ints = re.compile ("[0-9]+")
my_kwargs = {'calcname':           ('c2h6n4_lasscf_dr' + ['{:02.0F}','{:03.0F}'][dr_nn < 0]).format (dr_nn*10),
             'doLASSCF':           True,
             'debug_energy':       False,
             'debug_reloc':        False,
             'nelec_int_thresh':   1e-3,
             'num_mf_stab_checks': 0}
load_casscf_guess = False
dr_guess = None
bath_tol = 1e-8
for iargv in sys.argv[2:]:
    if iargv in my_kwargs and isinstance (my_kwargs[iargv], bool) and my_kwargs[iargv] == False:
        my_kwargs[iargv] = True
    elif iargv == 'load_CASSCF_guess':
        load_casscf_guess = True
    elif iargv[:9] == 'bath_tol=':
        bath_tol = float (iargv[9:])
    elif iargv[:16] == 'load_scan_guess=':
        dr_guess = float (iargv[16:])
    else: 
        CASlist = np.array ([int (i)-1 for i in ints.findall (iargv)])
        print ("CASlist (note subtraction by 1 for python's zero-indexing; cf. jmol's 1-indexing): {}".format (CASlist))

# Hartree--Fock calculation
# --------------------------------------------------------------------------------------------------------------------
mol = c2h6n4_struct.structure (dr_nn, dr_nn, basis, symmetry=False)
mf = scf.RHF (mol)
mf.verbose = 4
mf.kernel ()
if not mf.converged:
    mf = mf.newton ()
    mf.kernel ()
for i in range (my_kwargs['num_mf_stab_checks']):
    new_mo = mf.stability ()[0]
    dm0 = reduce (np.dot, (new_mo, np.diag (mf.mo_occ), new_mo.conjugate ().T))
    mf = scf.RHF (mol)
    mf.verbose = 4
    mf.kernel (dm0)
    if not mf.converged:
        mf = mf.newton ()
        mf.kernel ()

# Set up the localized AO basis
# --------------------------------------------------------------------------------------------------------------------
myInts = localintegrals.localintegrals(mf, range(mol.nao_nr ()), 'meta_lowdin')
myInts.molden( my_kwargs['calcname'] + '_locints.molden' )

# Build fragments from atom list
# --------------------------------------------------------------------------------------------------------------------
N2Ha = make_fragment_atom_list (myInts, list (range(3)), 'CASSCF(4,4)', name='N2Ha') #, active_orb_list = CASlist)
C2H4 = make_fragment_atom_list (myInts, list (range(3,9)), 'RHF', name='C2H4')
N2Hb = make_fragment_atom_list (myInts, list (range(9,12)), 'CASSCF(4,4)', name='N2Hb') #, active_orb_list = CASlist)
N2Ha.bath_tol = C2H4.bath_tol = N2Hb.bath_tol = bath_tol
fraglist = [N2Ha, C2H4, N2Hb]

# Load or generate active orbital guess 
# --------------------------------------------------------------------------------------------------------------------
c2h6n4_dmet = dmet (myInts, fraglist, **my_kwargs)
if load_casscf_guess:
    npyfile = 'c2h6n4_casano.{:.1f}.npy'.format (dr_nn)
    norbs_cmo = (mol.nelectron - 8) // 2
    norbs_amo = 8
    N2Ha.load_amo_guess_from_casscf_npy (npyfile, norbs_cmo, norbs_amo)
    N2Hb.load_amo_guess_from_casscf_npy (npyfile, norbs_cmo, norbs_amo)
elif dr_guess is not None:
    chkname = ('c2h6n4_lasscf_dr' + ['{:02.0F}','{:03.0F}'][dr_guess < 0]).format (dr_guess*10)
    c2h6n4_dmet.load_checkpoint (chkname + '.chk.npy')
else:
    c2h6n4_dmet.generate_frag_cas_guess (mf.mo_coeff, caslst=CASlist)

# Calculation
# --------------------------------------------------------------------------------------------------------------------
energy_result = c2h6n4_dmet.doselfconsistent ()
c2h6n4_dmet.save_checkpoint (my_kwargs['calcname'] + '.chk.npy')
print ("----Energy: {:.1f} {:.8f}".format (dr_nn, energy_result))

# Save natural-orbital moldens
# --------------------------------------------------------------------------------------------------------------------



