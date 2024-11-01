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
import me2n2_struct
import tracemalloc

assert (len (sys.argv) > 1), "Usage: python casdmet_calc.py r_nn [options]"
# r_nn is N=N internuclear distance in Angstrom (equilibrium geometry = 1.24414799)
# most important options:
#   load_scan_guess=r_nn_guess
#       Load active orbitals and chemical potential from a different point on the potential energy curve
#       r_nn_guess is N=N internuclear distance from another CAS-DMET calculation,
#       which checkpoint file exists in the current directory.
#   load_CASSCF_guess
#       Load active orbitals from a comparable CASSCF calculation.
#       Requires file named "me2n2_casano.${r_nn}.npy" in current directory.
#       File must contain 2d-ndarray of shape (norb+1, norb_active) where first
#       row contains active-space natural orbital occupancies, and remaining rows
#       contain the corresponding natural orbital coefficients in terms of atomic orbitals

# I/O
# --------------------------------------------------------------------------------------------------------------------
basis = '6-31g'
r_nn = float (sys.argv[1])
print ("Me2N2 at r_nn = {}".format (r_nn))
CASlist = None #np.empty (0, dtype=int)
ints = re.compile ("[0-9]+")
my_kwargs = {'calcname':           'me2n2_casdmet_r{:2.0f}'.format (r_nn*10),
             'do1SHOT':            True,
             'debug_energy':       False,
             'debug_reloc':        False,
             'nelec_int_thresh':   1e-5,
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
mol = me2n2_struct.structure (r_nn, basis)
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
N2 = make_fragment_atom_list (myInts, list (range(2)), 'CASSCF(4,4)', name="N2")
Me1 = make_fragment_atom_list (myInts, list (range(2,6)), 'RHF', name='Me1')
Me2 = make_fragment_atom_list (myInts, list (range(6,10)), 'RHF', name='Me2')
N2.bath_tol = Me1.bath_tol = Me2.bath_tol = bath_tol
N2.mol_output = my_kwargs['calcname'] + '_N2.log'
Me1.mol_output = my_kwargs['calcname'] + '_Me1.log'
Me2.mol_output = my_kwargs['calcname'] + '_Me2.log'
fraglist = [N2, Me1, Me2] 

# Load or generate active orbital guess
# --------------------------------------------------------------------------------------------------------------------
me2n2_dmet = dmet (myInts, fraglist, **my_kwargs)
if load_casscf_guess:
    npyfile = 'me2n2_casano.{:.1f}.npy'.format (r_nn)
    norbs_cmo = (mol.nelectron - 4) // 2
    norbs_amo = 4
    N2.load_amo_guess_from_casscf_npy (npyfile, norbs_cmo, norbs_amo)
elif dr_guess is not None:
    chkname = 'me2n2_casdmet_r{:2.0f}'.format (dr_guess*10)
    me2n2_dmet.load_checkpoint (chkname + '.chk.npy')
else:
    me2n2_dmet.generate_frag_cas_guess (mf.mo_coeff, caslst=CASlist)

# Calculation
# --------------------------------------------------------------------------------------------------------------------
energy_result = me2n2_dmet.doselfconsistent ()
me2n2_dmet.save_checkpoint (my_kwargs['calcname'] + '.chk.npy')
print ("----Energy: {:.1f} {:.8f}".format (r_nn, energy_result))

# Save natural-orbital moldens
# --------------------------------------------------------------------------------------------------------------------



