from pyscf import gto, scf, lib, df
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_sync_o1 import LASSCF

print ("WARNING: LASSCF_SYNC_O1 IS NOT FULLY DEBUGGED!")

lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = struct (3.0, 3.0, 'cc-pvtz', symmetry=False)
mol.verbose = lib.logger.INFO
mol.output = 'debug_tz_df_o1.log'
mol.build ()
my_aux = df.aug_etb (mol)
mf = scf.RHF (mol).density_fit (auxbasis = my_aux).run ()

# 1. Diamagnetic singlet
''' The constructor arguments are
    1) SCF object
    2) List or tuple of ncas for each fragment
    3) List or tuple of nelec for each fragment
    A list or tuple of total-spin multiplicity is supplied
    in "spin_sub".'''
las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
''' The class doesn't know anything about "fragments" at all.
    The active space is only "localized" provided one offers an
    initial guess for the active orbitals that is localized.
    That is the purpose of the localize_init_guess function.
    It requires a sequence of sequence of atom numbers, and it
    projects the orbitals in the ncore:nocc columns into the
    space of those atoms' AOs. The orbitals in the range
    ncore:ncore+ncas_sub[0] are the first active subspace,
    those in the range ncore+ncas_sub[0]:ncore+sum(ncas_sub[:2])
    are the second active subspace, and so on.'''
frag_atom_list = (list (range (3)), list (range (7, 10)))
mo_coeff = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
''' Right now, this function can only (roughly) reproduce the
    "force_imp=False, confine_guess=True" behavior of the old 
    orbital guess builder. I might add the complement later,
    but if you are doing potential energy scans or geometry
    optimizations I think the current implementation of
    pyscf.mcscf.addons.project_init_guess might actually be better.'''
las.kernel (mo_coeff)
print ("E(dia singlet) =", las.e_tot)

## 2. Antiferromagnetic quasi-singlet
#''' To change the spin projection quantum numbers of the
#    subspaces, instead of providing a list of nelec, provide
#    a list of tuples of (neleca,nelecb).'''
#las = LASSCF (mf, (4,4), ((4,0),(0,4)), spin_sub=(5,5))
#las.kernel (mo_coeff)
#print ("E(antiferro singlet) =", las.e_tot)
#
## 2. Ferromagnetic nonet
#''' If you are doing a high-spin ROHF calculation and you 
#    initialize with optimized ROHF orbitals (WITHOUT calling the
#    localize_init_guess function), the class will of course
#    immediately recognize that the gradient is zero and
#    happily exit on iteration 1. But if you initialize with
#    anything else, you might have problems because orbital
#    rotations between identically-occupied orbital subspaces are
#    redundant. This usually wasn't a problem in the old version
#    because nothing else was happening to those orbitals and
#    so there was no way for the Hessian-vector product to 
#    couple u <-> p and v <-> p into the undefined u <-> v
#    mode, but here, it poses a serious numerical problem.
#
#    I've dealt with this problem by turning the "ah_level_shift"
#    parameter (which defaults to 1e-8) into the stiffness of a
#    harmonic spring which I've attached to each conjugate 
#    gradient cycle. This is essentially a half-assed trust-radius
#    method, where I'm forcing the user to figure out how stiff to
#    make the spring instead of properly determining it on the fly
#    to put a ceiling on the norm of the step vector.
#
#    If optimizations aren't converging and the |x_orb| lines
#    in the iteration printout are going crazy, try increasing
#    the value of ah_level_shift.'''
#las = LASSCF (mf, (4,4), ((4,0),(4,0)), spin_sub=(5,5))
#las.set (ah_level_shift=1e-5).kernel (mo_coeff)
#print ("E(ferro nonet) =", las.e_tot)



