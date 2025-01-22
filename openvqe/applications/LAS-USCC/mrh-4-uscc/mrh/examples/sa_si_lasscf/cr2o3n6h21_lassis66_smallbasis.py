import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.lib import chkfile
from pyscf.data import nist
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import lassi
from mrh.my_pyscf.lassi import states as lassi_states

au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
def yamaguchi (e_roots, s2):
    '''The Yamaguchi formula for six unpaired electrons'''
    idx = np.argsort (e_roots)
    e_roots = e_roots[idx]
    s2 = s2[idx]
    idx_hs = (np.around (s2, 2) == 12)
    assert (np.count_nonzero (idx_hs)), 'high-spin ground state not found'
    idx_hs = np.where (idx_hs)[0][0]
    e_hs = e_roots[idx_hs]
    idx_ls = (np.around (s2, 2) == 0)
    assert (np.count_nonzero (idx_ls)), 'low-spin ground state not found'
    idx_ls = np.where (idx_ls)[0][0]
    e_ls = e_roots[idx_ls]
    j = (e_ls - e_hs) / 12
    return j*au2cm

basis={'C': 'sto-3g','H': 'sto-3g','O': 'sto-3g','N': 'sto-3g','Cr': 'cc-pvdz'}
mol=gto.M (atom='cr2o3n6h21.xyz', verbose=4, spin=6, charge=3, basis=basis,
           output='cr2o3n6h21_lassis66_smallbasis.log')
mf=scf.ROHF(mol)
mf.chkfile = 'cr2o3n6h21_lassis66_smallbasis.chk'
mf.init_guess = 'chk'
mf.kernel()
assert (mf.converged)

# Make sure the overall 2*Sz (total neleca - total nelecb) is as small as possible (0 or 1)
las = LASSCF(mf,(6,6),((3,0),(0,3)),spin_sub=(4,4))

# Find the initial guess orbitals
try: # We only want the orbitals, not any of the other information on the chkfile
    mo_coeff = chkfile.load (las.chkfile, 'las')['mo_coeff']
except (OSError, TypeError, KeyError) as e: # First time through you have to make them from scratch
    from pyscf.mcscf import avas
    ncas_avas, nelecas_avas, mo_coeff = avas.kernel (mf, ['Cr 3d', 'Cr 4d'], minao=mol.basis)
    mc_avas = mcscf.CASCI (mf, ncas_avas, nelecas_avas)
    mo_list = mc_avas.ncore + np.array ([5,6,7,8,9,10,15,16,17,18,19,20])
    mo_coeff = las.sort_mo (mo_list, mo_coeff)
    mo_coeff = las.localize_init_guess (([0],[1]), mo_coeff)

# Direct exchange only result
las = lassi_states.spin_shuffle (las) # generate direct-exchange states
las.weights = [1.0/las.nroots,]*las.nroots # set equal weights
las.kernel (mo_coeff) # optimize orbitals
e_roots, si = las.lassi ()
print (("Direct exchange only is modeled by {} states constructed with\n"
        "lassi_states.spin_shuffle.").format (las.nroots))
print ("J(LASSI, direct) = %.2f cm^-1" % yamaguchi (e_roots, si.s2))

# CASCI result for reference
mc = mcscf.CASCI (mf, 12, (6,0))
mc.kernel (las.mo_coeff)
e_roots = [mc.e_tot,]
s2 = [12,]
mc = mcscf.CASCI (mf, 12, (3,3))
mc.fix_spin_(ss=0)
mc.kernel (las.mo_coeff)
e_roots += [mc.e_tot]
s2 += [0,]
print ("J(CASCI) = %.2f cm^-1" % yamaguchi (np.asarray (e_roots), np.asarray (s2)))

# Direct exchange & kinetic exchange result
las = lassi_states.all_single_excitations (las) # generate kinetic-exchange states
las.lasci () # do not reoptimize orbitals at this step - not likely to converge
e_roots, si = las.lassi ()
print (("Use of lassi_states.all_single_excitations generates\n"
        "{} additional kinetic-exchange (i.e., charge-transfer)\n"
        "states.").format (las.nroots-4))
print ("J(LASSI, direct & kinetic) = %.2f cm^-1" % yamaguchi (e_roots, si.s2))

# Locally excited states
lroots = np.minimum (3, las.get_ugg ().ncsf_sub)
las.lasci (lroots=lroots)
e_roots, si = las.lassi (opt=0)
print (("Including up to second locally-excited states improves\n"
        "results still further"))
print ("J(LASSI, direct & kinetic, nmax=2) = %.2f cm^-1" % yamaguchi (e_roots, si.s2))
print ("See bottom of file {} for output of lassi.sitools.analyze".format (mol.output))
from mrh.my_pyscf.lassi.sitools import analyze
analyze (las, si, state=[0,1,2,3]) # Four states involved in this Yamaguchi manifold


