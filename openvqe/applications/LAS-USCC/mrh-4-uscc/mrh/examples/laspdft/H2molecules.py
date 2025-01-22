import sys
from pyscf import gto, scf, tools, dft, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_pyscf import mcpdft

# Initialization
mol = gto.M()
mol.atom='''H -5.10574 2.01997 0.00000; H -4.29369 2.08633 0.00000; H -3.10185 2.22603 0.00000; H -2.29672 2.35095 0.00000''' 
mol.basis='sto3g'
mol.build()

# Mean field calculation
mf = scf.ROHF(mol).newton().run()

'''
# Option-1: Perform LASSCF and then pass las object
# LASSCF Calculations
las = LASSCF(mf,(2, 2),(2, 2),spin_sub=(1, 1))
frag_atom_list = ([0, 1], [2, 3]) 
mo0 = las.localize_init_guess(frag_atom_list)
elas = las.kernel(mo0)[0]

# LAS-PDFT
mc = mcpdft.LASSCF(las, 'tPBE', (2, 2), (2,2), grids_level=1)
epdft = mc.kernel()[0]
'''
# Option-2: Feed the mean field object and fragment information to mcpdft.LASSCF
mc = mcpdft.LASSCF(mf, 'tPBE', (2, 2), (2, 2), spin_sub=(1,1), grids_level=1)
frag_atom_list = ([0, 1] , [2, 3])
mo0 = mc.localize_init_guess (frag_atom_list)
mc.kernel(mo0)

elas = mc.e_mcscf[0]
epdft = mc.e_tot

print ("E(LASSCF) =", elas)
print ("E(tPBE) =", epdft)
