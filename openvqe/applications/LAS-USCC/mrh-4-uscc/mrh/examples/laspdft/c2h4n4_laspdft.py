from pyscf import gto, scf, lib, mcscf, mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.tools.molden import from_lasscf
from c2h4n4_struct import structure as struct

# Mean field calculations
mol = struct(0, 0, '6-31g')
mol.output = 'c2h4n4_laspdft.log'
mol.verbose = lib.logger.INFO
mol.build()
mf = scf.RHF(mol).run()

# LASPDFT
from mrh.my_pyscf import mcpdft

# Option-1:
las = LASSCF(mf, (3,3), ((2,1),(1,2)))
mo0 = las.sort_mo([16,18,22,23,24,26])
mo0 = las.localize_init_guess((list(range(5)), list(range(5,10))), mo0)
las.kernel(mo0)

mc = mcpdft.LASSCF(las, 'tPBE', (3, 3), ((2,1),(1,2)))
mc.kernel()

'''
# Option-2
mc = mcpdft.LASSCF(mf, 'tPBE', (3, 3), ((2,1),(1,2)))
guess_mo = mc.sort_mo([16,18,22,23,24,26])
mo0 = mc.localize_init_guess((list(range(5)), list(range(5,10))), guess_mo)
mc.kernel(mo0)
'''

print("\n------Results-----\n")
print ("E(LASSCF) =", mc.e_mcscf)
print ("E(LAS-tPBE) =", mc.e_tot)
