from pyscf import gto, scf, lib, mcscf, mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.tools.molden import from_lasscf
from c2h4n4_struct import structure as struct

# Mean field calculations
mol = struct(0, 0, '6-31g')
mol.output = 'c2h4n4_sa_laspdft.log'
mol.verbose = lib.logger.INFO
mol.build()
mf = scf.RHF(mol).run()

'''
# SA-LASPDFT calculations: There are two ways
# 1. Peform the SA-LASSCF and pass that to PDFT
# 2. Pass the mf with required fragment and state information (I like this more :))
'''

from mrh.my_pyscf import mcpdft

'''
# Option-1:
las = LASSCF(mf, (3,3), ((2,1),(1,2)))
las = las.state_average([0.5,0.5],spins=[[1,-1],[-1,1]], smults=[[2,2],[2,2]], charges=[[0,0],[0,0]], wfnsyms=[[1,1],[1,1]])
mo0 = las.sort_mo([16,18,22,23,24,26])
mo0 = las.localize_init_guess((list(range(5)), list(range(5,10))), mo0)
las.kernel(mo0)

mc = mcpdft.LASSCF(las, 'tPBE', (3, 3), ((2,1),(1,2)))
mc = mc.state_average([0.5,0.5],spins=[[1,-1],[-1,1]],smults=[[2,2],[2,2]],charges=[[0,0],[0,0]], wfnsyms=[[1,1],[1,1]])
mc.kernel()
'''

# Option-2
mc = mcpdft.LASSCF(mf, 'tPBE', (3, 3), ((2,1),(1,2)))
guess_mo = mc.sort_mo([16,18,22,23,24,26])
mo0 = mc.localize_init_guess((list(range(5)), list(range(5,10))), guess_mo)
mc = mc.state_average([0.5,0.5],spins=[[1,-1],[-1,1]],smults=[[2,2],[2,2]],charges=[[0,0],[0,0]], wfnsyms=[[1,1],[1,1]])
mc.kernel(mo0)

# Results
print("\n----Results-------\n")
print("LASSCF energy for state-0 =", mc.e_mcscf[0])
print("LASPDFT energy for state-0 =", mc.e_tot[0])


