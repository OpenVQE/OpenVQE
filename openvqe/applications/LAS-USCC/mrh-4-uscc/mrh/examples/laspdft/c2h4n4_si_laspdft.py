from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf import lassi
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.tools.molden import from_lasscf
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.lassi.states import all_single_excitations

# Mean field calculation
mol = struct(0, 0, '6-31g')
mol.output = 'c2h4n4_si_laspdft.log'
mol.verbose = lib.logger.INFO
mol.build()
mf = scf.RHF(mol).run()

# Option: Perfom the SA-LASSCF calculations and feed that along with level of Charge transfer to MC-PDFT to obtain the LASSI and LASSI-PDFT energy

# SA-LASSCF: For orbitals
las = LASSCF(mf, (3,3), ((2,1),(1,2)))
las = las.state_average([0.5,0.5], spins=[[1,-1],[-1,1]], smults=[[2,2],[2,2]], charges=[[0,0],[0,0]],wfnsyms=[[1,1],[1,1]])
guess_mo = las.sort_mo([16,18,22,23,24,26])
mo0 = las.localize_init_guess((list(range (5)), list(range (5,10))), guess_mo)
las.kernel(mo0)
 
# LASSI-PDFT
mc = mcpdft.LASSI(las, 'tPBE', (3, 3), ((2,1),(1,2)))
mc = all_single_excitations(mc) # Level of charge transfer
mc.kernel(las.mo_coeff) # SA-LAS orbitals

# Results
print("\n----Results-------\n")
#print("State",' \t',  "LASSCF Energy",'\t\t',"LASSI Energy",'\t\t', "LASSI-PDFT Energy") 
#[print(sn,'\t',x,'\t', y,'\t', z) for sn, x, y, z in zip(list(range(mc.nroots)), mc.e_mcscf, mc.e_lassi, mc.e_tot)]
print("LASSI state-0 =", mc.e_mcscf[0])
print("LASSI-PDFT state-0 =", mc.e_tot[0])

