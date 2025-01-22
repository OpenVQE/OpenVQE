import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import lassi
from mrh.my_pyscf.tools import molden
from c2h4n4_struct import structure as struct

mol = struct (0, 0, '6-31g')
mol.output = 'c2h4n4_631g_automatic_singles.log'
mol.verbose = lib.logger.INFO
mol.build ()
mf = scf.RHF (mol).run ()

las = LASSCF (mf, (3,3), ((2,1),(1,2)))
las = las.state_average ([0.5,0.5],
    spins=[[1,-1],[-1,1]],
    smults=[[2,2],[2,2]],    
    charges=[[0,0],[0,0]],
    wfnsyms=[[1,1],[1,1]])
mo = las.sort_mo ([16,18,22,23,24,26])
mo = las.localize_init_guess ((list (range (5)), list (range (5,10))), mo)
las.kernel (mo)
molden.from_lasscf (las, 'c2h4n4_lasscf66_631g.molden')

mc = mcscf.CASCI (mf, 6, 6).set (fcisolver=csf_solver(mol,smult=1))
mc.kernel (las.mo_coeff)
molden.from_mcscf (mc, 'c2h4n4_casscf66_631g.molden', cas_natorb=True)

print ("LASSCF((3,3),(3,3)) energy =", las.e_tot)
print ("CASCI(6,6) energy =", mc.e_tot)

las2 = las.state_average ([0.5,0.5,0,0],
    spins=[[1,-1],[-1,1],[0,0],[0,0]],
    smults=[[2,2],[2,2],[1,1],[1,1]],    
    charges=[[0,0],[0,0],[-1,1],[1,-1]])
las2.lasci ()
las2.dump_spaces ()
# Using LASSI Kernel
#e_roots, si_hand = las2.lassi ()
mylassi_hand = lassi.LASSI(las2)
e_roots, si_hand = mylassi_hand.kernel()
print ("LASSI(hand) energy =", e_roots[0])
molden.from_lassi (las2, 'c2h4n4_las66si4_631g.molden', si=si_hand)

from mrh.my_pyscf.lassi.states import all_single_excitations
las = all_single_excitations (las)
las.lasci () # Optimize the CI vectors
las.dump_spaces () # prints all state tables in the output file
#e_roots, si_s = las.lassi ()
mylassi = lassi.LASSI(las)
e_roots, si_s = mylassi.kernel()

print ("LASSI(S) energy =", e_roots[0])
molden.from_lassi (las, 'c2h4n4_las66siS_631g.molden', si=si_s)

print ("SI vector (hand):")
print (si_hand[:,0])

print ("SI vector (singles):")
print (si_s[:,0])

