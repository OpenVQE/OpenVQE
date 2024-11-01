from pyscf import gto, scf, lib, mcscf
from mrh.my_pyscf import mcpdft

mol = gto.M (atom='C 0 0 0', basis='6-31g', output='debug_sa_casci.log', verbose=lib.logger.INFO)
mf = scf.RHF (mol).run ()
mc1 = mcpdft.CASSCF (mf, 'tPBE', 4, 4, grids_level=6).run ()
mc2 = mcpdft.CASCI (mf, 'tPBE', 4, 4, grids_level=6).set (mo_coeff=mc1.mo_coeff).run ()
print (mc1.e_tot-mc2.e_tot)
mc3 = mc2.state_average ([0.5,0.5]).run ()
print (mc3.e_states[0]-mc1.e_tot)

