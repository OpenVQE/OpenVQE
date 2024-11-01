from pyscf import scf
from mrh.tests.lasscf.c2h6n4_struct import structure as struct
from mrh.my_pyscf.mcscf import lasscf_sync_o0 as syn
from mrh.my_pyscf.mcscf import lasscf_async as asyn

mol = struct (1.0, 1.0, 'sto-3g', symmetry=False)
mol.verbose = 5
mol.output = 'c2h6n4_lasscf88_sto3g.log'
mol.build ()
mf = scf.RHF (mol).run ()
las_syn = syn.LASSCF (mf, (4,4), ((4,0),(0,4)), spin_sub=(5,5))
mo = las_syn.localize_init_guess ((list (range (3)), list (range (9,12))), mf.mo_coeff)
las_syn.state_average_(weights=[1,0,0,0,0],
                       spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                       smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
las_syn.kernel (mo)
print ("Synchronous calculation converged?", las_syn.converged)
las_asyn = asyn.LASSCF (mf, (4,4), ((4,0),(0,4)), spin_sub=(5,5))
mo = las_asyn.set_fragments_((list (range (3)), list (range (9,12))), mf.mo_coeff)
las_asyn.state_average_(weights=[1,0,0,0,0],
                        spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                        smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
las_asyn.kernel (mo)
print ("Asynchronous calculation converged?", las_asyn.converged)
print ("Final state energies:")
print ("{:>16s} {:>16s} {:>16s}".format ("Synchronous", "Asynchronous", "Difference"))
fmt_str = "{:16.9e} {:16.9e} {:16.9e}"
for es, ea in zip (las_syn.e_states, las_asyn.e_states): print (fmt_str.format (es, ea, ea-es))




