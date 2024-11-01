import numpy as np
from pyscf import gto, scf, tools
from c2h6n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

rnn0 = 1.23681571
mol = struct (3.0, 3.0, '6-31g', symmetry=False)
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
frag_atom_list = (list (range (3)), list (range (9,12)))
mo0 = las.localize_init_guess (frag_atom_list)
las.kernel (mo0)
las_scanner = las.as_scanner ()

pes = np.loadtxt ('c2h6n4_pes_old.dat')[:34,:]
pes = np.hstack ((pes, np.zeros ((34,1))))
pes[33,3] = las.e_tot

# ISN'T THIS SO MUCH BETTER RIDDHISH?????
for ix, dr_nn in enumerate (np.arange (2.9, -0.301, -0.1)):
    mol1 = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    pes[32-ix,3] = las_scanner (mol1)

print ("  r_NN  {:>11s}  {:>13s}  {:>13s}".format ("CASSCF", "vLASSCF(v1)", "vLASSCF(test)"))
for row in pes:
    print (" {:5.3f}  {:11.6f}  {:13.8f}  {:13.8f}".format (*row))



