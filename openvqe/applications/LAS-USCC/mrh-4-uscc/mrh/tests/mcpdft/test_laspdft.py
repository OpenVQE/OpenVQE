import sys
from pyscf import gto, scf, tools, dft, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_pyscf import mcpdft
import unittest

class KnownValues(unittest.TestCase):
    def test_h4 (self):
        mol = gto.M()
        mol.atom='''H -5.10574 2.01997 0.00000;
                    H -4.29369 2.08633 0.00000;
                    H -3.10185 2.22603 0.00000;
                    H -2.29672 2.35095 0.00000''' 
        mol.basis='sto3g'
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.build()
        mf = scf.ROHF(mol).newton().run()
        mc = mcpdft.LASSCF(mf, 'tPBE', (2, 2), (2, 2), spin_sub=(1,1), grids_level=1)
        frag_atom_list = ([0, 1] , [2, 3])
        mo0 = mc.localize_init_guess (frag_atom_list)
        mc.kernel(mo0)
        elas = mc.e_mcscf[0]
        epdft = mc.e_tot
        self.assertAlmostEqual (mc.e_tot, -2.285617754797544, 7)

if __name__ == "__main__":
    print("Full Tests for LAS-PDFT")
    unittest.main()


