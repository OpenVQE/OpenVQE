import sys
from pyscf import gto, scf, tools, dft, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_pyscf import mcpdft
import unittest

class KnownValues(unittest.TestCase):
    def test_ethene (self):
        mol = gto.M()
        mol.atom = '''C  -0.662958  0.000000  0.000000;
                            C   0.662958  0.000000  0.000000;
                            H  -1.256559  -0.924026  0.000000;
                            H  1.256559  -0.924026  0.000000;
                            H  -1.256559  0.924026  0.000000;
                            H  1.256559  0.924026  0.000000'''
        mol.basis='sto3g'
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.build()
        mf = scf.ROHF(mol).newton().run()
        mc = mcpdft.LASSI(mf, 'tPBE', (2, ), (2, ), grid_level=1).state_average(
            [1, 0], spins=[[0,], [2, ]], smults=[[1, ], [3, ]], charges=[[0, ],[0, ]])
        mo0 = mc.localize_init_guess(([0, 1],), mc.sort_mo([8, 9]))
        mc.kernel(mo0)
        elassi = mc.e_mcscf[0]
        epdft = mc.e_tot[0]
        self.assertAlmostEqual (elassi , -77.1154672717181, 7) # Reference values of CASSCF and CAS-PDFT
        self.assertAlmostEqual (epdft , -77.49805221093968, 7)

if __name__ == "__main__":
    print("Full Tests for LASSI-PDFT")
    unittest.main()

    

