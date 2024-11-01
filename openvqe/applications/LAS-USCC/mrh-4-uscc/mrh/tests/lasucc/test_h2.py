import numpy as np
from pyscf import gto, scf, mcscf, lib
from mrh.my_pyscf.fci import csf_solver
from mrh.exploratory.unitary_cc import uccsd_sym0
from mrh.exploratory.unitary_cc import uccsd_sym1
import unittest, math

mol = gto.M (atom = 'H 0 0 0; H 1.5 0 0', basis='6-31g', verbose=0, output='/dev/null')
rhf = scf.RHF (mol).run ()
uhf = scf.UHF (mol)
dm_a, dm_b = uhf.get_init_guess ()
dm_b[:2,:2] = 0.0
uhf.kernel ((dm_a, dm_b))
fci = mcscf.CASSCF (rhf, 4, 2).set (fcisolver = csf_solver (mol, smult=1)).run ()

def tearDownModule():
    global mol, rhf, uhf, fci
    mol.stdout.close ()
    del mol, rhf, uhf, fci

class KnownValues(unittest.TestCase):

    def test_sym0_uccs (self):
        np.random.seed (1) 
        x0 = (1 - 2*np.random.rand (28))*math.pi/2
        uccs = uccsd_sym0.UCCS (mol).run (x=x0)
        self.assertAlmostEqual (uhf.e_tot, uccs.e_tot, 6)

    def test_sym0_uccsd (self):
        uccsd = uccsd_sym0.UCCSD (mol).run ()
        self.assertAlmostEqual (uccsd.e_tot, fci.e_tot, 6)

    def test_sym1_uccs (self):
        uccs = uccsd_sym1.UCCS (mol).run ()
        self.assertAlmostEqual (rhf.e_tot, uccs.e_tot, 6)

    def test_sym1_uccsd (self):
        uccsd = uccsd_sym1.UCCSD (mol).run ()
        self.assertAlmostEqual (uccsd.e_tot, fci.e_tot, 6)

if __name__ == "__main__":
    print("Full Tests for UCCS & UCCSD/6-31g of H2 molecule")
    unittest.main()

