import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import direct_spin1
from mrh.exploratory.citools import fockspace
from mrh.exploratory.unitary_cc.uccsd_sym1 import *
from itertools import product
import unittest, math

c_list = [np.random.rand (2**(2*norb)) for norb in range (2,5)]
for c in c_list:
    c[:] /= linalg.norm (c)
uop_s_list = [get_uccs_op (norb) for norb in range (2,5)]
uop_sd_list = [get_uccsd_op (norb) for norb in range (2,5)]
for uop in (uop_s_list + uop_sd_list):
    x = uop.get_uniq_amps ()
    x = (1-(2*np.random.rand (*x.shape)))*math.pi/2
    uop.set_uniq_amps_(x)

class KnownValues(unittest.TestCase):

    def test_unitary (self):
        for norb, c, uop_s, uop_sd in zip (range (2,5), c_list, uop_s_list, uop_sd_list):
            for uop, l in zip ([uop_s, uop_sd], ['singles', 'singles and doubles']):
                uc = uop (c)
                uuc = uop (uc, transpose=True)
                with self.subTest (norb=norb, op=l):
                    self.assertAlmostEqual (linalg.norm (uc), 1.0, 8)
                    self.assertAlmostEqual (lib.fp (c), lib.fp (uuc), 8)
                    self.assertLessEqual (abs (c.conj ().dot (uc)), 1.0)

    def test_s2 (self):
        for norb, c, uop in zip (range (2,5), c_list, uop_s_list):
            ssc_ref = np.zeros_like (c)
            for nelec in product (range (norb+1), repeat=2):
                c_h = np.squeeze (fockspace.fock2hilbert (c, norb, nelec))
                ssc_h = direct_spin1.contract_ss (c_h, norb, nelec)
                ssc_ref += fockspace.hilbert2fock (ssc_h, norb, nelec).ravel ()
            ssc = contract_s2 (c, norb)
            with self.subTest (norb=norb, checking='s2 op'):
                self.assertAlmostEqual (lib.fp (ssc), lib.fp (ssc_ref), 6)
            uc = uop (c)
            ussc = uop (ssc)
            ssuc = contract_s2 (uc, norb)
            commutator = ussc - ssuc
            with self.subTest (norb=norb, checking='singles operator symmetry'):
                self.assertLessEqual (linalg.norm (commutator), 1e-8)

    def test_number_symmetry (self):
        for norb, c, uop_s, uop_sd in zip (range (2,5), c_list, uop_s_list, uop_sd_list):
            for uop, l in zip ([uop_s, uop_sd], ['singles', 'singles and doubles']):
                uc = uop (c)
                ac, bc = fockspace.number_operator (c, norb)
                auc, buc = fockspace.number_operator (uc, norb)
                uac, ubc = uop (ac), uop (bc)
                comm_a = auc - uac
                comm_b = buc - ubc
                with self.subTest (norb=norb, op=l):
                    self.assertLessEqual (linalg.norm (comm_a), 1e-8)
                    self.assertLessEqual (linalg.norm (comm_b), 1e-8)

if __name__ == "__main__":
    print("Full Tests for UCC partial spin symmetry module")
    unittest.main()

