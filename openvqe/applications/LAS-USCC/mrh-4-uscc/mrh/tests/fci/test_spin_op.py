import numpy as np
import unittest
from scipy import linalg
from mrh.my_pyscf.fci.spin_op import contract_sdown, contract_sup
from pyscf.fci.direct_spin1 import contract_2e
from pyscf.fci.spin_op import spin_square0
from pyscf.fci import cistring
from itertools import product

np.random.seed(1)
def get_cc_chc_smult (eri, c, norb, ne):
    cc = c.conj ().ravel ().dot (c.ravel ())
    chc = c.conj ().ravel ().dot (contract_2e (eri, c, norb, ne).ravel ())
    ss, smult = spin_square0 (c, norb, ne)
    return cc, chc, smult

def opstring (op, s, ms):
    parity = int(round(2*s))%2
    if parity:
        fmt = '{:.1f}'
        s = round (s,1)
        ms = round (ms,1)
    else:
        fmt = '{:d}'
        s = int (round (s))
        ms = int (round (ms))
    fmt = 'S{}|' + fmt + ',' + fmt + '>'
    return fmt.format (('-','z','+')[op+1], s, ms)

class KnownValues(unittest.TestCase):

    def test_ladder (self):
        for norb, nelec_tot in product ((6,7), repeat=2):
            nelec = (min (norb, nelec_tot), nelec_tot - min(norb, nelec_tot))
            smult = nelec[0]-nelec[1]+1
            cishape = [cistring.num_strings (norb, ne) for ne in nelec]
            ci = np.random.rand (*cishape)
            eri = np.random.rand (norb,norb,norb,norb)
            ci /= linalg.norm (ci)
            cc_ref, chc_ref, smult_ref = get_cc_chc_smult (eri, ci, norb, nelec)
            self.assertAlmostEqual (cc_ref, 1.0, 9)
            self.assertAlmostEqual (smult_ref, smult, 9)
            s = (smult-1)/2
            with self.subTest(norb=norb, nelec=nelec_tot, op=opstring (1,s,s)):
                self.assertEqual (contract_sup (ci, norb, nelec).size, 0)
            for ms in np.arange (s,-s,-1):
                ci = contract_sdown (ci, norb, nelec)
                nelec = (nelec[0]-1, nelec[1]+1)
                cc_test, chc_test, smult_test = get_cc_chc_smult (eri, ci, norb, nelec)
                with self.subTest(norb=norb, nelec=nelec_tot, op=opstring(-1,s,ms)):
                    self.assertAlmostEqual (cc_test, 1.0, 9)
                    self.assertAlmostEqual (smult_test, smult, 9)
                    self.assertAlmostEqual (chc_test, chc_ref, 9)
            with self.subTest(norb=norb, nelec=nelec_tot, op=opstring (-1,s,-s)):
                self.assertEqual (contract_sdown (ci, norb, nelec).size, 0)
            for ms in np.arange (-s,s,1):
                ci = contract_sup (ci, norb, nelec)
                nelec = (nelec[0]+1, nelec[1]-1)
                cc_test, chc_test, smult_test = get_cc_chc_smult (eri, ci, norb, nelec)
                with self.subTest(norb=norb, nelec=nelec_tot, op=opstring(1,s,ms)):
                    self.assertAlmostEqual (cc_test, 1.0, 9)
                    self.assertAlmostEqual (smult_test, smult, 9)
                    self.assertAlmostEqual (chc_test, chc_ref, 9)

if __name__ == "__main__":
    print("Full Tests for fci.spin_op")
    unittest.main()



