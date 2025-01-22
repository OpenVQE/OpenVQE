import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import direct_spin1
from mrh.exploratory.citools import fockspace
from mrh.exploratory.unitary_cc.uccsd_sym0 import *
from mrh.exploratory.unitary_cc.uccsd_sym0 import _op1h_spinsym
from itertools import product
import unittest

class KnownValues(unittest.TestCase):

    def test_unitary (self):
        for norb in range (3,7):
            c = np.random.rand (2**norb)
            c /= linalg.norm (c)
            tp = np.random.rand (norb)
            tph = np.random.rand (norb,norb)
            t2 = np.random.rand (norb,norb,norb,norb)
            uop_s = get_uccs_op (norb, tp=tp, tph=tph)
            uop_sd = get_uccsd_op (norb, tp=tp, tph=tph, t2=t2)
            for uop, l in zip ([uop_s, uop_sd], ['singles', 'singles and doubles']):
                uc = uop (c)
                uuc = uop (uc, transpose=True)
                with self.subTest (norb=norb, op=l):
                    self.assertAlmostEqual (linalg.norm (uc), 1.0, 6)
                    self.assertAlmostEqual (lib.fp (c), lib.fp (uuc), 6)
                    self.assertLessEqual (abs (c.conj ().dot (uc)), 1.0)

    def test_derivative (self):
        norb = 4
        c = np.random.rand (2**norb)
        uop = FSUCCOperator (norb, [2, 3], [1, 1])
        x = np.array ([0.5, -0.6])
        dx = 0.01
        uop.set_uniq_amps_(x)
        uc_0 = uop (c)
        for i in range (2):
            duc = uop.get_deriv1 (c, i)
            xp = x.copy ()
            xp[i] += dx
            uop.set_uniq_amps_(xp)
            uc_p = uop (c)
            xm = x.copy ()
            xm[i] -= dx
            uop.set_uniq_amps_(xm)
            uc_m = uop (c)
            uop.set_uniq_amps_(x)
            duc_num = (uc_p - uc_m) / (2*dx)
            with self.subTest (igen=i):
                self.assertAlmostEqual (lib.fp (duc), lib.fp (duc_num), 4)

    def test_ham (self):
        for norb in range (2,5):
            npair = norb*(norb+1)//2
            c = np.random.rand (2**(2*norb))
            c /= linalg.norm (c)
            h0 = np.random.rand (1)[0]
            h1 = np.random.rand (norb, norb)
            h2 = np.random.rand (npair, npair)
            h1 += h1.T
            h2 += h2.T
            hc_ref = np.zeros_like (c)
            for nelec in product (range (norb+1), repeat=2):
                c_h = np.squeeze (fockspace.fock2hilbert (c, norb, nelec))
                h2eff = direct_spin1.absorb_h1e (h1, h2, norb, nelec, 0.5)
                hc_h = h0 * c_h + direct_spin1.contract_2e (h2eff, c_h, norb, nelec)
                hc_ref += fockspace.hilbert2fock (hc_h, norb, nelec).ravel ()
            h1 = h1[np.tril_indices (norb)]
            hc_test = _op1h_spinsym (norb, [h0,h1,h2], c)
            with self.subTest (norb=norb):
                self.assertAlmostEqual (lib.fp (hc_test), lib.fp (hc_ref), 6)


if __name__ == "__main__":
    print("Full Tests for UCC no-symmetry module")
    unittest.main()

