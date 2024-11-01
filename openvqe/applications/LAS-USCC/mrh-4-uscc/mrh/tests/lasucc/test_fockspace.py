import numpy as np
from pyscf import lib
from mrh.my_pyscf.lassi.op_o0 import _ci_outer_product
from mrh.exploratory.citools.fockspace import *
import unittest

np.random.seed (0)
class KnownValues(unittest.TestCase):

    def test_f2h2f (self):
        nroots = 3
        norb = 8
        ci_f = np.random.rand (nroots, (2**norb)**2)
        s = np.diag (np.dot (ci_f.conj (), ci_f.T))
        ci_f /= np.sqrt (s)[:,None]
        ci_f = ci_f.reshape (nroots, (2**norb), (2**norb))
        ci_fhf = ci_f.copy ()
        norms = []
        neleca_avg = []
        nelecb_avg = []
        for (neleca, nelecb) in product (list (range (norb+1)), repeat=2):
            ci_h = fock2hilbert (ci_f, norb, (neleca, nelecb))
            n = (ci_h.conj () * ci_h).sum ((1,2))
            norms.append (n)
            neleca_avg.append (n*neleca)
            nelecb_avg.append (n*nelecb)
            ci_fhf -= hilbert2fock (ci_h, norb, (neleca, nelecb))
            smin = abs (neleca-nelecb)+1
            smax = min (neleca+nelecb, 2*norb-neleca-nelecb)+1
            nelec=(neleca,nelecb)
            w = 0
            for smult in range (smin, smax+1, 2):
                w += hilbert_sector_weight (ci_f, norb, nelec, smult)
            with self.subTest (check='sector weight nelec={}'.format (nelec)):
                self.assertAlmostEqual (lib.fp (n), lib.fp (w), 6)
        with self.subTest (check='reversible'):
            self.assertLessEqual (np.amax (np.abs (ci_fhf)), 1e-8)
        norms = np.stack (norms, axis=0).sum (0)
        neleca_avg = np.stack (neleca_avg, axis=0).sum (0)
        nelecb_avg = np.stack (nelecb_avg, axis=0).sum (0)
        na_ci, nb_ci = number_operator (ci_f, norb)
        neleca_test = np.diag (np.dot (ci_f.reshape (nroots,-1), na_ci.reshape(nroots,-1).T))
        nelecb_test = np.diag (np.dot (ci_f.reshape (nroots,-1), nb_ci.reshape(nroots,-1).T))
        for i in range (nroots):
            with self.subTest (check='root {} norm'.format (i)):
                self.assertAlmostEqual (norms[i], 1.0, 8)
            with self.subTest (check='root {} neleca'.format (i)):
                self.assertAlmostEqual (neleca_avg[i], 4.0, 1)
                self.assertAlmostEqual (neleca_avg[i], neleca_test[i], 9)
            with self.subTest (check='root {} nelecb'.format (i)):
                self.assertAlmostEqual (nelecb_avg[i], 4.0, 1)
                self.assertAlmostEqual (nelecb_avg[i], nelecb_test[i], 9)


    def test_outer_product (self):
        ci_h_f = [np.random.rand (6,6), np.random.rand (2,2), np.random.rand (6,6)]
        ci_h_f = [c / linalg.norm (c) for c in ci_h_f]
        ci_h_ref_gen = _ci_outer_product (ci_h_f, [4,2,4], [[2,2],[1,1],[2,2]])[0]
        ci_h_ref = [x.copy () for x in ci_h_ref_gen ()]
        ci_f = np.multiply.outer (np.squeeze (hilbert2fock (ci_h_f[2], 4, (2,2))),
                                  np.squeeze (hilbert2fock (ci_h_f[1], 2, (1,1))))
        ci_f = ci_f.transpose (0,2,1,3).reshape (2**6, 2**6)
        ci_f = np.multiply.outer (ci_f, np.squeeze (hilbert2fock (ci_h_f[0], 4, (2,2))))
        ci_f = ci_f.transpose (0,2,1,3).reshape (2**10, 2**10)
        ci_h_test = np.squeeze (fock2hilbert (ci_f, 10, (5,5)))
        self.assertAlmostEqual (lib.fp (ci_h_ref), lib.fp (ci_h_test), 9)


if __name__ == "__main__":
    print("Full Tests for fockspace module")
    unittest.main()

