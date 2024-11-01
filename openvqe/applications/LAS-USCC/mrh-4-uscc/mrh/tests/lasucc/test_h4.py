import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.fci import direct_spin1
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.exploratory.citools import lasci_ominus1, fockspace
import unittest

xyz = '''H 0.0 0.0 0.0
         H 1.0 0.0 0.0
         H 0.2 3.9 0.1
         H 1.159166 4.1 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='/dev/null', verbose=0)
mf = scf.RHF (mol).run ()
ref = mcscf.CASSCF (mf, 4, 4).run () # = FCI
las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
las.kernel (las.localize_init_guess (((0,1), (2,3)), mf.mo_coeff))

def tearDownModule():
    global mol, mf, ref, las
    mol.stdout.close ()
    del mol, mf, ref, las

class KnownValues(unittest.TestCase):

    def test_lasci_ominus1 (self):
        mc = mcscf.CASCI (mf, 4, 4)
        mc.mo_coeff = las.mo_coeff
        mc.fcisolver = lasci_ominus1.FCISolver (mol)
        mc.fcisolver.norb_f = [2,2]
        mc.kernel ()
        self.assertAlmostEqual (mc.e_tot, las.e_tot, 6)

    def test_lasuccsd_total_energy (self):
        mc = mcscf.CASCI (mf, 4, 4)
        mc.mo_coeff = las.mo_coeff
        mc.fcisolver = lasuccsd.FCISolver (mol)
        mc.fcisolver.norb_f = [2,2]
        mc.kernel ()
        with self.subTest (from_='reported'):
            self.assertAlmostEqual (mc.e_tot, ref.e_tot, 6)
        psi = mc.fcisolver.psi
        x = psi.x
        h1, h0 = mc.get_h1eff ()
        h2 = mc.get_h2eff ()
        h = [h0, h1, h2]
        energy, gradient = psi.e_de (x, h)
        with self.subTest (from_='obj fn'):
            self.assertAlmostEqual (energy, mc.e_tot, 9)
        c = np.squeeze (fockspace.fock2hilbert (psi.get_fcivec (x), 4, (2,2)))
        h2eff = direct_spin1.absorb_h1e (h1, h2, 4, (2,2), 0.5)
        hc = direct_spin1.contract_2e (h2eff, c, 4, (2,2))
        chc = c.conj ().ravel ().dot (hc.ravel ()) + h0
        with self.subTest (from_='h2 contraction'):
            self.assertAlmostEqual (chc, mc.e_tot, 9)
        c_f = psi.ci_f
        for ifrag, c in enumerate (c_f):
            heff = psi.get_dense_heff (x, h, ifrag)
            hc = np.dot (heff.full, c.ravel ())
            chc = np.dot (c.conj ().ravel (), hc)
            with self.subTest (from_='frag {} Fock-space dense effective Hamiltonian'.format (ifrag)):
                self.assertAlmostEqual (chc, mc.e_tot, 9)
            heff, _ = heff.get_number_block ((1,1),(1,1))
            c = np.squeeze (fockspace.fock2hilbert (c, 2, (1,1))).ravel ()
            hc = np.dot (heff, c)
            chc = np.dot (c.conj (), hc)
            with self.subTest (from_='frag {} Hilbert-space dense effective Hamiltonian'.format (ifrag)):
                self.assertAlmostEqual (chc, mc.e_tot, 9)
            


if __name__ == "__main__":
    print("Full Tests for LASCI(o-1) and LASUCCSD/sto-3g of H2 dimer")
    unittest.main()

