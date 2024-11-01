import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from pyscf.lib import temporary_env
from pyscf.mcscf import mc_ao2mo
from pyscf import mcpdft
from mrh.my_pyscf.mcpdft.pdft_feff import EotOrbitalHessianOperator
from mrh.my_pyscf.mcpdft.pdft_feff import vector_error
import unittest

h2 = scf.RHF (gto.M (atom = 'H 0 0 0; H 1.2 0 0', basis = '6-31g', 
    output='/dev/null', verbose=0)).run ()
lih = scf.RHF (gto.M (atom = 'Li 0 0 0; H 1.2 0 0', basis = 'sto-3g',
    output='/dev/null', verbose=0)).run ()

def case (kv, mc, mol, state, fnal):
    hop = EotOrbitalHessianOperator (mc, incl_d2rho=True)
    with kv.subTest (q='g_orb sanity'):
        veff1, veff2 = mc.get_pdft_veff (mc.mo_coeff, mc.ci, incl_coul=False,
                                         paaa_only=True)
        fcasscf = mcscf.CASSCF (mc._scf, mc.ncas, mc.nelecas)
        fcasscf.__dict__.update (mc.__dict__)
        casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)
        with lib.temporary_env (fcasscf, get_hcore=lambda:veff1):
            g_orb_alt = fcasscf.gen_g_hop (mc.mo_coeff, 1, casdm1, casdm2,
                                           veff2)[0]
            kv.assertAlmostEqual (lib.fp (hop.g_orb), lib.fp (g_orb_alt), 9)
    x0 = -hop.g_orb / hop.h_diag
    x0_norm = linalg.norm (x0)
    x0 = 2*np.random.rand(*x0.shape)-1
    x0 *= x0_norm / linalg.norm (x0)
    x0[hop.g_orb==0] = 0
    err_tab = np.zeros ((2,4))
    for ix, p in enumerate (range (17,19)):
        # For numerically unstable (i.e., translated) fnals,
        # it is somewhat difficult to find the convergence plateau
        # However, repeated calculations should show that
        # failure is rare and due only to numerical instability
        # and chance.
        x1 = x0 / (2**p)
        x1_norm = linalg.norm (x1)
        dg_test, de_test = hop (x1)
        dg_ref, de_ref = hop.seminum_orb (x1)
        de_err = abs ((de_test-de_ref)/de_ref)
        dg_err, dg_theta = vector_error (dg_test, dg_ref)
        err_tab[ix,:] = [x1_norm, de_err, dg_err, dg_theta]
    conv_tab = err_tab[1:,:] / err_tab[:-1,:]
    with kv.subTest (q='x'):
        kv.assertAlmostEqual (conv_tab[-1,0], 0.5, 9)
    with kv.subTest (q='de'):
        kv.assertAlmostEqual (conv_tab[-1,1], 0.5, delta=0.05)
    with kv.subTest (q='d2e'):
        kv.assertAlmostEqual (conv_tab[-1,2], 0.5, delta=0.05)
    

def tearDownModule():
    global h2, lih
    h2.mol.stdout.close ()
    lih.mol.stdout.close ()
    del h2, lih

class KnownValues(unittest.TestCase):

    def test_de_d2e (self):
        for mol, mf in zip (('H2', 'LiH'), (h2, lih)):
            for state, nel in zip (('Singlet', 'Triplet'), (2, (2,0))):
                for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
                    mc = mcpdft.CASSCF (mf, fnal, 2, nel, grids_level=1).run ()
                    with self.subTest (mol=mol, state=state, fnal=fnal):
                        case (self, mc, mol, state, fnal)


if __name__ == "__main__":
    print("Full Tests for MC-PDFT second fnal derivatives")
    unittest.main()






