import os
import numpy as np
from scipy import linalg
from pyscf import gto, scf, df, fci, lib
from pyscf.fci.addons import fix_spin_
from pyscf import mcpdft
#from mrh.my_pyscf.fci import csf_solver
import unittest
topdir = os.path.abspath (os.path.join (__file__, '..'))

h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol_nosymm = gto.M (atom = h2co_casscf66_631g_xyz, basis = 'sto-3g', symmetry = False, output='/dev/null', verbose = 0)
mol_symm = gto.M (atom = h2co_casscf66_631g_xyz, basis = 'sto-3g', symmetry = True, output='/dev/null', verbose = 0)
def get_mc_ref (mol, ri=False, sa2=False, mo0=None):
    mf = scf.RHF (mol)
    if ri: mf = mf.density_fit (auxbasis = df.aug_etb (mol))
    mc = mcpdft.CASSCF (mf.run (), 'tPBE', 2, 2, grids_level=1)
    if sa2:
        #fcisolvers = [csf_solver (mol, smult=((2*i)+1)) for i in (0,1)]
        fcisolvers = [fix_spin_(fci.solver (mol), ss=0),
                      fix_spin_(fci.solver (mol).set (spin=2), ss=2)]
        if mol.symmetry:
            fcisolvers[0].wfnsym = 'A1'
            fcisolvers[1].wfnsym = 'A2'
        mc = mc.state_average_mix_(fcisolvers, [0.5,0.5])
    return mc.run (mo0)

def setUpModule():
    global mol_nosymm, mol_symm, mo0
    mc_symm = get_mc_ref (mol_symm)
    mo0 = mc_symm.mo_coeff.copy ()
    del mc_symm

def tearDownModule():
    global mol_nosymm, mol_symm, mo0
    mol_nosymm.stdout.close ()
    mol_symm.stdout.close ()
    del mol_nosymm, mol_symm, mo0

class KnownValues(unittest.TestCase):

    def test_ss (self):
        ref_nosymm = [-0.14738492029847025, -0.14788287172179898]
        ref_symm = [-0.14738492029577735, -0.14788570886155353]
        for mol, ref in zip ((mol_nosymm, mol_symm), (ref_nosymm, ref_symm)):
            ref_conv, ref_df = ref
            mc_conv = get_mc_ref (mol, ri=False, sa2=False, mo0=mo0)
            mc_conv_grad = mc_conv.nuc_grad_method ()
            mc_df = get_mc_ref (mol, ri=True, sa2=False, mo0=mo0)
            mc_df_grad = mc_df.nuc_grad_method ()
            for lbl, mc_grad, ref in (('conv', mc_conv_grad, ref_conv), ('DF', mc_df_grad, ref_df)):
                with self.subTest (symm=mol.symmetry, eri=lbl):
                    test = mc_grad.kernel ()
                    self.assertAlmostEqual (lib.fp (test), ref, 4)
            
    def test_sa (self):
        ref_nosymm = [-0.5126958322662911, -0.5156542636903004]
        ref_symm = [-0.512695920915627, -0.5156482433418408]
        for mol, ref in zip ((mol_nosymm, mol_symm), (ref_nosymm, ref_symm)):
            ref_conv, ref_df = ref
            mc_conv = get_mc_ref (mol, ri=False, sa2=True, mo0=mo0)
            mc_conv_grad = mc_conv.nuc_grad_method ()
            mc_df = get_mc_ref (mol, ri=True, sa2=True, mo0=mo0)
            mc_df_grad = mc_df.nuc_grad_method ()
            for lbl, mc_grad, ref in (('conv', mc_conv_grad, ref_conv), ('DF', mc_df_grad, ref_df)):
                with self.subTest (symm=mol.symmetry, eri=lbl):
                    test = np.stack ((mc_grad.kernel (state=0),
                                      mc_grad.kernel (state=1)), axis=0)
                    self.assertAlmostEqual (lib.fp (test), ref, 4)


if __name__ == "__main__":
    print("Full Tests for MC-PDFT gradients of H2CO molecule")
    unittest.main()






