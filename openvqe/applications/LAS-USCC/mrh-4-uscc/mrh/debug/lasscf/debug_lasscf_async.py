import unittest
from pyscf import gto, scf, tools, mcscf,lib
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from mrh.my_pyscf.mcscf import lasscf_sync_o0 as syn
from pyscf.mcscf import avas

def setUpModule():
    global mf, frag_atom_list, mo0
    xyz='''H     -0.943967690125   0.545000000000   0.000000000000
    C      0.000000000000   0.000000000000   0.000000000000
    C      1.169134295109   0.675000000000   0.000000000000
    H      0.000000000000  -1.090000000000   0.000000000000
    H      1.169134295109   1.765000000000   0.000000000000
    C      2.459512146748  -0.070000000000   0.000000000000
    C      3.628646441857   0.605000000000   0.000000000000
    H      2.459512146748  -1.160000000000   0.000000000000
    H      3.628646441857   1.695000000000   0.000000000000
    H      4.572614131982   0.060000000000   0.000000000000'''
    mol=gto.M(atom=xyz,basis='sto3g',verbose=4,output='debug_lasscf_async.log')
    mf=scf.RHF(mol)
    mf.run()
    frag_atom_list=[list(range(1+4*ifrag,3+4*ifrag)) for ifrag in range(2)]
    ncas,nelecas,mo0 = avas.kernel(mf, ['C 2p'])
    
def tearDownModule():
    global mf, frag_atom_list, mo0
    mf.stdout.close ()
    del mf, frag_atom_list, mo0

def _run_mod (mod):
    las=mod.LASSCF(mf, (2,2), (2,2))
    localize_fn = getattr (las, 'set_fragments_', las.localize_init_guess)
    mo_coeff=localize_fn (frag_atom_list, mo0)
    las.state_average_(weights=[.2,]*5,
                       spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                       smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
    las.kernel(mo_coeff)
    return las

class KnownValues (unittest.TestCase):

    def test_implementations (self):
        las_syn = _run_mod (syn)
        with self.subTest ('synchronous calculation converged'):
            self.assertTrue (las_syn.converged)
        las_asyn = _run_mod (asyn)
        with self.subTest ('asynchronous calculation converged'):
            self.assertTrue (las_asyn.converged)
        with self.subTest ('average energy'):
            self.assertAlmostEqual (las_syn.e_tot, las_asyn.e_tot, 8)
        for i in range (5):
            with self.subTest ('energy', state=i):
                self.assertAlmostEqual (las_syn.e_states[i], las_asyn.e_states[i], 6)

if __name__ == "__main__":
    print("Full Tests for lasscf_async")
    unittest.main()
