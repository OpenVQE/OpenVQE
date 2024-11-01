import unittest
import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from pyscf.fci.direct_spin1 import _unpack_nelec
from c2h6n4_struct import structure as struct
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf import lassi_dms
from mrh.my_pyscf.mcscf.soc_int import compute_hso, amfi_dm
from mrh.my_pyscf.lassi.op_o0 import si_soc, ci_outer_product
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi.lassi import make_stdm12s, roots_make_rdm12s, ham_2q
import itertools

def setUpModule():
    global mol1, mf1, mol2, mf2, las2, las2_e, las2_si
    mol1 = gto.M (atom="""
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  -0.758602  0.000000  0.504284
    """, basis='631g',symmetry=True,
    output='test_lassi_soc1.log',
    verbose=lib.logger.DEBUG)
    mf1 = scf.RHF (mol1).run ()
   
    # NOTE: Test systems don't have to be scientifically meaningful, but they do need to
    # be "mathematically" meaningful. I.E., you can't just test zero. You need a test case
    # where the effect of the thing you are trying to test is numerically large enough
    # to be reproduced on any computer. Calculations that don't converge can't be used
    # as test cases for this reason.
    #mol2 = struct (2.0, 2.0, '6-31g', symmetry=False)
    #mol2.output = 'test_lassi_soc2.log'
    #mol2.verbose = lib.logger.DEBUG
    #mol2.build ()
    #mf2 = scf.RHF (mol2).run ()
    #las2 = LASSCF (mf2, (4,4), (4,4), spin_sub=(1,1))
    #las2.mo_coeff = las2.localize_init_guess ((list (range (3)), list (range (9,12))), mf2.mo_coeff)
    ## NOTE: for 2-fragment and above, you will ALWAYS need to remember to do the line above.
    ## If you skip it, you can expect your orbitals not to converge.
    #las2.state_average_(weights=[1,0,0,0,0],
    #                        spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
    #                        smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
    ## NOTE: Be careful about state selection. You have to select states that can actually be coupled
    ## by a 1-body SOC operator. For instance, spins=[0,0] and spins=[2,2] would need at least a 2-body
    ## operator to couple.
    #las2.kernel ()
    ## Light speed value chosen because it changes the ground state from a triplet to 
    ## a contaminated quasi-singlet.
    #with lib.light_speed (5):
    #    las2_e, las2_si = las2.lassi (opt=0, soc=True, break_symmetry=True)

def tearDownModule():
    global mol1, mf1#, mol2, mf2, las2, las2_e, las2_si
    mol1.stdout.close()
    #mol2.stdout.close()
    del mol1, mf1#, mol2, mf2, las2, las2_e, las2_si

class KnownValues (unittest.TestCase):

    # NOTE: In OpenMolcas, when using the ANO-RCC basis sets, the AMFI operator is switched from the Breit-Pauli
    # to the Douglass-Kroll Hamiltonian. There is no convenient way to switch this off; the only workaround
    # I've found is to "disguise" the basis set as something unrelated to ANO-RCC by copying and pasting it into
    # a separate file. Therefore, for now, we can only compare results from non-relativistic basis sets between
    # the two codes, until we implement Douglass-Kroll ourselves.

    def test_soc_int (self):
        # Obtained from OpenMolcas v22.02
        int_ref = 2*np.array ([0.0000000185242348, 0.0000393310222742, 0.0000393310222742, 0.0005295974407740]) 
        
        amfi_int = compute_hso (mol1, amfi_dm (mol1), amfi=True)
        amfi_int = amfi_int[2][amfi_int[2] > 0]
        amfi_int = np.sort (amfi_int.imag)
        self.assertAlmostEqual (lib.fp (amfi_int), lib.fp (int_ref), 8)

    def test_soc_1frag (self):
        # References obtained from OpenMolcas v22.10 (locally-modified to enable changing the speed of light,
        # see https://gitlab.com/MatthewRHermes/OpenMolcas/-/tree/amfi_speed_of_light)
        esf_ref = [0.0000000000,] + ([0.7194945289,]*3) + ([0.7485251565,]*3)
        eso_ref = [-0.0180900821,0.6646578117,0.6820416863,0.7194945289,0.7485251565,0.8033618737,0.8040680811]
        hso_ref = np.zeros ((7,7), dtype=np.complex128)
        hso_ref[1,0] =  0 - 10982.305j # T(+1)
        hso_ref[3,0] =  0 + 10982.305j # T(-1)
        hso_ref[4,2] =  10524.501 + 0j # T(+1)
        hso_ref[5,1] = -10524.501 + 0j # T(-1)
        hso_ref[5,3] =  10524.501 + 0j # T(+1)
        hso_ref[6,2] = -10524.501 + 0j # T(-1)
        hso_ref[5,0] =  0 - 18916.659j # T(0) < testing both this and T(+-1) is the reason I did 2 triplets
        
        las = LASSCF (mf1, (6,), (8,), spin_sub=(1,), wfnsym_sub=('A1',)).run (conv_tol_grad=1e-7)
        las.state_average_(weights=[1,0,0,0,0,0,0],
                           spins=[[0,],[2,],[0,],[-2,],[2,],[0,],[-2,],],
                           smults=[[1,],[3,],[3,],[3,],[3,],[3,],[3,],],
                           wfnsyms=([['A1',],]+([['B1',],]*3)+([['A2',],]*3)))
                           #wfnsyms=([['A1',],['B1',],['A2',],['B2',]]))
        las.lasci ()
        e0 = las.e_states[0]
        with self.subTest (deltaE='SF'):
            esf_test = las.e_states - e0
            self.assertAlmostEqual (lib.fp (esf_test), lib.fp (esf_ref), 6)
        with lib.light_speed (10):
            e_roots, si = las.lassi (opt=0, soc=True, break_symmetry=True)
            h0, h1, h2 = ham_2q (las, las.mo_coeff, soc=True)
        eso_test = e_roots - e0
        with self.subTest (deltaE='SO'):
            self.assertAlmostEqual (lib.fp (eso_test), lib.fp (eso_ref), 6)
        hso_test = (si * eso_test[None,:]) @ si.conj ().T
        from pyscf.data import nist
        au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
        def test_hso (hso_test, tag='kernel'):
            hso_test *= au2cm
            hso_test = np.around (hso_test, 8)
            # Align relative signs: 0 - 1,3,5 block (all imaginary; vide supra)
            for i in (1,3,5):
                if np.sign (hso_test.imag[i,0]) != np.sign (hso_ref.imag[i,0]):
                    hso_test[i,:] *= -1
                    hso_test[:,i] *= -1
            # Align relative signs: 2 - 4,6 block (all real; vide supra)
            for i in (4,6):
                if np.sign (hso_test.real[i,2]) != np.sign (hso_ref.real[i,2]):
                    hso_test[i,:] *= -1
                    hso_test[:,i] *= -1
            for i, j in zip (*np.where (hso_ref)):
                with self.subTest (tag, hso=(i,j)):
                    try:
                        self.assertAlmostEqual (hso_test[i,j],hso_ref[i,j],1)
                    except AssertionError as e:
                        if abs (hso_test[i,j]+hso_ref[i,j]) < 0.05:
                            raise AssertionError ("Sign fix failed for element",i,j)
                        raise (e)
                    # NOTE: 0.1 cm-1 -> 0.5 * 10^-6 au. These are actually tight checks.
        test_hso ((si * eso_test[None,:]) @ si.conj ().T)
        stdm1s, stdm2s = make_stdm12s (las, soc=True, break_symmetry=True, opt=0)
        n = las.ncas
        lbl = np.asarray ([["a'a", "b'a"],["a'b","b'b"]])
        for i,j in itertools.combinations (range (7), 2):
            neleca_i = int (round (np.trace (stdm1s[i,:n,:n,i])))
            nelecb_i = int (round (np.trace (stdm1s[i,n:,n:,i])))
            neleca_j = int (round (np.trace (stdm1s[j,:n,:n,j])))
            nelecb_j = int (round (np.trace (stdm1s[j,n:,n:,j])))
            t = stdm1s[i,:,:,j]
            t = [[linalg.norm (t[:n,:n]), linalg.norm (t[:n,n:])],
                 [linalg.norm (t[n:,:n]), linalg.norm (t[n:,n:])]]
            print (i,j)
            idx = np.asarray (t) > 0
            print ((neleca_i, nelecb_i), lbl[idx], (neleca_j, nelecb_j))
        stdm2 = stdm2s.sum ((1,4))
        e0eff = h0 - e0
        h0eff = np.eye (7) * e0eff
        h1eff = np.einsum ('pq,iqpj->ij', h1, stdm1s.conj ())
        h2eff = np.einsum ('pqrs,ipqrsj->ij', h2, stdm2) * .5
        test_hso (h0eff + h1eff + h2eff, 'make_stdm12s')
        rdm1s, rdm2s = roots_make_rdm12s (las, las.ci, si, soc=True, break_symmetry=True, opt=0)
        rdm2 = rdm2s.sum ((1,4))
        e1eff = np.einsum ('pq,iqp->i', h1, rdm1s)
        e2eff = np.einsum ('pqrs,ipqrs->i', h2, rdm2) * .5
        test_hso ((si * (e0eff+e1eff+e2eff)[None,:]) @ si.conj ().T, 'roots_make_rdm12s')


    #def test_soc_2frag (self):
    #    ## stationary test for >1 frag calc
    #    esf_ref = np.array ([-296.6356767693,-296.6354236887,-296.6354236887,-296.6354236887,-296.6354236887])
    #    eso_ref = np.array ([-296.6357061838,-296.6356871348,-296.6356871348,-296.6351604534,-296.6351310388])
    #    with self.subTest (deltaE='SF'):
    #        self.assertAlmostEqual (lib.fp (las2.e_states), lib.fp (esf_ref), 8)
    #    with self.subTest (deltaE='SO'):
    #        self.assertAlmostEqual (lib.fp (las2_e), lib.fp (eso_ref), 8)

    #def test_soc_stdm12s (self):
    #    stdm1s_test, stdm2s_test = make_stdm12s (las2, soc=True, opt=0)    
    #    with self.subTest ('2-electron'):
    #        self.assertAlmostEqual (linalg.norm (stdm2s_test), 12.835689640991259)
    #    with self.subTest ('1-electron'):
    #        self.assertAlmostEqual (linalg.norm (stdm1s_test), 5.719302474657559)
    #    dm1s_test = np.einsum ('ipqi->ipq', stdm1s_test)
    #    with self.subTest (oneelectron_sanity='diag'):
    #        # LAS states are spin-pure: there should be nothing in the spin-breaking sector
    #        self.assertAlmostEqual (np.amax(np.abs(dm1s_test[:,8:,:8])), 0)
    #        self.assertAlmostEqual (np.amax(np.abs(dm1s_test[:,:8,8:])), 0)
    #    dm2_test = np.einsum ('iabcdi->iabcd', stdm2s_test.sum ((1,4)))
    #    e0, h1, h2 = ham_2q (las2, las2.mo_coeff)
    #    # Teffanie: once you have modified ham_2q, delete the "block_diag" line below
    #    h1 = linalg.block_diag (h1,h1)
    #    e1 = np.einsum ('pq,ipq->i', h1, dm1s_test)
    #    e2 = np.einsum ('pqrs,ipqrs->i', h2, dm2_test) * .5
    #    e_test = e0 + e1 + e2
    #    with self.subTest (sanity='spin-free total energies'):
    #        self.assertAlmostEqual (lib.fp (e_test), lib.fp (las2.e_states), 6)
    #    # All the stuff below is about making sure that the nonzero part of this is in
    #    # exactly the right spot
    #    for i in range (1,5):
    #        ifrag, ispin = divmod (i-1, 2)
    #        jfrag = int (not bool (ifrag))
    #        with self.subTest (oneelectron_sanity='hermiticity', ket=i):
    #            self.assertAlmostEqual (lib.fp (stdm1s_test[0,:,:,i]),
    #                                    lib.fp (stdm1s_test[i,:,:,0].conj ().T), 16)
    #        d1 = stdm1s_test[0,:,:,i].reshape (2,2,4,2,2,4)
    #        with self.subTest (oneelectron_sanity='fragment-local', ket=i):
    #            self.assertAlmostEqual (np.amax(np.abs(d1[:,jfrag,:,:,:,:])), 0, 16)
    #            self.assertAlmostEqual (np.amax(np.abs(d1[:,:,:,:,jfrag,:])), 0, 16)
    #        d1 = d1[:,ifrag,:,:,ifrag,:]
    #        with self.subTest (oneelectron_sanity='sf sector zero', ket=i):
    #            self.assertAlmostEqual (np.amax(np.abs(d1[0,:,0,:])), 0, 16)
    #            self.assertAlmostEqual (np.amax(np.abs(d1[1,:,1,:])), 0, 16)
    #        with self.subTest (oneelectron_sanity='raising XOR lowering', ket=i):
    #            # NOTE: dumbass PySCF 1-RDM convention that ket is first
    #            if ispin:
    #                self.assertAlmostEqual (np.amax (np.abs (d1[1,:,0,:])), 0, 16)
    #                d1=d1[0,:,1,:]
    #            else:
    #                self.assertAlmostEqual (np.amax (np.abs (d1[0,:,1,:])), 0, 16)
    #                d1=d1[1,:,0,:]
    #        with self.subTest (oneelectron_sanity='nonzero S.O.C.', ket=i):
    #            self.assertAlmostEqual (linalg.norm (d1), 1.1539613201047167, 8)
    #    # lassi_dms.make_trans and total electron count
    #    ncas = las2.ncas
    #    nelec_fr = [[_unpack_nelec (fcibox._get_nelec (solver, nelecas))
    #                 for solver in fcibox.fcisolvers]
    #                for fcibox, nelecas in zip (las2.fciboxes, las2.nelecas_sub)]
    #    ci_r, nelec_r = ci_outer_product (las2.ci, las2.ncas_sub, nelec_fr)
    #    for i in range (5):
    #        nelec_r_test = (int (round (np.trace (stdm1s_test[i,:ncas,:ncas,i]))),
    #                        int (round (np.trace (stdm1s_test[i,ncas:,ncas:,i]))))
    #        with self.subTest ('electron count', state=i):
    #            self.assertEqual (nelec_r_test, nelec_r[i])
    #    def dm_sector (dm, m):
    #        # NOTE: dumbass PySCF 1-RDM convention that ket is first
    #        if m==-1: return dm[ncas:2*ncas,0:ncas]
    #        elif m==1: return dm[0:ncas,ncas:2*ncas]
    #        elif m==0:
    #            return (dm[0:ncas,0:ncas] - dm[ncas:2*ncas,ncas:2*ncas])*np.sqrt (0.5)
    #        else: assert (False)
    #    for i,j in itertools.product (range(5), repeat=2):
    #        for m in (-1,0,1):
    #            t_test = dm_sector (stdm1s_test[i,:,:,j], m)
    #            t_ref = lassi_dms.make_trans (m, ci_r[i], ci_r[j], 8, nelec_r[i], nelec_r[j])
    #            with self.subTest ('lassi_dms agreement', bra=i, ket=j, sector=m):
    #                self.assertAlmostEqual (lib.fp (t_test), lib.fp (t_ref), 9)

    #def test_soc_rdm12s (self):
    #    rdm1s_test, rdm2s_test = roots_make_rdm12s (las2, las2.ci, las2_si, opt=0)
    #    stdm1s, stdm2s = make_stdm12s (las2, soc=True, opt=0)    
    #    rdm1s_ref = np.einsum ('ir,jr,iabj->rab', las2_si.conj (), las2_si, stdm1s)
    #    rdm2s_ref = np.einsum ('ir,jr,isabtcdj->rsabtcd', las2_si.conj (), las2_si, stdm2s)
    #    with self.subTest (sanity='dm1s'):
    #        self.assertAlmostEqual (lib.fp (rdm1s_test), lib.fp (rdm1s_ref), 10)
    #    with self.subTest (sanity='dm2s'):
    #        self.assertAlmostEqual (lib.fp (rdm2s_test), lib.fp (rdm2s_ref), 10)
    #    # Stationary test has the issue that the 2nd and third states are degenerate.
    #    # Therefore their RDMs actually vary randomly. Average the second and third RDMs
    #    # together to deal with this.
    #    rdm1s_test[1:3] = rdm1s_test[1:3].sum (0) / 2
    #    rdm2s_test[1:3] = rdm2s_test[1:3].sum (0) / 2
    #    with self.subTest ('2-electron'):
    #        self.assertAlmostEqual (linalg.norm (rdm2s_test), 11.399865962223883)
    #    with self.subTest ('1-electron'):
    #        self.assertAlmostEqual (linalg.norm (rdm1s_test), 4.478325182276608)
    #    with lib.light_speed (5):
    #        e0, h1, h2 = ham_2q (las2, las2.mo_coeff, soc=True)
    #    rdm2_test = rdm2s_test.sum ((1,4))
    #    # NOTE: dumbass PySCF 1-RDM convention that ket is first
    #    e1 = np.einsum ('pq,iqp->i', h1, rdm1s_test)
    #    e2 = np.einsum ('pqrs,ipqrs->i', h2, rdm2_test) * .5
    #    e_test = e0 + e1 + e2 - las2.e_states[0]
    #    e_ref = las2_e - las2.e_states[0]
    #    for ix, (test, ref) in enumerate (zip (e_test, e_ref)):
    #        with self.subTest (sanity='spin-orbit coupled total energies', state=ix):
    #            self.assertAlmostEqual (test, ref, 6)

if __name__ == "__main__":
    print("Full Tests for SOC")
    unittest.main()
