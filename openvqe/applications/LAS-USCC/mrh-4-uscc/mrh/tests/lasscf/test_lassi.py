#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import copy
import unittest
import numpy as np
from scipy import linalg
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.tools import molden
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi.lassi import roots_make_rdm12s, root_make_rdm12s, make_stdm12s, ham_2q
topdir = os.path.abspath (os.path.join (__file__, '..'))

def setUpModule ():
    global mol, mf, las, e_roots, si, rdm1s, rdm2s
    dr_nn = 2.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG
    mol.output = 'test_lassi.log'
    mol.spin = 0
    mol.build ()
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
    las.state_average_(weights=[1.0/5.0,]*5,
        spins=[[0,0],[0,0],[2,-2],[-2,2],[2,2]],
        smults=[[1,1],[3,3],[3,3],[3,3],[3,3]])
    las.frozen = list (range (las.mo_coeff.shape[-1]))
    ugg = las.get_ugg ()
    las.mo_coeff = np.loadtxt (os.path.join (topdir, 'test_lassi_mo.dat'))
    las.ci = ugg.unpack (np.loadtxt (os.path.join (topdir, 'test_lassi_ci.dat')))[1]
    #las.set (conv_tol_grad=1e-8).run ()
    #np.savetxt ('test_lassi_mo.dat', las.mo_coeff)
    #np.savetxt ('test_lassi_ci.dat', ugg.pack (las.mo_coeff, las.ci))
    las.e_states = las.energy_nuc () + las.states_energy_elec ()
    e_roots, si = las.lassi ()
    rdm1s, rdm2s = roots_make_rdm12s (las, las.ci, si)

def tearDownModule():
    global mol, mf, las, e_roots, si, rdm1s, rdm2s
    mol.stdout.close ()
    del mol, mf, las, e_roots, si, rdm1s, rdm2s

class KnownValues(unittest.TestCase):
    def test_evals (self):
        self.assertAlmostEqual (lib.fp (e_roots), 153.47664766268417, 6)

    def test_si (self):
        # Arbitrary signage in both the SI and CI vector, sadly
        # Actually this test seems really inconsistent overall...
        dms = [np.dot (si[:,i:i+1], si[:,i:i+1].conj ().T) for i in range (7)]
        self.assertAlmostEqual (lib.fp (np.abs (dms)), 2.371964339437981, 6)

    def test_nelec (self):
        for ix, ne in enumerate (si.nelec):
            if ix == 1:
                self.assertEqual (ne, (6,2))
            else:
                self.assertEqual (ne, (4,4))

    def test_s2 (self):
        s2_array = np.zeros (7)
        s2_array[1] = 6
        s2_array[2] = 6
        s2_array[3] = 2
        self.assertAlmostEqual (lib.fp (si.s2), lib.fp (s2_array), 3)

    def test_tdms (self):
        stdm1s, stdm2s = make_stdm12s (las)
        nelec = float (sum (las.nelecas))
        for ix in range (stdm1s.shape[0]):
            d1 = stdm1s[ix,...,ix].sum (0)
            d2 = stdm2s[ix,...,ix].sum ((0,3))
            with self.subTest (root=ix):
                self.assertAlmostEqual (np.trace (d1), nelec,  9)
                self.assertAlmostEqual (np.einsum ('ppqq->',d2), nelec*(nelec-1), 9)
        rdm1s_test = np.einsum ('ar,asijb,br->rsij', si.conj (), stdm1s, si) 
        rdm2s_test = np.einsum ('ar,asijtklb,br->rsijtkl', si.conj (), stdm2s, si) 
        self.assertAlmostEqual (lib.fp (rdm1s_test), lib.fp (rdm1s), 9)
        self.assertAlmostEqual (lib.fp (rdm2s_test), lib.fp (rdm2s), 9)

    def test_rdms (self):    
        h0, h1, h2 = ham_2q (las, las.mo_coeff)
        d1_r = rdm1s.sum (1)
        d2_r = rdm2s.sum ((1, 4))
        nelec = float (sum (las.nelecas))
        for ix, (d1, d2) in enumerate (zip (d1_r, d2_r)):
            with self.subTest (root=ix):
                self.assertAlmostEqual (np.trace (d1), nelec,  9)
                self.assertAlmostEqual (np.einsum ('ppqq->',d2), nelec*(nelec-1), 9)
        e_roots_test = h0 + np.tensordot (d1_r, h1, axes=2) + np.tensordot (d2_r, h2, axes=4) / 2
        for e1, e0 in zip (e_roots_test, e_roots):
            self.assertAlmostEqual (e1, e0, 8)

    def test_lassi_singles_constructor (self):
        from mrh.my_pyscf.lassi.states import all_single_excitations
        las2 = all_single_excitations (las)
        las2.check_sanity ()
        # Meaning of tuple: (na+nb,smult)
        # from state 0, (1+2,2)&(3+2,2) & a<->b & l<->r : 4 states
        # from state 1, smults=((2,4),(4,2),(4,4)) permutations of above: 12 additional states
        # from states 2 and 3, (4+1,4)&(0+3,4) & a<->b & l<->r : 4 additional states
        # from state 4, (2+1,2)&(4+1,4) & (2+1,4)&(4+1,4) 
        #             & (3+0,4)&(3+2,2) & (3+0,4)&(3+2,4) & l<->r : 8 additional states
        # 5 + 4 + 12 + 4 + 8 = 33
        self.assertEqual (las2.nroots, 33)

    def test_lassi_spin_shuffle (self):
        from mrh.my_pyscf.lassi.states import spin_shuffle
        las3 = LASSCF (mf, (4,2,4), (4,2,4), spin_sub=(5,3,5))
        las3 = spin_shuffle (las3)
        las3.check_sanity ()
        # The number of states is the number of graphs connecting one number
        # in each row which sum to zero:
        # -2 -1 0 +1 +2
        #    -1 0 +1
        # -2 -1 0 +1 +2
        # For the first two rows, paths which sum to -3 and +3 are immediately
        # excluded. Two paths connecting the first two rows each sum to -2 and +2
        # and three paths each sum to -1, 0, +1. Each partial sum then has one
        # remaining option to complete the path, so
        # 2 + 3 + 3 + 3 + 2 = 13
        self.assertEqual (las3.nroots, 13)

    def test_casci_limit (self):
        from mrh.my_pyscf.lassi.states import all_single_excitations
        from mrh.my_pyscf.mcscf.lasci import get_space_info
        xyz='''H 0 0 0
        H 1 0 0
        H 3 0 0
        H 4 0 0'''
        rmol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=0, output='/dev/null')
        rmf = scf.RHF (rmol).run ()

        # Random Hamiltonian
        rng = np.random.default_rng ()
        rmf._eri = rng.random (rmf._eri.shape)
        hcore = rng.random ((4,4))
        rmf.get_hcore = lambda *args: hcore

        # CASCI limit
        mc = mcscf.CASCI (rmf, 4, 4).run ()
        casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, mc.ncas, mc.nelecas)

        # LASSCF
        rlas = LASSCF (rmf, (2,2), (2,2), spin_sub=(1,1))
        rlas.conv_tol_grad = rlas.conv_tol_self = 9e99

        # LASSI in the CASCI limit
        for i in range (2): rlas = all_single_excitations (rlas)
        charges, spins, smults, wfnsyms = get_space_info (rlas)
        lroots = 4 - smults
        idx = (charges!=0) & (lroots==3)
        lroots[idx] = 1
        rlas.lasci (lroots=lroots.T)
        e_roots, si = rlas.lassi (opt=0)
        with self.subTest ("total energy"):
            self.assertAlmostEqual (e_roots[0], mc.e_tot, 8)
        lasdm1s, lasdm2s = root_make_rdm12s (rlas, rlas.ci, si, state=0, opt=0)
        lasdm1 = lasdm1s.sum (0)
        lasdm2 = lasdm2s.sum ((0,3))
        with self.subTest ("casdm1"):
            self.assertAlmostEqual (lib.fp (lasdm1), lib.fp (casdm1), 8)
        with self.subTest ("casdm2"):
            self.assertAlmostEqual (lib.fp (lasdm2), lib.fp (casdm2), 8)
        stdm1s = make_stdm12s (rlas, opt=0)[0][9:13,:,:,:,9:13] # second rootspace
        with self.subTest("state indexing"):
            # column-major ordering for state excitation quantum numbers:
            # earlier fragments advance faster than later fragments
            self.assertAlmostEqual (lib.fp (stdm1s[0,:,:2,:2,0]),
                                    lib.fp (stdm1s[2,:,:2,:2,2]))
            self.assertAlmostEqual (lib.fp (stdm1s[0,:,2:,2:,0]),
                                    lib.fp (stdm1s[1,:,2:,2:,1]))

if __name__ == "__main__":
    print("Full Tests for SA-LASSI")
    unittest.main()

