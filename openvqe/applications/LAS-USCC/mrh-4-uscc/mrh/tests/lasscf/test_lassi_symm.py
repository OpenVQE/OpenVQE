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
from mrh.my_pyscf.lassi.lassi import roots_make_rdm12s, make_stdm12s, ham_2q
topdir = os.path.abspath (os.path.join (__file__, '..'))

dr_nn = 2.0
mol = struct (dr_nn, dr_nn, '6-31g', symmetry='Cs')
mol.verbose = lib.logger.DEBUG 
mol.output = 'test_lassi_symm.log'
mol.spin = 0 
mol.symmetry = 'Cs'
mol.build ()
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
las.state_average_(weights=[1.0/7.0,]*7,
    spins=[[0,0],[0,0],[2,-2],[-2,2],[0,0],[0,0],[2,2]],
    smults=[[1,1],[3,3],[3,3],[3,3],[1,1],[1,1],[3,3]],
    wfnsyms=[['A\'','A\''],]*4+[['A"','A\''],['A\'','A"'],['A\'','A\'']])
las.frozen = list (range (las.mo_coeff.shape[-1]))
ugg = las.get_ugg ()
las.mo_coeff = las.label_symmetry_(np.loadtxt (os.path.join (topdir, 'test_lassi_symm_mo.dat')))
las.ci = ugg.unpack (np.loadtxt (os.path.join (topdir, 'test_lassi_symm_ci.dat')))[1]
#las.set (conv_tol_grad=1e-8).run ()
las.e_states = las.energy_nuc () + las.states_energy_elec ()
e_roots, si = las.lassi ()
rdm1s, rdm2s = roots_make_rdm12s (las, las.ci, si)

def tearDownModule():
    global mol, mf, las
    mol.stdout.close ()
    del mol, mf, las

class KnownValues(unittest.TestCase):
    def test_evals (self):
        self.assertAlmostEqual (lib.fp (e_roots), -213.84185089228347, 6)

    def test_si (self):
        # Arbitrary signage in both the SI and CI vector, sadly
        # Actually this test seems really inconsistent overall...
        dms = [np.dot (si[:,i:i+1], si[:,i:i+1].conj ().T) for i in range (7)]
        self.assertAlmostEqual (lib.fp (np.abs (dms)), 2.5895141912171784, 6)

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

    def test_wfnsym (self):
        self.assertEqual (si.wfnsym, [0,]*5 + [1,]*2)

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

if __name__ == "__main__":
    print("Full Tests for SA-LASSI with pointgroup symmetry")
    unittest.main()

