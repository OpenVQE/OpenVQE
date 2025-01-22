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

import copy
import unittest
import numpy as np
from scipy import linalg
from copy import deepcopy
from itertools import product
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.tools import molden
from pyscf.fci.direct_spin1 import _unpack_nelec
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi.lassi import root_make_rdm12s, roots_make_rdm12s
from mrh.my_pyscf.lassi.lassi import make_stdm12s, ham_2q, las_symm_tuple
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi import op_o1

# State list contains a couple of different 4-frag interactions
states  = {'charges': [[0,0,0,0],[1,1,-1,-1],[2,-1,-1,0],[1,0,0,-1],[1,1,-1,-1],[2,-1,-1,0],[1,0,0,-1]],
           'spins':   [[0,0,0,0],[1,1,-1,-1],[0,1,-1,0], [1,0,0,-1],[-1,-1,1,1],[0,-1,1,0], [-1,0,0,1]],
           'smults':  [[1,1,1,1],[2,2,2,2],  [1,2,2,1],  [2,1,1,2], [2,2,2,2],  [1,2,2,1],  [2,1,1,2]],
           'wfnsyms': [[0,0,0,0],]*7}
weights = [1.0,] + [0.0,]*6
nroots = 7
xyz = '''6        2.215130000      3.670330000      0.000000000
1        3.206320000      3.233120000      0.000000000
1        2.161870000      4.749620000      0.000000000
6        1.117440000      2.907720000      0.000000000
1        0.141960000      3.387820000      0.000000000
1       -0.964240000      1.208850000      0.000000000
6        1.117440000      1.475850000      0.000000000
1        2.087280000      0.983190000      0.000000000
6        0.003700000      0.711910000      0.000000000
6       -0.003700000     -0.711910000      0.000000000
6       -1.117440000     -1.475850000      0.000000000
1        0.964240000     -1.208850000      0.000000000
1       -2.087280000     -0.983190000      0.000000000
6       -1.117440000     -2.907720000      0.000000000
6       -2.215130000     -3.670330000      0.000000000
1       -0.141960000     -3.387820000      0.000000000
1       -2.161870000     -4.749620000      0.000000000
1       -3.206320000     -3.233120000      0.000000000'''

mol = gto.M (atom = xyz, basis='STO-3G', symmetry=False, verbose=0, output='/dev/null')
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (2,2,2,2),((1,1),(1,1),(1,1),(1,1)))
las.state_average_(weights=weights, **states)
a = list (range (18))
frags = [a[:5], a[5:9], a[9:13], a[13:18]]
las.mo_coeff = las.localize_init_guess (frags, mf.mo_coeff)
np.random.seed (1)
las.ci = las.get_init_guess_ci (las.mo_coeff, las.get_h2eff (las.mo_coeff))
for c in las.ci:
    for iroot in range (len (c)):
        c[iroot] = np.random.rand (*c[iroot].shape)
        c[iroot] /= linalg.norm (c[iroot])
nelec_frs = np.array (
    [[_unpack_nelec (fcibox._get_nelec (solver, nelecas)) for solver in fcibox.fcisolvers]
     for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
)
rand_mat = np.random.rand (7,7)
rand_mat += rand_mat.T
e, si = linalg.eigh (rand_mat)
si = lib.tag_array (si, rootsym=las_symm_tuple (las)[0])

def tearDownModule():
    global mol, mf, las
    mol.stdout.close ()
    del mol, mf, las

class KnownValues(unittest.TestCase):
    def test_stdm12s (self):
        d12_o0 = make_stdm12s (las, opt=0)
        d12_o1 = make_stdm12s (las, opt=1)
        for r in range (2):
            for i, j in product (range (nroots), repeat=2):
                with self.subTest (rank=r+1, bra=i, ket=j):
                    self.assertAlmostEqual (lib.fp (d12_o0[r][i,...,j]),
                        lib.fp (d12_o1[r][i,...,j]), 9)

    def test_ham_s2_ovlp (self):
        h1, h2 = ham_2q (las, las.mo_coeff, veff_c=None, h2eff_sub=None)[1:]
        lbls = ('ham','s2','ovlp')
        mats_o0 = op_o0.ham (las, h1, h2, las.ci, nelec_frs)#, orbsym=orbsym, wfnsym=wfnsym)
        fps_o0 = [lib.fp (mat) for mat in mats_o0]
        mats_o1 = op_o1.ham (las, h1, h2, las.ci, nelec_frs)#, orbsym=orbsym, wfnsym=wfnsym)
        for lbl, mat, fp in zip (lbls, mats_o1, fps_o0):
            with self.subTest(matrix=lbl):
                self.assertAlmostEqual (lib.fp (mat), fp, 9)

    def test_rdm12s (self):
        d12_o0 = op_o0.roots_make_rdm12s (las, las.ci, nelec_frs, si)#, orbsym=orbsym, wfnsym=wfnsym)
        d12_o1 = op_o1.roots_make_rdm12s (las, las.ci, nelec_frs, si)#, orbsym=orbsym, wfnsym=wfnsym)
        for r in range (2):
            for i in range (nroots):
                with self.subTest (rank=r+1, root=i):
                    self.assertAlmostEqual (lib.fp (d12_o0[r][i]),
                        lib.fp (d12_o1[r][i]), 9)
                with self.subTest ('single matrix constructor', opt=0, rank=r+1, root=i):
                    d12_o0_test = root_make_rdm12s (las, las.ci, si, state=i, soc=False,
                                                    break_symmetry=False, opt=0)[r]
                    self.assertAlmostEqual (lib.fp (d12_o0_test), lib.fp (d12_o0[r][i]), 9)
                with self.subTest ('single matrix constructor', opt=1, rank=r+1, root=i):
                    d12_o1_test = root_make_rdm12s (las, las.ci, si, state=i, soc=False,
                                                    break_symmetry=False, opt=1)[r]
                    self.assertAlmostEqual (lib.fp (d12_o1_test), lib.fp (d12_o0[r][i]), 9)

if __name__ == "__main__":
    print("Full Tests for LASSI o1 4-fragment intermediates")
    unittest.main()

