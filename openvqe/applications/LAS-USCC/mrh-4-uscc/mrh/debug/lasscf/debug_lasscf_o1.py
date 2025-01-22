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
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.tools import molden
from c2h4n4_struct import structure as struct
from mrh.my_pyscf.mcscf.lasscf_sync_o1 import LASSCF

dr_nn = 2.0
mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
mol.verbose = lib.logger.DEBUG 
mol.output = '/dev/null'
mol.spin = 0 
mol.build ()
mf = scf.RHF (mol).run ()
las = LASSCF (mf, (4,4), ((3,1),(1,3)), spin_sub=(3,3))
las._debug_full_pspace = True
las._debug_o0 = False
las.max_cycle_macro = 1
las.kernel ()
las.mo_coeff = np.loadtxt ('test_lasci_mo.dat')
las.ci = [[np.loadtxt ('test_lasci_ci0.dat')], [-np.loadtxt ('test_lasci_ci1.dat').T]]
ugg = las.get_ugg ()
h_op = las.get_hop (ugg=ugg)
nmo, ncore, nocc = h_op.nmo, h_op.ncore, h_op.nocc
np.random.seed (0)
x = np.random.rand (ugg.nvar_tot)
offs_ci1 = ugg.nvar_orb
offs_ci2 = offs_ci1 + np.squeeze (ugg.ncsf_sub)[0]
xorb, xci = ugg.unpack (x)
xci0 = [[np.zeros (16),], [np.zeros (16),]]


def tearDownModule():
    global mol, mf, las, ugg, h_op, x
    mol.stdout.close ()
    del mol, mf, las, ugg, h_op, x

sectors = ["ac","aa","vc","va","ci1","ci2"]
def itsec (vorb, vci):
    yield vorb[ncore:nocc,:ncore]
    yield vorb[ncore:nocc,ncore:nocc]
    yield vorb[nocc:,:ncore]
    yield vorb[nocc:,ncore:nocc]
    yield vci[0][0]
    yield vci[1][0]

class KnownValues(unittest.TestCase):
    def test_grad (self):
        gorb0, gci0, gx0 = las.get_grad (ugg=ugg)
        grad0 = np.append (gorb0, gci0)
        grad1 = h_op.get_grad ()
        gx1 = h_op.get_gx () # "gx" is not even defined in this context
        self.assertAlmostEqual (lib.fp (grad0), -0.1547273632764783, 9)
        self.assertAlmostEqual (lib.fp (grad1), -0.1547273632764783, 9)
        self.assertAlmostEqual (lib.fp (gx0), 0.0, 9)
        self.assertAlmostEqual (lib.fp (gx1), 0.0, 9) 

    def test_hessian (self):
        hx = h_op._matvec (x)
        self.assertAlmostEqual (lib.fp (hx), 179.117392525215, 9)

    def test_hc2 (self):
        xp = x.copy ()
        xp[:offs_ci2] = 0.0
        hx = h_op._matvec (xp)[offs_ci2:]
        self.assertAlmostEqual (lib.fp (hx), 1.0607759066755826, 9)

    def test_hcc (self):
        xp = x.copy ()
        xp[:offs_ci2] = 0.0
        hx = h_op._matvec (xp)[offs_ci1:offs_ci2]
        self.assertAlmostEqual (lib.fp (hx), 0.00014830104777428284, 9)

    def test_hco (self):
        xp = x.copy ()
        xp[offs_ci1:] = 0.0
        hx = h_op._matvec (xp)[offs_ci1:]
        self.assertAlmostEqual (lib.fp (hx), -0.6543458685319448, 9)

    def test_hoc (self):
        xp = x.copy ()
        xp[:offs_ci1] = 0.0
        hx = h_op._matvec (xp)[:offs_ci1]
        self.assertAlmostEqual (lib.fp (hx), 0.21204376122818072, 9)

    def test_hoo (self):
        xp = x.copy ()
        xp[offs_ci1:] = 0.0
        hx = h_op._matvec (xp)[:offs_ci1]
        self.assertAlmostEqual (lib.fp (hx), 178.41916344898377, 9)

    def test_h_xcv (self):
        xorb0 = xorb.copy ()
        xorb0[ncore:nocc,:] = xorb0[:,ncore:nocc] = 0.0
        xp = ugg.pack (xorb0, xci0)
        hxorb, hxci = ugg.unpack (h_op._matvec (xp))
        refs = [-0.06544076804895266,0.20492877852377767,-76.54148711825971,0.001430987773399131,-0.023324408608589215,0.011004024379865151]
        for sec, test, ref in zip (sectors, itsec (hxorb,hxci), refs):
            with self.subTest (sector=sec):
                self.assertAlmostEqual (lib.fp (test), ref, 9)

    def test_h_xaa (self):
        xorb0 = np.zeros_like (xorb)
        xorb0[ncore:nocc,ncore:nocc] = xorb[ncore:nocc,ncore:nocc] 
        xp = ugg.pack (xorb0, xci0)
        hxorb, hxci = ugg.unpack (h_op._matvec (xp))
        refs = [0.23685465500076336,1.036052308111469,0.13584422544065847,0.09072092665879812,0.0015574081837203407,-0.0028940907410843902]
        for sec, test, ref in zip (sectors, itsec (hxorb,hxci), refs):
            with self.subTest (sector=sec):
                self.assertAlmostEqual (lib.fp (test), ref, 9)

    def test_h_xua (self):
        xorb0 = np.zeros_like (xorb)
        xorb0[ncore:nocc,:] = xorb[ncore:nocc,:] 
        xorb0[:,ncore:nocc] = xorb[:,ncore:nocc] 
        xp = ugg.pack (xorb0, xci0)
        hxorb, hxci = ugg.unpack (h_op._matvec (xp))
        refs = [32.59486557852766,0.8505552949164717,-1.5790308710536478,-2.7650296786870276,-0.09445253579668796,0.12861532657269764]
        for sec, test, ref in zip (sectors, itsec (hxorb,hxci), refs):
            with self.subTest (sector=sec):
                self.assertAlmostEqual (lib.fp (test), ref, 9)

    def test_prec (self):
        M_op = h_op.get_prec ()
        Mx = M_op._matvec (x)
        self.assertAlmostEqual (lib.fp (Mx), 8358.536413578968, 7)


if __name__ == "__main__":
    print("Full Tests for LASSCF Newton-CG module functions")
    unittest.main()

