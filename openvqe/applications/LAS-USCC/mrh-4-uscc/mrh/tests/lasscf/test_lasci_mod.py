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
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list
from mrh.my_pyscf.mcscf.lasci_sync import LASCI_HessianOperator, LASCI_UnitaryGroupGenerators
topdir = os.path.abspath (os.path.join (__file__, '..'))

def build (mf, m1=0, m2=0, ir1=0, ir2=0, CASlist=None, active_first=False, calcname='c2h4n4', **kwargs):
    # I/O
    # --------------------------------------------------------------------------------------------------------------------
    mol = mf.mol
    my_kwargs = {'calcname':           calcname,
                 'doLASSCF':           True,
                 'debug_energy':       False,
                 'debug_reloc':        False,
                 'nelec_int_thresh':   1e-3,
                 'num_mf_stab_checks': 0,
                 'do_conv_molden':     False,
                 'orb_maxiter':        0}
    bath_tol = 1e-8
    my_kwargs.update (kwargs)
    
    # Set up the localized AO basis
    # --------------------------------------------------------------------------------------------------------------------
    myInts = localintegrals.localintegrals(mf, range(mol.nao_nr ()), 'meta_lowdin')
    
    # Build fragments from atom list
    # --------------------------------------------------------------------------------------------------------------------
    N2Ha = make_fragment_atom_list (myInts, list (range(3)), 'CASSCF(4,4)', name='N2Ha')#, active_orb_list = CASlist)
    C2H2 = make_fragment_atom_list (myInts, list (range(3,7)), 'RHF', name='C2H2')
    N2Hb = make_fragment_atom_list (myInts, list (range(7,10)), 'CASSCF(4,4)', name='N2Hb')#, active_orb_list = CASlist)
    N2Ha.bath_tol = C2H2.bath_tol = N2Hb.bath_tol = bath_tol
    N2Ha.target_S = abs (m1)
    N2Ha.target_MS = m1
    #N2Ha.mol_output = calcname + '_N2Ha.log'
    N2Hb.target_S = abs (m2)
    N2Hb.target_MS = m2
    #N2Hb.mol_output = calcname + '_N2Hb.log'
    if mol.symmetry:
        N2Ha.wfnsym = ir1
        N2Hb.wfnsym = ir2
    fraglist = [N2Ha, C2H2, N2Hb]
    
    # Load or generate active orbital guess 
    # --------------------------------------------------------------------------------------------------------------------
    c2h4n4_dmet = dmet (myInts, fraglist, **my_kwargs)
    c2h4n4_dmet.generate_frag_cas_guess (mf.mo_coeff, caslst=CASlist, force_imp=active_first, confine_guess=(not active_first))
    
    # Calculation
    # --------------------------------------------------------------------------------------------------------------------
    return c2h4n4_dmet

dr_nn = 2.0
mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
mol.verbose = lib.logger.DEBUG 
mol.output = '/dev/null'
mol.spin = 0 
mol.build ()
mf = scf.RHF (mol).run ()
dmet = build (mf, 1, -1, active_first=False)
dmet.conv_tol_grad = 1e-5
try:
    e_tot = dmet.doselfconsistent ()
except RuntimeError as e:
    e_tot = 0.0

#np.savetxt ('test_lasci_mo.dat', dmet.las.mo_coeff)
#np.savetxt ('test_lasci_ci0.dat', dmet.las.ci0)
#np.savetxt ('test_lasci_ci1.dat', dmet.las.ci[1])
dmet.las.mo_coeff = np.loadtxt (os.path.join (topdir, 'test_lasci_mo.dat'))
dmet.las.ci[0] = [np.loadtxt (os.path.join (topdir, 'test_lasci_ci0.dat'))]
dmet.las.ci[1] = [-np.loadtxt (os.path.join (topdir, 'test_lasci_ci1.dat')).T]
ugg = LASCI_UnitaryGroupGenerators (dmet.las, dmet.las.mo_coeff, dmet.las.ci)
h_op = LASCI_HessianOperator (dmet.las, ugg)
np.random.seed (0)
x = np.random.rand (ugg.nvar_tot)

def tearDownModule():
    global mol, mf, dmet, ugg, h_op, x
    mol.stdout.close ()
    dmet.lasci_log.close ()
    del mol, mf, dmet, ugg, h_op, x


class KnownValues(unittest.TestCase):
    def test_grad (self):
        gorb0, gci0, gx0 = dmet.las.get_grad (ugg=ugg)
        grad0 = np.append (gorb0, gci0)
        grad1 = h_op.get_grad ()
        gx1 = h_op.get_gx ()
        self.assertAlmostEqual (lib.fp (grad0), 0.0116625439865621, 8)
        self.assertAlmostEqual (lib.fp (grad1), 0.0116625439865621, 8)
        self.assertAlmostEqual (lib.fp (gx0), -0.0005604501808183955, 8)
        self.assertAlmostEqual (lib.fp (gx1), -0.0005604501808183955, 8)

    def test_hessian (self):
        hx = h_op._matvec (x)
        self.assertAlmostEqual (lib.fp (hx), 179.57890716580786, 7)

    def test_hc2 (self):
        xp = x.copy ()
        xp[:-16] = 0.0
        hx = h_op._matvec (xp)[-16:]
        self.assertAlmostEqual (lib.fp (hx), -0.15501937126181198, 7)

    def test_hcc (self):
        xp = x.copy ()
        xp[:-16] = 0.0
        hx = h_op._matvec (xp)[-32:-16]
        self.assertAlmostEqual (lib.fp (hx), -0.0012479602465573338, 7)

    def test_hco (self):
        xp = x.copy ()
        xp[-32:] = 0.0
        hx = h_op._matvec (xp)[-32:]
        self.assertAlmostEqual (lib.fp (hx), 0.24146683733262314, 7)

    def test_hoc (self):
        xp = x.copy ()
        xp[:-32] = 0.0
        hx = h_op._matvec (xp)[:-32]
        self.assertAlmostEqual (lib.fp (hx), -0.043190112417823626, 7)

    def test_hoo (self):
        xp = x.copy ()
        xp[-32:] = 0.0
        hx = h_op._matvec (xp)[:-32]
        self.assertAlmostEqual (lib.fp (hx), 182.07818989609675, 7)

    def test_prec (self):
        M_op = h_op.get_prec ()
        Mx = M_op._matvec (x)
        self.assertAlmostEqual (lib.fp (Mx), 2.940197014418852, 7)


if __name__ == "__main__":
    print("Full Tests for LASCI module functions")
    unittest.main()

