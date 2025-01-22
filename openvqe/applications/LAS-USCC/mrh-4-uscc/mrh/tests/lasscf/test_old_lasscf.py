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
from c2h4n4_struct import structure as struct
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list

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
                 'do_conv_molden':     False}
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

dr_nn = 3.0
mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
mol.verbose = lib.logger.DEBUG 
mol.output = '/dev/null'
mol.spin = 8
mol.build ()
mf = scf.RHF (mol).run ()
dmet = build (mf, 1, -1, active_first=True)
dmet.conv_tol_grad = 1e-6
e_tot = dmet.doselfconsistent ()

def tearDownModule():
    global mol, mf, dmet
    mol.stdout.close ()
    dmet.lasci_log.close ()
    del mol, mf, dmet


class KnownValues(unittest.TestCase):
    def test_energies (self):
        self.assertAlmostEqual (e_tot, -295.44765266564866, 9)
        self.assertAlmostEqual (dmet.las.e_tot, e_tot, 9)
        self.assertAlmostEqual (dmet.fragments[0].E_imp, e_tot, 7)
        self.assertAlmostEqual (dmet.fragments[1].E_imp, e_tot, 7)
        self.assertAlmostEqual (dmet.fragments[2].E_imp, e_tot, 7)

    def test_active_orbitals (self):
        ncore, ncas = dmet.las.ncore, dmet.las.ncas
        mo_las = dmet.las.mo_coeff[:,ncore:][:,:ncas]
        mo_frag = np.append (dmet.fragments[0].loc2amo, dmet.fragments[2].loc2amo, axis=1)
        ovlp = mo_las.conj ().T @ dmet.ints.ao_ovlp @ dmet.ints.ao2loc @ mo_frag
        self.assertAlmostEqual (linalg.norm (ovlp - np.eye (8)), 0, 8)

    def test_ci (self):
        ci0 = dmet.las.ci
        ci1 = [dmet.fragments[0].ci_as, dmet.fragments[2].ci_as]
        for c0, c1 in zip (ci0, ci1):
            self.assertAlmostEqual (linalg.norm (c0-c1), 0, 8)

    def test_1rdm (self):
        sloc = dmet.ints.ao_ovlp @ dmet.ints.ao2loc
        dm1 = sloc.conj ().T @ dmet.las.make_rdm1 () @ sloc
        for f in (dmet.fragments[0], dmet.fragments[2]):
            dm1_test = f.imp2loc @ dm1 @ f.loc2imp
            self.assertAlmostEqual (linalg.norm (dm1_test - f.get_oneRDM_imp ()), 0, 8)

if __name__ == "__main__":
    print("Full Tests for (old) LASSCF module agreement")
    unittest.main()

