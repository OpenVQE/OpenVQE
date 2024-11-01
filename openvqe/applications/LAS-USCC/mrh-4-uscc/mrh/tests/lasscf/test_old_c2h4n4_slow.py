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
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.lib import linalg_helper
# More conservative lindep than default
# I can't override the parameter alone because the
# function gets loaded from linalg_helper into
# lib as soon as any PySCF module is loaded
def my_safe_eigh (h, s, lindep=1e-10):
    return linalg_helper.safe_eigh (h, s, lindep)
lib.safe_eigh = my_safe_eigh
from c2h4n4_struct import structure as struct
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list
import sys, traceback, tracemalloc, warnings
#def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#
#    log = file if hasattr(file,'write') else sys.stderr
#    traceback.print_stack(file=log)
#    log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
#warnings.showwarning = warn_with_traceback


def run (mf, m1=0, m2=0, ir1=0, ir2=0, CASlist=None, active_first=False, calcname='c2h4n4', **kwargs):
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
    N2Hb.target_S = abs (m2)
    N2Hb.target_MS = m2
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
    e = c2h4n4_dmet.doselfconsistent ()
    c2h4n4_dmet.lasci_log.close ()
    return e

dr_nn = 3.0
mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
mol.verbose = lib.logger.DEBUG
mol.output = '/dev/null'
mol.build ()
mf = scf.RHF (mol).run ()
mf_df = mf.density_fit (auxbasis = df.aug_etb (mol)).run ()

mol_hs = mol.copy ()
mol_hs.spin = 8
mol_hs.verbose = lib.logger.DEBUG
mol_hs.output = '/dev/null'
mol_hs.build ()
mf_hs = scf.RHF (mol_hs).run ()
mf_hs_df = mf_hs.density_fit (auxbasis = df.aug_etb (mol_hs)).run ()

def tearDownModule():
    global mol, mf, mf_df, mol_hs, mf_hs, mf_hs_df
    mol.stdout.close ()
    mol_hs.stdout.close ()
    del mol, mf, mf_df, mol_hs, mf_hs, mf_hs_df


class KnownValues(unittest.TestCase):
    def test_dia (self):
        self.assertAlmostEqual (run (mf, 0, 0, calcname='dia'), -295.44779578419946, 8)

    def test_dia_df (self):
        self.assertAlmostEqual (run (mf_df, 0, 0, calcname='dia_df'), -295.44716017803967, 8)

    def test_ferro (self):
        self.assertAlmostEqual (run (mf_hs, 2, 2, active_first=True, calcname='ferro'), mf_hs.e_tot, 8)

    def test_ferro_df (self):
        self.assertAlmostEqual (run (mf_hs_df, 2, 2, active_first=True, calcname='ferro_df'), mf_hs_df.e_tot, 8)

    def test_af (self):
        self.assertAlmostEqual (run (mf_hs, 2, -2, active_first=True, calcname='af'), -295.44724798042466, 8)

    def test_af_df (self):
        self.assertAlmostEqual (run (mf_hs_df, 2, -2, active_first=True, calcname='af_df'), -295.4466638852035, 7)


if __name__ == "__main__":
    print("Full Tests for (old) LASSCF c2h4n4")
    unittest.main()

