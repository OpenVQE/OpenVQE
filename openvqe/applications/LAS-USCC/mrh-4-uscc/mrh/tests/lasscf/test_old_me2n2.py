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
from me2n2_struct import structure as struct
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list

def run (mf, CASlist=None, **kwargs):
    # I/O
    # --------------------------------------------------------------------------------------------------------------------
    mol = mf.mol
    my_kwargs = {'calcname':           'me2n2_lasscf',
                 'doLASSCF':           True,
                 'debug_energy':       False,
                 'debug_reloc':        False,
                 'nelec_int_thresh':   1e-5,
                 'num_mf_stab_checks': 0}
    bath_tol = 1e-8
    my_kwargs.update (kwargs)
    
    # Set up the localized AO basis
    # --------------------------------------------------------------------------------------------------------------------
    myInts = localintegrals.localintegrals(mf, range(mol.nao_nr ()), 'meta_lowdin')
    
    # Build fragments from atom list
    # --------------------------------------------------------------------------------------------------------------------
    N2 = make_fragment_atom_list (myInts, list (range(2)), 'CASSCF(4,4)', name="N2")
    N2.target_S = N2.target_MS = mol.spin // 2
    Me1 = make_fragment_atom_list (myInts, list (range(2,6)), 'RHF', name='Me1')
    Me2 = make_fragment_atom_list (myInts, list (range(6,10)), 'RHF', name='Me2')
    N2.bath_tol = Me1.bath_tol = Me2.bath_tol = bath_tol
    fraglist = [N2, Me1, Me2] 
    
    # Generate active orbital guess 
    # --------------------------------------------------------------------------------------------------------------------
    me2n2_dmet = dmet (myInts, fraglist, **my_kwargs)
    me2n2_dmet.generate_frag_cas_guess (mf.mo_coeff, caslst=CASlist, force_imp=True, confine_guess=False)
    
    # Calculation
    # --------------------------------------------------------------------------------------------------------------------
    e = me2n2_dmet.doselfconsistent ()
    me2n2_dmet.lasci_log.close ()
    return e

r_nn = 3.0
mol = struct (3.0, '6-31g')
mol.verbose = lib.logger.DEBUG
mol.output = '/dev/null'
mol.build ()
mf = scf.RHF (mol).run ()
mc = mcscf.CASSCF (mf, 4, 4).run ()
mf_df = mf.density_fit (auxbasis = df.aug_etb (mol)).run ()
mc_df = mcscf.CASSCF (mf_df, 4, 4).run ()
mol_hs = mol.copy ()
mol_hs.spin = 4
mol_hs.build ()
mf_hs = scf.RHF (mol_hs).run ()
mf_hs_df = mf_hs.density_fit (auxbasis = df.aug_etb (mol_hs)).run ()

def tearDownModule():
    global mol, mf, mf_df, mc, mc_df, mol_hs, mf_hs, mf_hs_df
    mol.stdout.close ()
    mol_hs.stdout.close ()
    del mol, mf, mf_df, mc, mc_df, mol_hs, mf_hs, mf_hs_df


class KnownValues(unittest.TestCase):
    def test_lasscf (self):
        self.assertAlmostEqual (run (mf), mc.e_tot, 6)

    def test_lasscf_df (self):
        self.assertAlmostEqual (run (mf_df), mc_df.e_tot, 6)

    def test_lasscf_hs (self):
        self.assertAlmostEqual (run (mf_hs), mf_hs.e_tot, 8)

    def test_lasscf_hs_df (self):
        self.assertAlmostEqual (run (mf_hs_df), mf_hs_df.e_tot, 8)

if __name__ == "__main__":
    print("Full Tests for (old) LASSCF me2n2")
    unittest.main()

