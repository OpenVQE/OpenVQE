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
from c2h4n4_struct import structure as struct
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_dmet.fragments import make_fragment_atom_list, make_fragment_orb_list
import tracemalloc
tracemalloc.start ()

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
mol = struct (dr_nn, dr_nn, '6-31g', symmetry='Cs')
mol.verbose = lib.logger.DEBUG
mol.output = '/dev/null'
mol.build ()
mf = scf.RHF (mol).run ()
mf_df = mf.density_fit (auxbasis = df.aug_etb (mol)).run ()

def tearDownModule():
    global mol, mf, mf_df
    mol.stdout.close ()
    del mol, mf, mf_df 


class KnownValues(unittest.TestCase):

    def test_symm (self):
        self.assertAlmostEqual (run (mf, calcname='symm'), -295.44779578419946, 7)

    def test_symm_df (self):
        self.assertAlmostEqual (run (mf_df, calcname='symm_df'), -295.44716017803967, 7)
        

if __name__ == "__main__":
    print("Full Tests for (old) LASSCF c2h4n4 with symmetry")
    unittest.main()

