import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf
from pyscf.tools import molden
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
import unittest

lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = gto.M (atom = 'N 0 0 0; N 1.2 0 0', basis='cc-pvdz', verbose=lib.logger.INFO, output='debug_scf.log')
mf = scf.RHF (mol).run ()
mc_ecas = mcpdft.CASSCF (mf, 'tLDA,VWN3', 6, 6, ci_min='ecas').run ()
molden.from_mcscf (mc_ecas, 'debug_scf.molden')
casci_epdft = mcpdft.CIMCPDFT (mf, 'tLDA,VWN3', 6, 6).set (mo_coeff=mc_ecas.mo_coeff).run ()
mc_epdft = mcpdft.CIMCPDFT_SCF (mf, 'tLDA,VWN3', 6, 6).set (mo_coeff=mc_ecas.mo_coeff, ci=casci_epdft.ci).run ()

#rnn = np.arange (0.8, 4.81, 0.125)
#epdft = np.zeros_like (rnn)
#epdft_var = np.zeros_like (rnn)
#ci0_casscf = np.zeros_like (rnn)
#ci0_pdft_var = np.zeros_like (rnn)
#tab = np.vstack ((rnn, epdft, epdft_var, ci0_casscf, ci0_pdft_var)).T
#
#for row in tab:
#    mol = gto.M (atom = 'N 0 0 0; N {} 0 0'.format (row[0]), basis='cc-pvdz', verbose=lib.logger.INFO, output='debug_scf.log')
#    mf = scf.RHF (mol).run ()
#    mc_ecas = mcpdft.CASSCF (mf, 'tLDA,VWN3', 6, 6, ci_min='ecas').run ()
#    molden.from_mcscf (mc_ecas, 'debug_scf.molden')
#    mc_epdft = mcpdft.CASCI (mf, 'tLDA,VWN3', 6, 6, ci_min='epdft').set (mo_coeff=mc_ecas.mo_coeff).run ()
#    row[1] = mc_ecas.e_tot
#    row[2] = mc_epdft.e_tot
#    row[3] = mc_ecas.ci[0,0]**2
#    row[4] = mc_epdft.ci[0,0]**2
#    print (" {:>5.3f} {:>15.10f} {:>15.10f} {:>15.10f} {:>15.10f}".format (*row))
#
#print ("{:>6s} {:>15s} {:>15s} {:>19s} {:>21s}".format ("dr", "E(CASSCF)", "E(PDFT-SCF)", "|ci[0,0]|^2(CASSCF)", "|ci[0,0]|^2(PDFT-SCF)"))
#for row in tab:
#    print (" {:>5.3f} {:>15.10f} {:>15.10f} {:>19.10f} {:>21.10f}".format (*row))


