import numpy as np
import os, sys
from pyscf import gto, dft, scf, mcscf, lib, ci
from pyscf.tools import molden
from mrh.my_pyscf import mcpdft

# Usage: python H2_PES_decomp_tPBE.py basis ncas
#
#
# Writes a table to H2_PES_decomp_tPBE_cas2{ncas}_{basis}.dat with columns:
#
# r_HH E_PDFT E_MCSCF E_nuc E_core E_Coulomb E_OTx E_OTc E_WFNxc
#
# Where E_WFNxc is the MCSCF exchange-correlation energy and the on-top
# exchange and correlation contributions are separated

ncas = int (sys.argv[2])
nelecas = 2
basis = sys.argv[1] 
fnal = 'PBE'
hyb = 0.25
transl_type = 't'
HHrange = np.arange (0.5, 4.1, 0.1)
grids_level = 3
symmetry = False
verbose = lib.logger.INFO
gsbasis = None if len (sys.argv) < 4 else sys.argv[3]

logfile = os.path.basename (__file__)[:-3] + '_cas2{}_{}.log'.format (ncas, basis)
datfile = os.path.basename (__file__)[:-3] + '_cas2{}_{}.dat'.format (ncas, basis)
mofile = os.path.basename (__file__)[:-3] + '_cas2{}_{}.mo.npy'.format (ncas, basis)
if gsbasis:
    gsmofile = os.path.basename (__file__)[:-3] + '_cas2{}_{}.mo.npy'.format (ncas, gsbasis)
moldenfile = os.path.basename (__file__)[:-3] + '_cas2{}_{}.molden'.format (ncas, basis)
mol = gto.M (atom = 'H 0 0 0; H 0 0 {:.6f}'.format (HHrange[0]), basis = basis, symmetry = symmetry, verbose = verbose, output = logfile) 
if gsbasis:
    gsmol = gto.M (atom = 'H 0 0 0; H 0 0 {:.6f}'.format (HHrange[0]), basis = gsbasis, symmetry = symmetry, verbose = 0)

# Build spin-broken guess density matrix
# There is probably a PySCF function to do this better I just felt like doing it like this
nao = mol.nao_nr ()
if nao == 2:
    dma = np.array ([[1.0,0],[0,0]])
    dmb = np.array ([[0,0],[0,1.0]])
else:
    idx_h0_1s = mol.search_ao_label ('0 H 1s')[0]
    idx_h1_1s = mol.search_ao_label ('1 H 1s')[0]
    idx_h0_2s = mol.search_ao_label ('0 H 2s')[0]
    idx_h1_2s = mol.search_ao_label ('1 H 2s')[0]
    dma = np.zeros ((nao, nao))
    dmb = np.zeros ((nao, nao))
    dma[idx_h0_1s,idx_h0_1s] = dmb[idx_h1_1s,idx_h1_1s] = dma[idx_h0_2s,idx_h0_2s] = dmb[idx_h1_2s,idx_h1_2s] = 1
dm0 = [dma, dmb]

# Restricted mean-field base of MC-SCF objects
mf = scf.RHF (mol).run ()

# MC-PDFT objects
if gsbasis:
    gsmo = np.load (gsmofile)
    mc = mcpdft.CASSCF (mf, transl_type + fnal, ncas, nelecas, grids_level = 3)
    mo = mcscf.project_init_guess (mc, gsmo, prev_mol=gsmol)
    molden.from_mo (mol, 'check_projection.molden', mo)
    mc.kernel (mo)
    mc = mc.as_scanner ()
else:  
    mc = mcpdft.CASSCF (mf, transl_type + fnal, ncas, nelecas, grids_level = 3).run ().as_scanner ()
np.save (mofile, mc.mo_coeff)

# Do MCSCF scan forwards
table = np.zeros ((HHrange.size, 9))
table[:,0] = HHrange
for ix, HHdist in enumerate (HHrange):
    geom = 'H 0 0 0; H 0 0 {:.6f}'.format (HHdist)
    table[ix,1] = mc (geom)
    table[ix,2] = mc.e_mcscf
    table[ix,3:] = list (mc.get_energy_decomposition ())

# Print file
fmt_str = '{:.1f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'
with open (datfile, 'w') as f:
    for row in table: # Print the table from short distance to long distance
        line = fmt_str.format (*row)
        print (line[:-1])
        f.write (line)



