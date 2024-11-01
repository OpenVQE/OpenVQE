import numpy as np
import os, sys
from pyscf import gto, dft, scf, mcscf, lib, ci, mcpdft
from pyscf.tools import molden

# Usage: python H2_PES_hybrid_tPBE.py basis ncas
#
#
# Writes a table to H2_PES_hybrid_tPBE_cas2{ncas}_{basis}.dat with columns:
#
# r_HH E_FCI E_CASSCF(2,ncas) E_PBE E_PBE0 E_tPBE(2,ncas) E_tPBE0(2,ncas,b=0) E_tPBE0(2,ncas,b=a) E_tPBE0(2,ncas,b=a**2)
#
# where the three different versions of 'tPBE0' are
#
# E_tPBE0 = T + V_ne + J + a*E_xc^CASSCF + (1-a)*E_x^tPBE + (1-b)*E_c^tPBE
#
# with different cases for the relationship of 'b' to 'a'. See also the docstring of mcpdft.otfnal.make_hybrid_fnal
#
#
# Note that CISD == FCI because there are only 2 electrons.
# If you change 'hyb' below, 'PBE0' also changes to the corresponding KS-DFT hybrid functional.

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
kshfnal = mcpdft.hyb (fnal, hyb, 'diagram') # recovers KS-DFT-type hybrid
otfnal0 = transl_type + mcpdft.hyb (fnal, hyb, 'translation')
otfnal1 = transl_type + mcpdft.hyb (fnal, hyb, 'average')
otfnal2 = transl_type + mcpdft.hyb (fnal, hyb, 'lambda')
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

# 'Full CI' object
fci = ci.CISD (mf).run ().as_scanner ()

# UKS comparison (initial guess must break symmetry!)
pks = scf.UKS (mol)
pks.xc = fnal
pks.kernel (dm0)
pks = pks.as_scanner ()
hks = scf.UKS (mol)
hks.xc = kshfnal
hks.kernel (dm0)
hks = hks.as_scanner ()

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
mc0 = mcpdft.CASSCF (mf, otfnal0, ncas, nelecas, grids_level = 3).run ().as_scanner ()
mc1 = mcpdft.CASSCF (mf, otfnal1, ncas, nelecas, grids_level = 3).run ().as_scanner ()
mc2 = mcpdft.CASSCF (mf, otfnal2, ncas, nelecas, grids_level = 3).run ().as_scanner ()
molden.from_mcscf (mc0, moldenfile)
np.save (mofile, mc.mo_coeff)

# Do MCSCF scan forwards
table = np.zeros ((HHrange.size, 9))
table[:,0] = HHrange
for ix, HHdist in enumerate (HHrange):
    geom = 'H 0 0 0; H 0 0 {:.6f}'.format (HHdist)
    efci = fci (geom)
    epdft = mc (geom)
    mc0.mo_coeff = mc1.mo_coeff = mc2.mo_coeff = mc.mo_coeff
    molden.from_mo (mol, logfile[:-3] + '{:.2f}.molden'.format (HHdist), mc.mo_coeff)
    epdft0 = mc0 (geom)
    epdft1 = mc1 (geom) 
    epdft2 = mc2 (geom) 
    emcscf = mc.e_mcscf
    table[ix,1:] = [efci, emcscf, 0.0, 0.0, epdft, epdft0, epdft1, epdft2]

# Do UKS scan backwards (more orbital stability & smoother potential energy curves)
geom = 'H 0 0 0; H 0 0 {:.6f}'.format (HHrange[-1])
euks = pks (geom, dm0 = dm0)
euks = hks (geom, dm0 = dm0)
for ix, HHdist in enumerate (HHrange[::-1]):
    geom = 'H 0 0 0; H 0 0 {:.6f}'.format (HHdist)
    table[len (HHrange)-1-ix, 3] = pks (geom)
    table[len (HHrange)-1-ix, 4] = hks (geom)

# Print file
#table[:,2:] = table[:,2:] - table[:,1:2]
fmt_str = '{:.1f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'
with open (datfile, 'w') as f:
    for row in table: # Print the table from short distance to long distance
        line = fmt_str.format (*row)
        print (line[:-1])
        f.write (line)



