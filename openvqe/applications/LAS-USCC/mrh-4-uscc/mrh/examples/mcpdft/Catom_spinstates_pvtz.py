import sys
sys.path.append ('../../..')
from pyscf import gto, mcscf, scf, fci, dft, mcpdft

''' C atom triplet-singlet gap reported in JCTC 2014, 10, 3669
    CASSCF(4,4):    1.6 eV
    tPBE:           1.1 eV
    tBLYP:          1.0 eV
    'Vertical' means triplet orbitals for singlet state
    'Relaxed' means optimized orbitals for both states
'''

mol = gto.M (atom = 'C 0 0 0', basis='cc-pvtz', spin = 2, symmetry='D2h')
mf = scf.RHF (mol)
mf.kernel ()
hs = mcscf.CASSCF (mf, 4, (3, 1))
ehs = hs.kernel ()[0]

mf.mo_coeff = hs.mo_coeff
ls_vert = mcscf.CASCI (mf, 4, (2,2))
ls_vert.fcisolver = fci.solver (mf.mol, singlet=True)
els_vert = ls_vert.kernel ()[0]

ls_rel = mcscf.CASSCF (mf, 4, (2,2))
ls_rel.fcisolver = fci.solver (mf.mol, singlet=True)
els_rel = ls_rel.kernel ()[0]

print ("CASSCF high-spin energy: {:.8f}".format (ehs))
print ("CASSCF (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("CASSCF (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("CASSCF vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("CASSCF relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))

ls_vert = mcpdft.CASCI (ls_vert, 'tPBE', 4, (2, 2), grids_level=9)
ls_rel = mcpdft.CASCI (ls_rel, 'tPBE', 4, (2, 2), grids_level=9)
hs = mcpdft.CASCI (hs, 'tPBE', 4, (3, 1), grids_level=9)

els_vert = ls_vert.compute_pdft_energy_()[0]
els_rel = ls_rel.compute_pdft_energy_()[0]
ehs = hs.compute_pdft_energy_()[0]
print ("MC-PDFT (tPBE) high-spin energy: {:.8f}".format (ehs))
print ("MC-PDFT (tPBE) (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("MC-PDFT (tPBE) (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("MC-PDFT (tPBE) vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("MC-PDFT (tPBE) relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))

els_vert = ls_vert.compute_pdft_energy_(otxc='tBLYP')[0]
els_rel = ls_rel.compute_pdft_energy_(otxc='tBLYP')[0]
ehs = hs.compute_pdft_energy_(otxc='tBLYP')[0]
print ("MC-PDFT (tBLYP) high-spin energy: {:.8f}".format (ehs))
print ("MC-PDFT (tBLYP) (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("MC-PDFT (tBLYP) (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("MC-PDFT (tBLYP) vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("MC-PDFT (tBLYP) relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))

