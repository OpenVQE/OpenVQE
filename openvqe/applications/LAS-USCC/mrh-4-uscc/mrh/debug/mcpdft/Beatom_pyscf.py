from pyscf import gto, mcscf, scf, fci, dft
from mrh.my_pyscf.mcpdft import mcpdft, otfnal
from mrh.my_pyscf.fci import csf_solver
import numpy as np
from math import sqrt
from pyscf.fci.spin_op import spin_square0


mol_ls = gto.M (atom = 'Be 0 0 0', basis='cc-pvtz', spin = 0, symmetry=False)
mol_hs = gto.M (atom = 'Be 0 0 0', basis='cc-pvtz', spin = 2, symmetry=False)
mf = scf.RHF (mol_ls)
mf.kernel ()
ls = mcscf.CASSCF (mf, 4, 2)
ls.fcisolver = csf_solver (mf.mol, smult=1)
ls.conv_tol = 1e-10
els = ls.kernel ()[0]

mf = scf.RHF (mol_hs)
mf.kernel ()
hs_vert = mcscf.CASCI (mf, 4, 2)
hs_vert.fcisolver = csf_solver (mf.mol, smult=3)
ehs_vert = hs_vert.kernel ()[0]

hs_rel = mcscf.CASSCF (mf, 4, 2)
hs_rel.fcisolver = csf_solver (mf.mol, smult=3)
hs_rel.conv_tol = 1e-10
#hs_rel.ah_conv_tol = 1e-12
ehs_rel = hs_rel.kernel ()[0]

print ("CASSCF high-spin energy: {:.8f}".format (els))
print ("CASSCF (vertical) low-spin energy: {:.8f}".format (ehs_vert))
print ("CASSCF (relaxed) low-spin energy: {:.8f}".format (ehs_rel))
print ("CASSCF vertical excitation energy: {:.8f}".format (27.2114 * (ehs_vert - els)))
print ("CASSCF relaxed excitation energy: {:.8f}".format (27.2114 * (ehs_rel - els)))

ks = dft.UKS (mol_ls)
ks.xc = 'pbe'
ks.grids.level = 9
ot = otfnal.transfnal (ks)

els = mcpdft.kernel (ls, ot)[0]
ehs_vert = mcpdft.kernel (hs_vert, ot)[0]
ehs_rel = mcpdft.kernel (hs_rel, ot)[0]
print ("MC-PDFT (tPBE) high-spin energy: {:.8f}".format (els))
print ("MC-PDFT (tPBE) (vertical) low-spin energy: {:.8f}".format (ehs_vert))
print ("MC-PDFT (tPBE) (relaxed) low-spin energy: {:.8f}".format (ehs_rel))
print ("MC-PDFT (tPBE) vertical excitation energy: {:.8f}".format (27.2114 * (ehs_vert - els)))
print ("MC-PDFT (tPBE) relaxed excitation energy: {:.8f}".format (27.2114 * (ehs_rel - els)))

ks = dft.UKS (mol_ls)
ks.xc = 'blyp'
ks.grids.level = 9
ot = otfnal.transfnal (ks)

els = mcpdft.kernel (ls, ot)[0]
ehs_vert = mcpdft.kernel (hs_vert, ot)[0]
ehs_rel = mcpdft.kernel (hs_rel, ot)[0]
print ("MC-PDFT (tBLYP) high-spin energy: {:.8f}".format (els))
print ("MC-PDFT (tBLYP) (vertical) low-spin energy: {:.8f}".format (ehs_vert))
print ("MC-PDFT (tBLYP) (relaxed) low-spin energy: {:.8f}".format (ehs_rel))
print ("MC-PDFT (tBLYP) vertical excitation energy: {:.8f}".format (27.2114 * (ehs_vert - els)))
print ("MC-PDFT (tBLYP) relaxed excitation energy: {:.8f}".format (27.2114 * (ehs_rel - els)))

