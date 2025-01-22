import sys
from pyscf import gto, mcscf, scf, fci, dft
from mrh.my_pyscf.mcpdft import mcpdft, otfnal
from mrh.my_pyscf.gto.ano_contractions import contract_ano_basis

mol = gto.M (atom = 'Mn 0 0 0', basis='ano', spin = 5, symmetry = 'Dooh', output = 'Mnatom_pyscf.log', verbose = 4)
mol = contract_ano_basis (mol, 'vtzp')
mf = scf.RHF (mol).sfx2c1e ()
mf.kernel ()
hs = mcscf.CASSCF (mf, 9, (6, 1))
hs.conv_tol = 1e-10
hs.ah_conv_tol = 1e-12
hs.fix_spin_(ss=8.75)
amo_irrep = {'A1g': 2, 'E2gy': 1, 'E2gx': 1, 'E1gx': 1, 'E1gy': 1, 'A1u': 1, 'E1ux': 1, 'E1uy': 1}
imo_irrep = {'A1g': 3, 'A1u': 2, 'E1ux': 2, 'E1uy': 2}
mo = hs.sort_mo_by_irrep (amo_irrep, cas_irrep_ncore=imo_irrep)
hs.wfnsym = 'A1g'
ehs = hs.kernel (mo)[0]
if not hs.converged:
    hs = hs.newton ()
    ehs = hs.kernel (hs.mo, ci0=hs.ci)[0]
assert (hs.converged)

mf.mo_coeff = hs.mo_coeff
ls_vert = mcscf.CASCI (mf, 9, (7,0))
ls_vert.fcisolver = fci.solver (mf.mol, symm=True, singlet=False)
ls_vert.wfnsym = 'E1ux'
#ls_vert.fcisolver.conv_tol = 1e-12
els_vert = ls_vert.kernel ()[0]
assert (ls_vert.converged)

ls_rel = mcscf.CASSCF (mf, 9, (7,0))
ls_rel.fcisolver = fci.solver (mf.mol, symm=True, singlet=False)
ls_rel.conv_tol = 1e-10
ls_rel.ah_conv_tol = 1e-12
ls_rel.fcisolver.conv_tol = 1e-12
ls_rel.wfnsym = 'E1ux'
els_rel = ls_rel.kernel (mo)[0]
assert (ls_rel.converged)

print ("CASSCF high-spin energy: {:.8f}".format (ehs))
print ("CASSCF (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("CASSCF (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("CASSCF vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("CASSCF relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))

ks = dft.UKS (mol)
ks.xc = 'pbe'
ks.grids.level = 9
ot = otfnal.transfnal (ks.sfx2c1e ())

els_vert = mcpdft.kernel (ls_vert, ot)
els_rel = mcpdft.kernel (ls_rel, ot)
ehs = mcpdft.kernel (hs, ot)
print ("MC-PDFT (tPBE) high-spin energy: {:.8f}".format (ehs))
print ("MC-PDFT (tPBE) (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("MC-PDFT (tPBE) (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("MC-PDFT (tPBE) vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("MC-PDFT (tPBE) relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))

ks = dft.UKS (mol)
ks.xc = 'blyp'
ks.grids.level = 9
ot = otfnal.transfnal (ks.sfx2c1e ())

els_vert = mcpdft.kernel (ls_vert, ot)
els_rel = mcpdft.kernel (ls_rel, ot)
ehs = mcpdft.kernel (hs, ot)
print ("MC-PDFT (tBLYP) high-spin energy: {:.8f}".format (ehs))
print ("MC-PDFT (tBLYP) (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("MC-PDFT (tBLYP) (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("MC-PDFT (tBLYP) vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("MC-PDFT (tBLYP) relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))

