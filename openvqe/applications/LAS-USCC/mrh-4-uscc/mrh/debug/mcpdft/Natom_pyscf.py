from pyscf import gto, mcscf, scf, fci, dft, lib
from mrh.my_pyscf.mcpdft import mcpdft, otfnal
from mrh.my_pyscf.fci import csf_solver

mol_hs = gto.M (atom = 'N 0 0 0', basis='cc-pvtz', spin = 3, symmetry='Dooh', verbose=lib.logger.DEBUG, output='Natom_hs.log')
mol_ls = gto.M (atom = 'N 0 0 0', basis='cc-pvtz', spin = 1, symmetry='Dooh', verbose=lib.logger.DEBUG, output='Natom_ls.log')
mf = scf.RHF (mol_hs)
mf.kernel ()
hs = mcscf.CASSCF (mf, 4, 5)
hs.fcisolver = csf_solver (mf.mol, smult=4)
hs.conv_tol = 1e-10
ehs = hs.kernel ()[0]

mf = scf.RHF (mol_ls)
mf.kernel ()
ls_vert = mcscf.CASCI (mf, 4, 5)
#ls_vert.fcisolver.conv_tol = 1e-12
ls_vert.fcisolver = csf_solver (mf.mol, smult=2)
ls_vert.conv_tol = 1e-10
els_vert = ls_vert.kernel (hs.mo_coeff)[0]

ls_rel = mcscf.CASSCF (mf, 4, 5)
ls_rel.fcisolver = fci.solver (mf.mol, singlet=False)
ls_rel.conv_tol = 1e-10
ls_rel.fcisolver = csf_solver (mf.mol, smult=2)
els_rel = ls_rel.kernel (hs.mo_coeff)[0]
if not ls_rel.converged:
    ls_rel = ls_rel.newton ()
    ls_rel.ci = None
    els_rel = ls_rel.kernel ()[0]

print ("CASSCF high-spin energy: {:.8f}".format (ehs))
print ("CASSCF (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("CASSCF (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("CASSCF vertical excitation energy: {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("CASSCF relaxed excitation energy: {:.8f}".format (27.2114 * (els_rel - ehs)))

ks = dft.UKS (mol_ls)
ks.xc = 'pbe'
ks.grids.level = 9
ot = otfnal.transfnal (ks)

ehs = mcpdft.kernel (hs, ot)[0]
els_vert = mcpdft.kernel (ls_vert, ot)[0]
els_rel = mcpdft.kernel (ls_rel, ot)[0]
print ("MC-PDFT (tPBE) high-spin energy: {:.8f}".format (ehs))
print ("MC-PDFT (tPBE) (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("MC-PDFT (tPBE) (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("MC-PDFT (tPBE) vertical excitation energy: {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("MC-PDFT (tPBE) relaxed excitation energy: {:.8f}".format (27.2114 * (els_rel - ehs)))

ks = dft.UKS (mol_ls)
ks.xc = 'blyp'
ks.grids.level = 9
ot = otfnal.transfnal (ks)

ehs = mcpdft.kernel (hs, ot)[0]
els_vert = mcpdft.kernel (ls_vert, ot)[0]
els_rel = mcpdft.kernel (ls_rel, ot)[0]
print ("MC-PDFT (tBLYP) high-spin energy: {:.8f}".format (ehs))
print ("MC-PDFT (tBLYP) (vertical) low-spin energy: {:.8f}".format (els_vert))
print ("MC-PDFT (tBLYP) (relaxed) low-spin energy: {:.8f}".format (els_rel))
print ("MC-PDFT (tBLYP) vertical excitation energy: {:.8f}".format (27.2114 * (els_vert - ehs)))
print ("MC-PDFT (tBLYP) relaxed excitation energy: {:.8f}".format (27.2114 * (els_rel - ehs)))

