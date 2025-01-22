from pyscf import gto, scf, mcscf

print ("Computing Fe(II) total energy...")
from mrh.my_pyscf.gto import ANO_RCC_VDZP
#                            ^          ^ This is a dict, not a str.
# Do not use quotation marks when specifying it:
mol0 = gto.M (atom='Fe 0 0 0', basis=ANO_RCC_VDZP, charge=2, verbose=0)
mf0 = scf.RHF (mol0).sfx2c1e ().run ()
#                    ^     ^ apply scalar relativistic correction
# This is similar to the DKH correction applied by default in OpenMolcas
# but not identical. You can specify X2C in OpenMolcas for something more
# directly comparable. The ANO-RCC bases are designed to be used with
# scalar relativistic corrections; they are not used without them.
mc0 = mcscf.CASSCF (mf0, 5, 6).run ()
print ('E[RHF] = {:.9f} ; E[CAS(6,5)] = {:.9f}'.format (mf0.e_tot, mc0.e_tot))

# An older, more complicated implementation. PySCF itself has the full ANO
# basis but no convenient way to specify the conventional MB and V*ZP 
# subsets of it. 
print ("\nSanity check...")
from mrh.my_pyscf.gto import contract_ano_basis
mol1 = gto.M (atom='Fe 0 0 0', basis='ano', charge=2, verbose=0)
mol1 = contract_ano_basis (mol1, 'VDZP')
mf1 = scf.RHF (mol1).sfx2c1e ().run (mf0.make_rdm1 ())
mc1 = mcscf.CASSCF (mf1, 5, 6).run (mc0.mo_coeff)
print ('E[RHF] = {:.9f} ; E[CAS(6,5)] = {:.9f}'.format (mf1.e_tot, mc1.e_tot))

# Sadly PySCF is missing some data for the ANO basis starting at francium
print ("\nTrying to do a radium(II) ion results in...")
mol2 = gto.M (atom='Ra 0 0 0', basis=ANO_RCC_VDZP, charge=2, verbose=0)
try:
    mf2 = scf.RHF (mol2).sfx2c1e ().run ()
except Exception as e:
    print (type (e), str (e))

