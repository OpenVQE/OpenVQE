import numpy as np
from pyscf import gto, scf, mcscf, lib, fci, df
from pyscf.fci.addons import fix_spin_
from pyscf.data.nist import BOHR
from mrh.my_pyscf import mcpdft
import unittest

print (BOHR)
def auto_energy (r=1.5):
    xyz = 'Li 0 0 0\nH {} 0 0'.format (r)
    mol_nosym = gto.M (atom = xyz, basis = 'sto3g', symmetry=True,
                     output = '/dev/null', verbose = 0)
    mol_sym = gto.M (atom = xyz, basis = 'sto3g', symmetry=True,
                     output = '/dev/null', verbose = 0)
    mf_nosym = scf.RHF (mol_nosym).run ()
    mf_sym = scf.RHF (mol_sym).run ()
    mc_nosym = mcscf.CASSCF (mf_nosym, 5, 2).run ()
    mc_sym = mcscf.CASSCF (mf_sym, 5, 2).run ()
    mcp_ss_nosym = mcpdft.CASSCF (mc_nosym, 'ftLDA,VWN3', 5, 2,
                                  grids_level=1).run (conv_tol=1e-12)
    mcp_ss_sym = mcpdft.CASSCF (mc_sym, 'ftLDA,VWN3', 5, 2,
                                grids_level=1)
    mcp_ss_sym.fcisolver = fci.solver (mol_sym).set (wfnsym='A1')
    mcp_ss_sym.run (conv_tol=1e-12)
    solver_A1 = fci.solver (mol_sym).set (wfnsym='A1', nroots=3)
    solver_E1x = fci.solver (mol_sym).set (wfnsym='E1x', nroots=1, spin=2)
    solver_E1y = fci.solver (mol_sym).set (wfnsym='E1y', nroots=1, spin=2)
    mcp_sa_2 = mcp_ss_sym.state_average_mix (
        [solver_A1,solver_E1x,solver_E1y], [1.0/5,]*5).run (conv_tol=1e-12)
    energies = np.asarray ([mcp_ss_sym.e_tot,]+mcp_sa_2.e_states)
    return energies, mcp_ss_sym, mcp_sa_2

def gradcomp (r=1.5):
    e_tab = np.zeros ((21,7))
    for p in range (20):
        delta = 1.0 / (2**p)
        ep = auto_energy (r+(delta/2))[0]
        em = auto_energy (r-(delta/2))[0]
        de = (ep-em)/delta
        e_tab[p,0] = delta
        e_tab[p,1:] = BOHR * de
    mc0, mc1 = auto_energy (r)[1:]
    mc0_grad = mc0.nuc_grad_method ()
    mc1_grad = mc1.nuc_grad_method ()
    e_tab[20,1] = -mc0_grad.kernel ()[0,0]
    for i in range (5):
        e_tab[20,i+2] = -mc1_grad.kernel (state=i)[0,0]
    return e_tab

de_tab = gradcomp(1.5)
err_tab = de_tab[:-1,:].copy ()
err_tab[:,1:] -= de_tab[-1,1:]
relerr_tab = err_tab[1:,:].copy ()
relerr_tab[:,1:] /= err_tab[:-1,1:]
fmt_str = ' '.join (['{:.8e}',]*7)
print ("de:")
for row in de_tab:
    print (fmt_str.format (*row))
print ("err:")
for row in err_tab:
    print (fmt_str.format (*row))
print ("relerr:")
for row in relerr_tab:
    print (fmt_str.format (*row))

