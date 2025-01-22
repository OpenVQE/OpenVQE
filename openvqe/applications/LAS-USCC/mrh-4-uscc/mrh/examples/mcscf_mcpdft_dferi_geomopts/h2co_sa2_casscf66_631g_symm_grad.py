#from rhf import monkeypatch_setup
#monkeypatch_teardown = monkeypatch_setup ()
import math
import numpy as np
from scipy import linalg
from pyscf import scf, gto, lib, mcscf, df
from pyscf.df.grad import sacasscf as casscf_grad
from mrh.my_pyscf.grad import numeric as numeric_grad
from mrh.my_pyscf.fci import csf_solver

def bond_length (carts, i, j):
    return linalg.norm (carts[i] - carts[j])
def bond_angle (carts, i, j, k):
    rij = carts[i] - carts[j]
    rkj = carts[k] - carts[j]
    res = max (min (1.0, np.dot (rij, rkj) / linalg.norm (rij) / linalg.norm (rkj)), -1.0)
    return math.acos (res) * 180 / math.pi
def out_of_plane_angle (carts, i, j, k, l):
    eji = carts[j] - carts[i]
    eki = carts[k] - carts[i]
    eli = carts[l] - carts[i]
    eji /= linalg.norm (eji)
    eki /= linalg.norm (eki)
    eli /= linalg.norm (eli)
    return -math.asin (np.dot (eji, (np.cross (eki, eli) / math.sin (bond_angle (carts, j, i, k) * math.pi / 180)))) * 180 / math.pi
def h2co_geom_analysis (carts):
    print ("rCO = {:.4f} Angstrom".format (bond_length (carts, 1, 0)))
    print ("rCH1 = {:.4f} Angstrom".format (bond_length (carts, 2, 0)))
    print ("rCH2 = {:.4f} Angstrom".format (bond_length (carts, 3, 0)))
    print ("tOCH1 = {:.2f} degrees".format (bond_angle (carts, 1, 0, 2)))
    print ("tOCH2 = {:.2f} degrees".format (bond_angle (carts, 1, 0, 3)))
    print ("tHCH = {:.2f} degrees".format (bond_angle (carts, 3, 0, 2)))
    print ("eta = {:.2f} degrees".format (out_of_plane_angle (carts, 0, 2, 3, 1)))
def my_call (env):
    carts = env['mol'].atom_coords () * lib.param.BOHR
    h2co_geom_analysis (carts)
conv_params = {
    'convergence_energy': 1e-6,  # Eh
    'convergence_grms': 5.0e-5,  # Eh/Bohr
    'convergence_gmax': 7.5e-5,  # Eh/Bohr
    'convergence_drms': 1.0e-4,  # Angstrom
    'convergence_dmax': 1.5e-4,  # Angstrom
}

lib.logger.TIMER_LEVEL = lib.logger.INFO
h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = True, verbose = lib.logger.INFO, output = 'h2co_sa2_casscf66_631g_symm_grad.log')
mf_conv = scf.RHF (mol).run ()
mc_conv = mcscf.CASSCF (mf_conv, 6, 6)
fcisolvers = [csf_solver (mol, smult=1) for i in (0,1)]
fcisolvers[0].wfnsym = 'A1'
fcisolvers[1].wfnsym = 'A2'
mc_conv = mcscf.addons.state_average_mix(mc_conv, fcisolvers, [0.5,0.5])
mc_conv.conv_tol = 1e-10
mc_conv.kernel ()

mf = scf.RHF (mol).density_fit (auxbasis = df.aug_etb (mol)).run ()
mc_df = mcscf.CASSCF (mf, 6, 6)
fcisolvers = [csf_solver (mol, smult=1) for i in (0,1)]
fcisolvers[0].wfnsym = 'A1'
fcisolvers[1].wfnsym = 'A2'
mc_df = mcscf.addons.state_average_mix(mc_df, fcisolvers, [0.5,0.5])
mc_df.conv_tol = 1e-10
mc_df.kernel ()

from mrh.my_pyscf.tools.molden import from_sa_mcscf
for i in (0, 1):
    from_sa_mcscf (mc_df, 'h2co_sa2_casscf66_631g_symm_state{}.molden'.format (i), state=i)

try:
    de_num = np.load ('h2co_sa2_casscf66_631g_grad_num.npy')
    de_conv_0_num, de_conv_1_num, de_df_0_num, de_df_1_num = list (de_num)
except OSError as e:
    conv_num = numeric_grad.Gradients (mc_conv).run ()
    de_conv_0_num, de_conv_1_num = list (conv_num.de_states)
    df_num = numeric_grad.Gradients (mc_df).run ()
    de_df_0_num, de_df_1_num = list (df_num.de_states)
    np.save ('h2co_sa2_casscf66_631g_grad_num.npy', np.append (conv_num.de_states, df_num.de_states, axis=0))

de_conv_0 = mc_conv.nuc_grad_method ().kernel (state = 0)
de_df_0 = casscf_grad.Gradients (mc_df).kernel (state = 0)
de_conv_1 = mc_conv.nuc_grad_method ().kernel (state = 1)
de_df_1 = casscf_grad.Gradients (mc_df).kernel (state = 1)

def printable_grad (arr):
    arr[np.abs (arr)<1e-10] = 0.0
    line_fmt = ' '.join (('{:13.9f}',)*3)
    return '\n'.join ((line_fmt.format (*row) for row in arr))

print ("Analytic gradient of state 0 with conventional ERIs:\n", printable_grad (de_conv_0)[1:])
print ("Numeric gradient of state 0 with conventional ERIs:\n", printable_grad (de_conv_0_num)[1:])
print ("Gradient error of state 0 with conventional ERIs = {:.6e}".format (linalg.norm (de_conv_0 - de_conv_0_num)))
print ("Analytic gradient of state 0 with DF ERIs:\n", printable_grad (de_df_0)[1:])
print ("Numeric gradient of state 0 with DF ERIs:\n", printable_grad (de_df_0_num)[1:])
print ("Gradient error of state 0 with DF ERIs = {:.6e}".format (linalg.norm (de_df_0 - de_df_0_num)))
print ("Gradient disagreements of state 0: Analytic = {:.6e} ; Numeric = {:.6e}".format (
    linalg.norm (de_df_0 - de_conv_0), linalg.norm (de_df_0_num - de_conv_0_num)))
print ("Analytic gradient of state 1 with conventional ERIs:\n", printable_grad (de_conv_1)[1:])
print ("Numeric gradient of state 1 with conventional ERIs:\n", printable_grad (de_conv_1_num)[1:])
print ("Gradient error of state 1 with conventional ERIs = {:.6e}".format (linalg.norm (de_conv_1 - de_conv_1_num)))
print ("Analytic gradient of state 1 with DF ERIs:\n", printable_grad (de_df_1)[1:])
print ("Numeric gradient of state 1 with DF ERIs:\n", printable_grad (de_df_1_num)[1:])
print ("Gradient error of state 1 with DF ERIs = {:.6e}".format (linalg.norm (de_df_1 - de_df_1_num)))
print ("Gradient disagreements of state 1: Analytic = {:.6e} ; Numeric = {:.6e}".format (
    linalg.norm (de_df_1 - de_conv_1), linalg.norm (de_df_1_num - de_conv_1_num)))

