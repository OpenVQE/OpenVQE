#from rhf import monkeypatch_setup
#monkeypatch_teardown = monkeypatch_setup ()
import numpy as np
from scipy import linalg
from pyscf import scf, gto, lib, mcscf, df
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.df.grad import dfmcpdft as mcpdft_grad
from mrh.my_pyscf.grad import numeric as numeric_grad
from mrh.my_pyscf.fci import csf_solver

h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = lib.logger.DEBUG, output = 'h2co_sa2_tpbe66_631g_grad_debug.log')
mf_conv = scf.RHF (mol).run ()
mc_conv = mcpdft.CASSCF (mf_conv, 'tPBE', 6, 6, grids_level=6)
mc_conv.fcisolver = csf_solver (mol, smult=1)
mc_conv = mc_conv.state_average_([0.5,0.5])
mc_conv.conv_tol = 1e-10
mc_conv.kernel ()

mf = scf.RHF (mol).density_fit (auxbasis = df.aug_etb (mol)).run (mf_conv.make_rdm1 ())
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6, grids_level=6)
mc.fcisolver = csf_solver (mol, smult=1)
mc = mc.state_average_([0.5,0.5])
mc.conv_tol = 1e-10
mc.kernel (mc_conv.mo_coeff, mc_conv.ci)


numgrad_conv = numeric_grad.Gradients (mc_conv).run ()
print ("First state")
de_conv = mc_conv.nuc_grad_method ().kernel (state=0)
print ("Conventional ERI analytic:\n", de_conv)
de_conv_num = numgrad_conv.de_states[0]
#de_conv_num = np.array ([[-6.29268207e-02,  1.76592986e-04, -7.39246008e-06],
# [ 6.86265793e-02, -1.74918975e-04, -1.91981128e-06],
# [-2.70814282e-03, -1.13363633e-05, -5.35862378e-04],
# [-2.70418672e-03, -5.80733661e-06,  5.42638644e-04]])
print ("Conventional ERI numeric:\n", de_conv_num)
print ("Conventional ERI a-n:\n", de_conv-de_conv_num)
print ("Error norm =", linalg.norm (de_conv-de_conv_num))
de_df = mcpdft_grad.Gradients (mc).kernel (state=0)
print ("DF-ERI analytic:\n", de_df)
numgrad_df = numeric_grad.Gradients (mc).run ()
de_df_num = numgrad_df.de_states[0]
#de_df_num = np.array ([[-6.38140226e-02,  1.89626409e-04, -4.01996112e-06],
# [ 6.93076243e-02, -1.80151614e-04, -7.67915486e-06],
# [-2.58932334e-03, -4.25798608e-06, -7.22031258e-04],
# [-2.58924004e-03, -3.09638490e-06,  7.22522807e-04]])
print ("DF-ERI numeric:\n", de_df_num)
print ("DF-ERI a-n:\n", de_df-de_df_num)
print ("Error norm =", linalg.norm (de_df-de_df_num))
print ("Conventional-DF analytic disagreement:\n", de_df-de_conv)
print ("Disagreement norm =", linalg.norm (de_df-de_conv))

print ("Second state")
de_conv = mc_conv.nuc_grad_method ().kernel (state=1)
print ("Conventional ERI analytic:\n", de_conv)
de_conv_num = numgrad_conv.de_states[1]
#de_conv_num = np.array ([[-1.90536923e-01, -1.76554799e-04,  7.42319456e-06],
# [ 2.01381137e-01,  1.74965419e-04,  1.91157681e-06],
# [-5.56435594e-03,  1.14374404e-05, -4.47110500e-03],
# [-5.56835620e-03,  5.90008904e-06,  4.46427367e-03]])
print ("Conventional ERI numeric:\n", de_conv_num)
print ("Conventional ERI a-n:\n", de_conv-de_conv_num)
print ("Error norm =", linalg.norm (de_conv-de_conv_num))
de_df = mcpdft_grad.Gradients (mc).kernel (state=1)
print ("DF-ERI analytic:\n", de_df)
de_df_num = numgrad_df.de_states[1]
#de_df_num = np.array ([[-1.90633745e-01, -1.89581634e-04,  4.03479068e-06],
# [ 2.01922264e-01,  1.80202096e-04,  7.67101816e-06],
# [-5.80234283e-03,  4.35132507e-06, -4.37469615e-03],
# [-5.80242899e-03,  3.19417576e-06,  4.37420349e-03]])
print ("DF-ERI numeric:\n", de_df_num)
print ("DF-ERI a-n:\n", de_df-de_df_num)
print ("Error norm =", linalg.norm (de_df-de_df_num))
print ("Conventional-DF analytic disagreement:\n", de_df-de_conv)
print ("Disagreement norm =", linalg.norm (de_df-de_conv))

