#from rhf import monkeypatch_setup
#monkeypatch_teardown = monkeypatch_setup ()
from scipy import linalg
from pyscf import scf, gto, lib, mcscf, df
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.df.grad import dfmcpdft as mcpdft_grad
from mrh.my_pyscf.grad import numeric as numeric_grad

h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = lib.logger.INFO, output = 'h2co_tpbe66_631g_grad.log')
mf_conv = scf.RHF (mol).run ()
mc_conv = mcpdft.CASSCF (mf_conv, 'tPBE', 6, 6, grids_level=3)
mc_conv.conv_tol = 1e-10
mc_conv.kernel ()

mf = scf.RHF (mol).density_fit (auxbasis = df.aug_etb (mol)).run (mf_conv.make_rdm1 ())
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6, grids_level=6)
mc.conv_tol = 1e-10
mc.kernel (mc_conv.mo_coeff, mc_conv.ci)

de_conv = mc_conv.nuc_grad_method ().kernel ()
print ("Conventional ERI analytic:\n", de_conv)
de_conv_num = numeric_grad.Gradients (mc_conv).kernel ()
print ("Conventional ERI numeric:\n", de_conv_num)
print ("Conventional ERI a-n:\n", de_conv-de_conv_num)
print ("Error norm =", linalg.norm (de_conv-de_conv_num))
de_df = mcpdft_grad.Gradients (mc).kernel ()
print ("DF-ERI analytic:\n", de_df)
de_df_num = numeric_grad.Gradients (mc).kernel ()
print ("DF-ERI numeric:\n", de_df_num)
print ("DF-ERI a-n:\n", de_df-de_df_num)
print ("Error norm =", linalg.norm (de_df-de_df_num))
print ("Conventional-DF analytic disagreement:\n", de_df-de_conv)
print ("Disagreement norm =", linalg.norm (de_df-de_conv))

