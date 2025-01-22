import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, df
from pyscf.data.nist import BOHR
from mrh.my_pyscf.fci import csf_solver
from pyscf import mcpdft
from pyscf.df.grad import mspdft as mspdft_grad

# 1. Energy calculation
mol = gto.M (atom = 'Li 0 0 0; H 2.5 0 0', basis='STO-3G',
    output='LiH_cms2ftlda22_sto3g_df.log', verbose=lib.logger.DEBUG)
mf = scf.RHF (mol).density_fit (auxbasis = df.aug_etb (mol)).run ()
mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 2, 2, grids_level=9)
mc.fcisolver = csf_solver (mol, smult=1) # Important: singlets only
mc = mc.multi_state ([0.5,0.5], 'CMS')
mc.kernel ()
print ("CMS-PDFT energies are: {}".format (mc.e_states))
print ("Examine LiH_cms2ftlda22_sto3g_df.log for more information")

# 2. Gradient calculation
mc_grad = mspdft_grad.Gradients (mc)
de0 = mc_grad.kernel (state = 0)
de1 = mc_grad.kernel (state = 1)
print ("The ground state gradient array in Hartree/Angs is")
print (de0/BOHR)
print ("That of the first singlet excited state is")
print (de1/BOHR)

# 3. Geometry optimization
def print_rLiH (env):
    carts = env['mol'].atom_coords () * BOHR
    print ("R(Li-H):", linalg.norm (carts[1,:]-carts[0,:]))
print ("Optimizing the geometry of the first excited state:")
mc_opt = mc_grad.as_scanner (state=1).optimizer ()
mc_opt.callback = print_rLiH
mc_opt.kernel ()


