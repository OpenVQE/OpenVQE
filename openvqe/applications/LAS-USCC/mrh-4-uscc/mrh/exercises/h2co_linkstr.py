import numpy as np
from scipy import linalg
from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from pyscf.lib import logger

# This exercise is for extracting the single- and double-excitation parts of an MC-SCF FCI vector using gen_linkstr!

h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = logger.INFO, output = 'h2co_casscf66_631g_grad.log')
mf = scf.RHF (mol).run ()
mc = mcscf.CASSCF (mf, 6, 6)
mc.kernel ()

linkstr = mc.fcisolver.gen_linkstr (mc.ncas, mc.nelecas, False)
addr = np.zeros (mc.ci.shape, dtype=np.bool_) # Boolean index array for CI vector
ref = np.array ([0]) # HF determinant is usually addr = 0
addr[ref[0],ref[0]] = True
sing_addrs = []
for links, nelec, sp in zip (linkstr, mc.nelecas, ('up', 'down')):
    redun = ref
    sing = np.setdiff1d (links[ref,:,2], redun) # E_pq|ref> = |single> (unique only)
    redun = np.append (ref, sing)
    doub = np.setdiff1d (links[sing,:,2], redun) # E_pq|single> = |double> (unique only)
    print ("Reference, single excitation, and double excitation addresses for {}-spin electrons:".format (sp), ref, sing, doub)
    print ("Reference string for {}-spin:".format (sp), bin (cistring.addrs2str (mc.ncas, nelec, ref)[0]))
    print ("Single-excitation strings for {}-spin:".format (sp), [bin (x) for x in cistring.addrs2str (mc.ncas, nelec, sing)])
    print ("Double-excitation strings for {}-spin:".format (sp), [bin (x) for x in cistring.addrs2str (mc.ncas, nelec, doub)])
    if sp == 'up': # Same-spin single and double excitations
        addr[sing,0] = True
        addr[doub,0] = True
    elif sp == 'down':
        addr[0,sing] = True
        addr[0,doub] = True
    sing_addrs.append (sing)
addr[np.ix_(*sing_addrs)] = True # Combine the two spin singles to get ab-> doubles
cisd = mc.ci[addr]
print ("Norm of ref+singles+doubles part of FCI vector:", linalg.norm (cisd)) 

