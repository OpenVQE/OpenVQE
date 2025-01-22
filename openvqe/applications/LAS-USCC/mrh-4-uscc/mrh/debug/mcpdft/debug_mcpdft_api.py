from pyscf import gto, scf, mcscf, lib
from pyscf.tools import molden
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.tools.molden import from_sa_mcscf

mol = gto.M (atom = 'Li 0 0 0\nH 1.5 0 0', basis = 'sto3g', symmetry=False,
             output = 'LiH.log', verbose = lib.logger.INFO)
mf = scf.RHF (mol).run ()
mc0 = mcscf.CASSCF (mf, 5, 2).run ()

### SS-energies ###

mc = []
mc.append (mcpdft.CASSCF (mol, 'tPBE', 5, 2).set (mo_coeff=mf.mo_coeff).run ())
mc.append (mcpdft.CASSCF (mf, 'tPBE', 5, 2).run ())
mc.append (mcpdft.CASSCF (mc0, 'tPBE', 5, 2).run (conv_tol=1e-12))
mc.append (mcpdft.CASCI (mol, 'tPBE', 5, 2).set (mo_coeff=mc[-1].mo_coeff).run ())
mc.append (mcpdft.CASCI (mf, 'tPBE', 5, 2).set (mo_coeff=mc[-1].mo_coeff).run ())
mc.append (mcpdft.CASCI (mc0, 'tPBE', 5, 2).run ())
for m in mc:
    print ('{:.9f} {}'.format (m.e_tot, m.converged))

### Gradients ###

mc_grad = []
for i, m in enumerate (mc):
    try:
        m_grad = m.nuc_grad_method ()
        de = m_grad.kernel ()
        mc_grad.append ('{}\n{}'.format (m_grad.converged, de))
    except NotImplementedError as e:
        mc_grad.append (str (e))
for m in mc_grad:
    print (m)

#  CASSCF spectrum:
#  S0 A1 E = -7.88201653164191
#  T1 A1 E = -7.76028550406524
#  S1 A1 E = -7.74394083816439
#  T2 E1x (B1) E = -7.71247967093247
#  T3 E1y (B2) E = -7.71247967093247
#  S2 E1x (B1) E = -7.69084161562994
#  S3 E1y (B2) E = -7.69084161562994

nroots = 5
mc = mc[2]
mc_sa = mc.state_average ([1.0/nroots,]*nroots).run (conv_tol=1e-12)
for state in range (nroots):
    print (mc_sa.e_states[state])
    from_sa_mcscf (mc_sa, 'LiH.{}.molden'.format (state),
                   state=state, cas_natorb=True)

#print (('       ref = np.array ([{},{},{},\n'
#        '                        {},{}])').format (*mc_sa.e_states))

import numpy as np
ref = mc.get_energy_decomposition ()
ref_nuc, ref_wfn = ref[0], np.array (ref[1:])
print (ref_wfn.shape)
print (ref_wfn)
ref_wfn = np.append (ref_wfn[None,:], np.array (mc_sa.get_energy_decomposition ()[1:]).T, axis=0)
print (ref_wfn.shape)
print (ref_wfn)
print ('ref_nuc = {:.10f}'.format (ref_nuc))
print ('ref_wfn = np.array ([[{:.10f}, {:.10f}, {:.10f}, {:.10f}, {:.10f}],'.format (*ref_wfn[0]))
print ('                     [{:.10f}, {:.10f}, {:.10f}, {:.10f}, {:.10f}],'.format (*ref_wfn[1]))
print ('                     [{:.10f}, {:.10f}, {:.10f}, {:.10f}, {:.10f}],'.format (*ref_wfn[2]))
print ('                     [{:.10f}, {:.10f}, {:.10f}, {:.10f}, {:.10f}],'.format (*ref_wfn[3]))
print ('                     [{:.10f}, {:.10f}, {:.10f}, {:.10f}, {:.10f}],'.format (*ref_wfn[4]))
print ('                     [{:.10f}, {:.10f}, {:.10f}, {:.10f}, {:.10f}]])'.format (*ref_wfn[5]))


