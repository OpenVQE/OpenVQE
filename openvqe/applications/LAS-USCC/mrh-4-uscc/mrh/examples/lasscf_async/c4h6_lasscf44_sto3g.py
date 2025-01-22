import pyscf
from pyscf import gto, scf, tools, mcscf,lib
from mrh.my_pyscf.mcscf import lasscf_async as asyn
from mrh.my_pyscf.mcscf import lasscf_sync_o0 as syn
from pyscf.mcscf import avas
lib.logger.TIMER_LEVEL = lib.logger.INFO

outputfile='c4h6_lasscf44_sto3g.log'
xyz='''H     -0.943967690125   0.545000000000   0.000000000000
C      0.000000000000   0.000000000000   0.000000000000
C      1.169134295109   0.675000000000   0.000000000000
H      0.000000000000  -1.090000000000   0.000000000000
H      1.169134295109   1.765000000000   0.000000000000
C      2.459512146748  -0.070000000000   0.000000000000
C      3.628646441857   0.605000000000   0.000000000000
H      2.459512146748  -1.160000000000   0.000000000000
H      3.628646441857   1.695000000000   0.000000000000
H      4.572614131982   0.060000000000   0.000000000000'''
mol=gto.M(atom=xyz,basis='sto3g',verbose=4,output=outputfile)
mf=scf.RHF(mol)
mf.run()
frag_atom_list=[list(range(1+4*ifrag,3+4*ifrag)) for ifrag in range(2)]
ncas,nelecas,mo0 = avas.kernel(mf, ['C 2p'])

las_syn=syn.LASSCF(mf, (2,2), (2,2))
mo_coeff=las_syn.localize_init_guess (frag_atom_list, mo0)
las_syn.state_average_(weights=[.2,]*5,
                        spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                        smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
las_syn.kernel(mo_coeff)
print ("Synchronous calculation converged?", las_syn.converged)

las_asyn=asyn.LASSCF(mf, (2,2), (2,2))
mo_coeff=las_asyn.set_fragments_(frag_atom_list, mo0)
las_asyn.state_average_(weights=[.2,]*5,
                        spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                        smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
las_asyn.kernel(mo_coeff)
print ("Asynchronous calculation converged?", las_asyn.converged)

print ("Final state energies:")
print ("{:>16s} {:>16s} {:>16s}".format ("Synchronous", "Asynchronous", "Difference"))
fmt_str = "{:16.9e} {:16.9e} {:16.9e}"
for es, ea in zip (las_syn.e_states, las_asyn.e_states): print (fmt_str.format (es, ea, ea-es))


