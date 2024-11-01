### functions taken from github.com/hczhai/fci-siso/blob/master/fcisiso.py ##

import numpy as np
from pyscf import fci, lib
from pyscf.fci import cistring

def make_trans_rdm1(dspin, cibra, ciket, norb, nelec_bra, nelec_ket):
    
    nelabra, nelbbra = nelec_bra
    nelaket, nelbket = nelec_ket

    if dspin == 'ba':
        cond = nelabra == nelaket - 1 and nelbbra == nelbket + 1 and nelaket != 0 and nelbbra != 0
    elif dspin == 'ab':
        cond = nelabra == nelaket + 1 and nelbbra == nelbket - 1 and nelbket != 0 and nelabra != 0
    elif dspin == 'aa':
        cond = nelabra == nelaket and nelaket != 0 and nelabra !=0
    else:
        cond = nelbbra == nelbket and nelbbra !=0 and nelbket !=0
        
    if not cond:
        return np.array(0)
   
    nabra = fci.cistring.num_strings(norb, nelabra)
    nbbra = fci.cistring.num_strings(norb, nelbbra)
    naket = fci.cistring.num_strings(norb, nelaket)
    nbket = fci.cistring.num_strings(norb, nelbket)
    cibra = cibra.reshape(nabra, nbbra)
    ciket = ciket.reshape(naket, nbket)
    #lidxbra = fci.cistring.gen_des_str_index(range(norb), nelabra if dspin[0] == 'a' else nelbbra)
    if dspin[1] == 'a':
        lidxket = fci.cistring.gen_des_str_index(range(norb), nelaket)
        naketd = fci.cistring.num_strings(norb, nelaket - 1)
        t1 = np.zeros((norb, naketd, nbket))
        for str0 in range(naket):
            for _, i, str1, sign in lidxket[str0]:
                t1[i, str1, :] += sign * ciket[str0, :]
    else:
        lidxket = fci.cistring.gen_des_str_index(range(norb), nelbket)
        nbketd = fci.cistring.num_strings(norb, nelbket - 1)
        t1 = np.zeros((norb, naket, nbketd))
        for str0 in range(nbket):
            for _, i, str1, sign in lidxket[str0]:
                t1[i, :, str1] += sign * ciket[:, str0]
        if nelaket % 2 == 1:
            t1 = -t1
    if dspin[0] == 'a':
        lidxbra = fci.cistring.gen_des_str_index(range(norb), nelabra)
        nabrad = fci.cistring.num_strings(norb, nelabra - 1)
        t2 = np.zeros((norb, nabrad, nbbra))
        for str0 in range(nabra):
            for _, i, str1, sign in lidxbra[str0]:
                t2[i, str1, :] += sign * cibra[str0, :]
    else:
        lidxbra = fci.cistring.gen_des_str_index(range(norb), nelbbra)
        nbbrad = fci.cistring.num_strings(norb, nelbbra - 1)
        t2 = np.zeros((norb, nabra, nbbrad))
        for str0 in range(nbbra):
            for _, i, str1, sign in lidxbra[str0]:
                t2[i, :, str1] += sign * cibra[:, str0]
        if nelabra % 2 == 1:
            t2 = -t2
    
    rdm1 = np.tensordot(t1, t2, axes=((1,2), (1,2)))

    # This appears to return rdm1[p,q] = <ket|p'q|bra> = <bra|q'p|ket>
    return rdm1

def make_trans(m, ciket, cibra, norb, nelec_ket, nelec_bra):
    # MRH NOTE: this has been modified to consistency with spinless dm1[p,q] = <q'p>

    if m == -1:
        # MRH NOTE: the factor of -1 that used to be here makes no sense
        return make_trans_rdm1('ab', cibra, ciket, norb, nelec_bra, nelec_ket).T
    elif m == 1:
        return make_trans_rdm1('ba', cibra, ciket, norb, nelec_bra, nelec_ket).T
    else:
        return (make_trans_rdm1('aa', cibra, ciket, norb, nelec_bra, nelec_ket)
                - make_trans_rdm1('bb', cibra, ciket, norb, nelec_bra, nelec_ket)).T


if __name__=='__main__':
    from pyscf import gto, scf, mcscf
    from mrh.my_pyscf.fci import csf_solver
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    molh2o = gto.M (atom="""
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.758602  0.000000  -0.504284
    """, basis='631g',symmetry=True,
    output='test_lassi_soc.log',
    verbose=lib.logger.DEBUG)
    mfh2o = scf.RHF (molh2o).run ()
    with lib.temporary_env (mfh2o.mol, charge=2):
        mc = mcscf.CASSCF (mfh2o, 8, 4).set (conv_tol=1e-12)
        mc.fcisolver = csf_solver (mfh2o.mol, smult=3).set (wfnsym='A1')
        mc.kernel ()
        # The result is very sensitive to orbital basis, so I optimize orbitals
        # tightly using CASSCF, which is a more stable implementation
        las = LASSCF (mfh2o, (8,), (4,), spin_sub=(3,), wfnsym_sub=('A1',))
        las.mo_coeff = mc.mo_coeff
        las.state_average_(weights=[1/4,]*4,
                           spins=[[2,],[0,],[-2,],[0]],
                           smults=[[3,],[3,],[3,],[1]],
                           wfnsyms=(([['B1',],]*3)+[['A1',],]))
        las.lasci ()
        from mrh.my_pyscf.mcscf import lassi_op_o1
        ints = lassi_op_o1.make_ints (las, las.ci, np.array (list (range (4))))[1][0]
        tdm_test = make_trans (1, las.ci[0][0], las.ci[0][3], 8, (3,1), (2,2))
        tdm_ref = ints.get_sp (0,3)
        print ("ab:")
        print (lib.fp (tdm_test))
        print (lib.fp (tdm_ref))
        tdm_test = make_trans (-1, las.ci[0][2], las.ci[0][3], 8, (1,3), (2,2))
        tdm_ref = ints.get_sm (2,3)
        print ("ba:")
        print (lib.fp (tdm_test))
        print (lib.fp (tdm_ref))
        e_roots, si = las.lassi (opt=0, soc=True, break_symmetry=True)




