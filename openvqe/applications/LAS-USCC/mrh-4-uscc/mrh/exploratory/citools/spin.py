import numpy as np
from pyscf import lib
from scipy import linalg
from pyscf.fci import cistring
from pyscf.fci.spin_op import spin_square0, contract_ss
from pyscf.fci.addons import _unpack_nelec
import itertools

def spin_square_diag (norb, nelec):
    neleca, nelecb = _unpack_nelec (nelec)
    strsa = strsb = cistring.make_strings (range(norb), neleca)
    na = nb = len (strsa)
    strsa_uint8 = strsb_uint8 = strsa.view ('uint8').reshape (na, -1).T
    if neleca != nelecb:
        strsb = cistring.make_strings (range(norb), nelecb)
        nb = len (strsb)
        strsb_uint8 = strsb.view ('uint8').reshape (nb, -1).T
    sdiag = np.zeros ((na, nb), dtype=np.uint8)
    for ix, (taba, tabb) in enumerate (zip (strsa_uint8, strsb_uint8)):
        if (ix*8) > norb: break
        tab = np.bitwise_xor.outer (taba, tabb)
        for j in range (8):
            tab, acc = np.divmod (tab, 2)
            sdiag += acc
    tab = taba = tabb = strsa = strsb = strsa_uint = strsb_uint = None
    #print (sdiag)
    sdiag = sdiag.astype ('float64')*.5
    sz = (neleca-nelecb)*.5
    sdiag += sz*sz
    return sdiag

def spin_square_diag_check (norb, nelec):
    neleca, nelecb = _unpack_nelec (nelec)
    na = cistring.num_strings (norb, neleca)
    nb = cistring.num_strings (norb, nelecb)
    sdiag = np.zeros ((na,nb))
    for ia in range (na):
        for ib in range (nb):
            ci0 = np.zeros_like (sdiag)
            ci0[ia,ib] = 1.0
            ci0 = contract_ss (ci0, norb, nelec)
            sdiag[ia,ib] = ci0[ia,ib]
            #sdiag[ia,ib] = spin_square0 (ci0, norb, nelec) [0]
    return sdiag

def spin_4th_diag_check (norb, nelec):
    neleca, nelecb = _unpack_nelec (nelec)
    na = cistring.num_strings (norb, neleca)
    nb = cistring.num_strings (norb, nelecb)
    sdiag = np.zeros ((na,nb))
    for ia in range (na):
        for ib in range (nb):
            ci0 = np.zeros_like (sdiag)
            ci0[ia,ib] = 1.0
            ci0 = contract_ss (ci0, norb, nelec)
            ci0 = contract_ss (ci0, norb, nelec)
            sdiag[ia,ib] = ci0[ia,ib]
    return sdiag

if __name__=='__main__':
    for norb in range (65):
        for neleca in range (norb+1):
            for nelecb in range (norb+1):
                if norb != 6: continue 
                #if neleca != 3: continue
                #if nelecb != 2: continue
                sdiag_test = spin_square_diag (norb, (neleca,nelecb))
                sdiag_ref = spin_square_diag_check (norb, (neleca,nelecb))
                sdiag2_test = sdiag_test*sdiag_test
                sdiag2_ref = spin_4th_diag_check (norb, (neleca,nelecb))
                #print (sdiag2_ref-sdiag2_test)
                print (norb, (neleca, nelecb), 
                       np.amax (np.abs (sdiag_test-sdiag_ref)),
                       np.amax (np.abs (sdiag2_test-sdiag2_ref)),
                       linalg.norm (sdiag_test))

