import numpy as np
from mrh.exploratory.unitary_cc import uccsd_sym1
from mrh.exploratory.unitary_cc import usccsd_sym1
from mrh.exploratory.unitary_cc import uccsd_sym0
from mrh.exploratory.unitary_cc import usccsd_sym0
from mrh.exploratory.citools import lasci_ominus1
from itertools import combinations, combinations_with_replacement
import time
from scipy.optimize import minimize


def gen_uccsd_op (norb, nlas, t1_s2sym=True):
    ''' Build the fragment-interaction singles and doubles UCC operator!! '''
    t1_idx = np.zeros ((norb, norb), dtype=np.bool_)
    nfrag = len (nlas)
    for ifrag, afrag in combinations (range (nfrag), 2):
        i = sum (nlas[:ifrag])
        a = sum (nlas[:afrag])
        j = i + nlas[ifrag]
        b = a + nlas[afrag]
        t1_idx[a:b,i:j] = True
    t1_idx = np.where (t1_idx)
    a_idxs, i_idxs = list (t1_idx[0]), list (t1_idx[1])
    pq = [[p, q] for p, q in zip (*np.tril_indices (norb))]
    frag_idx = np.concatenate ([[ix,]*n for ix, n in enumerate (nlas)])
    for ab, ij in combinations_with_replacement (pq, 2):
        abij = np.concatenate ([ab, ij])
        nfint = len (np.unique (frag_idx[abij]))
        if nfint > 1:
            a_idxs.append (ab)
            i_idxs.append (ij)
    uop = uccsd_sym1.FSUCCOperator (norb, a_idxs, i_idxs, s2sym=t1_s2sym)
    #uop = uccsd_sym0.FSUCCOperator (norb, a_idxs, i_idxs)
    return uop
       
def gen_usccsd_op(norb, nlas, a_idxs, i_idxs,t1_s2sym=False):  
    uop = usccsd_sym1.FSUCCOperator (norb, a_idxs, i_idxs, s2sym=False)
    return uop
        
class FCISolver (lasci_ominus1.FCISolver):
    def get_uop (self, norb, nlas, t1_s2sym=None):
        frozen = str (getattr (self, 'frozen', None))
        if t1_s2sym is None:
            t1_s2sym = getattr (self, 't1_s2sym', True)
        if frozen.upper () == 'CI':
            return uccsd_sym1.get_uccsd_op (norb, s2sym=t1_s2sym)
        return gen_uccsd_op (norb, nlas, t1_s2sym=t1_s2sym)
        
class FCISolver2(lasci_ominus1.FCISolver):
    def __init__(self, mol, a_idxs, i_idxs):
        super().__init__(mol)
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs

    def get_uop(self, norb, nlas, t1_s2sym=None):
        if getattr(self, 'frozen', '').upper() == 'CI':
            return uccsd_sym1.get_uccsd_op2(norb, epsilon=0.0)
        else:
            return gen_usccsd_op(norb, nlas, self.a_idxs, self.i_idxs)
            #Just pass in norb, nlas, a_idxs_lst, i_idxs_lst

