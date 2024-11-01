import numpy as np
from scipy import linalg
from pyscf import lib
from mrh.exploratory.unitary_cc import uccsd_sym1, lasuccsd
from itertools import product, combinations, combinations_with_replacement
import unittest, math

def spin (norb, a, i):
    sa = np.array ([a1 // norb for a1 in a])
    si = np.array ([i1 // norb for i1 in i])
    naa = np.count_nonzero (sa==0)
    nba = np.count_nonzero (sa==1)
    nai = np.count_nonzero (si==0)
    nbi = np.count_nonzero (si==1)
    errstr = '{} {} {} {} {} {}'.format (a,i,naa,nba,nai,nbi)
    assert (naa+nba == len (a)), errstr
    assert (nai+nbi == len (i)), errstr
    assert (len (a) == len (i)), errstr
    ap = np.array ([a1 % norb for a1 in a])
    ip = np.array ([i1 % norb for i1 in i])
    return (naa!=nai), list(np.append (ap, ip))

def frag (las0, las1, *q):
    frags = []
    for qi in q:
        for i, (p0, p1) in enumerate (zip (las0, las1)):
            if p0 <= qi and qi < p1:
                frags.append (i)
                break
    return len(list(set(frags)))<=1

def uccsd_countop (norb, nlas=None):
    if nlas is not None:
        nlas1 = np.cumsum (nlas)
        nlas0 = nlas1-nlas1[0]
    n1 = 0
    nsporb = norb*2
    for a, i in combinations (range (nsporb), 2):
        skip, p = spin (norb, [a], [i])
        if skip: continue
        if nlas is not None:
            if frag (nlas0, nlas1, *p): continue
        n1 += 1
    assert ((n1%2)==0)
    n2 = 0
    for (a,b), (i,j) in combinations (combinations (range (nsporb), 2), 2):
        skip, p = spin (norb, [a, b], [i, j])
        if skip: continue
        if nlas is not None:
            if frag (nlas0, nlas1, *p): continue
        n2 += 1
    ngen = n1 + n2
    ngen_uniq = (n1//2) + n2
    return ngen, ngen_uniq

class KnownValues(unittest.TestCase):

    def test_uccsd_opnum (self):
        uop = uccsd_sym1.get_uccsd_op (7)
        ngen, ngen_uniq = uccsd_countop (7)
        self.assertEqual (uop.ngen, ngen)
        self.assertEqual (uop.ngen_uniq, ngen_uniq)

    def test_lasuccsd_opnum (self):
        uop = lasuccsd.gen_uccsd_op (7, [4,3])
        ngen, ngen_uniq = uccsd_countop (7, [4,3])
        self.assertEqual (uop.ngen, ngen)
        self.assertEqual (uop.ngen_uniq, ngen_uniq)

if __name__ == "__main__":
    print("Full Tests for UOP generation")
    unittest.main()

