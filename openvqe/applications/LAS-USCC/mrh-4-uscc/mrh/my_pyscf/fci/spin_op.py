#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MRH note: copied to mrh.my_pyscf on 11/04/2020

from functools import reduce
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec

librdm = lib.load_library('libfci')

######################################################
# Spin squared operator
######################################################
# S^2 = (S+ * S- + S- * S+)/2 + Sz * Sz
# S+ = \sum_i S_i+ ~ effective for all beta occupied orbitals.
# S- = \sum_i S_i- ~ effective for all alpha occupied orbitals.
# There are two cases for S+*S-
# 1) same electron \sum_i s_i+*s_i-, <CI|s_i+*s_i-|CI> gives
#       <p|s+s-|q> \gammalpha_qp = trace(\gammalpha) = neleca
# 2) different electrons for \sum s_i+*s_j- (i\neq j, n*(n-1) terms)
# As a two-particle operator S+*S-
#       = <ij|s+s-|kl>Gamma_{ik,jl} = <iajb|s+s-|kbla>Gamma_{iakb,jbla}
#       = <ia|s+|kb><jb|s-|la>Gamma_{iakb,jbla}
# <CI|S+*S-|CI> = neleca + <ia|s+|kb><jb|s-|la>Gamma_{iakb,jbla}
#
# There are two cases for S-*S+
# 1) same electron \sum_i s_i-*s_i+
#       <p|s+s-|q> \gammabeta_qp = trace(\gammabeta) = nelecb
# 2) different electrons
#       = <ij|s-s+|kl>Gamma_{ik,jl} = <ibja|s-s+|kalb>Gamma_{ibka,jalb}
#       = <ib|s-|ka><ja|s+|lb>Gamma_{ibka,jalb}
# <CI|S-*S+|CI> = nelecb + <ib|s-|ka><ja|s+|lb>Gamma_{ibka,jalb}
#
# Sz*Sz = Msz^2 = (neleca-nelecb)^2
# 1) same electron
#       <p|ss|q>\gamma_qp = <p|q>\gamma_qp = (neleca+nelecb)/4
# 2) different electrons
#       <ij|2s1s2|kl>Gamma_{ik,jl}/2
#       =(<ia|ka><ja|la>Gamma_{iaka,jala} - <ia|ka><jb|lb>Gamma_{iaka,jblb}
#       - <ib|kb><ja|la>Gamma_{ibkb,jala} + <ib|kb><jb|lb>Gamma_{ibkb,jblb})/4

# set aolst for local spin expectation value, which is defined as
#       <CI|ao><ao|S^2|CI>
# For a complete list of AOs, I = \sum |ao><ao|, it becomes <CI|S^2|CI>

def contract_sladder(fcivec, norb, nelec, op=-1):
    ''' Contract spin ladder operator S+ or S- with fcivec.
        Changes neleca - nelecb without altering <S2>
        Obtained by modifying pyscf.fci.spin_op.contract_ss
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)
    assert (op in (-1,1)), 'op = -1 or 1'
    if ((op==-1 and (neleca==0 or nelecb==norb)) or
        (op==1 and (neleca==norb or nelecb==0))): return np.zeros ((0,0))
    # ^ Annihilate vacuum state ^

    def gen_map(fstr_index, nelec, des=True):
        a_index = fstr_index(range(norb), nelec)
        amap = np.zeros((a_index.shape[0],norb,2), dtype=np.int32)
        if des:
            for k, tab in enumerate(a_index):
                amap[k,tab[:,1]] = tab[:,2:]
        else:
            for k, tab in enumerate(a_index):
                amap[k,tab[:,0]] = tab[:,2:]
        return amap

    if op==-1:
        aindex = gen_map(cistring.gen_des_str_index, neleca)
        bindex = gen_map(cistring.gen_cre_str_index, nelecb, False)
    else:
        aindex = gen_map(cistring.gen_cre_str_index, neleca, False)
        bindex = gen_map(cistring.gen_des_str_index, nelecb)

    ci1 = np.zeros((cistring.num_strings(norb,neleca+op),
                   cistring.num_strings(norb,nelecb-op)))
    for i in range(norb):
        signa = aindex[:,i,1]
        signb = bindex[:,i,1]
        maska = np.where(signa!=0)[0]
        maskb = np.where(signb!=0)[0]
        addra = aindex[maska,i,0]
        addrb = bindex[maskb,i,0]
        citmp = lib.take_2d(fcivec, maska, maskb)
        citmp *= signa[maska].reshape(-1,1)
        citmp *= signb[maskb]
        #: ci1[addra.reshape(-1,1),addrb] += citmp
        lib.takebak_2d(ci1, citmp, addra, addrb)
    ci1 /= linalg.norm (ci1) # ???
    return ci1


def contract_sdown (ci, norb, nelec): return contract_sladder (ci, norb, nelec, op=-1)
def contract_sup (ci, norb, nelec): return contract_sladder (ci, norb, nelec, op=1)

if __name__ == '__main__':
    import sys
    import time
    from pyscf.fci.direct_spin1 import contract_2e
    from pyscf.fci.spin_op import spin_square0
    t0, w0 = lib.logger.process_clock (), lib.logger.perf_counter ()
    nelec, norb = (int (argv) for argv in sys.argv[1:])
    nelec = (min (norb, nelec), nelec - min(norb, nelec))
    smult = nelec[0]-nelec[1]+1
    print ("Testing the spin ladder operators for a {}e, {}o s={} space".format (sum (nelec), norb, (smult-1)/2))
    cishape = [cistring.num_strings (norb, ne) for ne in nelec]
    np.random.seed(1)
    ci = np.random.rand (*cishape)
    eri = np.random.rand (norb,norb,norb,norb)
    ci /= linalg.norm (ci)
    print (" neleca nelecb ndeta ndetb {:>5s} {:>13s} {:>5s} {:>5s}".format ("cc","chc","ss","2s+1"))
    def print_line (c, ne):
        try:
            ss, smult = spin_square0 (c, norb, ne)
            hc = contract_2e (eri, c, norb, ne)
            chc = c.conj ().ravel ().dot (hc.ravel ())
        except AssertionError as e:
            assert (any ([n<0 for n in ne]))
            ss, smult, chc = (0.0, 1.0, 0.0)
        cc = linalg.norm (c)
        ndeta, ndetb = c.shape
        print (" {:>6d} {:>6d} {:>5d} {:>5d} {:5.3f} {:13.9f} {:5.3f} {:5.3f}".format (ne[0], ne[1], ndeta, ndetb, cc, chc, ss, smult))
    print_line (ci, nelec)
    for ndown in range (smult-1):
        ci = contract_sdown (ci, norb, nelec)
        nelec = (nelec[0]-1, nelec[1]+1)
        print_line (ci, nelec)
    for nup in range (smult):
        ci = contract_sup (ci, norb, nelec)
        nelec = (nelec[0]+1, nelec[1]-1)
        print_line (ci, nelec)
    print ("Time elapsed {} clock ; {} wall".format (lib.logger.process_clock () - t0, lib.logger.perf_counter () - w0))

