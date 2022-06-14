import numpy as np
import openfermion

# spin_complement_gsd:
#  here just to check the open_Fermion
def spin_complement_gsd_open_fermion_twin(n_elec, orbital_number):
    alpha_orbs = [2 * i for i in range(orbital_number)]
    beta_orbs = [2 * i + 1 for i in range(orbital_number)]

    spin_complement_gsd_twin = []
    term_a = []

    # aa
    for p in alpha_orbs:
        for q in alpha_orbs:
            if p >= q:
                continue
            # if abs(hamiltonian_op.one_body_tensor[p,q]) < 1e-8:
            #    print(" Dropping term %4i %4i" %(p,q), " V= %+6.1e" %hamiltonian_op.one_body_tensor[p,q])
            #    continue

            term_a = openfermion.FermionOperator(((q, 1), (p, 0)))
            term_a += openfermion.FermionOperator(((q + 1, 1), (p + 1, 0)))
            term_a -= openfermion.hermitian_conjugated(term_a)
            term_a = openfermion.normal_ordered(term_a)
            if term_a.many_body_order() > 0:
                spin_complement_gsd_twin.append(term_a)
    #             spin_complement_gsd_twin.append(term_a)

    # aa
    pq = 0
    term_b = []
    for p in alpha_orbs:
        for q in alpha_orbs:
            if p > q:
                continue
            rs = 0
            for r in alpha_orbs:
                for s in alpha_orbs:
                    if r > s:
                        continue
                    if pq < rs:
                        continue
                    term_b = openfermion.FermionOperator(
                        ((r, 1), (p, 0), (s, 1), (q, 0))
                    )
                    term_b += openfermion.FermionOperator(
                        ((r + 1, 1), (p + 1, 0), (s + 1, 1), (q + 1, 0))
                    )
                    term_b -= openfermion.hermitian_conjugated(term_b)
                    term_b = openfermion.normal_ordered(term_b)
                    if term_b.many_body_order() > 0:
                        spin_complement_gsd_twin.append(term_b)
                    #                     spin_complement_gsd_twin.append(term_b)

                    rs += 1
            pq += 1

    pq = 0
    for p in alpha_orbs:
        for q in beta_orbs:
            rs = 0
            for r in alpha_orbs:
                for s in beta_orbs:
                    if pq < rs:
                        continue
                    term_b = openfermion.FermionOperator(
                        ((r, 1), (p, 0), (s, 1), (q, 0))
                    )
                    if p > q:
                        continue
                    term_b += openfermion.FermionOperator(
                        ((s - 1, 1), (q - 1, 0), (r + 1, 1), (p + 1, 0))
                    )
                    term_b -= openfermion.hermitian_conjugated(term_b)
                    term_b = openfermion.normal_ordered(term_b)
                    if term_b.many_body_order() > 0:
                        spin_complement_gsd_twin.append(term_b)

                    rs += 1
            pq += 1

    return len(spin_complement_gsd_twin), spin_complement_gsd_twin


def UCCSD_openFermion(n_elec, orbital_number):
    sd = []
    n_occ = int(np.ceil(n_elec / 2))
    n_vir = orbital_number - n_occ

    for i in range(0, n_occ):
        ia = 2 * i
        ib = 2 * i + 1

        for a in range(0, n_vir):
            aa = 2 * n_occ + 2 * a
            ab = 2 * n_occ + 2 * a + 1

            term_a = openfermion.FermionOperator(((aa, 1), (ia, 0)), 1)
            term_a += openfermion.FermionOperator(((ab, 1), (ib, 0)), 1)

            term_a -= openfermion.hermitian_conjugated(term_a)

            term_a = openfermion.normal_ordered(term_a)

            # Normalize
            coeff_a = 0
            for t in term_a.terms:
                coeff_t = term_a.terms[t]
                coeff_a += coeff_t * coeff_t
            #         print("term_a.many_body_order()",term_a.many_body_order())
            #         for t in term_a:
            # #             print(term_a)
            if term_a.many_body_order() > 0:
                term_a = term_a / np.sqrt(coeff_a)
                sd.append(term_a)

    for i in range(0, n_occ):
        ia = 2 * i
        ib = 2 * i + 1

        for j in range(i, n_occ):
            ja = 2 * j
            jb = 2 * j + 1

            for a in range(0, n_vir):
                aa = 2 * n_occ + 2 * a
                ab = 2 * n_occ + 2 * a + 1

                for b in range(a, n_vir):
                    ba = 2 * n_occ + 2 * b
                    bb = 2 * n_occ + 2 * b + 1

                    term_a = openfermion.FermionOperator(
                        ((aa, 1), (ba, 1), (ia, 0), (ja, 0)), 2 / np.sqrt(12)
                    )
                    #                 term_a += openfermion.FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                    term_a += openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (ia, 0), (jb, 0)), 1 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((ab, 1), (ba, 1), (ib, 0), (ja, 0)), 1 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (ib, 0), (ja, 0)), 1 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((ab, 1), (ba, 1), (ia, 0), (jb, 0)), 1 / np.sqrt(12)
                    )

                    term_b = openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (ia, 0), (jb, 0)), 1 / 2
                    )
                    term_b += openfermion.FermionOperator(
                        ((ab, 1), (ba, 1), (ib, 0), (ja, 0)), 1 / 2
                    )
                    term_b += openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (ib, 0), (ja, 0)), -1 / 2
                    )
                    term_b += openfermion.FermionOperator(
                        ((ab, 1), (ba, 1), (ia, 0), (jb, 0)), -1 / 2
                    )

                    term_a -= openfermion.hermitian_conjugated(term_a)
                    term_b -= openfermion.hermitian_conjugated(term_b)

                    term_a = openfermion.normal_ordered(term_a)
                    term_b = openfermion.normal_ordered(term_b)
                    # Normalize
                    coeff_a = 0
                    coeff_b = 0
                    for t in term_a.terms:
                        coeff_t = term_a.terms[t]
                        coeff_a += coeff_t * coeff_t
                    for t in term_b.terms:
                        coeff_t = term_b.terms[t]
                        coeff_b += coeff_t * coeff_t
                    # coeff A and coeff B tends to 1
                    if term_a.many_body_order() > 0:
                        term_a = term_a / np.sqrt(coeff_a)
                        sd.append(term_a)

                    if term_b.many_body_order() > 0:
                        term_b = term_b / np.sqrt(coeff_b)
                        sd.append(term_b)
    print("term_b: ", term_b)
    return len(sd), sd


# spin_complement_gsd:
#  here just to check the open_Fermion
def spin_complement_gsd_openFermion(n_elec, orbital_number):
    spin_complement_gsd = []
    # n_elec = 2
    # orbital_number  = 2
    for p in range(0, orbital_number):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, orbital_number):
            qa = 2 * q
            qb = 2 * q + 1

            term_a = openfermion.FermionOperator(((pa, 1), (qa, 0)))
            term_a += openfermion.FermionOperator(((pb, 1), (qb, 0)))

            term_a -= openfermion.hermitian_conjugated(term_a)
            term_a = openfermion.normal_ordered(term_a)

            if term_a.many_body_order() > 0:
                spin_complement_gsd.append(term_a)

    pq = -1
    for p in range(0, orbital_number):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, orbital_number):
            qa = 2 * q
            qb = 2 * q + 1

            pq += 1

            rs = -1
            for r in range(0, orbital_number):
                ra = 2 * r
                rb = 2 * r + 1

                for s in range(r, orbital_number):
                    sa = 2 * s
                    sb = 2 * s + 1

                    rs += 1

                    if pq > rs:
                        continue

                    term_a = openfermion.FermionOperator(
                        ((ra, 1), (pa, 0), (sa, 1), (qa, 0))
                    )
                    term_a += openfermion.FermionOperator(
                        ((rb, 1), (pb, 0), (sb, 1), (qb, 0))
                    )

                    term_b = openfermion.FermionOperator(
                        ((ra, 1), (pa, 0), (sb, 1), (qb, 0))
                    )
                    term_b += openfermion.FermionOperator(
                        ((rb, 1), (pb, 0), (sa, 1), (qa, 0))
                    )

                    term_c = openfermion.FermionOperator(
                        ((ra, 1), (pb, 0), (sb, 1), (qa, 0))
                    )
                    term_c += openfermion.FermionOperator(
                        ((rb, 1), (pa, 0), (sa, 1), (qb, 0))
                    )

                    term_a -= openfermion.hermitian_conjugated(term_a)
                    term_b -= openfermion.hermitian_conjugated(term_b)
                    term_c -= openfermion.hermitian_conjugated(term_c)

                    term_a = openfermion.normal_ordered(term_a)
                    term_b = openfermion.normal_ordered(term_b)
                    term_c = openfermion.normal_ordered(term_c)

                    if term_a.many_body_order() > 0:
                        spin_complement_gsd.append(term_a)

                    if term_b.many_body_order() > 0:
                        spin_complement_gsd.append(term_b)

                    if term_c.many_body_order() > 0:
                        spin_complement_gsd.append(term_c)

    return len(spin_complement_gsd), spin_complement_gsd


# THIS is from the open_fermion package:
def singlet_uccsd_open_fermion(n_elec, orbital_number):
    singlet_sd = []
    n_occ = int(np.ceil(n_elec / 2))
    n_vir = orbital_number - n_occ

    for i in range(0, n_occ):
        ia = 2 * i
        ib = 2 * i + 1

        for a in range(0, n_vir):
            aa = 2 * n_occ + 2 * a
            ab = 2 * n_occ + 2 * a + 1

            term_a = openfermion.FermionOperator(((aa, 1), (ia, 0)), 1 / np.sqrt(2))
            term_a += openfermion.FermionOperator(((ab, 1), (ib, 0)), 1 / np.sqrt(2))

            term_a -= openfermion.hermitian_conjugated(term_a)

            term_a = openfermion.normal_ordered(term_a)

            # Normalize
            coeff_a = 0
            for t in term_a.terms:
                coeff_t = term_a.terms[t]
                coeff_a += coeff_t * coeff_t
            #         print("term_a.many_body_order()",term_a.many_body_order())
            #         for t in term_a:
            # #             print(term_a)
            if term_a.many_body_order() > 0:
                term_a = term_a / np.sqrt(coeff_a)
                singlet_sd.append(term_a)

    for i in range(0, n_occ):
        ia = 2 * i
        ib = 2 * i + 1

        for j in range(i, n_occ):
            ja = 2 * j
            jb = 2 * j + 1

            for a in range(0, n_vir):
                aa = 2 * n_occ + 2 * a
                ab = 2 * n_occ + 2 * a + 1

                for b in range(a, n_vir):
                    ba = 2 * n_occ + 2 * b
                    bb = 2 * n_occ + 2 * b + 1

                    term_a = openfermion.FermionOperator(
                        ((aa, 1), (ba, 1), (ia, 0), (ja, 0)), 2 / np.sqrt(12)
                    )
                    #                 term_a += openfermion.FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                    term_a += openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (ia, 0), (jb, 0)), 1 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((ab, 1), (ba, 1), (ib, 0), (ja, 0)), 1 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (ib, 0), (ja, 0)), 1 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((ab, 1), (ba, 1), (ia, 0), (jb, 0)), 1 / np.sqrt(12)
                    )

                    term_b = openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (ia, 0), (jb, 0)), 1 / 2
                    )
                    term_b += openfermion.FermionOperator(
                        ((ab, 1), (ba, 1), (ib, 0), (ja, 0)), 1 / 2
                    )
                    term_b += openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (ib, 0), (ja, 0)), -1 / 2
                    )
                    term_b += openfermion.FermionOperator(
                        ((ab, 1), (ba, 1), (ia, 0), (jb, 0)), -1 / 2
                    )

                    term_a -= openfermion.hermitian_conjugated(term_a)
                    #                 print("term_a",term_a)
                    term_b -= openfermion.hermitian_conjugated(term_b)
                    #                 print("term_b",term_b)
                    term_a = openfermion.normal_ordered(term_a)
                    term_b = openfermion.normal_ordered(term_b)
                    #                 print("term_a",term_a)
                    #                 print("term_b",term_b)
                    # Normalize
                    coeff_a = 0
                    coeff_b = 0
                    for t in term_a.terms:
                        coeff_t = term_a.terms[t]
                        coeff_a += coeff_t * coeff_t
                    for t in term_b.terms:
                        coeff_t = term_b.terms[t]
                        coeff_b += coeff_t * coeff_t

                    if term_a.many_body_order() > 0:
                        term_a = term_a / np.sqrt(coeff_a)
                        singlet_sd.append(term_a)

                    if term_b.many_body_order() > 0:
                        term_b = term_b / np.sqrt(coeff_b)
                        singlet_sd.append(term_b)
    return len(singlet_sd), singlet_sd


# print("Pool size:",len(singlet_sd))
# print(singlet_sd)


# In[2]:


# Singlet Generalized Singles and Doubles
def singlet_generalized_singles_doubles_openFermion(n_elec, orbital_number):
    singlet_gsd = []
    # n_elec = 2
    # orbital_number  = 2
    for p in range(0, orbital_number):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, orbital_number):
            qa = 2 * q
            qb = 2 * q + 1

            term_a = openfermion.FermionOperator(((pa, 1), (qa, 0)))
            term_a += openfermion.FermionOperator(((pb, 1), (qb, 0)))

            term_a -= openfermion.hermitian_conjugated(term_a)
            term_a = openfermion.normal_ordered(term_a)

            # Normalize
            coeff_a = 0
            for t in term_a.terms:
                coeff_t = term_a.terms[t]
                coeff_a += coeff_t * coeff_t

            if term_a.many_body_order() > 0:
                term_a = term_a / np.sqrt(coeff_a)
                singlet_gsd.append(term_a)

    pq = -1
    for p in range(0, orbital_number):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, orbital_number):
            qa = 2 * q
            qb = 2 * q + 1

            pq += 1

            rs = -1
            for r in range(0, orbital_number):
                ra = 2 * r
                rb = 2 * r + 1

                for s in range(r, orbital_number):
                    sa = 2 * s
                    sb = 2 * s + 1

                    rs += 1

                    if pq > rs:
                        continue

                    term_a = openfermion.FermionOperator(
                        ((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12)
                    )
                    term_a += openfermion.FermionOperator(
                        ((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12)
                    )

                    term_b = openfermion.FermionOperator(
                        ((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2
                    )
                    term_b += openfermion.FermionOperator(
                        ((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2
                    )
                    term_b += openfermion.FermionOperator(
                        ((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2
                    )
                    term_b += openfermion.FermionOperator(
                        ((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2
                    )

                    term_a -= openfermion.hermitian_conjugated(term_a)
                    term_b -= openfermion.hermitian_conjugated(term_b)
                    #                 print("term_bherm",term_b)
                    term_a = openfermion.normal_ordered(term_a)
                    term_b = openfermion.normal_ordered(term_b)

                    # Normalize
                    coeff_a = 0
                    coeff_b = 0
                    for t in term_a.terms:
                        coeff_t = term_a.terms[t]
                        #                     print("coeff_t",coeff_t)
                        coeff_a += coeff_t * coeff_t
                    #                 print("term_b.terms",term_b.terms)
                    for t in term_b.terms:
                        coeff_t = term_b.terms[t]
                        #                     print("coeff_t",coeff_t)
                        coeff_b += coeff_t * coeff_t
                    #                 print("coeff_b",coeff_b)
                    #                 print(term_a.many_body_order())
                    if term_a.many_body_order() > 0:
                        term_a = term_a / np.sqrt(coeff_a)
                        singlet_gsd.append(term_a)
                    #                 print(term_b.many_body_order())
                    #                 print("term_b",term_b)
                    if term_b.many_body_order() > 0:
                        term_b = term_b / np.sqrt(coeff_b)
                        singlet_gsd.append(term_b)
    return len(singlet_gsd), singlet_gsd
