import numpy as np
import itertools
import math
from mrh.util.basis import represent_operator_in_basis, project_operator_into_subspace, is_basis_orthonormal, measure_basis_olap, get_complementary_states, is_basis_orthonormal_and_complete
from pyscf.lo import boys
from pyscf.scf.hf import get_ovlp
from functools import reduce
from pyscf.tools import molden

def debug_ofc_oneRDM ( dmet_obj ):

    # Make sure its idempotent part is what it is supposed to be
    oneRDM_loc = dmet_obj.helper.construct1RDM_loc( dmet_obj.doSCF, dmet_obj.umat )
    evals, evecs = np.linalg.eigh (oneRDM_loc)
    idem_idx = np.logical_or (np.isclose (evals, 2.0 * np.ones (dmet_obj.ints.norbs_tot)), np.isclose (evals, np.zeros (dmet_obj.ints.norbs_tot)))
    corr_idx = np.logical_not (idem_idx)
    print ("oneRDM_loc eigenvalues: {0}\neigenvectors:\n{1}".format (evals, evecs))
    print ("Testing guide 1RDM: found {0} idempotent states of {1} expected".format(
        np.count_nonzero (idem_idx), dmet_obj.ints.loc2idem.shape[1]))
    loc2tidem = evecs[:,idem_idx]
    l2t = np.asmatrix (loc2tidem)
    l2i = np.asmatrix (dmet_obj.ints.loc2idem)
    t2l = l2t.H
    i2l = l2i.H
    Pidem_loc  = np.asarray (l2i * i2l)
    Ptidem_loc = np.asarray (l2t * t2l)
    print ("error matrix of idempotent states (norm = {1}):\n{0}".format (
        Ptidem_loc - Pidem_loc, np.linalg.norm (Ptidem_loc - Pidem_loc)))
    loc2tcorr = evecs[:,corr_idx]
    l2t = np.asmatrix (loc2tcorr)
    l2c = np.asmatrix (np.concatenate ([frag.loc2amo for frag in dmet_obj.fragments], axis=1))
    t2l = l2t.H
    c2l = l2c.H
    Pcorr_loc  = np.asarray (l2c * c2l)
    Ptcorr_loc = np.asarray (l2t * t2l)
    print ("error matrix of correlated states (norm = {1}):\n{0}".format (
        Ptcorr_loc - Pcorr_loc, np.linalg.norm (Ptcorr_loc - Pcorr_loc)))

def debug_Eimp (dmet_obj, frag):
    raise NotImplementedError ("Hopelessly broken since last used.")
    E0 = dmet_obj.ints.const ()
    E1 = 0.0
    E2 = 0.0

    OEI = dmet_obj.ints.loc_rhf_fock_bis (0.5 * frag.oneRDM_loc)
    TEI = frag.impham_TEI
    Eimp = (dmet_obj.ints.const () + np.einsum ('ij,ij->', OEI, frag.oneRDM_loc)
                             + 0.5 * np.einsum ('ijkl,ijkl->', TEI, frag.twoCDM_imp))
    for loc2tb, twoCDM in zip (frag.loc2tbc, frag.twoCDMfroz_tbc):
        V     = dmet_obj.ints.dmet_tei (loc2tb)
        Eimp += 0.5 * np.einsum ('ijkl,ijkl->', V, twoCDM)

    print ("debug_Eimp :: fragment {0} impurity energy = {1:.5f}, test energy = {2:.5f}, difference = {3:.5f}".format(
            frag.frag_name, frag.E_imp, Eimp, frag.E_imp - Eimp))
    return Eimp

def debug_Etot (dmet_obj):
    raise NotImplementedError ("Hopelessly broken since last used.")
    if dmet_obj.CC_E_TYPE == 'CASCI':
        print ("debug_Etot :: CASCI calculation; passing to debug_Eimp")
        return debug_Eimp (dmet_obj, dmet_obj.fragments[0])
    frags = dmet_obj.fragments 
    for f in frags:
        debug_Eimp (dmet_obj, f)
        f.E1_test = 0.0
        f.E2_test = 0.0
    E0 = dmet_obj.ints.const ()
    E1 = 0.0
    E2 = 0.0
    print ("debug_Etot :: constant = {0}".format (E0))

    for f in itertools.product (frags, frags):
        fname    = "{0} + {1}".format (f[0].frag_name, f[1].frag_name)
        loc2frag = [i.loc2frag for i in f] 
        OEI_i    = [represent_operator_in_basis (dmet_obj.ints.loc_rhf_fock_bis (0.5 * i.oneRDM_loc), *loc2frag) for i in f]
        oneRDM_i = [0.5 * represent_operator_in_basis (i.oneRDM_loc, *loc2frag) for i in f]
        E1_i     = [np.einsum ('ij,ij->', OEI, oneRDM) for OEI, oneRDM in zip (OEI_i, oneRDM_i)]
        E1      += sum(E1_i)
        print ("debug_Etot :: fragments {0} E1 = {1}".format (fname, sum(E1_i)))
        for E, i in zip (E1_i, f):
            i.E1_test += E
    print ("debug_Etot :: one-body = {0}".format (E1))

    for f in itertools.product (frags, frags, frags, frags):
        fname    = "{0} + {1} + {2} + {3}".format (f[0].frag_name, f[1].frag_name, f[2].frag_name, f[3].frag_name) 
        loc2frag = [i.loc2frag for i in f]
        TEI      = dmet_obj.ints.general_tei (loc2frag)
        twoRDM_i = [0.25 * i.get_twoCDM(*loc2frag) for i in f]
        E2_i     = [0.5 * np.einsum ('ijkl,ijkl->', TEI, twoRDM) for twoRDM in twoRDM_i]
        E2      += sum(E2_i)
        print ("debug_Etot :: fragments {0} E2 = {1}".format (fname, sum(E2_i)))
        for E, i in zip (E2_i, f):
            i.E2_test += E
    print ("debug_Etot :: two-body = {0}".format (E2))
    Etot = E0 + E1 + E2
    print ("debug_Etot :: object energy = {0:.5f}, test energy = {1:.5f}, difference = {2:.5f}".format(
            dmet_obj.energy, Etot, dmet_obj.energy - Etot))
    print ("debug_Etot :: fragment energy decomposition:")
    for f in frags:
        E_test = f.E1_test + f.E2_test
        E_diff = f.E_frag - E_test
        print ("{0} fragment energy = {1:.5f}, test E1 = {2:.5f}, test E2 = {3:.5f}, test Etot = {4:.5f}, difference = {5:.5f}".format(
               f.frag_name, f.E_frag, f.E1_test, f.E2_test, E_test, E_diff))
        del f.E1_test
        del f.E2_test

    return Etot

def examine_ifrag_olap (dmet_obj):
    frags = dmet_obj.fragments
    for f1, f2 in itertools.combinations_with_replacement (frags, 2):
        if f1 is not f2:
            olap_mag = measure_basis_olap (f1.loc2emb[:,:f1.norbs_frag], f2.loc2emb[:,:f2.norbs_frag])[0]
            print ("Quasi-fragment overlap magnitude between {0} and {1}: {2:.2f}".format(
                f1.frag_name, f2.frag_name, olap_mag))
            olap_mat = np.dot (f1.loc2emb[:,:f1.norbs_frag].T, f2.loc2emb[:,:f2.norbs_frag])
            #print (np.array2string (olap_mat, precision=2, suppress_small=True))
        else:
            if not is_basis_orthonormal (f1.loc2imp[:,:f1.norbs_frag]):
                raise RuntimeError ("{0} quasi-fragment basis not orthonormal?? Overlap=\n{1}".format (
                    f1.frag_name, np.dot (f1.imp2loc[:f1.norbs_frag,:],f1.loc2imp[:,:f1.norbs_frag])))

def examine_wmcs (dmet_obj):
    loc2wmas = np.concatenate ([f.loc2amo for f in dmet_obj.fragments], axis=1)
    loc2wmcs = get_complementary_states (loc2wmas)

    print ("Examining whole-molecule active space:")
    loc2wmas = dmet_obj.ints.compare_basis_to_loc (loc2wmas, dmet_obj.fragments, quiet=False)[0]
    norbs_wmas = np.array ([f.norbs_as for f in dmet_obj.fragments])

    print ("Examining whole-molecule core space:")
    norbs_wmcs_before = dmet_obj.ints.compare_basis_to_loc (loc2wmcs, dmet_obj.fragments, quiet=True)[1]
    norbs_before = norbs_wmas + norbs_wmcs_before
    ao2loc = dmet_obj.ints.ao2loc
    loc2ao = ao2loc.conjugate ().T
    ao2wmcs = np.dot (ao2loc, loc2wmcs)
    ao2wmcs_new = boys.Boys (dmet_obj.ints.mol, ao2wmcs).kernel ()
    aoOao_inv = np.linalg.inv (np.dot (ao2loc, loc2ao))
    loc2wmcs_new = reduce (np.dot, [loc2ao, aoOao_inv, ao2wmcs_new])
    loc2wmcs_new, norbs_wmcs_after = dmet_obj.ints.compare_basis_to_loc (loc2wmcs_new, dmet_obj.fragments, quiet=False)
    norbs_after = norbs_wmas + norbs_wmcs_after

    loc2new = np.append (loc2wmas, loc2wmcs_new, axis=1)
    active_flag = np.append (np.ones (loc2wmas.shape[1]), np.zeros (loc2wmcs_new.shape[1]))
    print ("Is the new basis orthonormal and complete? {0}".format (is_basis_orthonormal_and_complete (loc2new)))
    print ("Fragment-orbital assignment breakdown:")
    print ("Frag Before After")
    for frag, bef, aft in zip (dmet_obj.fragments, norbs_before, norbs_after):
        print ('{:>4s} {:6d} {:5d}'.format (frag.frag_name, int (bef), int (aft)))

    filename = 'relocalized_basis.molden'
    mol = dmet_obj.ints.mol.copy ()
    ao2new = np.dot (dmet_obj.ints.ao2loc, loc2new)
    molden.from_mo (mol, filename, ao2new, occ=active_flag)
    '''
    for idx, frag in enumerate (dmet_obj.fragments):
        print ("{0} fragment impurity orbitals".format (frag.frag_name))
        ao2imp = np.dot (ao2loc, frag.loc2imp)
        ao2imp_new = boys.Boys (dmet_obj.ints.mol, ao2imp).kernel ()
        loc2imp_new = reduce (np.dot, [loc2ao, aoOao_inv, ao2imp_new])
        norbs_imp = dmet_obj.ints.compare_basis_to_loc (loc2imp_new, dmet_obj.fragments, quiet=False)[1]
        print ("{0} fragment-localized impurity orbitals compared to {1} user-specified fragment orbitals".format (norbs_imp[idx], frag.norbs_frag))
    '''

    
