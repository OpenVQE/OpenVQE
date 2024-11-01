import numpy as np
from scipy import linalg
from sympy import S
from sympy.physics.quantum.cg import CG
from pyscf.lib import logger
from itertools import product

def _assert_selection_rule (cond, errmsg, j2, m2):
    errmsg = errmsg + '\nj2 = {}\nm2 = {}'.format (j2, m2)
    assert (cond), errmsg

def _assert_j2m2 (j2, m2, mcoup):
    mtag = ("m","j'")[int(mcoup)]
    _assert_selection_rule (len (j2) == len (m2), 'j, {} length mismatch'.format (mtag), j2, m2)
    _assert_selection_rule (np.all (j2>=0), 'invalid j: must be nonnegative', j2, m2)
    _assert_selection_rule (np.all (np.abs (m2) <= j2), '|{}| > j'.format (mtag), j2, m2)
    _assert_selection_rule (np.all ((j2%2) == (m2%2)), 'j, {} parity mismatch'.format (mtag), j2, m2)
    if mcoup:
        # |j2a - j2b| <= j2c <= j2a + j2b
        j2c = np.cumsum (m2)
        j2a = j2c - m2
        j2b = j2
        j2c_min = np.abs (j2a - j2b)
        j2c_max = j2a + j2b
        errmsg = ("invalid j': |j'[:k].sum () - j[k]| > j'[k]\n"
                  "note this means j'[k] must = j[k] wherever\n"
                  "j[:k].sum () == 0, including for k=0!")
        _assert_selection_rule (np.all (j2c >= j2c_min), errmsg, j2, m2)
        _assert_selection_rule (np.all (j2c <= j2c_max), "invalid j': j'[:k].sum () + j[k] < j'[k]", j2, m2)

def _selection_rules (j2, m2):
    parity = np.all ((j2%2) == (m2%2))
    m = m2[2] == m2[0] + m2[1]
    mag = np.all (np.abs (m2) <= j2)
    cparity = (j2%2).sum () == 0
    return (parity and m and mag and cparity)
    # TODO: Figure out why the SymPy calculator gives nonzero coefficients for 2J%2 != 2M%2
    # Will this affect us for spin-orbit coupling or Sz-broken orbital bases?
               

def cg_prod (jmag, jpol, mpol, fac=2, log=None):
    r'''Compute a single coefficient :math:`<\vec{j}\vec{m}|\vec{j}\vec{j}^\prime>`,
        where :math:`\sum_k j_k^\prime=j_{\rm{tot}}`, :math:`\sum_k m_k=m_{\rm{tot}}`,
        and :math:`-j_{\rm{tot}}\le m_{\rm{tot}}\le j_{\rm{tot}}` by multiplying Clebsch-Gordan coefficients.

        Args:
            jmag: list of N nonnegative spin magnitudes, :math:`\vec{j}`
            jpol: list of N spin magnitude couplings from left to right
                Must have jcoup[0] == jmag[0] and sum (jcoup[:k]) >= 0 for all k<N
            mpol: list of N spin polarizations, -sum (jcoup) <= sum (mpol) <= sum(jcoup)

        Kwargs:
            fac: integer
                Divide jmag, jpol, and mpol by this to get the actual quantum #s.
                Default (2) means that you actually provided 2*j, 2*m, etc., so
                as to avoid data type ambiguities and floating-point rounding
                issues with half-integer quantum numbers. (Internally, everything
                is multiplied by 2 until we get to the actual Clebsch-Gordan
                cruncher.)
            log: PySCF logger object
                For logging output

        Returns:
            coeff: floating-point
    '''
    ### Throat-clearing ###
    fac = 2 // fac 
    jmag = np.around (np.asarray (jmag) * fac).astype (np.int32)
    jpol = np.around (np.asarray (jpol) * fac).astype (np.int32)
    mpol = np.around (np.asarray (mpol) * fac).astype (np.int32)
    _assert_j2m2 (jmag, jpol, True)
    _assert_j2m2 (jmag, mpol, False)
    mtot = mpol.sum ()
    jtot = jpol.sum ()
    assert (mtot in range (-jtot, jtot+1, 2)), 'mpol and jpol mismatch'
    if log is None: log = logger.Logger (verbose=logger.DEBUG4)

    coeff = S(1)
    jrun = np.cumsum (jpol)
    mrun = np.cumsum (mpol)
    for i in range (1, len(jmag)):
        coeff *= CG (S(jrun[i-1])/2, S(mrun[i-1])/2,       # j1, m1
                     S(jmag[i])/2,   S(mpol[i])/2,         # j2, m2
                     S(jrun[i])/2,   S(mrun[i])/2).doit () # j3, m3 = m1 + m2
    return coeff.evalf ()

def cg_prod_vec (jmag, jpol, mtot, fac=2, log=None):
    r'''Compute unitary vector :math:`<\vec{j}\vec{m}|\vec{j}\vec{j}^\prime>` for all :math:`\vec{m}`,
        where :math:`\sum_k j_k^\prime=j_{\rm{tot}}`, :math:`\sum_k m_k=m_{\rm{tot}}`,
        and :math:`-j_{\rm{tot}}\le m_{\rm{tot}}\le j_{\rm{tot}}` by multiplying Clebsch-Gordan coefficients.

        Args:
            jmag: list of N nonnegative spin magnitudes, :math:`\vec{j}`
            jpol: list of N spin magnitude couplings from left to right
                Must have jcoup[0] == jmag[0] and sum (jcoup[:k]) >= 0 for all k<N
            mtot: overall spin polarization, -sum (jcoup) <= mtot <= sum(jcoup)

        Kwargs:
            fac: integer
                Divide jmag, jpol, and mtot by this to get the actual quantum #s.
                Default (2) means that you actually provided 2*j, 2*m, etc., so
                as to avoid data type ambiguities and floating-point rounding
                issues with half-integer quantum numbers. (Internally, everything
                is multiplied by 2 until we get to the actual Clebsch-Gordan
                cruncher.)
            log: PySCF logger object
                For logging output

        Returns:
            mpol_list: integer ndarray, shape (*,N)
                m * fac for the uncoupled basis functions :math:`|\vec{j}\vec{m}>`
                with nonzero CG products
            coeffs: real ndarray, shape (*)
                The unitary vector :math:`<\vec{j}\vec{m}|\vec{j}\vec{j}^\prime>`
                for :math:`|\vec{j}\vec{m}>` identified by mpol_list

    '''
    ### Throat-clearing ###
    fac = 2 // fac
    jmag = np.around (np.asarray (jmag) * fac).astype (np.int32)
    jpol = np.around (np.asarray (jpol) * fac).astype (np.int32)
    mtot = mtot * fac
    _assert_j2m2 (jmag, jpol, True)
    jtot = jpol.sum ()
    if log is None: log = logger.Logger (verbose=logger.DEBUG3)
 
    ### Recursion function ###
    def _recurse (m2a_str, coeffs_in, j2, m2c_max, m2c_min):    
        m2c_str = []
        coeffs_out = []
        logger.debug4 (log, 'Entering _recurse with {} strings, {} coeffs, and {} <= m2c <= {}'.format (len (m2a_str), len (coeffs_in), m2c_min, m2c_max))
        for m2a_vec, coeff in zip (m2a_str, coeffs_in):
            m2a = m2a_vec.sum ()
            m2b_max = min (m2c_max - m2a, j2[1])
            m2b_min = max (m2c_min - m2a, -j2[1])
            logger.debug4 (log, 'For this string, {} <= m2b <= {}'.format (m2b_min, m2b_max))
            for m2b in range (m2b_min, m2b_max+1, 2):
                m2c = m2a + m2b
                m2 = np.array ([m2a, m2b, m2c])
                cgfac = CG (S(j2[0])/2, S(m2a)/2,
                            S(j2[1])/2, S(m2b)/2,
                            S(j2[2])/2, S(m2c)/2).doit ()
                logger.debug4 (log, 'Computing <{},{};{},{}|{},{}> = {}'.format (
                    str(S(j2[0])/2), str(S(m2a)/2),
                    str(S(j2[1])/2), str(S(m2b)/2),
                    str(S(j2[2])/2), str(S(m2c)/2), str (cgfac)))
                m2c_str.append (np.append (m2a_vec, m2b))
                try:
                    coeffs_out.append (float (coeff * cgfac.evalf ()))
                except TypeError as e:
                    assert (j2.sum () % 2 == 0), 'Coupling parity error? {}'.format (j2)
        return m2c_str, coeffs_out 

    ### Initialize and recurse ###
    mpol_list = [np.zeros (0, dtype=np.int32)]
    coeffs = [np.asarray([1.0])]
    j2a = 0
    mrem = jmag.sum ()
    for j2b, dj in zip (jmag, jpol):
        j2c = j2a + dj
        mrem -= j2b
        m2c_max = mtot + mrem
        m2c_min = mtot - mrem
        mpol_list, coeffs = _recurse (mpol_list, coeffs, 
            np.asarray ([j2a, j2b, j2c], dtype=np.int32),
            m2c_max, m2c_min)
        logger.debug4 (log, 'Spinner coupled; {} tags and {} coeffs for running j = {} with {} units of m left to play with'.format (len (mpol_list), len (coeffs), j2c, mrem))
        j2a = j2c

    ### Throat-clearing ###
    mpol_list = np.stack (mpol_list, axis=0) // fac
    return mpol_list, np.asarray (coeffs).astype (np.float64)


if __name__ == '__main__':
    print ("Three triplets")
    jmag=[2,2,2]
    mtot=0
    for j1, j2 in product ((-2,-1,0,1,2), repeat=2):
        jpol = [2,j1,j2]
        try:
            mpol_list, coeffs = cg_prod_vec (jmag, jpol, mtot)
            print (jpol, linalg.norm (coeffs))
        except AssertionError as e:
            print (jpol, 'breaks selection rules')
    print ("Antiferromagnetic quintets")
    mpol_list, coeffs = cg_prod_vec ([4,4],[4,-4],0)
    print ("jpol = [2,-2]; coeffs norm =", linalg.norm (coeffs))
    for mpol, coeff in zip (mpol_list, coeffs):
        print (mpol, coeff)


