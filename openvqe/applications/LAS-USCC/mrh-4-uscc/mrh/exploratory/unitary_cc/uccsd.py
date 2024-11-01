import time
import numpy as np
from scipy import linalg
from pyscf.lib import logger
from pyscf.fci.addons import _unpack_nelec
from pyscf.fci.cistring import num_strings
from pyscf.fci.direct_nosym import FCISolver

# There is no way this will ever be a practical method for production calculations
# However, it may be a useful utility to test certain things

def contract_sd (mol, t1, t2, ci, norb, nelec, hermi=-1, unit_tol=1e-8, tci_tol=1e-12, maxiter=171):
    ''' (Attempt to) evaluate |uci> = exp(t1+t2-t1'-t2')|ci>
    TODO: link_index support?

    Args:
        mol : PySCF mole object
        t1 : ndarray of shape (norb,norb)
        t2 : ndarray of shape (norb,norb,norb,norb)
        ci : list/ndarray broadcastable to shape (*,ndeta,ndetb)
            ndeta = norb choose nelec[0]
            ndetb = norb choose nelec[1]
            CI vector(s) to which to (attempt to) apply the unitary
            correlator
        norb : integer
        nelec : integer or pair of integers

    Kwargs:
        hermi : int
            If -1, t1 and t2 are assumed already antisymmetrized
            If 0, t1 and t2 are explicitly antisymmetrized
            If 1, an exception is raised
        unit_tol : float
            Maximum permitted deviation from orthonormality
        tci_tol : float
            Maximum permitted norm of the last term evaluated
            from the Taylor series
        maxiter : int
            Maximum number of terms from the Taylor series to
            compute. Note that the IEEE 754 float64 format
            rounds 1/171! to zero.

    Returns:
        conv : bool
            convergence status of the Taylor series on exit
        uci : ndarray of shape (*,ndeta,ndetb)
            CI vector(s) after unitary transformation
    '''
    t0 = (time.process_time (), time.time ())
    nelec = _unpack_nelec (nelec)
    if hermi == 0:
        t1 = t1 - t1.T
        t2 = t2 - t2.transpose (1,0,3,2)
    elif hermi != -1: raise RuntimeError ("hermi = 0 or -1 (unitary generator cannot be hermitian)")
    ndeta = num_strings (norb, nelec[0]) 
    ndetb = num_strings (norb, nelec[1]) 
    ci = np.asarray (ci).reshape (-1,ndeta*ndetb)
    tci_eps = np.finfo (ci.dtype).eps 
    nroots = ci.shape[0]

    fci = FCISolver (mol)
    gen_op = fci.absorb_h1e (t1, t2, norb, nelec, 0.5) 
    scale_t = np.amax (np.abs (gen_op))
    tci = ci.copy ()
    uci = ci.copy ()
    for it in range (maxiter):
        norm_tci = np.amax (np.diag (np.dot (tci.conj (), tci.T)))
        norm_uci = np.dot (uci.conj (), uci.T)
        err_unit = np.amax (np.abs (norm_uci - np.eye (nroots)))
        ovlp = np.trace (np.dot (uci.conj (), ci.T))
        logger.debug (mol, 'term %d err_unit = %9.3e ; norm_tci = %9.3e ; ovlp = %9.3e', it, err_unit, norm_tci, ovlp)
        conv = (norm_tci < tci_tol) and (err_unit < unit_tol)
        # Let's make sure that tci rounds to zero only through comparison to the accumulant,
        # not through comparison to the operator (whose magnitude is supposed to be meaningless)
        amax_tci = np.amax (np.abs (tci))
        if conv or (amax_tci<tci_eps): break
        scale = np.amax (np.abs (tci)) / scale_t
        tci = np.stack ([fci.contract_2e (gen_op, c / scale, norb, nelec).ravel () for c in tci], axis=0)
        tci *= scale / (it+1)
        uci += tci

    if conv:
        logger.info (mol, "exp(t1+t2-t1'-t2')|ci> converged after %d terms", it)
    else:
        logger.info (mol, "exp(t1+t2-t1'-t2')|ci> not converged after %d terms", it)
        logger.info (mol, "Nonunitary error: %9.3e", err_unit)
        logger.info (mol, "<ci|(t'-t)**%d (t-t')**%d|ci> / %d!**2 = %9.3e", it, it, it, norm_tci)

    logger.timer (mol, "exp(t1+t2-t1'-t2')|ci> evaluation", *t0)
    return conv, uci.reshape (nroots, ndeta, ndetb)


