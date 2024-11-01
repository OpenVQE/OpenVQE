import numpy as np
from scipy import linalg
from functools import reduce

def get_translational_coordinates (carts, masses):
    ''' Construct mass-weighted translational coordinate vectors '''
    natm = carts.shape[0]
    u = np.stack ([np.eye (3) for iatm in range (natm)], axis=0)
    u *= np.sqrt (masses)[:,None,None]
    norm_u = (u*u).sum ((0,1))
    u /= np.sqrt (norm_u)[None,None,:]
    return u

def get_rotational_coordinates (carts, masses):
    ''' Construct mass-weighted rotational coordinate vectors '''
    natm = carts.shape[0]
    # Translate to center of mass
    carts = carts - (np.einsum ('i,ij->j', masses, carts) / np.sum (masses))[None,:]
    # Generate and diagonalize moment-of-inertia vector
    rad2 = (carts * carts).sum (1)
    I = rad2[:,None,None] * np.stack ([np.eye (3) for iatm in range (natm)], axis=0)
    I -= np.stack ([np.outer (icart, icart) for icart in carts], axis=0)
    I = np.einsum ('m,mij->ij', masses, I)
    mI, X = linalg.eigh (I)
    # Generate rotational coordinates: cross-product of X axes with radial displacement from X axes
    u = np.zeros ((natm, 3, 3))
    RXt = np.dot (carts, X)
    for iatm in range (natm):
        iRXt = np.stack ([RXt[iatm].copy () for i in range (3)], axis=0)
        iRXt[np.diag_indices (3)] = 0.0
        iRXt = iRXt @ X.T
        u[iatm] = np.stack ([np.cross (i, j) for i, j in zip (iRXt, X.T)], axis=-1)
    u *= np.sqrt (masses)[:,None,None]
    # Remove norm = 0 modes (linear molecules)
    norm_u = (u * u).sum ((0,1))
    idx = norm_u > 1e-8
    u = u[:,:,idx] / np.sqrt (norm_u[idx])[None,None,:]
    mI = mI[idx]
    return mI, u

class InternalCoords (object):
    def __init__(self, mol):
        self.mol = mol
        self.masses = mol.atom_mass_list ()
        self.carts = mol.atom_coords ()
    def get_coords (self, carts=None, include_inertia=False, mass_weighted=True, guess_uvib=None):
        if carts is None: carts = self.carts
        utrans = get_translational_coordinates (carts, self.masses)
        mI, urot = get_rotational_coordinates (carts, self.masses)
        nrot = urot.shape[-1]
        ntrans = 3
        nvib = 0
        uall = np.append (urot, utrans, axis=-1)
        if guess_uvib is not None:
            uall = np.append (uall, guess_uvib, axis=-1)
            nvib = guess_uvib.shape[-1]
        uall = linalg.qr (uall.reshape (3*self.mol.natm,nrot+ntrans+nvib))[0]
        uvib = uall[:,nrot+ntrans:].reshape (self.mol.natm, 3, -1)
        if not mass_weighted:
            utrans /= np.sqrt (self.masses)[:,None,None]
            urot /= np.sqrt (self.masses)[:,None,None]
            uvib /= np.sqrt (self.masses)[:,None,None]
        if include_inertia: return utrans, urot, uvib, mI
        return utrans, urot, uvib
    def transform_1body (self, vec, carts=None):
        utrans, urot, uvib = self.get_coords (carts=carts, mass_weighted=False)
        vec_t = np.tensordot (vec, utrans, axes=((0,1),(0,1)))
        vec_r = np.tensordot (vec, urot,   axes=((0,1),(0,1)))
        vec_v = np.tensordot (vec, uvib,   axes=((0,1),(0,1)))
        return vec_t, vec_r, vec_v
    def _project_1body (self, vec, carts=None, idx=None, mass_weighted=False):
        if not mass_weighted:
            vec = vec.copy () / np.sqrt (self.masses)[:,None]
        uvib = self.get_coords (carts=carts)[idx]
        vec = np.tensordot (vec, uvib, axes=((0,1),(0,1)))
        vec = np.dot (uvib.conjugate (), vec)
        if not mass_weighted:
            vec *= np.sqrt (self.masses)[:,None]
        return vec
    def project_1body_trans (self, vec, carts=None, mass_weighted=False):
        return self._project_1body (vec, carts=carts, idx=0,
            mass_weighted=mass_weighted)
    def project_1body_rot (self, vec, carts=None, mass_weighted=False):
        return self._project_1body (vec, carts=carts, idx=1,
            mass_weighted=mass_weighted)
    def project_1body_vib (self, vec, carts=None, mass_weighted=False):
        return self._project_1body (vec, carts=carts, idx=2,
            mass_weighted=mass_weighted)




