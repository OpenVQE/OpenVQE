import numpy as np
from pyscf import gto

def structure (dnn1=0, dnn2=0, basis='6-31g', symmetry=False):
    f = open ('c2h6n4.xyz', 'r')
    equilgeom = f.read ()
    mol = gto.M (atom = equilgeom, basis = basis, symmetry=True, unit='au')
    atoms = tuple(mol.atom_symbol (i) for i in range (mol.natm))
    coords = mol.atom_coords ()
    
    idx_nn = np.asarray ([[0, 1], [10, 11]])
    nn_vec = coords[[1,10],:] - coords[[2,9],:]
    nn_equil = np.mean (np.linalg.norm (nn_vec, axis=1))
    print ("r_nn equilbrium distance: {}".format (np.linalg.norm (nn_vec, axis=1)))
    nn_vec /= np.linalg.norm (nn_vec, axis=1)[:,np.newaxis]
    delta_coords = np.zeros_like (coords)
    delta_coords[idx_nn[0],:] = np.broadcast_to (nn_vec[0,:], (2,3))
    delta_coords[idx_nn[1],:] = np.broadcast_to (nn_vec[1,:], (2,3))
    scale = np.zeros (12, dtype=coords.dtype)
    scale[idx_nn[0]] = dnn1
    scale[idx_nn[1]] = dnn2
    newcoords = coords + scale[:,np.newaxis] * delta_coords
    carts = [[atoms[i]] + list(newcoords[i,:]) for i in range(12)]
    dummymol = gto.M (atom = carts, basis=basis, symmetry=True)
    newcoords = dummymol.atom_coords ()
    carts = [[atoms[i]] + list(newcoords[i,:]) for i in range(12)]
    return gto.M (atom = carts, basis=basis, symmetry=symmetry, unit='au')


