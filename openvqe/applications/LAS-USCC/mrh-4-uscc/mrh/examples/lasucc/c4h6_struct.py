import numpy as np
from pyscf import gto
from pyscf.lib.parameters import BOHR

def structure (dnn1=0, dnn2=0, basis='6-31g', symmetry=False, _xyz='c4h6.xyz', output=None, verbose=None):
    with open (_xyz, 'r') as f:
        equilgeom = ''.join (f.readlines ()[2:])
    
    mol = gto.M (atom = equilgeom, basis=basis, symmetry=True, spin=0)
    atoms = tuple(mol.atom_symbol (i) for i in range (mol.natm))
    coords = mol.atom_coords () * BOHR
    
    idx_nn = np.asarray ([[0, 1, 2], [7, 8, 9]])
    nn_vec = coords[[0,7],:] - coords[[3,5],:]
    nn_equil = np.mean (np.linalg.norm (nn_vec, axis=1))
    nn_vec /= np.linalg.norm (nn_vec, axis=1)[:,np.newaxis]
    delta_coords = np.zeros_like (coords)
    delta_coords[idx_nn[0],:] = np.broadcast_to (nn_vec[0,:], (3,3))
    delta_coords[idx_nn[1],:] = np.broadcast_to (nn_vec[1,:], (3,3))
    scale = np.zeros (10, dtype=coords.dtype)
    scale[idx_nn[0]] = dnn1
    scale[idx_nn[1]] = dnn2
    newcoords = coords + scale[:,np.newaxis] * delta_coords
    carts = [[atoms[i]] + list(newcoords[i,:]) for i in range(10)]
    dummymol = gto.M (atom = carts, basis=basis, symmetry=True)
    newcoords = dummymol.atom_coords ()
    carts = [[atoms[i]] + list(newcoords[i,:]) for i in range(10)]
    return gto.M (atom = carts, basis=basis, symmetry=symmetry, unit='au', output=output, verbose=verbose)

def r0 (_xyz='c4h6.xyz'):
    with open (_xyz, 'r') as f:
        equilgeom = ''.join (f.readlines ()[2:])
    
    mol = gto.M (atom = equilgeom, basis='sto-3g', symmetry=True, spin=0)
    atoms = tuple(mol.atom_symbol (i) for i in range (mol.natm))
    coords = mol.atom_coords () * BOHR
    
    idx_nn = np.asarray ([[0, 1, 2], [7, 8, 9]])
    nn_vec = coords[[0,7],:] - coords[[3,5],:]
    nn_equil = np.mean (np.linalg.norm (nn_vec, axis=1))
    return nn_equil

