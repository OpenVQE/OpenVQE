#########################
# Script to run LASVQE after setting up a mean field object
#########################

import numpy as np
from argparse import ArgumentParser
# PySCF imports
from pyscf import gto, scf, lib

from get_geom import get_geom
from las_vqe import LASVQE
from selected_las_vqe import LASVQE as selected_LASVQE

parser = ArgumentParser(description='Do LAS-VQE for (H2)_2, specifying distance between H2 molecules')
parser.add_argument('--dist', type=float, default=1.0, help='distance of H2s from one another')
args = parser.parse_args()


# Define molecule: (H2)_2
xyz = get_geom('close', dist=args.dist)
#xyz = '''H 0.0 0.0 0.0
#             H 1.0 0.0 0.0
#             H 0.2 1.6 0.1
#             H 1.159166 1.3 -0.1'''
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g_{}.log'.format(args.dist),
    symmetry=False, verbose=lib.logger.DEBUG)

# Do RHF
mf = scf.RHF(mol).run()
print("HF energy: ", mf.e_tot)

# Fragments can be chosen based on all orbitals associated with an atom or specific orbitals
frag_atom_list = ((0,1),(2,3))

# Create LASVQE object
lasvqe = selected_LASVQE(mf, (2,2), (2,2), frag_atom_list, spin_sub=(1,1))

vqe_en, vqe_result = lasvqe.run(epsilon = 0.003727593720314938)
print("LAS-VQE Energy: ", vqe_en)

