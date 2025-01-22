"""Custom excitations for the system VQE during LAS-VQE.

This uses the inter-fragment generalized UCC excitation generator located in 
the MRH code and returns the excitations as the List needed by qiskit's VQE 
class.
"""

from typing import Tuple, List, Any
from mrh.exploratory.unitary_cc import lasuccsd

# Gets all acceptable operators for UCCSD
# excluding intra-fragment ones
def custom_excitations(num_spatial_orbitals: int,
                           num_particles: Tuple[int, int],
                           num_sub: List[int]
                           ) -> List[Tuple[Tuple[Any, ...], ...]]:
    excitations = []
    norb = num_spatial_orbitals
    uop = lasuccsd.gen_uccsd_op (norb, num_sub)
    a_idxs = uop.a_idxs
    i_idxs = uop.i_idxs
    for a,i in zip(a_idxs,i_idxs):
        excitations.append((tuple(i),tuple(a[::-1])))
    
    return excitations

