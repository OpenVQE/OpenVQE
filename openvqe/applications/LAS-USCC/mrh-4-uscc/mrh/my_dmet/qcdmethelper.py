'''
    QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

from mrh.my_dmet import localintegrals
import mrh.my_dmet.rhf
from mrh.util.rdm import get_1RDM_from_OEI_in_subspace
from mrh.util.basis import represent_operator_in_basis, project_operator_into_subspace
import numpy as np
import ctypes
from mrh.lib.helper import load_library
lib_qcdmet = load_library ('libqcdmet')

class qcdmethelper:

    def __init__( self, theLocalIntegrals, list_H1, altcf, minFunc ):
    
        self.locints = theLocalIntegrals
        #assert( self.locints.nelec_tot % 2 == 0 )
        self.numPairs = int (self.locints.nelec_tot / 2)
        self.altcf = altcf
        self.minFunc = None

        if self.altcf:
            assert (minFunc == 'OEI' or minFunc == 'FOCK_INIT')
            self.minFunc = minFunc
        
        # Variables for c gradient calculation
        #self.list_H1 = list_H1
        H1start, H1row, H1col = list_H1#self.convertH1sparse()
        self.H1start = H1start
        self.H1row = H1row
        self.H1col = H1col
        self.Nterms = len( self.H1start ) - 1
        
    def convertH1sparse( self ):
    
        H1start = []
        H1row   = []
        H1col   = []
        H1start.append( 0 )
        totalsize = 0
        for count in range( len( self.list_H1 ) ):
            rowco, colco = np.where( self.list_H1[count] == 1 )
            totalsize += len( rowco )
            H1start.append( totalsize )
            for count2 in range( len( rowco ) ):
                H1row.append( rowco[ count2 ] )
                H1col.append( colco[ count2 ] )
        H1start = np.array( H1start, dtype=ctypes.c_int )
        H1row   = np.array( H1row,   dtype=ctypes.c_int )
        H1col   = np.array( H1col,   dtype=ctypes.c_int )
        return ( H1start, H1row, H1col )

    def construct1RDM_loc( self, doSCF, umat_loc ):
        
        # Everything in this functions works in the original local AO / lattice basis!
        if doSCF:
            return self.locints.get_wm_1RDM_from_scf_on_OEI (self.locints.loc_oei ()      + umat_loc)
        elif self.altcf and self.minFunc == 'OEI' :
            return self.locints.get_wm_1RDM_from_OEI        (self.locints.loc_oei ()      + umat_loc)
        else:
            return self.locints.get_wm_1RDM_from_OEI        (self.locints.loc_rhf_fock () + umat_loc)
    
    def construct1RDM_response( self, doSCF, umat_loc, NOrotation ):

        # This part is local-basis        
        if doSCF:
            oneRDM = self.locints.get_wm_1RDM_from_scf_on_OEI (self.loc_oei () + umat_loc)
            OEI    = self.locints.loc_rhf_fock_bis (oneRDM)
        else:
            OEI    = self.locints.loc_rhf_fock() + umat_loc

        # Do I need to project this into the into the idempotent subspace?
        # The overall derivative is going to be -2 * dg_mol/du . |g_imps - g_mol|
        # If you separate g and g_err into the idem and corr spaces
        #   P_idem |g_imps - g_mol| P_idem is the thing I really want to differentiate
        #   P_idem |g_imps - g_mol| P_corr is small but not generally zero due to active spaces changing between iterations
        #   P_corr |g_imps - g_mol| P_corr is zero by construction
        #   P_idem dg_mol/du P_idem needs to be zero but might not be if OEI is defined in the entire space
        #   However I think I can solve this by projecting the ~final~ derivative up in main_object into the working space

        # This part works in the rotated NO basis if NOrotation is specified
        rdm_deriv_rot = np.ones( [ self.locints.norbs_tot * self.locints.norbs_tot * self.Nterms ], dtype=ctypes.c_double )
        if ( NOrotation != None ):
            OEI = np.dot( np.dot( NOrotation.T, OEI ), NOrotation )
        OEI = np.array( OEI.reshape( (self.locints.norbs_tot * self.locints.norbs_tot) ), dtype=ctypes.c_double )
        
        lib_qcdmet.rhf_response( ctypes.c_int( self.locints.norbs_tot ),
                                 ctypes.c_int( self.Nterms ),
                                 ctypes.c_int( self.numPairs ),
                                 self.H1start.ctypes.data_as( ctypes.c_void_p ),
                                 self.H1row.ctypes.data_as( ctypes.c_void_p ),
                                 self.H1col.ctypes.data_as( ctypes.c_void_p ),
                                 OEI.ctypes.data_as( ctypes.c_void_p ),
                                 rdm_deriv_rot.ctypes.data_as( ctypes.c_void_p ) )
        
        rdm_deriv_rot = rdm_deriv_rot.reshape( (self.Nterms, self.locints.norbs_tot, self.locints.norbs_tot), order='C' )
        return rdm_deriv_rot
        
    def constructbath( self, OneDM, impurityOrbs, numBathOrbs, threshold=1e-13 ):
    
        embeddingOrbs = 1 - impurityOrbs
        embeddingOrbs = np.atleast_2d( embeddingOrbs )
        if (embeddingOrbs.shape[0] > 1):
            embeddingOrbs = embeddingOrbs.T # Now certainly row-like matrix (shape = 1 x len(vector))
        isEmbedding = np.dot( embeddingOrbs.T , embeddingOrbs ) == 1
        numEmbedOrbs = np.sum( embeddingOrbs )
        embedding1RDM = np.reshape( OneDM[ isEmbedding ], ( numEmbedOrbs , numEmbedOrbs ) )

        numImpOrbs   = np.sum( impurityOrbs )
        numTotalOrbs = len( impurityOrbs )
        
        eigenvals, eigenvecs = np.linalg.eigh( embedding1RDM )
        idx = np.maximum( -eigenvals, eigenvals - 2.0 ).argsort() # Occupation numbers closest to 1 come first
        tokeep = np.sum( -np.maximum( -eigenvals, eigenvals - 2.0 )[idx] > threshold )
        if ( tokeep < numBathOrbs ):
            print("DMET::constructbath : Throwing out", numBathOrbs - tokeep, "orbitals which are within", threshold, "of 0 or 2.")
        numBathOrbs = min(np.sum( tokeep ), numBathOrbs)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx]
        pureEnvironEigVals = -eigenvals[numBathOrbs:]
        pureEnvironEigVecs = eigenvecs[:,numBathOrbs:]
        idx = pureEnvironEigVals.argsort()
        eigenvecs[:,numBathOrbs:] = pureEnvironEigVecs[:,idx]
        pureEnvironEigVals = -pureEnvironEigVals[idx]
        coreOccupations = np.hstack(( np.zeros([ numImpOrbs + numBathOrbs ]), pureEnvironEigVals ))
    
        for counter in range(0, numImpOrbs):
            eigenvecs = np.insert(eigenvecs, counter, 0.0, axis=1) #Stack columns with zeros in the beginning
        counter = 0
        for counter2 in range(0, numTotalOrbs):
            if ( impurityOrbs[counter2] ):
                eigenvecs = np.insert(eigenvecs, counter2, 0.0, axis=0) #Stack rows with zeros on locations of the impurity orbitals
                eigenvecs[counter2, counter] = 1.0
                counter += 1
        assert( counter == numImpOrbs )
    
        # Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
        assert( np.linalg.norm( np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs) ) < 1e-12 )

        # eigenvecs[ : , 0:numImpOrbs ]                      = impurity orbitals
        # eigenvecs[ : , numImpOrbs:numImpOrbs+numBathOrbs ] = bath orbitals
        # eigenvecs[ : , numImpOrbs+numBathOrbs: ]           = pure environment orbitals in decreasing order of occupation number
        return ( numBathOrbs, eigenvecs, coreOccupations )
        
