import numpy as np
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import param, logger

STEPSIZE_DEFAULT=0.001
SCANNER_VERBOSE_DEFAULT=4

# MRH 05/04/2020: I don't know why I have to present the molecule instead
# of just the coordinates, but somehow I can't get the units right any other
# way.

def _make_mol (mol, coords):
    return [[mol.atom_symbol (i), coords[i,:]] for i in range (mol.natm)]

def _numgrad_1df (mol, scanner, coords, iatm, icoord, delta=0.002):
    coords[iatm,icoord] += delta
    ep = scanner (_make_mol (mol, coords))
    ep_states = np.array (getattr (scanner, 'e_states', [ep]))
    coords[iatm,icoord] -= 2*delta
    em = scanner (_make_mol (mol, coords))
    em_states = np.array (getattr (scanner, 'e_states', [em]))
    coords[iatm,icoord] += delta
    de_states = (ep_states - em_states) / (2*delta)*param.BOHR
    return (ep-em) / (2*delta)*param.BOHR, de_states


class Gradients (rhf_grad.GradientsMixin):

    def __init__(self, method, stepsize=STEPSIZE_DEFAULT, scanner_verbose=SCANNER_VERBOSE_DEFAULT):
        self.stepsize = stepsize
        self.scanner = None 
        # MRH 05/04/2020: there must be a better way to do this
        if hasattr (self.scanner, '_scf'):
            self.scanner._scf.verbose = scanner_verbose
        rhf_grad.GradientsMixin.__init__(self, method)
        self.scanner = self.base.as_scanner ()
        self.scanner.verbose = scanner_verbose

    def _numgrad_1df (self, iatm, icoord):
        return _numgrad_1df (self.mol, self.scanner, self.mol.atom_coords () * param.BOHR,
            iatm, icoord, delta=self.stepsize)

    def kernel (self, atmlst=None, stepsize=None, state=None):
        if atmlst is None:
            atmlst = self.atmlst
        if stepsize is None:
            stepsize = self.stepsize
        else:
            self.stepsize = stepsize
        if atmlst is None:
            atmlst = list (range (self.mol.natm))
        
        coords = self.mol.atom_coords () * param.BOHR
        de = [[self._numgrad_1df (i, j) for j in range (3)] for i in atmlst]
        self.scanner (_make_mol (self.mol, coords)) # Reset!
        self.de = np.asarray ([[i for i,j in k] for k in de])
        self.de_states = np.asarray ([[j for i,j in k] for k in de]).transpose (2,0,1)
        if state is not None: self.de = self.de_states[state]
        return self.de

    def grad_elec (self, atmlst=None, stepsize=None):
        # This is just computed backwards from full gradients
        if atmlst is None:
            atmlst = self.atmlst
        if stepsize is None:
            stepsize = self.stepsize
        else:
            self.stepsize = stepsize
        if atmlst is None:
            atmlst = list (range (self.mol.natm))

        if getattr (self, 'de', None) is not None:
            de = self.de = self.kernel (atmlst=atmlst, stepsize=stepsize)
        de_elec = de - self.grad_nuc (atmlst=atmlst)
        return de_elec

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '----------- %s numeric gradients -------------',
                        self.base.__class__.__name__)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')


