import re, os

ctr_re = re.compile ('[0-9]+[spdfgh]')
lm_qn = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
ano_rcc_ctrs = ['MB', 'VDZ', 'VDZP', 'VTZ', 'VTZP', 'VQZP']

def contract_ano_basis (mol, contractions):
    ''' Get a smaller version of the ANO-RCC basis.

        Args:
            mol: pyscf.gto.mole object
            contractions: str or dict of type {str: str}
                identifies contractions to keep, e.g.: {'C': '4s3p2d'} keeps the first 4, 3, and 2 s, p, and d
                contractions of primitive gaussians for the 'C' atom

    '''
    dummy = {mol.atom_pure_symbol (atm): 'ano' for atm in range (mol.natm)}
    new_basis = mol.format_basis (dummy)
    if isinstance (contractions, dict):
        contr = {key: val for key, val in contractions.items () if key in new_basis}
    elif isinstance (contractions, str):
        contr = {key: contractions for key in new_basis}
    contr = parse_basis_tbl (contr)

    for atm_key, ctr_str in contr.items ():
        nmax_cgos = [0 for i in range (6)]
        nacc_cgos = [0 for i in range (6)]
        ctrs = ctr_re.findall (ctr_str)
        for ctr in ctrs:
            shell = lm_qn[ctr[-1]]
            n_ctrs = int (ctr[:-1])
            nmax_cgos[shell] = n_ctrs
        for bl in new_basis[atm_key]:
            ishell = bl[0]
            if nacc_cgos[ishell] == nmax_cgos[ishell]:
                bl[0] = -1
                continue
            n_ctrs = min (nmax_cgos[ishell] - nacc_cgos[ishell], len (bl[1]) - 1)
            nacc_cgos[ishell] += n_ctrs
            for irow in range (1, len (bl)):
                bl[irow] = bl[irow][:n_ctrs+1]

        new_basis[atm_key] = [bl for bl in new_basis[atm_key] if bl[0] > -1]

    mol.basis = new_basis
    mol.build ()
    return mol

def parse_basis_tbl (contr):
    ''' Read OpenMolcas's basis.tbl to get the meaning of strings like MB, VTZP, etc. '''
    splitter = re.compile ('\.|\ ')
    with open (os.path.join (os.path.dirname (__file__), 'basis.tbl'), 'r') as f:
        for line in f:
            if not 'ANO-RCC' in line:
                continue
            cols = splitter.split (line)
            if cols[0] in contr and cols[1] == 'ANO-RCC-' + contr[cols[0]].upper ():
                contr[cols[0]] = cols[-2]
    return contr

BREAK_ELEMENT = {'VDZP': 'Fr',
                 'VTZP': 'Fr',
                 'VQZP': 'Fr'}
def ano_rcc_(level='MB'):
    ''' Read OpenMolcas's basis.tbl and build a complete PySCF basis dictionary
        for MB, VTZP, etc. '''
    splitter = re.compile ('\.|\ ')
    basis = {}
    break_element = BREAK_ELEMENT.get (level.upper ())
    with open (os.path.join (os.path.dirname (__file__), 'basis.tbl'), 'r') as f:
        for line in f:
            if not 'ANO-RCC' in line:
                continue
            cols = splitter.split (line)
            if cols[0] == break_element:
                break
            if cols[1] == 'ANO-RCC-' + level.upper ():
                basis[cols[0]] = 'ano@' + cols[-2]
    return basis
                



