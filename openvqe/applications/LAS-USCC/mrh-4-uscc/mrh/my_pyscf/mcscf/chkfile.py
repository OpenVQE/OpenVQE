import h5py
from pyscf.lib.chkfile import load
from pyscf.lib.chkfile import load_mol, save_mol

keys_config = ['ncas', 'nelecas', 'ncore', 'ncas_sub', 'nelecas_sub']
keys_saconstr = ['weights', 'charges', 'spins', 'smults', 'wfnsyms']
keys_results = ['e_states', 'states_converged', 'e_tot', 'mo_coeff']

def load_las_(mc, chkfile=None):
    if chkfile is None: chkfile = mc.chkfile
    if chkfile is None: raise RuntimeError ('chkfile not specified')
    data = load (chkfile, 'las')
    if data is None: raise KeyError ('LAS record not in chkfile')

    # conditionals for backwards compatibility with older chkfiles that
    # only stored mo_coeff and ci
    for key in keys_config:
        if key in data:
            setattr (mc, key, data[key])
    # this needs to happen before some of the results attributes
    if all ([key in data for key in keys_saconstr]):
        sakwargs = {key: data[key] for key in keys_saconstr}
        mc.state_average_(**sakwargs)
    for key in keys_results:
        if key in data:
            setattr (mc, key, data[key])
    if 'frags_orbs' in data: mc.frags_orbs = data['frags_orbs']
    # special handling for ragged CI vector
    ci = data['ci']
    mc.ci = []
    for i in range (mc.nfrags):
        mc.ci.append ([])
        cii = ci[str(i)]
        for j in range (mc.nroots):
            mc.ci[-1].append (cii[str(j)])
    # special handling for ragged frags_orbs
    if 'frags_orbs' in data:
        mc.frags_orbs = []
        frags_orbs = data['frags_orbs']
        for i in range (mc.nfrags):
            mc.frags_orbs.append (list (frags_orbs[str(i)]))
    # if mo_coeff has tagged orbsym, save it, in case someone decides
    # to change PySCF symmetry convention again
    if 'mo_coeff_orbsym' in data:
        from pyscf.lib import tag_array
        mc.mo_coeff = tag_array (mc.mo_coeff, orbsym=data['mo_coeff_orbsym'])
    return mc

def dump_las (mc, chkfile=None, method_key='las', mo_coeff=None, ci=None,
              overwrite_mol=True, **kwargs):
    if chkfile is None: chkfile = mc.chkfile
    if not chkfile: return mc
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    kwargs['mo_coeff'] = mo_coeff
    if ci is None: ci = mc.ci

    keys = keys_config + keys_results
    data = {key: kwargs.get (key, getattr (mc, key)) for key in keys}
    from mrh.my_pyscf.mcscf.lasci import get_space_info
    data_saconstr = get_space_info (mc)
    data_saconstr = [mc.weights,] + list (data_saconstr)
    for key, val in zip (keys_saconstr, data_saconstr):
        data[key] = kwargs.get (key, val)

    with h5py.File (chkfile, 'a') as fh5:
        if 'mol' not in fh5:
            fh5['mol'] = mc.mol.dumps()
        elif overwrite_mol:
            del (fh5['mol'])
            fh5['mol'] = mc.mol.dumps()
        if method_key in fh5:
            del (fh5[method_key])
        chkdata = fh5.create_group (method_key)

        for key, val in data.items (): chkdata[key] = val
        # special handling for ragged CI vector
        for i, cii in enumerate (ci):
            chkdata_ci_i = chkdata.create_group ('ci/'+str(i))
            for j, ciij in enumerate (cii):
                chkdata_ci_i[str(j)] = ciij
        # special handling for ragged frags_orbs
        if getattr (mc, 'frags_orbs', None) is not None:
            chkdata_frags_orbs = chkdata.create_group ('frags_orbs')
            for i, frag_orbs in enumerate (mc.frags_orbs):
                chkdata_frags_orbs[str(i)] = frag_orbs
        # if mo_coeff has tagged orbsym, save it, in case someone decides
        # to change PySCF symmetry convention again
        if getattr (mo_coeff, 'orbsym', None) is not None:
            chkdata['mo_coeff_orbsym'] = mo_coeff.orbsym
    return mc


if __name__=='__main__':
    from pyscf import gto, scf, mcscf
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
    xyz='''Li 0 0 0,
           H 2 0 0,
           Li 10 0 0,
           H 12 0 0'''
    mol = gto.M (atom=xyz, basis='6-31g', symmetry=False, output='chkfile.log')
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (2,2), (2,2))
    mc = mcscf.CASSCF (mf, 4, 4).run ()
    mo = las.localize_init_guess (([0,1],[2,3]), mc.mo_coeff, freeze_cas_spaces=True)
    las.kernel (mo)
    dump_las (las, chkfile='chkfile.chk')
    load_las_(las, 'chkfile.chk')



