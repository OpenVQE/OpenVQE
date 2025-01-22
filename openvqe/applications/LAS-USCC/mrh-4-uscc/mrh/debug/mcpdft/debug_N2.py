import numpy as np
from scipy import linalg
from pyscf import gto, scf, mcscf, lib
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.tfnal_derivs import contract_fot

def N2_xyz (r=1.6):
    return 'N -{0:.6f} 0 0 ; N {0:.6f} 0 0'.format (abs (r)/2)

mol = gto.M (atom = N2_xyz(), basis='sto-3g', symmetry=False,
    output='debug_N2.log', verbose=lib.logger.DEBUG)
mf = scf.RHF (mol).run ()

# tLDA: 0.1841335132506145
# ftLDA: 0.18645076318377468
# tPBE: 0.21204047652310842

def test_molgrad (otxc):
    mc = mcpdft.CASSCF (mf, otxc, 6, 6).set (conv_tol=1e-10).run ().as_scanner ()
    mc_grad = mc.nuc_grad_method ().run ()
    de_an = (mc_grad.de[1,0]-mc_grad.de[0,0])/2
    de_num = (mc (N2_xyz (1.601)) - mc (N2_xyz (1.599)))*lib.param.BOHR/0.002
    print (otxc, "Numeric gradient:", de_num)
    print (otxc, "Analytical gradient:", de_an)
    print (otxc, "Gradient error:", de_an-de_num)

def test_densgrad (otxc):
    # throat-clearing
    mc = mcpdft.CASSCF (mf, otxc, 6, 6).set (conv_tol=1e-10).run ().as_scanner ()
    ncore, ncas = mc.ncore, mc.ncas
    nocc, nao = ncore + ncas, mc.mol.nao_nr ()
    mo_coeff = mc.mo_coeff
    mo_core = mc.mo_coeff[:,:ncore]
    mo_cas = mc.mo_coeff[:,ncore:nocc]
    ot = mc.otfnal
    if ot.grids.coords is None:
        ot.grids.build(with_non0tab=True)
    ngrids = ot.grids.coords.shape[0]
    ni = ot._numint
    dens_deriv = ot.dens_deriv
    def get_dens (d1, d2):
        make_rho, nset, nao = ni._gen_rho_evaluator (ot.mol, d1, 1)
        for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, nao, ot.dens_deriv, mc.max_memory, blksize=ngrids):
            rho = np.asarray ([make_rho (i, ao, mask, ot.xctype) for i in range(2)])
            Pi = get_ontop_pair_density (ot, rho, ao, d2, mo_cas, ot.dens_deriv, mask)
            return rho, Pi, weight

    # densities and potentials for the wave function
    dm1s, dm2s = mc.fcisolver.make_rdm12s (mc.ci, mc.ncas, mc.nelecas)
    dm1 = dm1s[0] + dm1s[1]
    dm2 = dm2s[0] + dm2s[1] + dm2s[1].transpose (2,3,0,1) + dm2s[2]
    dm2 -= np.multiply.outer (dm1, dm1)
    dm2 += np.multiply.outer (dm1s[0], dm1s[0]).transpose (0,3,2,1)
    dm2 += np.multiply.outer (dm1s[1], dm1s[1]).transpose (0,3,2,1)
    dm1s = [(mo_core @ mo_core.T) + (mo_cas @ d @ mo_cas.T) for d in dm1s]
    rho0, Pi0, weight = get_dens (dm1s, dm2)
    eot0, vot0, fot0 = ot.eval_ot (rho0, Pi0, dderiv=2, weights=weight)
    vrho0, vPi0 = vot0

    # Perturbed densities
    ddm1s = np.zeros ((2,nao,nao), dtype = dm1s[0].dtype)
    ddm1s[:,ncore:,:nocc] = 0.0001
    ddm1s += ddm1s.transpose (0,2,1)
    ddm1s = [mo_coeff @ d @ mo_coeff.T for d in ddm1s]
    ddm2 = np.ones_like (dm2) / 10000
    ddm2[:3,:,:,:] *= -1
    rhoD, PiD = get_dens (ddm1s, ddm2)[:2]

    def sector (rhoT, PiT):
        # Evaluate energy and potential at dens0 + densD = dens1
        rho1 = rho0 + rhoT
        Pi1 = Pi0 + PiT
        eot1 = ot.eval_ot (rho1, Pi1, weights=weight)[0]
        eotD = eot1 - eot0

        vot1 = contract_fot (ot, fot0, rho0.sum (0), Pi0, rhoT.sum (0), PiT)

        # Polynomial expansion
        if PiT.ndim == 1: PiT = PiT[None,:]
        nPi = min (PiT.shape[0], vPi0.shape[0])
        vrho = vrho0# + vot1[0]
        vPi = vPi0# + vot1[1]
        eotD_lin = (vrho*rhoT.sum (0)).sum (0) + (vPi[:nPi]*PiT[:nPi]).sum (0)
        E = np.dot (eot0, weight)
        dE_num = np.dot (eotD, weight)
        dE_lin = np.dot (eotD_lin, weight)
        eotD_quad = (vot1[0]*rhoT.sum (0)).sum (0) + (vot1[1][:nPi]*PiT[:nPi]).sum (0)
        dE_quad = np.dot (eotD_quad, weight) / 2 # Taylor series!!!
        return dE_num, dE_lin, dE_quad

    rhoT, PiT = np.zeros_like (rhoD), np.zeros_like (PiD)
    if rhoT.ndim > 2:
        rhoT[:,0,:] = rhoD[:,0,:]
    else:
        rhoT[:] = rhoD[:]
    num, lin, quad = sector (rhoT, PiT)
    rho_rat = (lin-num)/num
    rho2_rat = (lin+quad-num)/num

    rhoT[:] = PiT[:] = 0.0
    if PiT.ndim > 1:
        PiT[0,:] = PiD[0,:]
    else:
        PiT[:] = PiD[:]
    num, lin, quad = sector (rhoT, PiT)
    Pi_rat = (lin-num)/num
    Pi2_rat = (lin+quad-num)/num

    rhoT[:] = PiT[:] = 0.0
    if rhoT.ndim > 2:
        rhoT[:,1:4,:] = rhoD[:,1:4,:] 
        num, lin, quad = sector (rhoT, PiT)
        rhop_rat = (lin-num)/num
        rhop2_rat = (lin+quad-num)/num
    else:
        rhop_rat = rhop2_rat = 0.0

    rhoT[:] = PiT[:] = 0.0
    if PiT.ndim > 1 and vPi0.shape[0] > 1:
        PiT[1:4,:] = PiD[1:4,:] 
        num, lin, quad = sector (rhoT, PiT)
        Pip_rat = (lin-num)/num
        Pip2_rat = (lin+quad-num)/num
    else:
        Pip_rat = Pip2_rat = 0.0

    print (("{:>8s} " + ' '.join (['{:8.5f}',]*8)).format (otxc, rho_rat, rho2_rat, Pi_rat, Pi2_rat, rhop_rat, rhop2_rat, Pip_rat, Pip2_rat))

print ("Breakdown of on-top functional derivative error by sector and on-top functional type")
print ((' '.join (['{:>8s}',]*9)).format ('err','rho','rho2','Pi','Pi2','rhop','rhop2','Pip','Pip2'))
test_densgrad ('tLDA')
test_densgrad ('ftLDA')
test_densgrad ('tPBE')
test_densgrad ('ftPBE')

#test_molgrad ('tLDA')
#test_molgrad ('ftLDA')
#test_molgrad ('tPBE')
#test_molgrad ('ftPBE')

