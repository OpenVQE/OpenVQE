from pyscf.mcscf import casci
from mrh.my_pyscf.scf import hf_as

class CASCI (casci.CASCI)

    def __init__(self, mf, ncas, nelecas, ncore=None):
        assert (isinstance (mf, hf_as.RHF))
        casci.CASCI.__init__(self, mf, ncas, nelecas, ncore)
        self.max_cycle = 50

    def kernel (self, mo_coeff=None, ci0=None, tol=1e-10, conv_tol_ddm=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci
        if conv_tol_ddm is None:
            conv_tol_ddm = numpy.sqrt(conv_tol)
            logger.info(mf, 'Set density matrix conv threshold to %g', conv_tol_grad)


        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore
        nocc = ncore + ncas
        scf_conv = False
        cycle = 0
        dm1_old = 0
        e_old = 0 
        dm2_old = 0
        ci = ci0

        while not scf_conv and cycle < max (1, self.max_cycle):
            e_totcasci, e_cas, ci, mo_coeff, mo_energy = casci.CASCI.kernel (self, mo_coeff=mo_coeff, ci0=ci)
            casdm1, casdm2 = self.fcisolver.make_rdm12 (ci, self.ncas, self.nelecas)

            dm1 = self.make_rdm1 (mo_coeff=mo_coeff, ci=ci)
            self._scf.build_frozen_from_mo (mo_coeff, ncore, ncas, casdm1, casdm2)
            scf_conv, e_scf, mo_energy, mo_coeff, mo_occ = self.kernel (dm0=dm1)
            dm1 = self._scf.make_rdm1 (mo_coeff=mo_coeff)

            assert ((e_totcasci - e_scf) > -1e-8), "e_casci = {0}; e_hf_as = {1}; diff = {2}".format (e_totcasci, e_scf, e_totcasci - e_scf)

            norm_ddm1 = np.linalg.norm (dm1 - dm1_old)
            dm1_old = dm1

            de = e_scf - e_old
            e_old = e_scf

            norm_ddm2 = numpy.linalg.norm (casdm2 - dm2_old)
            dm2_old = casdm2            

            logger.info(self, 'cycle= %d E= %.15g  delta_E= %4.3g  |ddm1|= %4.3g  |ddm2|= %4.3g',
                    cycle+1, e_scf, de, norm_ddm1, norm_ddm2)

            if abs (de) < tol and norm_ddm1 < conv_tol_ddm and norm_ddm2 < conv_tol_ddm:
                scf_conv = True

        # One last casci
        e_tot, e_cas, ci, mo_coeff, mo_energy = casci.CASCI.kernel (self, mo_coeff=mo_coeff, ci0=ci0
        return e_tot, e_cas, ci, mo_coeff, mo_energy, scf_conv


