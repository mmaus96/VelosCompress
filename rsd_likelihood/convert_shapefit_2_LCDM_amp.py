import numpy as np
from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from make_pkclass import make_pkclass
from EH98_funcs import*
from getdist.mcsamples    import loadMCSamples
from classy import Class
from scipy import interpolate, linalg
from copy import deepcopy
import scipy.constants as conts
# from pnw_dst import pnw_dst

class shapefit_2_LCDM(Likelihood):
    zfid: float
    # covfn: str
    chains_fn: str
    basedir: str
    # fid_fn: str
    shapefit: bool
    unpack_chains: bool
    ext_prior: bool
    prior_cov_fn: str
    prior_means: list
    # par_means: list
    
    def initialize(self):
        """Sets up the class."""
        
        self.loadData()
        
    def get_requirements(self):
        
        req = {'theory_pars': None,\
               'H0': None,\
               'omega_b': None,\
               'omega_cdm': None,\
               # 'sigma8': None,\
               'omegam': None,\
               'logA': None}
        return(req)
        
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        pp   = self.provider
        theory_pars = pp.get_result('theory_pars')
        diff = self.par_means - theory_pars
        
        chi2 = np.dot(diff,np.dot(self.cinv,diff))
        
        # print(theory_pars,chi2)
        if self.ext_prior:
            H0 = pp.get_param('H0')
            # logA = pp.get_param('logA')
            omega_b = pp.get_param('omega_b')
            omega_cdm = pp.get_param('omega_cdm')
            
            d_prop = np.array([omega_b,omega_cdm,H0])
            d_true = np.array(self.prior_means)
            diff_prior = d_prop-d_true
            
            chi2 += np.dot(diff_prior,np.dot(self.prior_cinv,diff_prior))
            print(d_prop,chi2)
        else:
            print(theory_pars,chi2)
            
        return(-0.5*chi2)
    
    def loadData(self):
        """
        Loads the required data (shapefit/template mcmc chains).
        
        Covariance matrix is computed fromt the chains.
        
        """
        
        if self.unpack_chains:
            samples = loadMCSamples(self.basedir + self.chains_fn,no_cache=True, \
                            settings={'ignore_rows':0.3, 'contours': [0.68, 0.95],\
                                     })

            p = samples.getParams().__dict__
            fsig8_dat = p['f_sig8']
            apar_dat = p['apar']
            aperp_dat = p['aperp']

            if self.shapefit:
                par_means = samples.getMeans()[:4]
                m_dat = p['m']
                chains_all = np.zeros((len(fsig8_dat),4))
                chains_all[:,0] = fsig8_dat
                chains_all[:,1] = apar_dat
                chains_all[:,2] = aperp_dat
                chains_all[:,3] = m_dat
            else:
                par_means = samples.getMeans()[:3]
                chains_all = np.zeros((len(fsig8_dat),3))
                chains_all[:,0] = fsig8_dat
                chains_all[:,1] = apar_dat
                chains_all[:,2] = aperp_dat
                
            print(par_means)    
            covmat = np.cov(chains_all, rowvar=False)
            self.cinv = np.linalg.inv(covmat)
            print(self.cinv)
            self.par_means = par_means
        else:
            # burntin = np.load(self.basedir + self.chains_fn)
            burntin = deepcopy(np.loadtxt(self.basedir + self.chains_fn))
            aperp = deepcopy(burntin[:, 0])
            apar = deepcopy(burntin[:, 1])
            fsig8 = deepcopy(burntin[:, 2])
            m = deepcopy(burntin[:, 3])
            chain = np.zeros(np.shape(burntin[:,:4]))
            chain[:,0] = fsig8
            chain[:,1] = apar
            chain[:,2] = aperp
            chain[:,3] = m

            cov_inv = np.linalg.inv(np.cov(chain, rowvar=False))
            self.cinv = deepcopy(cov_inv)
            self.par_means = np.mean(chain,axis=0)
            # dat = np.loadtxt(self.basedir + self.chains_fn,encoding='utf-8')
            # if self.shapefit:
            #     chains_all = dat[:,:4]
            #     par_means = dat[:4,4]
            # else:
            #     chains_all = dat[:,:3]
            #     par_means = dat[:3,3]
        if self.ext_prior:
            prior_cov = deepcopy(np.loadtxt(self.prior_cov_fn))
            # prior_mn = self.
            self.prior_cinv = np.linalg.inv(prior_cov)
        
        
class Compute_theory_pars(Theory):
    """
    A class to return a set of shapefit parameters from LCDM params using CLASS.
    """
    zfid: float
    shapefit: bool
    
    def initialize(self):
        
        #Fiducial Parameters
        kmpiv = 0.03
        speed_of_light = 2.99792458e5
        kvec = np.logspace(-2.0, 0.0, 300)
        zfid = self.zfid
        As_fid = np.exp(3.0364)/1e10
        fid_class = Class()
        
        fid_class.set({
        "A_s": As_fid,
        "n_s": float(0.9649),
        "H0": 100.0*float(0.6736),
        "omega_b": float(0.02237),
        "omega_cdm": float(0.1200),
        "N_ur": float(2.0328),
        "N_ncdm": int(1),
        "m_ncdm": float(0.06),
        "Omega_k": float(0.0),
        "output": "mPk",
        "P_k_max_1/Mpc": float(1.0),
        "z_max_pk": zfid + 0.5
         })
        
        
        fid_class.compute()
        
        
        # fid_class = make_pkclass(zfid)
        h_fid = fid_class.h()
        Hz_fid = fid_class.Hubble(zfid) * conts.c/1000.0
        Chiz_fid = fid_class.angular_distance(zfid) * (1.+zfid)
        rd_fid = fid_class.rs_drag()
        pi_fid = np.array( [fid_class.pk_cb(k*h_fid, zfid ) * h_fid**3 for k in kvec] )
        
        fAmp_fid = fid_class.scale_independent_growth_factor_f(zfid)*np.sqrt(fid_class.pk_lin(kmpiv*h_fid,zfid)*h_fid**3)
        # Amp_fid = fid_class.scale_independent_growth_factor_f(zfid)*np.sqrt(fid_class.pk_lin(kmpiv*h_fid,zfid)*h_fid**3)
        
        transfer_fid = EH98(kvec, zfid, 1.0, cosmo=fid_class)*h_fid**3
        # transfer_fid = EH98_transfer(kvec, zfid, 1.0, cosmo = fid_class)
        # P_r_fid = Primordial(kvec*h_fid, As_fid, 0.9649, k_p = 0.05)
        # _,P_nw_fid = pnw_dst(kvec,pi_fid)
        # transfer_fid = P_nw_fid/P_r_fid
        
        fsigma8_fid = fid_class.scale_independent_growth_factor_f(zfid)*fid_class.sigma(8.0/fid_class.h(), zfid)
        # Pk_ratio_fid = EHpk_fid
        print(zfid, fsigma8_fid)
        
        self.fiducial_all = [h_fid, Hz_fid, rd_fid, Chiz_fid, fAmp_fid, transfer_fid, fsigma8_fid]
    
    def get_requirements(self):
        """What we need in order to provide theory params (fsig8, apar,aperp,m)."""
        
        req = {\
               'omega_b': None,\
               'omega_cdm': None,\
               'H0': None,\
               'logA': None,\
              }
        return(req)
    def get_can_provide(self):
        """What do we provide: an array of shapefit parameters."""
        return ['theory_pars']
    def get_can_provide_params(self):
        return ['omegam']
    
    def calculate(self, state, want_derived=False, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider
        
        h = pp.get_param('H0') / 100.
        logA = pp.get_param('logA')
        omega_b = pp.get_param('omega_b')
        omega_cdm = pp.get_param('omega_cdm')
        z = self.zfid
        kmpiv = 0.03
        kvec = np.logspace(-2.0, 0.0, 300)
        
        speed_of_light = 2.99792458e5

        # omega_b = 0.02237

        As =  np.exp(logA)*1e-10
        ns = 0.9649

        nnu = 1
        nur = 2.0328
        mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106

        # omega_c = (OmM - omega_b/h**2 - omega_nu/h**2) * h**2
        OmegaM = (omega_cdm + omega_b + omega_nu) / h**2

        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 1.,
            'z_max_pk': z + 0.5,
            # 'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            # 'omega_ncdm': omega_nu,
            'm_ncdm': mnu,
            'tau_reio': 0.0544,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_k': float(0.0)}

        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        # Caluclate AP parameters
        Hz = pkclass.Hubble(z) *conts.c/1000.0
        Chiz = pkclass.angular_distance(z) * (1.+z)
         
        rd = pkclass.rs_drag()
        pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in kvec] )
        
        #The fiducial values. 
        h_fid, Hz_fid, rd_fid, Chiz_fid, fAmp_fid, transfer_fid, fsigma8_fid = self.fiducial_all
        s = rd/rd_fid 
        f = pkclass.scale_independent_growth_factor_f(z)
        sigma8 = pkclass.sigma(s * 8.0 / pkclass.h(), z)
        # sigma8 = pkclass.sigma(8.0 / pkclass.h(), z)
        self.fsigma8_unsc = f*sigma8
        
        #Compute the ratio of fAmp in the Shapefit paper in order to find the corresponding fsigma8 parameter. 
        # theo_fAmp = pkclass.scale_independent_growth_factor_f(z)*np.sqrt(pkclass.pk_lin(kmpiv*h_fid*rd_fid/(rd),z)*(h*rd_fid/(rd))**3.)/fAmp_fid
        theo_fAmp = pkclass.scale_independent_growth_factor_f(z)*np.sqrt(pkclass.pk_lin(kmpiv*h_fid*rd_fid/(rd),z)*(h_fid*rd_fid/(rd))**3.)/fAmp_fid
        theo_aperp = (Chiz) / Chiz_fid / rd * rd_fid
        theo_apara = Hz_fid/ (Hz) / rd * rd_fid
        
        if self.shapefit:
            #Compute the transfer function with the EH98 formula. 
            transfer_new = EH98(kvec*h_fid*rd_fid/(rd*h), z ,1.0, cosmo=pkclass)*(rd_fid/rd)**3
            
            # P_r = Primordial(kvec*h*(rd_fid/rd), As, ns, k_p = 0.05)
            # _,P_nw = pnw_dst(kvec*h_fid*rd_fid/(rd*h),pi)
            # transfer_new = P_nw/P_r
        
            #Find the slope at the pivot scale.  
            ratio_transfer = slope_at_x(np.log(kvec), np.log(transfer_new/transfer_fid))
            theo_mslope = interpolate.interp1d(kvec, ratio_transfer, kind='cubic')(kmpiv)
            
            fAmp_mfac = np.exp(theo_mslope/(1.2) * np.tanh(0.6*np.log((rd_fid*h_fid)/(8.0*h)) ))
            
            #Find the model fsigma8, a_par, a_perp, and m.             
            theory_pars = [theo_fAmp*fsigma8_fid*fAmp_mfac, theo_apara, theo_aperp, theo_mslope]
        else:
            theory_pars = [ self.fsigma8_unsc, theo_apara, theo_aperp]
            
        state['theory_pars'] = theory_pars
        state['derived'] = {'omegam': OmegaM}