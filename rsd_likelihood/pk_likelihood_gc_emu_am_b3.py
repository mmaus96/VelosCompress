import numpy as np
import time
import json
import yaml

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval

from scipy.special import spherical_jn
from scipy.integrate import simps

from taylor_approximation import taylor_approximate
# from compute_sigma8_class import Compute_Sigma8

# from make_pkclass import make_pkclass
from copy import deepcopy

# Class to have a shape-fit likelihood for a bunch of pieces of data from both galactic caps in the same z bin
# Currently assumes all data have the same fiducial cosmology etc.
# If not I suggest changing the theory class so that instead of being labelled by "zstr" it gets labelled by sample name.
# And each sample name indexing the fiducial cosmology numbers (chi, Hz etc) in dictionaries. For another time...

class PkLikelihood(Likelihood):
    
    zfid: float
    sig8: float
    
    basedir: str
    
    fs_sample_names: list

    # optimize: turn this on when optimizng/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool 

    fs_datfns: list

    covfn: str
    
    fs_kmins: list
    fs_mmaxs: list
    fs_qmaxs: list
    fs_matMfns: list
    fs_matWfns: list
    
    kmax_spline: list
    
    cov_fac: float


    def initialize(self):
        """Sets up the class."""
        # Redshift Label for theory classes
        self.zstr = "%.2f" %(self.zfid)
        print(self.fs_sample_names,self.fs_datfns)

	# Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute
        self.linear_param_dict = yaml.load(open(self.basedir+self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict)
        
        print("We are here!")
        # self.sig8=sig8
        
        self.pconv = {}
        self.xith = {}
        
        self.sp_kmax = {}
        
        for ll, fs_sample_name in enumerate(self.fs_sample_names):
            self.sp_kmax[fs_sample_name] = self.kmax_spline[ll]
         
        
        self.loadData()
        #

    def get_requirements(self):
        
        req = {'taylor_pk_ell_mod': None,\
               'f_sig8': None,\
               'apar': None,\
               'aperp': None,\
              'm': None}
        
        for fs_sample_name in self.fs_sample_names:
            req_bias = { \
                   'bsig8_' + fs_sample_name: None,\
                   'b2_' + fs_sample_name: None,\
                   'bs_' + fs_sample_name: None,\
                   'b3_' + fs_sample_name: None,\
                  # 'alpha0_' + fs_sample_name: None,\
                  #'alpha2_' + fs_sample_name: None,\
                  # 'SN0_' + fs_sample_name: None,\
                   #'SN2_' + fs_sample_name: None\
                   }
            req = {**req, **req_bias}

        return(req)

    def full_predict(self, thetas=None):
        
        thy_obs = []

        if thetas is None:
            thetas = self.linear_param_means
        
        for fs_sample_name in self.fs_sample_names:
            fs_thy  = self.fs_predict(fs_sample_name,thetas=thetas)
            fs_obs  = self.fs_observe(fs_thy, fs_sample_name)
            thy_obs = np.concatenate( (thy_obs,fs_obs) )
        
        return thy_obs
    
    def logp(self,**params_values):
        """Return a log-likelihood."""

        # Compute the theory prediction with lin. params. at prior mean
        #t1 = time.time()
        thy_obs_0 = self.full_predict()
        self.Delta = self.dd - thy_obs_0
        #t2 = time.time()
        
        # Now compute template
        self.templates = []
        for param in self.linear_param_dict.keys():
            thetas = self.linear_param_means.copy()
            thetas[param] += 1.0
            self.templates += [ self.full_predict(thetas=thetas) - thy_obs_0 ]
        
        self.templates = np.array(self.templates)
        #t3 = time.time()
        
        # Make dot products
        self.Va = np.dot(np.dot(self.templates, self.cinv), self.Delta)
        self.Lab = np.dot(np.dot(self.templates, self.cinv), self.templates.T) + self.include_priors * np.diag(1./self.linear_param_stds**2)
        #self.Va = np.einsum('ij,jk,k', self.templates, self.cinv, self.Delta)
        #self.Lab = np.einsum('ij,jk,lk', self.templates, self.cinv, self.templates) + np.diag(1./self.linear_param_stds**2)
        self.Lab_inv = np.linalg.inv(self.Lab)
        #t4 = time.time()
        
        # Compute the modified chi2
        lnL  = -0.5 * np.dot(self.Delta,np.dot(self.cinv,self.Delta)) # this is the "bare" lnL
        lnL +=  0.5 * np.dot(self.Va, np.dot(self.Lab_inv, self.Va)) # improvement in chi2 due to changing linear params
        if not self.optimize:
            lnL += - 0.5 * np.log( np.linalg.det(self.Lab) ) + 0.5 * self.Nlin * np.log(2*np.pi) # volume factor from the determinant
        
        #t5 = time.time()
        
        #print(t2-t1, t3-t2, t4-t3, t5-t4)
        
        return lnL
    
    def get_best_fit(self):
        try:
            self.p0_nl  = self.dd - self.Delta
            self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
            self.p0_lin = np.einsum('i,il', self.bf_thetas, self.templates)
            return self.p0_nl + self.p0_lin
        except:
            print("Make sure to first compute the posterior.")
        
    def loadData(self):
        """
        Loads the required data.
        
        Do this in two steps... first load full shape data then xirecon, concatenate after.
        
        The covariance is assumed to already be joint in the concatenated format.
        
        """
        # First load the data
        
        self.kdats = {}
        self.p0dats = {}
        self.p2dats = {}
        self.fitiis = {}
        
        for ii, fs_datfn in enumerate(self.fs_datfns):
            fs_sample_name = self.fs_sample_names[ii]
            fs_dat = np.loadtxt(self.basedir+fs_datfn)
            self.kdats[fs_sample_name] = fs_dat[:,0]
            self.p0dats[fs_sample_name] = fs_dat[:,1]
            self.p2dats[fs_sample_name] = fs_dat[:,2]
            
            # Make a list of indices for the monopole and quadrupole only in Fourier space
            # This is specified to each sample in case the k's are different.
            yeses = self.kdats[fs_sample_name] > 0
            nos   = self.kdats[fs_sample_name] < 0
            self.fitiis[fs_sample_name] = np.concatenate( (yeses, nos, yeses, nos, nos ) )
        

        
        # Join the data vectors together
        self.dd = []        
        for fs_sample_name in self.fs_sample_names:
            self.dd = np.concatenate( (self.dd, self.p0dats[fs_sample_name], self.p2dats[fs_sample_name]) )

        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.basedir+self.covfn)/self.cov_fac
        
        # We're only going to want some of the entries in computing chi^2.
        
        # this is going to tell us how many indices to skip to get to the nth multipole
        startii = 0
        
        for ss, fs_sample_name in enumerate(self.fs_sample_names):
            
            kcut = (self.kdats[fs_sample_name] > self.fs_mmaxs[ss])\
                          | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:     # FS Monopole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
            kcut = (self.kdats[fs_sample_name] > self.fs_qmaxs[ss])\
                       | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
        
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        #print(self.sample_name, np.diag(self.cinv)[:10])
        
        # Finally load the window function matrix.
        self.matMs = {}
        self.matWs = {}
        for ii, fs_sample_name in enumerate(self.fs_sample_names):
            self.matMs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matMfns[ii])
            self.matWs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matWfns[ii])
        
        #
        
    def combine_bias_terms_pkell(self,bvec, p0ktable, p2ktable, p4ktable):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        
        # corr = 1.011
        
        return p0, p2, p4
    
    def fs_predict(self, fs_sample_name,thetas=None):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        
        taylorPTs = pp.get_result('taylor_pk_ell_mod')
        kv, p0ktable, p2ktable, p4ktable = taylorPTs[self.zstr]
        
        # pkclass = make_pkclass(self.zfid)

        #
        sig8 = self.sig8
        # sig8 = pp.get_param('sigma8')
        #sig8 = pp.get_result('sigma8')
        b1   = pp.get_param('bsig8_' + fs_sample_name)/sig8 - 1
        b2   = pp.get_param('b2_' + fs_sample_name)
        bs   = pp.get_param('bs_' + fs_sample_name)
        b3   = pp.get_param('b3_' + fs_sample_name)
        # print(type(sig8))


       # Instead of calling the linear parameters directly we will now analytically marginalize over them
        
        if thetas is None:
            alp0 = self.linear_param_means['alpha0_' + fs_sample_name]
            alp2 = self.linear_param_means['alpha2_' + fs_sample_name]
            sn0 = self.linear_param_means['SN0_' + fs_sample_name]
            sn2 = self.linear_param_means['SN2_' + fs_sample_name]
        else:
            alp0 = thetas['alpha0_' + fs_sample_name]
            alp2 = thetas['alpha2_' + fs_sample_name]
            sn0 = thetas['SN0_' + fs_sample_name]
            sn2 = thetas['SN2_' + fs_sample_name]
 
        bias = [b1, b2, bs, b3]
        cterm = [alp0,alp2,0,0]
        stoch = [sn0, sn2, 0]
        bvec = bias + cterm + stoch
        self.bvec = bvec
        
        #print(self.zstr, b1, sig8)
        
        p0, p2, p4 = self.combine_bias_terms_pkell(bvec, p0ktable, p2ktable, p4ktable)
        # print('P0 shape: ',np.shape(p0))
        
        #np.savetxt('pells_' + self.zstr + '_' + self.sample_name + '.txt',[kv,p0,p2,p4])
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],kv),np.append([0.0,],p0)
        p2 = np.append([0.,],p2)
        p4 = np.append([0.0,],p4)
        tt    = np.array([kv,p0,p2,p4]).T
        # print(type(tt))
        
        if np.any(np.isnan(tt)):
            f_sig8 = self.provider.get_param('f_sig8')
            apar = self.provider.get_param('apar')
            aperp = self.provider.get_param('aperp')
            m = self.provider.get_param('m')
            print("NaN's encountered in PREDICT. Parameter values are: {},{},{},{}".format(f_sig8,apar,aperp,m))
        
        return(tt)
        #

    def fs_observe(self,tt,fs_sample_name):
        """Apply the window function matrix to get the binned prediction."""
        
        
        
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        # kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        maxk = self.sp_kmax[fs_sample_name]
        kv  = np.linspace(0.0,maxk,int(maxk/0.001),endpoint=False) + 0.0005
        # kv  = self.kdats[fs_sample_name] #np.linspace(0.0,0.5,100,endpoint=False) 
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        
        if np.any(np.isnan(thy)) or np.max(thy) > 1e8:
            f_sig8 = self.provider.get_param('f_sig8')
            apar = self.provider.get_param('apar')
            aperp = self.provider.get_param('aperp')
            m = self.provider.get_param('m')
            plist = [f_sig8,apar,aperp,m] + self.bvec
            print("NaN's encountered in OBSERVE. Parameter values are: {}".format(plist))
        
        # wide angle
        expanded_model = np.matmul(self.matMs[fs_sample_name], thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        # Multiply by ad-hoc factor
        convolved_model = np.matmul(self.matWs[fs_sample_name], expanded_model )
        
        # keep only the monopole and quadrupole
        convolved_model = convolved_model[self.fitiis[fs_sample_name]]
        
        # Save the model:
        self.pconv[fs_sample_name] = convolved_model
    
        return convolved_model


class Taylor_pk_theory_zs(Theory):
    """
    A class to return a set of derivatives for the Taylor series of Pkell.
    """
    zfids: list
    pk_filenames: list
    s8_filename: str
    basedir: str
    
    def initialize(self):
        """Sets up the class by loading the derivative matrices."""
        
        print("Loading Taylor series.")
        
        # Load sigma8
        # self.compute_sigma8 = Compute_Sigma8(self.basedir + self.s8_filename)
        
        # Load clustering
        self.taylors_pk = {}
        self.taylors_xi = {}
        
        for zfid, pk_filename in zip(self.zfids, self.pk_filenames):
            zstr = "%.2f"%(zfid)
            taylors_pk = {}
            
            # Load the power spectrum derivatives
            json_file = open(self.basedir+pk_filename, 'r')
            emu = json.load( json_file )
            json_file.close()
            
            x0s = emu['x0']
            kvec = emu['kvec']
            derivs_p0 = [np.array(ll) for ll in emu['derivs0']]
            derivs_p2 = [np.array(ll) for ll in emu['derivs2']]
            derivs_p4 = [np.array(ll) for ll in emu['derivs4']]
            
            taylors_pk['x0'] = np.array(x0s)
            taylors_pk['kvec'] = np.array(kvec)
            taylors_pk['derivs_p0'] = derivs_p0
            taylors_pk['derivs_p2'] = derivs_p2
            taylors_pk['derivs_p4'] = derivs_p4


            self.taylors_pk[zstr] = taylors_pk
            
            del emu
    
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zmax = max(self.zfids)
        zg  = np.linspace(0,zmax,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {\
               'f_sig8': None,\
               'apar': None,\
               'aperp': None,\
               'm': None
              }
        
        return(req)
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        # return ['taylor_pk_ell_mod','taylor_xi_ell_mod']
        return ['taylor_pk_ell_mod']
    
    
    def get_can_provide_params(self):
        return ['f_sig8']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider
        
        f_sig8 = pp.get_param('f_sig8')
        apar = pp.get_param('apar')
        aperp = pp.get_param('aperp')
        m = pp.get_param('m')
        #sig8 = pp.get_param('sigma8')
        # OmM = pp.get_param('omegam')
        # sig8 = self.compute_sigma8.compute_sigma8(OmM,hub,logA)
        cosmopars = [f_sig8, apar, aperp,m]
        
        ptables = {}
        
        for zfid in self.zfids:
            zstr = "%.2f" %(zfid)
            
            # Load pktables
            x0s = self.taylors_pk[zstr]['x0']
            derivs0 = self.taylors_pk[zstr]['derivs_p0']
            derivs2 = self.taylors_pk[zstr]['derivs_p2']
            derivs4 = self.taylors_pk[zstr]['derivs_p4']
            
            kv = self.taylors_pk[zstr]['kvec']
            p0ktable = taylor_approximate(cosmopars, x0s, derivs0, order=3)
            p2ktable = taylor_approximate(cosmopars, x0s, derivs2, order=3)
            p4ktable = taylor_approximate(cosmopars, x0s, derivs4, order=3)
            
            ptables[zstr] = (kv, p0ktable, p2ktable, p4ktable)
            
        #state['sigma8'] = sig8
        state['derived'] = {'f_sig8': f_sig8}
        state['taylor_pk_ell_mod'] = ptables
