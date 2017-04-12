from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import useful_functions as uf

def make_param_dict(k_0, v_0, mu_0, Sigma_0,d):
    '''make a dictionary with the prior parameters for Gibbs sampling.
    updatated parameters will be the 'up_...' entries of the dictionary,
    they begin with the same values as the priors.
    
    Parameters
    --------
    k_0: float-like
    prior for the 'confidence' (pseudo-counts) on the prior mean.
    v_0: float-like
    prior for the 'confidence' (pseudo-counts) on the prior covariance matrix.
    mu_0: array-like
    prior for the mean. An n-dimensional vector representing the center of the distribution.
    Sigma_0: array like.
    Prior for the covariance matrix. An nxn positive definite matrix.

    Output
    --------
    Parameter dictionary with all the relevant parameters for the Normal Wishart model.
    '''
    
    return {'k_0': k_0, 'up_k_0': k_0 , 'v_0': v_0, 'up_v_0': v_0, 'mu_0':mu_0, 'up_mu_0':mu_0, \
    'Sigma_0':Sigma_0,  'up_Sigma_0':Sigma_0, 'd':d}


def update_param_dict(X, param_dict, chol=False):
    '''Update the parameter dictionary with sufficient statistics for the multivariate 
    Gaussian model, obtained from some data (assumed to be iid mv Gaussian).
    
    Parameters
    --------
    X: array-like
    Vector of observations assumed to be iid mvGaussian. len(X) should return the number of observations, and len(X[0]) 
    the dimensionallity. (shape = (n, d)). 
    param_dict: python-dict
    dictionary created by the function 'make_param_dict' with the relevant prior values.
    
    Output
    --------
    updates the 'up_...' entries of the param_dict and if chol, returns the Cholesky decomposition
    of the covariance matrix.'''
    
    SS = uf.store_sufficient_statistics_mvGaussian(X)
    k_n = param_dict['k_0']+SS['n']
    v_n = param_dict['v_0']+SS['n']
    
    mu_prec = SS['E_mu']-param_dict['mu_0']
    mu_prec.shape = (len(param_dict['mu_0']),1)
    mu_prec = mu_prec.dot(mu_prec.T)

    mu_n = ((param_dict['k_0']*param_dict['mu_0'])+(k_n*SS['E_mu']))/k_n
    Sigma_n = param_dict['Sigma_0'] + SS['S_m'] + ((param_dict['k_0']*SS['n'])/k_n)*mu_prec
    Prec_n = np.linalg.inv(Sigma_n)
    if chol:
        chky_Sigma = np.linalg.cholesky(Prec_n)
        param_dict['chky_Sigma'] = chky_Sigma
    param_dict['up_k_0']= k_n
    param_dict['up_v_0']= v_n
    param_dict['up_mu_0']= mu_n
    param_dict['up_Sigma_0']= Sigma_n
    param_dict['up_Prec_0'] = Prec_n


def Gibbs_sample(param_dict, chol=False):
    
    if chol:
        Chol_prec_m = uf.Wishart_rvs(param_dict['up_v_0'], param_dict['chky_Sigma'], chol=1)
        mu = uf.multivariate_Gaussian_rvs(param_dict['up_mu_0'], (param_dict['up_k_0'])*Chol_prec_m, chol=1)
        return uf.multivariate_Gaussian_rvs(mu, Chol_prec_m, chol=1)
    else:
        
        prec_m = uf.Wishart_rvs(df=param_dict['up_v_0'], S=param_dict['up_Prec_0'])
        mu = uf.multivariate_Gaussian_rvs(mu=param_dict['up_mu_0'], prec_m=(param_dict['up_k_0'])*prec_m)
        return uf.multivariate_Gaussian_rvs(mu, prec_m)
    #    prec_m = sts.invwishart(df=param_dict['up_v_0'], scale=param_dict['up_Prec_0']).rvs()
    #
    #    mu = sts.multivariate_normal(mean=param_dict['up_mu_0'], cov=(1./param_dict['up_k_0'])*prec_m).rvs()
    #    return sts.multivariate_normal(mean=mu, cov=prec_m).rvs()


def Gibbs_sampler(param_dict, t, chol=1):
    if chol:
        s_dict = {i: Gibbs_sample(param_dict, chol=1) for i in xrange(t)}
    else:
        s_dict = {i: Gibbs_sample(param_dict, chol=0) for i in xrange(t)}        
    return s_dict

d = 200
n=500
A = np.random.rand(d,d)
A = A.dot(A.T)
mu = np.random.uniform(0,1000,d)

#A = np.array([[11, 0.5*(sqrt(11*2))],[0.5*(sqrt(11*2)), 2]])

#mu = np.array([2, 33])

X = np.random.multivariate_normal(mu, A, n)
param_dict = make_param_dict(k_0=1, v_0=1, mu_0=np.zeros(d), Sigma_0=np.eye(d),d=d)

update_param_dict(X, param_dict, chol=1)

np.random.seed(666)
s_dict = Gibbs_sampler(param_dict, 500, chol=1)

np.random.seed(666)
s2_dict = Gibbs_sampler(param_dict, 500, chol=0)
