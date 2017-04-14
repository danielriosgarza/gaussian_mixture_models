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
    Vector of observations assumed to be iid mvGaussian. len(X) should return the number of observations,
    and len(X[0]) the dimensionallity. (shape = (n, d)). 
    param_dict: python-dict
    dictionary created by the function 'make_param_dict' with the relevant prior values.
    
    Output
    --------
    updates the 'up_...' entries of the param_dict and if chol, returns the Cholesky decomposition
    of the covariance matrix.'''
    
    #obtain sufficient statistics
    SS = uf.store_sufficient_statistics_mvGaussian(X)
    k_n = param_dict['k_0']+SS['n']
    v_n = param_dict['v_0']+SS['n']
    
    mu_prec = SS['E_mu']-param_dict['mu_0']
    mu_prec.shape = (len(param_dict['mu_0']),1)
    mu_prec = mu_prec.dot(mu_prec.T)

    mu_n = ((param_dict['k_0']*param_dict['mu_0'])+(k_n*SS['E_mu']))/k_n
    Sigma_n = param_dict['Sigma_0'] + SS['S_m'] + ((param_dict['k_0']*SS['n'])/k_n)*mu_prec
    
    #invert the precision matrix and take the Cholesky decomposition 
    if chol:
        chol_Sigma, chol_Prec, Prec_n = uf.inv_and_chol(Sigma_n, chol_of_A = 1, chol_of_invA=1)
        
    else:
        Prec_n =np.linalg.inv(Sigma_n)
    
    #update the parameters
    
    param_dict['up_k_0']= k_n
    param_dict['up_v_0']= v_n
    param_dict['up_mu_0']= mu_n
    param_dict['up_Sigma_0']= Sigma_n
    param_dict['up_Prec_0'] = Prec_n
    param_dict['chol_Sigma'] = chol_Sigma
    param_dict['chol_Prec'] = chol_Prec
    param_dict['d'] = len(mu_n)
    param_dict['n'] = SS['n']


def Gibbs_sample(param_dict, chol=False):
    
    if chol:
        Chol_prec_m = uf.Wishart_rvs(param_dict['up_v_0'], S = param_dict['chol_Prec'], chol=1)
        mu = uf.multivariate_Gaussian_rvs(param_dict['up_mu_0'], sqrt((param_dict['up_k_0']))*Chol_prec_m, chol=1)
        return uf.multivariate_Gaussian_rvs(mu, Chol_prec_m, chol=1)
    else:
        
        prec_m = uf.Wishart_rvs(df=param_dict['up_v_0'], S=param_dict['up_Prec_0'])
        mu = uf.multivariate_Gaussian_rvs(mu=param_dict['up_mu_0'], prec_m=(param_dict['up_k_0'])*prec_m)
        return uf.multivariate_Gaussian_rvs(mu, prec_m)
    #    prec_m = sts.invwishart(df=param_dict['up_v_0'], scale=param_dict['up_Prec_0']).rvs()
    #
    #    mu = sts.multivariate_normal(mean=param_dict['up_mu_0'], cov=(1./param_dict['up_k_0'])*prec_m).rvs()
    #    return sts.multivariate_normal(mean=mu, cov=prec_m).rvs()



def Gibbs_sample_slow_scipy_version(param_dict):
    Prec_m = sts.wishart(df = param_dict['up_v_0'], scale=param_dict['up_Prec_0']).rvs()
    Sigma_m = np.linalg.inv(Prec_m)
    mu = sts.multivariate_normal(mean= param_dict['up_mu_0'], cov=(1./param_dict['up_k_0'])*Sigma_m).rvs()
    return sts.multivariate_normal(mean=mu, cov = Sigma_m).rvs()


def collapsed_Gibbs_sampler(t, param_dict, chol=False):
    df = param_dict['up_v_0']-param_dict['d']+1.
    var_mul = (param_dict['up_k_0']+1.)/(param_dict['up_k_0']*df)
    if chol:
        return uf.multivariate_t_rvs_chol(param_dict['up_mu_0'], sqrt(var_mul)*param_dict['chol_Sigma'], df, n=t)
    else:
        return uf.multivariate_t_rvs(param_dict['up_mu_0'], var_mul*param_dict['up_Sigma_0'], df=df, n=t)


def Gibbs_sampler(param_dict, t, v='chol'):
    if v=='chol':
        s_dict = {i: Gibbs_sample(param_dict, chol=1) for i in xrange(t)}
    elif v=='full':
        s_dict = {i: Gibbs_sample(param_dict, chol=0) for i in xrange(t)}        
    elif v=='slow':
        s_dict = {i: Gibbs_sample_slow_scipy_version(param_dict) for i in xrange(t)}
    elif v=='coll_chol':
        s = collapsed_Gibbs_sampler(t,param_dict, chol=1)
        s_dict={i:s[i] for i in xrange(t)}
    elif v=='coll':
        s = collapsed_Gibbs_sampler(t,param_dict, chol=0)
        s_dict={i:s[i] for i in xrange(t)}
    return np.array([s_dict[i] for i in xrange(t)])



d = 20
n=500
A = np.random.rand(d,d)
A = A.dot(A.T)
mu = np.random.uniform(0,1000,d)

#A = np.array([[11, 0.5*(sqrt(11*2))],[0.5*(sqrt(11*2)), 2]])

#mu = np.array([2, 33])

X = np.random.multivariate_normal(mu, A, n)
param_dict = make_param_dict(k_0=1, v_0=1, mu_0=np.zeros(d), Sigma_0=np.eye(d),d=d)

update_param_dict(X, param_dict, chol=1)

import time

t= time.time()


s1 = Gibbs_sampler(param_dict, 500, v = 'coll')


s2 = Gibbs_sampler(param_dict, 500, v = 'coll_chol')

s3 = Gibbs_sampler(param_dict, 500, v = 'chol')

print time.time() -t

#np.random.seed(666)
#s2_dict = Gibbs_sampler(param_dict, 5000, chol=0)

#t = time.time()

#s3_dict = Gibbs_sampler(param_dict, 500, v='slow')

#print time.time()-t
