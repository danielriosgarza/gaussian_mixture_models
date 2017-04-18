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
    updates the 'up_...' entries of the param_dict many of the features may be commented
    out if only a specific subset is needed'''
    
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
    
    param_dict['up_k_0']= k_n #relative precision of the mean
    param_dict['up_v_0']= v_n #relative precision of the variance
    param_dict['up_mu_0']= mu_n #model mean
    param_dict['up_Sigma_0']= Sigma_n #model covm
    param_dict['up_Prec_0'] = Prec_n #model precision matrix
    param_dict['chol_Sigma'] = chol_Sigma #Cholesky dec of the model covm
    param_dict['chol_Prec'] = chol_Prec #Cholesky dec of the model precision matrix
    param_dict['d'] = len(mu_n) #dimensions
    param_dict['n'] = SS['n'] #number of samples
    param_dict['E_mu'] = SS['E_mu'] #empirical mean for non-informative priors
    param_dict['S_m'] = SS['S_m'] #scatter matrix for non-informative priors
    param_dict['chol_S_m'], param_dict['chol_invS_m'], param_dict['invS_m'] = uf.inv_and_chol(SS['S_m'], chol_of_A=1, chol_of_invA=1) #Cholesky dec
                #of the inverse of the scatter matrix and the inverse of the scatter matrix.


def Gibbs_sample(param_dict, chol=False, non_inf = False):
    '''Draw a sample from the Normal-Wishart model.
    Based on Gelman et al.(2013, 3 ed., chapter 4).
    
    Parameters
    ---------
    param_dict: python-dict
    dictionary with sufficient statistics and parameters from a multivariate data-set,
    obtained through the functions 'make_param_dict' and 'update_param_dict'.
    
    chol: Boolean.
    Assume the the input matrices were provided in the form of the lower Cholesky decomposition.
    
    non_inf: Boolean.
    This models tends weight heavily the distance from the prior to the empirical mean.
    If a non-informative prior is used, it's advisable to use this option. The parametrization
    will correspond to the  multivariate Jeffreys prior density. 
    
    Output
    --------
    Returns a d-dimensional draw from the Normal-Wishart model. For multiple samples,
    use the function 'Gibbs_sampler' with one of options:
    ('full'): chol=False, non_inf = False
    ('nonInfFull'): chol=False, non_inf = True
    ('chol'): chol=True, non_inf = False
    ('nonInfChol'): chol=True, non_inf = True'''
    
    if chol and non_inf:#sample from a non-informative prior, using the Cholesky decomposition
        Chol_prec_m = uf.Wishart_rvs(df = param_dict['n']-1., S = param_dict['chol_invS_m'], chol=1)
        mu = uf.multivariate_Gaussian_rvs(param_dict['E_mu'], sqrt((param_dict['n']))*Chol_prec_m, chol=1)
        return uf.multivariate_Gaussian_rvs(mu, Chol_prec_m, chol=1)
    
    elif chol and not non_inf: #sample from an informative prior using the Cholesky decomposition
        Chol_prec_m = uf.Wishart_rvs(param_dict['up_v_0'], S = param_dict['chol_Prec'], chol=1)
        mu = uf.multivariate_Gaussian_rvs(param_dict['up_mu_0'], sqrt((param_dict['up_k_0']))*Chol_prec_m, chol=1)
        return uf.multivariate_Gaussian_rvs(mu, Chol_prec_m, chol=1)
    
    elif non_inf and not chol: #sample from a non-informative prior
        prec_m = uf.Wishart_rvs(df=param_dict['n']-1, S=param_dict['invS_m'])
        mu = uf.multivariate_Gaussian_rvs(mu=param_dict['E_mu'], prec_m=(param_dict['n'])*prec_m)
        return uf.multivariate_Gaussian_rvs(mu, prec_m)
         
    else: #sample from an informative prior 
        
        prec_m = uf.Wishart_rvs(df=param_dict['up_v_0'], S=param_dict['up_Prec_0'])
        mu = uf.multivariate_Gaussian_rvs(mu=param_dict['up_mu_0'], prec_m=(param_dict['up_k_0'])*prec_m)
        return uf.multivariate_Gaussian_rvs(mu, prec_m)



def Gibbs_sample_slow_scipy_version(param_dict, non_inf=False):        
    '''Draw a sample from the Normal-Wishart model using the buit-in scipy distributions( significantly slower).
    Based on Gelman et al.(2013, 3 ed., chapter 4).  
    Parameters
    ---------
    param_dict: python-dict
    dictionary with sufficient statistics and parameters from a multivariate data-set,
    obtained through the functions 'make_param_dict' and 'update_param_dict'.
    
    non_inf: Boolean.
    This models tends weight heavily the distance from the prior to the empirical mean.
    If a non-informative prior is used, it's advisable to use this option. The parametrization
    will correspond to the  multivariate Jeffreys prior density. 
    
    Output
    --------
    Returns a d-dimensional draw from the Normal-Wishart model. For multiple samples,
    use the function 'Gibbs_sampler' with one of the options:
    ('slow'): non_inf = False
    ('nonInfSlow'): non_inf = True
    '''
    if non_inf:
        Prec_m = sts.wishart(df = param_dict['n']-1., scale=param_dict['invS_m']).rvs()
        Sigma_m = np.linalg.inv(Prec_m)
        mu = sts.multivariate_normal(mean= param_dict['E_mu'], cov=(1./param_dict['n'])*Sigma_m).rvs()
        return sts.multivariate_normal(mean=mu, cov = Sigma_m).rvs()
        
        
    else:
                    
        Prec_m = sts.wishart(df = param_dict['up_v_0'], scale=param_dict['up_Prec_0']).rvs()
        Sigma_m = np.linalg.inv(Prec_m)
        mu = sts.multivariate_normal(mean= param_dict['up_mu_0'], cov=(1./param_dict['up_k_0'])*Sigma_m).rvs()
        return sts.multivariate_normal(mean=mu, cov = Sigma_m).rvs()


def collapsed_Gibbs_sampler(t, param_dict, chol=False, non_inf=False):
    '''Draw a sample from the Normal-Wishart model using a collpased Gibbs sampler.
     Based on Gelman et al.(2013, 3 ed., chapter 4) and Murphy (2007, eq. 240 for the non-informative prior).  
    
    Parameters
    ---------
    param_dict: python-dict
    dictionary with sufficient statistics and parameters from a multivariate data-set,
    obtained through the functions 'make_param_dict' and 'update_param_dict'.
    
    non_inf: Boolean.
    This models tends weight heavily the distance from the prior to the empirical mean.
    If a non-informative prior is used, it's advisable to use this option. The parametrization
    will correspond to the  multivariate Jeffreys prior density. 
    
    Output
    --------
    Returns t draws d-dimensional from the collapesed version of Normal-Wishart model. 
    use the function 'Gibbs_sampler' with one of options:
    ('ColpsedFull'): chol=False, non_inf = False
    ('ColpsedNonInfFull'): chol=False, non_inf = True
    ('ColpsedChol'): chol=True, non_inf = False
    ('ColpsedNonInfChol'): chol=True, non_inf = True'''
    
    if chol and non_inf:#sample from a non-informative prior, using the Cholesky decomposition
        df = param_dict['n']-param_dict['d']
        var_coeff = math.sqrt((param_dict['n']+1.)/(param_dict['n']*df))
        return uf.multivariate_t_rvs_chol(mu = param_dict['E_mu'], L = var_coeff*param_dict['chol_S_m'], df=df, n=t)
        
    elif chol and not non_inf: #sample from an informative prior using the Cholesky decomposition
        df = param_dict['up_v_0']-param_dict['d']+1.
        var_coeff = math.sqrt(param_dict['up_k_0']+1.)/(param_dict['up_k_0']*df)
        return uf.multivariate_t_rvs_chol(mu=param_dict['up_mu_0'], L= var_coeff*param_dict['chol_Sigma'], df=df, n=t)
    
    elif non_inf and not chol: #sample from a non-informative prior
        df = param_dict['n']-param_dict['d']
        var_coeff = (param_dict['n']+1.)/(param_dict['n']*df)
        return uf.multivariate_t_rvs(mu = param_dict['E_mu'], S = var_coeff*param_dict['S_m'], df=df, n=t)
        
    else: #sample from an informative prior 
        df = param_dict['up_v_0']-param_dict['d']+1.
        var_coeff = (param_dict['up_k_0']+1.)/(param_dict['up_k_0']*df)
        return uf.multivariate_t_rvs(param_dict['up_mu_0'], var_coeff*param_dict['up_Sigma_0'], df=df, n=t)





def Gibbs_sampler(param_dict, t, v='chol'):
    '''
    Gibbs_sample(param_dict, chol=False, non_inf = False)
    ('full'): chol=False, non_inf = False
    ('nonInfFull'): chol=False, non_inf = True
    ('chol'): chol=True, non_inf = False
    ('nonInfChol'): chol=True, non_inf = True
    
    Gibbs_sample_slow_scipy_version(param_dict, non_inf=False)
    ('slow'): non_inf = False
    ('nonInfSlow'): non_inf = True
    
    collapsed_Gibbs_sampler(t, param_dict, chol=False, non_inf=False)
    ('ColpsedFull'): chol=False, non_inf = False
    ('ColpsedNonInfFull'): chol=False, non_inf = True
    ('ColpsedChol'): chol=True, non_inf = False
    ('ColpsedNonInfChol'): chol=True, non_inf = True'''
    
    if v=='full':
        s_dict = {i: Gibbs_sample(param_dict, chol=0, non_inf=0) for i in xrange(t)}
        
    elif v=='nonInfFull':
        s_dict = {i: Gibbs_sample(param_dict, chol=0, non_inf=1) for i in xrange(t)}
    
    elif v== 'chol':
        s_dict = {i: Gibbs_sample(param_dict, chol=1, non_inf=0) for i in xrange(t)}
    
    elif v== 'nonInfChol':
        s_dict = {i: Gibbs_sample(param_dict, chol=1, non_inf=1) for i in xrange(t)}

    elif v=='slow':
        s_dict = {i: Gibbs_sample_slow_scipy_version(param_dict, non_inf=0) for i in xrange(t)}
    
    elif v=='nonInfSlow':
        s_dict = {i: Gibbs_sample_slow_scipy_version(param_dict, non_inf=1) for i in xrange(t)}


    elif v=='ColpsedFull':
        s = collapsed_Gibbs_sampler(t=t, param_dict=param_dict, chol=0, non_inf=0)
        s_dict={i:s[i] for i in xrange(t)}
    
    elif v=='ColpsedNonInfFull':
        s = collapsed_Gibbs_sampler(t=t, param_dict=param_dict, chol=0, non_inf=1)
        s_dict={i:s[i] for i in xrange(t)}

    elif v=='ColpsedChol':
        s = collapsed_Gibbs_sampler(t=t, param_dict=param_dict, chol=1, non_inf=0)
        s_dict={i:s[i] for i in xrange(t)}
    
    elif v=='ColpsedNonInfChol':
        s = collapsed_Gibbs_sampler(t=t, param_dict=param_dict, chol=1, non_inf=1)
        s_dict={i:s[i] for i in xrange(t)}

    return s_dict, np.array([s_dict[i] for i in xrange(t)])




d = 2
n=600
#A = np.random.randint(2,10)*np.random.rand(d,d)
#A = A.dot(A.T)
#A = np.diag(np.random.uniform(0,1000,d))
mu = np.random.uniform(0,1000,d)

A = np.array([[11, 0.5*(sqrt(11*2))],[0.5*(sqrt(11*2)), 2]])

#mu = np.array([2, 33])

X = np.random.multivariate_normal(mu, A, n)
param_dict = make_param_dict(k_0=1, v_0=-1, mu_0=np.zeros(d), Sigma_0=np.eye(d),d=d)

update_param_dict(X, param_dict, chol=1)

import time

t= time.time()






draws=5000

s1, s_d1 = Gibbs_sampler(param_dict, draws, v = 'full')
s2, s_d2 = Gibbs_sampler(param_dict, draws, v = 'nonInfFull')
s3, s_d3 = Gibbs_sampler(param_dict, draws, v = 'chol')
s4, s_d4 = Gibbs_sampler(param_dict, draws, v = 'nonInfChol')

s5, s_d5 = Gibbs_sampler(param_dict, draws, v = 'slow')
s6, s_d6 = Gibbs_sampler(param_dict, draws, v = 'nonInfSlow')

s7, s_d7 = Gibbs_sampler(param_dict, draws, v = 'ColpsedFull')
s8, s_d8 = Gibbs_sampler(param_dict, draws, v = 'ColpsedNonInfFull')
s9, s_d9 = Gibbs_sampler(param_dict, draws, v = 'ColpsedChol')
s10, s_d10 = Gibbs_sampler(param_dict, draws, v = 'ColpsedNonInfChol')




print time.time() -t


for i in xrange(len(X.T)):
    print var(s1.T[i]), '\t', var(s2.T[i]), '\t', var(s3.T[i]), '\t', var(s4.T[i]), '\t', var(X.T[i])

scatter(s2.T[0], s2.T[1], c='r', alpha=0.2);scatter(s3.T[0], s3.T[1], c='r', alpha=0.2); scatter(s1.T[0], s1.T[1], c='g', alpha=0.2);scatter(X.T[0], X.T[1])
#np.random.seed(666)
#s2_dict = Gibbs_sampler(param_dict, 5000, chol=0)

#t = time.time()

#s3_dict = Gibbs_sampler(param_dict, 500, v='slow')

#print time.time()-t
