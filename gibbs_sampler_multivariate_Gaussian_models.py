from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts

def get_correlation_matrix(covm):
    diag = np.diag(np.diag(covm)**-0.5)
    return np.dot(np.dot(diag, covm), diag)


def log_multivariate_t_dist(x_vec, d, v, mu, precision_matrix):
    mu.shape= (d,1)
    x.shape = (d,1)
    c1 = 0.5*(v+d) 
    c2= x-mu
    p1 = gam_log(c1)
    p2 = gam_log(0.5*v)
    p3 = 0.5*d*log(v)
    p4 = 0.5*d*log(pi)
    p5 = 1./(2*np.linalg.slogdet(precision_matrix)[1])
    p6 = c1*(log(1+np.dot(c2.T,np.dot(precision_matrix, c2))))
    return p1-p2+p3+p4+p5-p6


def multivariate_Gaussian_rvs(mean, covariance_matrix, n):
    mean = np.array(mean)
    dim = len(mean)
    mean.shape =(dim,1)
    L = np.linalg.cholesky(covariance_matrix)
    z = np.random.multivariate_normal(np.zeros(dim).T, np.eye(dim), size=n)
    xa = np.array([np.dot(L, z[i]) for i in xrange(len(z))])
    return np.array([mean.T[0]+xa[i] for i in xrange(len(xa))])


    
def multivariate_t_rvs_a(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable


    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

        
            
                
                    
                            
def multivariate_t_rvs_b(mu,Sigma,N,M):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    N = degrees of freedom
    M = # of samples to produce
    '''
    d = len(Sigma)
    g = np.tile(np.random.gamma(N/2.,2./N,M),(d,1)).T
    Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
    return mu + Z/np.sqrt(g)



def scatter_matrix(data_set):
    ds = np.array(data_set)
    N = len(ds) #number of samples
    dim = len(ds[0]) #dimensions
    C = np.eye(N) - (1./N)*np.ones(N) #centering matrix
    return np.dot(np.dot(ds.T, C), ds)


def update_parameters(param_dict, data_set):
    ds = np.array(data_set)
    N = len(data_set) #number of samples
    D = len(data_set[0]) #dimensions of samples
    
    #prior_values
    k_0 = param_dict['prior_k0']
    mu_0 = param_dict['prior_mu']
    mu_0 = np.array(mu_0)
    mu_0.shape = (D,1)
    n_0 = param_dict['prior_n0']
    covm_0 = param_dict['prior_covm']
    
    k_m = k_0+N #mean prior plus number of samples
   
    empirical_mean = np.mean(ds,axis=0)
    empirical_mean.shape = (D,1)
    
    mu_m = (1./k_m)*((k_0*mu_0)+N*empirical_mean) #updated mean prior

    scat_m = scatter_matrix(ds)
    n_m = n_0+N #precision prior plus number of samples
    precision = np.dot((empirical_mean-mu_0), (empirical_mean-mu_0).T)
    
    covm_m = covm_0 + scat_m + ((k_0*N)/k_m)*precision   

    param_dict['up_mu_0'] = mu_m.T[0]
    param_dict['up_k0'] = k_m
    param_dict['up_covm_0'] = covm_m
    param_dict['up_n0'] = n_m

def draw_covm(param_dict):
    covm = sts.invwishart(df=param_dict['up_n0'], scale=param_dict['up_covm_0']).rvs()
    return covm #invwishart takes as input a covariance matrix and returns a covariance matrix

def draw_mu(param_dict, covm):
    return sts.multivariate_normal(mean=param_dict['up_mu_0'],cov=(1./param_dict['up_k0'])*covm).rvs()

def Gibbs_sampler(param_dict):
    covm = draw_covm(param_dict)
    mu = draw_mu(param_dict, covm)
    return sts.multivariate_normal(mean=mu, cov= covm).rvs()

def draw_from_collapsed_Gibbs_sampler(param_dict):
    D = param_dict['D']
    k_m = param_dict['up_k0']
    n_m = param_dict['up_n0']
    mu_m = param_dict['up_mu_0']
    df = n_m-D+1
    covm = param_dict['up_covm_0']
    pcovm = ((k_m+1.)/(k_m*df))*covm
    return multivariate_t_rvs_a(mu_m, pcovm, df,1)[0]
#example_data

#Constants

D = 2 # dimension of the MVN
N = 4000 # number of data points

#Bulding test data

#s1 = np.random.rand(D,D)*np.random.uniform(1,5)
#s1 = np.dot(s1, s1.T)

s1 = np.array([[3, 0.2*(sqrt(3*1.5))], [0.1*(sqrt(3*1.5)), 1.5]])

data_set = sts.multivariate_normal(np.zeros(D),s1).rvs(N)

param_dict = {'D' : D, 'prior_mu':np.zeros(D), 'prior_k0':1, 'prior_covm':np.eye(D), 'prior_n0':1,'up_mu_0':np.zeros(D), 'up_k0':1, 'up_covm_0':np.eye(D), 'up_n0':1 }


update_parameters(param_dict, data_set)

gibbs_sample = np.array([Gibbs_sampler(param_dict) for i in xrange(5000)])
collapsed_gibbs_sample = np.array([draw_from_collapsed_Gibbs_sampler(param_dict) for i in xrange(5000)])
