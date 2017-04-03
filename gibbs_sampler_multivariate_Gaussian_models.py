# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 03:46:28 2017

@author: user
"""


from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import math

def get_correlation_matrix(covm):
    diag = np.diag(np.diag(covm)**-0.5)
    return np.dot(np.dot(diag, covm), diag)




def multivariate_t_distribution(x,mu,Sigma,df,d):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    '''
    Num = gamma(1. * (d+df)/2)
    Denom = ( gamma(1.*df/2) * pow(df*pi,1.*d/2) * pow(np.linalg.det(Sigma),1./2) * pow(1 + (1./df)*np.dot(np.dot((x - mu),np.linalg.inv(Sigma)), (x - mu)),1.* (d+df)/2))
    d = 1. * Num / Denom 
    return d
    
def multivariate_student_t(X, mu, Sigma, df):    
    #multivariate student T distribution

    [n,d] = X.shape
    Xm = X-mu
    V = df * Sigma
    V_inv = np.linalg.inv(V)
    (sign, logdet) = slogdet(np.pi * V)

    logz = -gamma(df/2.0 + d/2.0) + gamma(df/2.0) + 0.5*logdet
    logp = -0.5*(df+d)*np.log(1+ np.sum(np.dot(Xm,V_inv)*Xm,axis=1))

    logp = logp - logz            

    return logp

def log_multivariate_t_dist(x_vec, d, v, mu, precision_matrix):
    mu.shape= (d,1)
    x.shape = (d,1)
    c1 = 0.5*(v+d) 
    c2= x-mu
    p1 = gam_log(c1)
    p2 = gam_log(0.5*v)
    p3 = 0.5*d*log(v)
    p4 = 0.5*d*log(math.pi)
    p5 = 1./(2*np.linalg.slogdet(precision_matrix)[1])
    p6 = c1*(log(1+np.dot(c2.T,np.dot(precision_matrix, c2))))
    p=p1-p2+p3+p4+p5-p6
    return float(p[0])


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

def initial_assigment(mk_dict, N, K):
    indices =np.arange(N)
    np.random.shuffle(indices)
    pvals = np.array([1./K]*K)
    group_assigments = np.cumsum(np.random.multinomial(N, pvals=pvals, size=1)[0])
    for i in xrange(len(group_assigments)):
        if i==0:
            mk_dict[i] = indices[0:group_assigments[i]]
        else:
            mk_dict[i] = indices[group_assigments[i-1]:group_assigments[i]]



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


def one_k_param_update(param_dict, data_set, gaussian_dict, k):
    update_parameters(param_dict, data_set)
    covm_k = draw_covm(param_dict)
    mu_k = draw_mu(param_dict, covm_k)
    gaussian_dict[k] = sts.multivariate_normal(mean=mu_k, cov = covm_k)


def collapsed_one_k_param_update(param_dict, data_set, coll_param_dict,k):
    update_parameters(param_dict, data_set)
    coll_param_dict[k] = param_dict.copy()

def update_alpha(alpha_0, mk_dict, K):
    return np.array([alpha_0[i]+len(mk_dict[i]) for i in xrange(K)])

def draw_pi(alpha):
    return sts.dirichlet(alpha).rvs()[0]
    

def assign_to_nk(pi, gaussian_dict, K, data_point):
    p =np.array([gaussian_dict[i].logpdf(data_point) for i in xrange(K)])
    p = p-max(p)
    p = exp(p)
    p = pi*p
    p = p/sum(p)
    return int(np.arange(K)[np.random.multinomial(1, p)==1])

def update_mk_dict(data_set, pi, gaussian_dict, mk_dict, K, N):
    mk = {i:[] for i in xrange(K)}
    for i in xrange(N):
        mk[assign_to_nk(pi, gaussian_dict, K, data_set[i])].append(i)
    for i in xrange(K):
        mk_dict[i]=np.array(mk[i])


def coll_log_probability(data_point, coll_param_dict, k):
    D = coll_param_dict[k]['D']
    k_m = coll_param_dict[k]['up_k0']
    n_m = coll_param_dict[k]['up_n0']
    mu_m = coll_param_dict[k]['up_mu_0']
    df = n_m-D+1
    covm = coll_param_dict[k]['up_covm_0']
    pcovm = ((k_m+1.)/(k_m*df))*covm
    prec_m = np.linalg.inv(pcovm)
    return multivariate_t_distribution(data_point, mu_m, pcovm, df,D)

def coll_assign_to_nk(pi, coll_param_dict, K, data_point):
    p =np.array([coll_log_probability(data_point, coll_param_dict, i) for i in xrange(K)])
    #p = p-max(p)
    # p = exp(p)
    p = pi*p
    p = p/sum(p)
    return int(np.arange(K)[np.random.multinomial(1, p)==1])

def coll_update_mk_dict(data_set, pi, coll_param_dict, mk_dict, K, N):
    mk = {i:[] for i in xrange(K)}
    for i in xrange(N):
        mk[coll_assign_to_nk(pi, coll_param_dict, K, data_set[i])].append(i)
    for i in xrange(K):
        mk_dict[i]=np.array(mk[i])





def gibbs_sampler(number_of_runs, data_set, param_dict, mk_dict, gaussian_dict, alpha_0, K, N):
    run_control_dict = {'draws':[], 'pis':[], 'alphas' : []}
    for i in xrange(number_of_runs):
        alpha = update_alpha(alpha_0, mk_dict, K)
        run_control_dict['alphas'].append(alpha)
        pi = draw_pi(alpha)
        run_control_dict['pis'].append(pi)
        [one_k_param_update(param_dict, data_set[mk_dict[k]], gaussian_dict, k)  for k in xrange(K) if len(mk_dict[k])>0 ]
        run_control_dict['draws'].append([gaussian_dict[k].rvs() for i in xrange(k)])
        update_mk_dict(data_set, pi, gaussian_dict, mk_dict, K, N)
    return run_control_dict

def collapsed_gibbs_sampler(number_of_runs, data_set, param_dict, mk_dict, coll_param_dict, alpha_0, K, N):
    run_control_dict = { 'pis':[], 'alphas' : []}
    for i in xrange(number_of_runs):
        alpha = update_alpha(alpha_0, mk_dict, K)
        run_control_dict['alphas'].append(alpha)
        pi = draw_pi(alpha)
        run_control_dict['pis'].append(pi)
        [collapsed_one_k_param_update(param_dict, data_set[mk_dict[k]], coll_param_dict, k)  for k in xrange(K) if len(mk_dict[k])>0 ]
        coll_update_mk_dict(data_set, pi, coll_param_dict, mk_dict, K, N)
    return run_control_dict



#microbiome
f=file('big_table.txt')
sample_list = f.readline().replace('\n', '').split('\t')[1::]

bac_prof = {}

for i in f:
    a=i.replace('\n','').split('\t')
    v = a[1::]
    bac_prof[a[0]] = np.array(v, dtype=np.float)    

bac = bac_prof.keys()

for i in bac:
    bac_prof[i][bac_prof[i]<(1./100)]=0
    

bac_data_set = np.array([bac_prof[i] for i in bac])
s = np.sum(bac_data_set, axis=1)
bac_data_set = bac_data_set[s>0]
bac_data_set = bac_data_set.T
bac_data_set = np.array([bac_data_set[i]/sum(bac_data_set[i]) for i in xrange(len(bac_data_set))])

#example_data

#Constants

D = len(bac_data_set[0]) # dimension of the MVN
N = len(bac_data_set) # number of data points
K = 2
alpha = 4000 #concentration parameter for the Dirichlet disrtibution
#Bulding test data

s1 = np.array([[3, 0.2*(sqrt(3*1.5))], [0.2*(sqrt(3*1.5)), 1.5]])
s2 = np.array([[1, 0.9*(sqrt(1*6))], [0.9*(sqrt(1*6)), 6]])
s3 = np.array([[5, 0.4*(sqrt(5*2))], [0.4*(sqrt(5*2)), 2]])
s4 = np.array([[6, 0.1*(sqrt(6*1.5))], [0.1*(sqrt(6*1.5)), 1.5]])

mu_1 = np.array([10,4])
mu_2 = np.array([2,-9])
mu_3 = np.array([-7,5])
mu_4 = np.array([18,-1])


true_pi = np.array([0.2,0.1,0.5, 0.2])

samples_d = np.random.multinomial(N, true_pi, 1)[0]

ds1 = sts.multivariate_normal(mu_1,s1).rvs(samples_d[0])
ds2 = sts.multivariate_normal(mu_2,s2).rvs(samples_d[1])
ds3 = sts.multivariate_normal(mu_3,s3).rvs(samples_d[2])
ds4 = sts.multivariate_normal(mu_4,s4).rvs(samples_d[3])


data_set = np.concatenate([ds1, ds2, ds3,ds4])
np.random.shuffle(data_set)
np.random.shuffle(data_set)

param_dict = {'D' : D, 'prior_mu':np.zeros(D), 'prior_k0':D, 'prior_covm':np.eye(D), 'prior_n0':D,'up_mu_0':np.zeros(D), 'up_k0':1, 'up_covm_0':np.eye(D), 'up_n0':1 }
coll_param_dict = {i: param_dict for i in xrange(K)}

alpha_0 = np.array([alpha/K]*K)

mk_dict = {}
gaussian_dict={}

initial_assigment(mk_dict, N, K)

print mk_dict

rcd = gibbs_sampler(2000, bac_data_set, param_dict, mk_dict, gaussian_dict, alpha_0, K, N)
