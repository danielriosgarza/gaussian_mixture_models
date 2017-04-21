from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import useful_functions as uf
import math

def initial_assigment(X, n, pvals, K):
    ki_dict={i:np.zeros(n, dtype=np.bool) for i in xrange(int(K))}
    ik_dict = {}
    im_dict={}
    indices =np.arange(n)
    np.random.shuffle(indices)
    group_assigments = np.cumsum(np.random.multinomial(n, pvals=pvals, size=1)[0])
    for k in xrange(len(group_assigments)):
        if k==0:
            ki_dict[k][indices[0:group_assigments[k]]]=1
            for i in indices[0:group_assigments[k]]:
                ik_dict[i]=k
                im_dict[i]=np.einsum('i,j',X[i],X[i])
        else:
            ki_dict[k][indices[group_assigments[k-1]:group_assigments[k]]]=1
            for i in indices[group_assigments[k-1]:group_assigments[k]]:
                ik_dict[i]=k
                im_dict[i]=np.einsum('i,j',X[i],X[i])
    return ki_dict, ik_dict, im_dict

def draw_Gaussian_parameters(ss_dict, K,pi_m):
    for k in xrange(int(K)):
        chol_prec_m = uf.Wishart_rvs(df = ss_dict[k]['n']-1., S = ss_dict[k]['invChol_sm'], chol=1)
        mu = uf.multivariate_Gaussian_rvs(ss_dict[k]['em'], math.sqrt(ss_dict[k]['n'])*chol_prec_m, chol=1)
        ss_dict[k]['ch_prec_m']=chol_prec_m
        ss_dict[k]['mean_factor'] = ss_dict[k]['n']*np.einsum('i,j',mu,mu)#mu*mu'
        ss_dict[k]['sum_factor'] =2*np.einsum('i,j',ss_dict[k]['sX'],mu)
        ss_dict[k]['joint_likelihood'] = uf.multivariate_Gaussian_likelihood_cached_suff_stat(ss_dict[k])+math.log(pi_m[k])
        ss_dict[k]['mu']=mu
        ss_dict[k]['joint_likelihood']+= uf.multivariate_Gaussian_pdf(ss_dict[k]['mu'], ss_dict[k]['em'],ss_dict[k]['ch_prec_m'], log_form=1, chol=1)
        ss_dict[k]['joint_likelihood']+=uf.wishart_pdf(ss_dict[k]['ch_prec_m'], ss_dict[k]['invChol_sm'],ss_dict[k]['n']-1,ss_dict[k]['d'], chol=1)
        
def get_initial_sufficient_statistics_dict(ki_dict, X, K, pi_m):
    ss_dict={k:uf.store_sufficient_statistics_for_Gaussian_likelihood(X[ki_dict[k]]) for k in xrange(int(K))}
    draw_Gaussian_parameters(ss_dict, K, pi_m)
    return ss_dict

def leave_one_out_likelihood(ss_dict, im_dict, ki_dict, X, K, k):
    k_ind = np.arange(K)
    k_ind=k_ind[k_ind!=k]
    l={}
    ind = np.arange(n)[ki_dict[k]]
    for i in xrange(len(ind)):
        cu_C_m = ss_dict[k]['C_m'].copy()
        cu_sX = ss_dict[k]['sX'].copy()
        cu_sf = ss_dict[k]['sum_factor'].copy()
        ss_dict[k]['n']-=1
        ss_dict[k]['C_m']-=im_dict[ind[i]]
        ss_dict[k]['sX']-=2*X[ind[i]]
        ss_dict[k]['sum_factor'] =2*np.einsum('i,j',ss_dict[k]['sX'],ss_dict[k]['mu'])
        l1 = uf.multivariate_Gaussian_likelihood_cached_suff_stat(ss_dict[k])
        l1+=uf.multivariate_Gaussian_pdf(ss_dict[k]['mu'],  ss_dict[k]['sX']/ ss_dict[k]['n'],ss_dict[k]['ch_prec_m'], log_form=1, chol=1)
        l1+=uf.wishart_pdf(ss_dict[k]['ch_prec_m'], ss_dict[k]['invChol_sm'],ss_dict[k]['n']-1,ss_dict[k]['d'], chol=1)
        l[ind[i]]=[ss_dict[k]['joint_likelihood']-l1]
        ss_dict[k]['C_m'] = cu_C_m.copy()
        ss_dict[k]['sX'] = cu_sX.copy()
        ss_dict[k]['sum_factor']  = cu_sf.copy()
        ss_dict[k]['n']+=1
        Lo = [add_one_likelihood(ss_dict, im_dict, X, ki, ind[i]) for ki in k_ind]
        l[ind[i]].extend(Lo) 
    return l
def add_one_likelihood(ss_dict, im_dict, X, k, i):
    cu_C_m = ss_dict[k]['C_m'].copy()
    cu_sX = ss_dict[k]['sX'].copy()
    cu_sf = ss_dict[k]['sum_factor'].copy()
    ss_dict[k]['n']+=1
    ss_dict[k]['C_m']+=im_dict[i]
    ss_dict[k]['sX']+=2*X[i]
    ss_dict[k]['sum_factor'] =2*np.einsum('i,j',ss_dict[k]['sX'],ss_dict[k]['mu'])
    l1 = uf.multivariate_Gaussian_likelihood_cached_suff_stat(ss_dict[k])
    l1+=uf.multivariate_Gaussian_pdf(ss_dict[k]['mu'],ss_dict[k]['sX']/ ss_dict[k]['n'],ss_dict[k]['ch_prec_m'], log_form=1, chol=1)
    l1+=uf.wishart_pdf(ss_dict[k]['ch_prec_m'], ss_dict[k]['invChol_sm'],ss_dict[k]['n']-1,ss_dict[k]['d'], chol=1)
    L = l1 - ss_dict[k]['joint_likelihood']
    ss_dict[k]['C_m'] = cu_C_m.copy()
    ss_dict[k]['sX'] = cu_sX.copy()
    ss_dict[k]['sum_factor']  = cu_sf.copy()
    ss_dict[k]['n']-=1  
    return L
d = 2 #indexed by j
n=600 #indexed by i
K=3. #indexed by k
alpha_0 = 10.

pi_0 = np.array([alpha_0/K for i in xrange(int(K))])


true_pi= np.array([300., 200.,100.])
p = true_pi/sum(true_pi)

A1 = np.array([[11, 0.5*(sqrt(11*2))],[0.5*(sqrt(11*2)), 2]])
A2 = np.array([[5, 0.9*(sqrt(5*12))],[0.9*(sqrt(5*12)), 12]])
A3 = np.array([[8, 0.1*(sqrt(8*7))],[0.1*(sqrt(8*7)), 7]])

mu1 = np.random.uniform(0,1000, d)
mu2 = np.random.uniform(0,1000, d)
mu3 = np.random.uniform(0,1000, d)

cvms = np.array([A1, A2, A3])
mus = np.array([mu1, mu2, mu3])

X = []

for i in xrange(n):
    ind = np.random.choice(np.arange(3), p=p)
    X.append(np.random.multivariate_normal(mean=mus[ind], cov=cvms[ind]))

X = np.array(X)
    
pi_m = np.random.dirichlet(pi_0)
ki_dict, ik_dict, im_dict = initial_assigment(X,n, p, K)
ss_dict = get_initial_sufficient_statistics_dict(ki_dict, X, K, pi_m)

