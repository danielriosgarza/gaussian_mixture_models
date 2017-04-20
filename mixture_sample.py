from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import useful_functions as uf



def retrieve_sufficient_statistics_Gaussian_mix(X):
    n = len(X)
    d = len(X[0])
    p1 = np.zeros((d,d))
    for i in xrange(n):
        b= X[i].copy()
        b.shape=(d,1)
        p1+= b.dot(b.T)
    return p1



def initial_assigment(n, pvals, K):
    ki_dict={i:np.zeros(n, dtype=np.bool) for i in xrange(int(K))}
    ik_dict = {}
    indices =np.arange(n)
    np.random.shuffle(indices)
    group_assigments = np.cumsum(np.random.multinomial(n, pvals=pvals, size=1)[0])
    for k in xrange(len(group_assigments)):
        if k==0:
            ki_dict[k][indices[0:group_assigments[k]]]=1
            for i in indices[0:group_assigments[k]]:
                ik_dict[i]=k
        else:
            ki_dict[k][indices[group_assigments[k-1]:group_assigments[k]]]=1
            for i in indices[group_assigments[k-1]:group_assigments[k]]:
                ik_dict[i]=k
    return ki_dict, ik_dict

def store_chol_dict(ki_dict, X):
    chol_dict={}
    for i in xrange(len(ki_dict)):
        a = retrieve_sufficient_statistics_Gaussian_mix(X[ki_dict[i]])
        chol_dict[i] = np.linalg.cholesky(a)
    return chol_dict

def get_sufficient_statistics_from_chol(ss_dict, ki_dict, chol_dict, X, k):
    ind = ki_dict[k]
    n = sum(ki_dict[k])
    if n<3:
        return ss_dict[k]
    else:
            
        em = np.mean(X[ind], axis=0)
        sqn=math.sqrt(n)
        sm = uf.cholesky_r1_update(chol_dict[k], sqn*em, down=1)
        cpm = uf.chol_of_the_inverse(sm)
        chol_prec_m = uf.Wishart_rvs(df = n-1, S = cpm, chol=1)
        mu = uf.multivariate_Gaussian_rvs(em, sqn*chol_prec_m, chol=1)
        return {'cP_m' : chol_prec_m, 'n':n, 'mu':mu}
         
    
    
    

    
def assign_n_to_k(pi, ss_dict, K, data_point):
    p =np.array([uf.multivariate_Gaussian_pdf(data_point, ss_dict[k]['mu'], \
ss_dict[int(k)]['cP_m'], chol=1, log_form=1) for k in xrange(int(K))])
   
    p = p-max(p)
    p = exp(p)
    p = pi*p
    p = p/sum(p)
    return int(np.arange(K)[np.random.multinomial(1, p)==1])         
                

def update_sample_to_chol(ki_dict, chol_dict, i, k, X, remove=False):
    if remove:
        ki_dict[k][i]=0
        chol_dict[k] = uf.cholesky_r1_update(chol_dict[k], X[i], down=1)
    else:
        ki_dict[k][i]=1
        chol_dict[k] = uf.cholesky_r1_update(chol_dict[k], X[i], down=1)

    
def draw_pi_0(alpha, K, ki_dict):
    alpha_up=np.array([sum(ki_dict[k])+alpha/K for k in xrange(int(K))]) 
    return np.random.dirichlet(alpha_up)


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
    

ki_dict, ik_dict = initial_assigment(n, p, K)
chol_dict = store_chol_dict(ki_dict, X)


pi_0 = np.random.dirichlet(pi_0)
s={}
ss_dict = {i:get_sufficient_statistics_from_chol(s, ki_dict, chol_dict, X, i) for i in xrange(int(K))}



def update_k_assignment(ik_dict,  ki_dict, chol_dict, ss_dict, pi_0, K, X):
    for i in xrange(len(ik_dict)):
       
        current_assign = ik_dict[i]
        new_assign = assign_n_to_k(pi_0, ss_dict, K,X[i])
        if current_assign==new_assign:
            pass
        else:
            update_sample_to_chol(ki_dict, chol_dict, i, current_assign, X, remove=True)
            ss_dict[current_assign]=get_sufficient_statistics_from_chol(ss_dict, ki_dict, chol_dict, X, current_assign)
            ik_dict[i] = new_assign
            update_sample_to_chol(ki_dict, chol_dict, i, new_assign, X, remove=False)
            ss_dict[new_assign]=get_sufficient_statistics_from_chol(ss_dict, ki_dict, chol_dict, X, new_assign)


def Gibbs_sampler(t, ik_dict,  ki_dict, chol_dict, ss_dict, pi_0, alpha_0, K, X):
    pis = pi_0.copy()
    for i in xrange(t):
        update_k_assignment(ik_dict,  ki_dict, chol_dict, ss_dict, pis, K, X)
        pis= draw_pi_0(alpha_0, K, ki_dict)
    return pis

pg = Gibbs_sampler(40, ik_dict,  ki_dict, chol_dict, ss_dict, pi_0, alpha_0, K, X)
