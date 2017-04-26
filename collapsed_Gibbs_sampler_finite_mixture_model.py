from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import useful_functions as uf
import math
from scipy.special import gammaln as lggm
import scipy.linalg as linalg


def initial_assigment(X, pvals, K):
    
    N = len(X)
    nk_array = np.zeros(N, dtype=int)
    indices =np.arange(N)
    np.random.shuffle(indices)#start with a random assignment of indices to k
    group_assigments = np.cumsum(np.random.multinomial(N, pvals=pvals, size=1)[0])
    for k in xrange(len(group_assigments)):
        if k==0:
            nk_array[indices[0:group_assigments[k]]]=k
        else:
            nk_array[indices[group_assigments[k-1]:group_assigments[k]]]=k
    
    return nk_array 



def cache_sufficient_statistics_for_n(X, K, alpha_0, kappa_0, v_0, mu_0):
    N = X.shape[0]
    D = X.shape[1]
    
    #global constants
    k_coeff = K/(K*(N+alpha_0-1))
    alpha_coeff = alpha_0/(K*(N+alpha_0-1))
    
    Zconst = -0.5*D*math.log(pi)
    
    
    #indexed by n
    l_kappa_p_1 = np.array([-0.5*D*math.log(kappa_0+n+1) for n in xrange(N)])
    
    l_kappa = np.array([0.5*D*math.log(kappa_0+n) for n in xrange(N)])
    
    l = []
    for n in xrange(N):
        m = sum([lggm((v_0+n+2-i)/2.) for i in xrange(n)])
        l.append(m)
    l_gamma_s_2 = np.array(l)
    
    l = []
    for n in xrange(N):
        m = -sum([lggm((v_0+n+1-i)/2.) for i in xrange(n)])
        l.append(m)
    l_gamma_s_1 = np.array(l)

    v_n_p_1 = np.array([-0.5*(v_0+n+1) for n in xrange(N)])

    v_n = np.array([0.5*(v_0+n) for n in xrange(N)])

    mu_factor = np.array([((kappa_0+mu_0)/(kappa_0+n), (1./(kappa_0+n))) for n in xrange(N)])

    mu_0_factor = kappa_0*np.einsum('i,j->ij',mu_0, mu_0)

    return {'k_coeff':k_coeff , 'alpha_coeff': alpha_coeff, 'Zconst': Zconst, 'l_kappa_p_1':l_kappa_p_1, \
    'l_kappa': l_kappa, 'l_gamma_s_2':l_gamma_s_2 , 'l_gamma_s_1':l_gamma_s_1 ,\
    'v_n_p_1': v_n_p_1 , 'v_n': v_n, 'mu_factor': mu_factor, 'kappa_0':kappa_0, 'mu_0_factor':mu_0_factor, 'mu_0':mu_0}




def cache_sufficient_statistics_for_k(nk_array, K, n_SS, X, S_0):
    k_SS = {k:{} for k in xrange(K)}
    
    for k in xrange(int(K)):
        samp = X[nk_array==k].copy()
        n = len(samp)
        sX = np.einsum('nd->d',samp)
        mu_n = (n_SS['kappa_0']*n_SS['mu_0']+sX)/(n_SS['kappa_0']+n)
        S = S_0+np.einsum('ij, iz->jz', samp, samp)+n_SS['mu_0_factor']-(n_SS['kappa_0']+n)*np.einsum('i,j->ij',mu_n, mu_n)
        
        k_SS[k]['sX']=sX
        k_SS[k]['mu_n']=mu_n
        k_SS[k]['chol_S'] = linalg.cholesky(S, lower=1, overwrite_a=1)
        k_SS[k]['n']=n
    
    return k_SS

def downdate_chol_S(X_n, k,  k_SS, n_SS):
    n = k_SS[k]['n']
    k_0 = n_SS['kappa_0']
    mu_0 = k_SS[k]['mu_n']
    
    mu_n = (((k_0+n)*mu_0)-X_n)/(k_0+n+1)
    
    u2 = math.sqrt(k_0+n)*mu_0
    
    u3 = math.sqrt(k_0+n+1)*mu_n
    
    k_SS[k]['chol_S'] = uf.cholesky_r1_update(k_SS[k]['chol_S'], X_n, down=True, pychud_im=0)
    
    k_SS[k]['chol_S'] = uf.cholesky_r1_update(k_SS[k]['chol_S'], u2, down=True, pychud_im=0)
    
    k_SS[k]['chol_S'] = uf.cholesky_r1_update(k_SS[k]['chol_S'], u3, down=False, pychud_im=0)
    
    k_SS[k]['n']+=1
    
    k_SS[k]['mu_n']=mu_n




def update_chol_S(X_n, k,  k_SS, n_SS):
    n = k_SS[k]['n']
    k_0 = n_SS['kappa_0']
    mu_0 = k_SS[k]['mu_n']
    
    mu_n = (((k_0+n)*mu_0)+X_n)/(k_0+n+1)
    
    u2 = math.sqrt(k_0+n)*mu_0
    
    u3 = math.sqrt(k_0+n+1)*mu_n
    
    k_SS[k]['chol_S'] = uf.cholesky_r1_update(k_SS[k]['chol_S'], X_n, down=False, pychud_im=0)
    
    k_SS[k]['chol_S'] = uf.cholesky_r1_update(k_SS[k]['chol_S'], u2, down=False, pychud_im=0)
    
    k_SS[k]['chol_S'] = uf.cholesky_r1_update(k_SS[k]['chol_S'], u3, down=True, pychud_im=0)
    
    k_SS[k]['n']+=1
    
    k_SS[k]['mu_n']=mu_n


D = 2 #indexed by d
N=1000 #indexed by n
K=3 #indexed by k
alpha_0 = 100.

pi_0 = np.array([alpha_0/K for i in xrange(int(K))])


true_pi= np.array([300., 200.,100.])
p = true_pi/sum(true_pi)

A1 = np.array([[11, 0.5*(sqrt(11*2))],[0.5*(sqrt(11*2)), 2]])
A2 = np.array([[5, 0.9*(sqrt(5*12))],[0.9*(sqrt(5*12)), 12]])
A3 = np.array([[8, 0.1*(sqrt(8*7))],[0.1*(sqrt(8*7)), 7]])


mu1 = np.random.uniform(-100,1000, D)
mu2 = np.random.uniform(-100,1000, D)
mu3 = np.random.uniform(-100,1000, D)

cvms = np.array([A1, A2, A3])
mus = np.array([mu1, mu2, mu3])


X = []

for i in xrange(N):
    ind = np.random.choice(np.arange(K), p=p)
    X.append(np.random.multivariate_normal(mean=mus[ind], cov=cvms[ind]))

X = np.array(X)
pvals = np.random.dirichlet(pi_0)

nk_array = initial_assigment(X, pvals, K)

n_SS = cache_sufficient_statistics_for_n(X, K, alpha_0, 1, 1, np.array([0,0]))

k_SS = cache_sufficient_statistics_for_k(nk_array, K, n_SS, X, np.eye(D))


X_n = X[98].copy()
c1 = k_SS[0]['chol_S'].copy()
downdate_chol_S(X_n, 0, k_SS, n_SS)
c2 = k_SS[0]['chol_S'].copy()
nk_array[98]=2
k_SS = cache_sufficient_statistics_for_k(nk_array, K, n_SS, X, np.eye(D))
c3 = k_SS[0]['chol_S'].copy()
