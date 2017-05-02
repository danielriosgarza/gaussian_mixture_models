from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import useful_functions as uf
import math
from scipy.special import gammaln as lggm
import scipy.linalg as linalg
from scipy.misc import logsumexp

def initial_assigment(X, pvals, K):
    '''create a random initial assignment of samples in X to K clusters,
    based on some probability.
    
    Parameters
    --------
    X: array-like
    nxd multidimensional data. n represents the number of samples (len (X)) and d the dimensionality (len(X[0])).
    pvals: array-like
      vector of probabilities initially assigned to each cluster K (consists of K numbers between 0 and 1 that 
      add to 1). 
    K: int
      number of clusters.

    Outputs
    --------
    nk_array: array-like
    mapping of samples in position i to in in range(K).'''
    
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
    N = X.shape[0]+1000
    D = X.shape[1]
    
    #global constants
    k_coeff = K/(K*(N+alpha_0-1))
    alpha_coeff = alpha_0/(K*(N+alpha_0-1))
    
    Zconst = np.array([n*(0.5)*D*math.log(pi) for n in xrange(1,N)])
        
    
    #indexed by n
    l_kappa_p_1 = np.array([0.5*D*math.log(kappa_0+n+1) for n in xrange(1,N)])
    
    l_kappa = np.array([0.5*D*math.log(kappa_0+n) for n in xrange(1,N)])
    
    l = []
    for n in xrange(1,N):
        m = sum([lggm((v_0+n+2-i)/2.) for i in xrange(1,D+1)])
        l.append(m)
    l_gamma_s_2 = np.array(l)
    
    l = []
    for n in xrange(1,N):
        m = -sum([lggm((v_0+n+1-i)/2.) for i in xrange(1,D+1)])
        l.append(m)
    l_gamma_s_1 = np.array(l)

    v_n_p_1 = np.array([0.5*(v_0+n+1) for n in xrange(1,N)])

    v_n = np.array([0.5*(v_0+n) for n in xrange(1,N)])

    mu_0_factor = kappa_0*np.einsum('i,j->ij',mu_0, mu_0)

    return {'k_coeff':k_coeff , 'alpha_coeff': alpha_coeff, 'Zconst': Zconst, 'l_kappa_p_1':l_kappa_p_1, \
    'l_kappa': l_kappa, 'l_gamma_s_2':l_gamma_s_2 , 'l_gamma_s_1':l_gamma_s_1 ,\
    'v_n_p_1': v_n_p_1 , 'v_n': v_n,  'kappa_0':kappa_0, 'mu_0_factor':mu_0_factor, 'mu_0':mu_0}




def cache_sufficient_statistics_for_k(nk_array, K, n_SS, X, S_0):
    k_SS = {k:{} for k in xrange(K)}
    
    for k in xrange(int(K)):
        samp = X[nk_array==k].copy()
        if len(samp)==0:
            k_SS[k]['mu_n']=n_SS['mu_0']
            k_SS[k]['chol_S'] = S_0
            k_SS[k]['n']=1
        else:
            n = len(samp)
            sX = np.einsum('nd->d',samp)
            mu_n = (n_SS['kappa_0']*n_SS['mu_0']+sX)/(n_SS['kappa_0']+n)
            S = S_0+np.einsum('ij, iz->jz', samp, samp)+n_SS['mu_0_factor']-(n_SS['kappa_0']+n)*np.einsum('i,j->ij',mu_n, mu_n)
                
            k_SS[k]['mu_n']=mu_n
            k_SS[k]['chol_S'] = linalg.cholesky(S, lower=1, overwrite_a=1)
            k_SS[k]['n']=n
    
    return k_SS

def downdate_chol_S(X_n, k,  k_SS, n_SS):
    n = k_SS[k]['n']
    k_0 = n_SS['kappa_0']
    mu_0 = k_SS[k]['mu_n'].copy()
        
    mu_n = (((k_0+n)*mu_0)-X_n)/(k_0+n-1)
        
    u2 = math.sqrt(k_0+n)*mu_0
        
    u3 = math.sqrt(k_0+n-1)*mu_n
        
    k_SS[k]['chol_S'] = uf.cholesky_r1_update(k_SS[k]['chol_S'], X_n, down=True)
        
    k_SS[k]['chol_S'] = uf.cholesky_r1_update(k_SS[k]['chol_S'], u2, down=False)
        
    k_SS[k]['chol_S'] = uf.cholesky_r1_update(k_SS[k]['chol_S'], u3, down=True)
    
    if n>1:
        k_SS[k]['n']-=1
    else:
        k_SS[k]['n']=1
        
    k_SS[k]['mu_n']=mu_n




def update_chol_S(X_n, k,  k_SS, n_SS):
    n = k_SS[k]['n']
    k_0 = n_SS['kappa_0']
    mu_0 = k_SS[k]['mu_n']
    
    mu_n = (((k_0+n)*mu_0)+X_n)/(k_0+n+1)
    
    u2 = math.sqrt(k_0+n)*mu_0
    
    u3 = math.sqrt(k_0+n+1)*mu_n
    
    chol_S = k_SS[k]['chol_S'].copy()
    
    chol_S = uf.cholesky_r1_update(chol_S, X_n, down=False)
    
    chol_S = uf.cholesky_r1_update(chol_S, u2, down=False)
    
    chol_S = uf.cholesky_r1_update(chol_S, u3, down=True)
    
    return chol_S,mu_n
    

def normalize_L(L):
    m = L.copy()
    m = m-max(m)
    m= exp(m)
    return m/sum(m)
    




D = 2 #indexed by d
N=100 #indexed by n
K=3 #indexed by k
alpha_0 = 1000.

pi_0 = np.array([alpha_0/K for i in xrange(int(K))])
pi_0=pi_0/sum(pi_0)

true_pi= np.array([3., 2.,5.])
p = true_pi/sum(true_pi)

A1 = np.array([[11, 0.5*(sqrt(11*2))],[0.5*(sqrt(11*2)), 2]])
A2 = np.array([[5, 0.9*(sqrt(5*12))],[0.9*(sqrt(5*12)), 12]])
A3 = np.array([[8, 0.1*(sqrt(8*7))],[0.1*(sqrt(8*7)), 7]])


mu1 = np.random.uniform(-10,100, D)
mu2 = np.random.uniform(-10,100, D)
mu3 = np.random.uniform(-10,100, D)

cvms = np.array([A1, A2, A3])
mus = np.array([mu1, mu2, mu3])


X = []

for i in xrange(N):
    ind = np.random.choice(np.arange(K), p=p)
    X.append(np.random.multivariate_normal(mean=mus[ind], cov=cvms[ind]))

X = np.array(X)

K=10
#D = 2           # dimensions
#N = 100          # number of points to generate
#K_true = 4      # the true number of components
#
## Model parameters
#alpha_0 = 1.
#K = 4           # number of components
#n_iter = 1000
#pi_0 = np.array([alpha_0/K for i in xrange(int(K))])
#pi_0 = pi_0/sum(pi_0)
## Generate data
#mu_scale = 4.0
#covar_scale = 0.7
#z_true = np.random.randint(0, K_true, N)
#mu = np.random.randn(D, K_true)*mu_scale
#X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
#X = X.T


#pvals = np.random.dirichlet(pi_0)

nk_array = initial_assigment(X, pi_0, K)

n_SS = cache_sufficient_statistics_for_n(X, K, alpha_0, 1, 1, np.array([0,0]))

k_SS = cache_sufficient_statistics_for_k(nk_array, K, n_SS, X, np.eye(D))





for t in xrange(100):
    n = np.arange(N)
    np.random.shuffle(n)
    
    
    for num in n:
        chols=[]
        muns=[]
        downdate_chol_S(X[num], nk_array[num],  k_SS, n_SS)
        p=np.zeros(K)
    
        for k in xrange(K):
            
            nid =  k_SS[k]['n']+1
            cup, mu_n = update_chol_S(X[num], k,  k_SS, n_SS)
            
            up_det = uf.chol_log_determinant(cup)
            
            if np.isnan(up_det):
                cup = uf.cholesky_approx_positive_semidefinite(cup)
                up_det = uf.chol_log_determinant(cup)
            chols.append(cup)
            muns.append(mu_n)
            low_det = uf.chol_log_determinant(k_SS[k]['chol_S'])
            if np.isnan(low_det):
                A = uf.cholesky_approx_positive_semidefinite(k_SS[k]['chol_S'])
                low_det = uf.chol_log_determinant(A)
            
            p_z_i =math.log((nid-1)+(alpha_0/K))
            p_x_i = -n_SS['Zconst'][nid] + n_SS['l_kappa'][nid]+n_SS['v_n'][nid]*low_det-n_SS['l_kappa'][nid+1]-n_SS['v_n'][nid+1]*up_det+n_SS['l_gamma_s_1'][nid+1]-n_SS['l_gamma_s_1'][nid]
               
            p[k]=p_z_i+p_x_i
            #print p, '\t', k
            
        
        assign_p = np.exp(p - max(p)) 
        assign_p= assign_p/sum(assign_p)
        #print assign_p    
        new_k = np.random.choice(K, p=assign_p)
        #print new_k
        k_SS[new_k]['chol_S'] = chols[new_k].copy()
        k_SS[new_k]['mu_n']=muns[new_k].copy()
        k_SS[new_k]['n']+=1
        nk_array[num]=new_k
        b=[ len(X[nk_array==0]),  len(X[nk_array==1]), len(X[nk_array==2]), len(X[nk_array==3]), len(X[nk_array==4]), len(X[nk_array==5])]            
        #print b
        

scatter(X.T[0][nk_array==0], X.T[1][nk_array==0]);scatter(X.T[0][nk_array==1], X.T[1][nk_array==1], c='g');scatter(X.T[0][nk_array==2], X.T[1][nk_array==2], c='y');scatter(X.T[0][nk_array==3], X.T[1][nk_array==3], c='k')
