from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import useful_functions as uf
import math
from scipy.special import gammaln as lggm

#some np.einsum recipes: ('ik,kj,ij->i', X,A,X)- the quadratic form for subarrays in X and matrix A1
#('i,j',X[i],X[i]) -XiXi'   ('ik, kj, zj, iz->i', X,cA,cA,X) - Xi'LL'Xi


def initial_assigment(X, pvals, K):
    '''Create initial data structures for a Bayesian Gaussian mixture model.
    
     Parameters
     --------
     
      X: array-like
      nxd multidimensional data. n represents the number of samples (len (X)) and d the dimensionality (len(X[0])).
      pvals: array-like
      vector of probabilities initially assigned to each cluster K (consists of K numbers between 0 and 1 that 
      add to 1). 
      K: int
      number of clusters.
      
      Output
      --------
      ki_dict: dict
      boolean mapping of clusters to indices of X.
      L_dict_prot: dict
      mapping of n indices of X to k zeros that will be replaced for likelihood values.
      sX: array-like
      n matrices consisting of XiXi' for each data-point'''

    n = len(X)
    ki_dict={i:np.zeros(n, dtype=np.bool) for i in xrange(int(K))} #boolean mapping samples to cluster k
    zeros=np.zeros(int(K)) #for likelihood prototype dict
    indices =np.arange(n)
    np.random.shuffle(indices)#start with a random assignment of indices to k
    group_assigments = np.cumsum(np.random.multinomial(n, pvals=pvals, size=1)[0])
    for k in xrange(len(group_assigments)):
        if k==0:
            ki_dict[k][indices[0:group_assigments[k]]]=1
        else:
            ki_dict[k][indices[group_assigments[k-1]:group_assigments[k]]]=1
    
    return ki_dict, {i:zeros.copy() for i in xrange(n)}, np.einsum('ik, ij -> ikj', X,X) 


def draw_Gaussian_parameters(ss_dict, L_dict_p, K,pi_m,X, ppd=False, collapsed=False):
    L_dict = L_dict_p.copy()
    for k in xrange(int(K)):
        if collapsed:
            chol_prec_m = ss_dict[k]['invChol_sm']
            mu = ss_dict[k]['em']
            df = ss_dict[k]['n']-ss_dict[k]['d']
            if df<1:
                #print 'setting df to 1'
                df=1
            chol_prec_m = math.sqrt((ss_dict[k]['n']*df)/(ss_dict[k]['n']+1))*chol_prec_m
            prec_m = chol_prec_m.dot(chol_prec_m.T)
            x_mu = X-mu
            m_dist_vec = np.einsum('ik,kj,ij->i', x_mu ,prec_m,x_mu ) 
            m_dist_vec = 1+(1./df)*m_dist_vec
            p1 = lggm(0.5*ss_dict[k]['n'])
            p2 = -lggm(0.5*df)
            p3 = -0.5*ss_dict[k]['d']*math.log(df)
            p4 = Zconst
            p5 = 0.5*uf.chol_log_determinant(chol_prec_m)
            p6 = -(0.5*ss_dict[k]['n'])*np.log(m_dist_vec)
            for i in xrange(len(X)):
                L_dict[i][k] = p1+p2+p3+p4+p5+p6[i]+pi_m[k]
            if ppd:
                chol_prec_m = uf.Wishart_rvs(df = ss_dict[k]['n']-1., S = ss_dict[k]['invChol_sm'], chol=1) #draw chol decomposition of prec_m
                mu = uf.multivariate_Gaussian_rvs(ss_dict[k]['em'], math.sqrt(ss_dict[k]['n'])*chol_prec_m, chol=1) #draw mean
                prec_m = chol_prec_m.dot(chol_prec_m.T)
                ss_dict[k]['Sigma'] = np.linalg.inv(prec_m)
                ss_dict[k]['mu']=mu
        else:
                                                   
            chol_prec_m = uf.Wishart_rvs(df = ss_dict[k]['n']-1., S = ss_dict[k]['invChol_sm'], chol=1) #draw chol decomposition of prec_m
            mu = uf.multivariate_Gaussian_rvs(ss_dict[k]['em'], math.sqrt(ss_dict[k]['n'])*chol_prec_m, chol=1) #draw mean
            #for likelihood
            prec_m = chol_prec_m.dot(chol_prec_m.T)
            x_mu = X-mu
            m_dist_vec = 0.5*np.einsum('ik,kj,ij->i', x_mu ,prec_m,x_mu ) #faster than  np.einsum('ik, kj, zj, iz->i', x_mu,chol_prec_m ,chol_prec_m ,x_mu)
            det = 0.5*uf.chol_log_determinant(chol_prec_m)
            for i in xrange(len(X)):
                L_dict[i][k] = Zconst+det-m_dist_vec[i]+pi_m[k]

            if ppd:
                ss_dict[k]['Sigma'] = np.linalg.inv(prec_m)
                ss_dict[k]['mu']=mu
        
    return L_dict
        
def update_Likelihood_dict(ki_dict,L_dict_p, X,cX, K, pi_m, collapsed=False):
    ss_dict={k:uf.store_sufficient_statistics_for_Gaussian_likelihood(X[ki_dict[k]],cX[ki_dict[k]]) for k in xrange(int(K))}
    if collapsed:
        L_dict = draw_Gaussian_parameters(ss_dict, L_dict_p, K, pi_m,X, collapsed=True)
    else:
        L_dict = draw_Gaussian_parameters(ss_dict, L_dict_p, K, pi_m,X)
    return L_dict        

def normalize_L(L):
    m = L.copy()
    m = m-max(m)
    m= exp(m)
    return m/sum(m)
    

def assign_Xi_to_k(L_dict, K,n):
    D = np.array([np.random.multinomial(1, pvals=normalize_L(L_dict[i])) for i in xrange(n)])
    return {k: D.T[k].astype(np.bool) for k in xrange(int(K))}


def draw_pi(alpha_0, ki_dict,K):
    alpha_up = np.array([sum(ki_dict[k])+alpha_0/K for k in xrange(int(K))])
    p = np.random.dirichlet(alpha_up)
    return np.log(p)


def Gibbs_sampler(T, X, cX, ki_dict, L_dict, K, n, alpha_0,collapsed=False):
    pi_m = draw_pi(alpha_0, ki_dict, K)
    K_up = int(K)
    w = 0

    if collapsed:
            
        for t in xrange(T):
            L_dict = update_Likelihood_dict(ki_dict, L_dict,X, cX, K_up, pi_m, collapsed=True) #0.002
            ki_dict = assign_Xi_to_k(L_dict, K_up, n)#0.006
            if 0 in [sum(ki_dict[k]) for k in xrange(K_up)]: #0.0001
                ki_dict, L_dict, cX = initial_assigment(X,p, K)#0.0007
                w+=1
            pi_m = draw_pi(alpha_0, ki_dict, K_up)#0.0002
            if w>50:
                K_up-=1
                print K_up
        posterior_param_dict = {k:uf.store_sufficient_statistics_for_Gaussian_likelihood(X[ki_dict[k]], cX[ki_dict[k]]) for k in xrange(int(K))}#0.001
        L_dict = draw_Gaussian_parameters(posterior_param_dict, L_dict, K, pi_m,X, ppd=1, collapsed=True)#0.003
        return np.exp(pi_m), posterior_param_dict
        
    else:
    
        for t in xrange(T):
            
            L_dict = update_Likelihood_dict(ki_dict, L_dict,X, cX, K_up, pi_m) #0.002
            ki_dict = assign_Xi_to_k(L_dict, K_up, n)#0.006
            if 0 in [sum(ki_dict[k]) for k in xrange(K_up)]: #0.0001
                ki_dict, L_dict, cX = initial_assigment(X, p, K)#0.0007
                w+=1
                print w
            pi_m = draw_pi(alpha_0, ki_dict, K_up)#0.0002
            if w>50:
                K_up-=1
                print K_up
        posterior_param_dict = {k:uf.store_sufficient_statistics_for_Gaussian_likelihood(X[ki_dict[k]], cX[ki_dict[k]]) for k in xrange(int(K))}#0.001
        L_dict = draw_Gaussian_parameters(posterior_param_dict, L_dict, K, pi_m,X, ppd=1)#0.003
        return np.exp(pi_m), posterior_param_dict
    
    


d = 50 #indexed by j
Zconst = -0.5*d*math.log(2*math.pi) #constant for the mvGaussian
n=1000 #indexed by i
K=3. #indexed by k
alpha_0 = 10.

pi_0 = np.array([alpha_0/K for i in xrange(int(K))])


true_pi= np.array([300., 200.,100.])
p = true_pi/sum(true_pi)

#A1 = np.array([[11, 0.5*(sqrt(11*2))],[0.5*(sqrt(11*2)), 2]])
#A2 = np.array([[5, 0.9*(sqrt(5*12))],[0.9*(sqrt(5*12)), 12]])
#A3 = np.array([[8, 0.1*(sqrt(8*7))],[0.1*(sqrt(8*7)), 7]])

A1 = 100*np.random.rand(d,d)
A1 = A1.dot(A1.T)

A2 = 100*np.random.rand(d,d)
A2 = A2.dot(A2.T)

A3 = 100*np.random.rand(d,d)
A3 = A3.dot(A3.T)

mu1 = np.random.uniform(-100,1000, d)
mu2 = np.random.uniform(-100,1000, d)
mu3 = np.random.uniform(-100,1000, d)

cvms = np.array([A1, A2, A3])
mus = np.array([mu1, mu2, mu3])


X = []

for i in xrange(n):
    ind = np.random.choice(np.arange(3), p=p)
    X.append(np.random.multivariate_normal(mean=mus[ind], cov=cvms[ind]))

X = np.array(X)
    
pi_m = np.random.dirichlet(pi_0)
pi_m  = np.log(pi_m)
ki_dict, L_dict, cX = initial_assigment(X,p, K)
L_dict = update_Likelihood_dict(ki_dict, L_dict,X, cX, K, pi_m)

