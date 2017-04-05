from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import math
import scipy.special as spe


def cholesky_r1_update(L, X):
    '''perform a rank 1 update of
    the cholesky decomposition of positive definite
    a matrix.
    
    L --> dxd lower triangular matrix, such that A = LL'
    X---> d dimensional vector, such that A* = A + XX'
    '''
    
    l = L.copy()
    x = X.copy()
    d = len(x)
    for k in xrange(d):
        r = math.sqrt((l[k,k]**2)+(x[k]**2))
        c = r/l[k,k]
        s = x[k]/l[k,k]
        l[k,k] = r
        for i in xrange(k+1, d):
            l[i, k] = (l[i,k]+ (s*x[i]))/c
            x[i] = (x[i]*c)-(s*l[i,k])
    return l


def gamma_pdf(x, alpha, theta, log_form=False):
    '''PDF of the gamma distribution. 
    is equivalent to scipy.stats.gamma(a=alpha, scale=theta).pdf(x)
    
    x--> scalar quantity representing an observation
    alpha, theta ---> scalar parameters'''
    
    t1 = -spe.gammaln(alpha)
    t2 = -alpha*math.log(theta)
    t3 = (alpha-1)*math.log(x)
    t4 = -x/theta
    if log_form:
        return t1+t2+t3+t4
    else:
        return math.exp(t1+t2+t3+t4)



def Gaussian_pdf(x, mu, precision, log_form = False):
    delta = (x-mu)**2
    t1 = math.log(0.5*(precision/math.pi))
    t2 = -precision*delta
    
    if log_form:
        return 0.5*(t1+t2)
    else:
        return math.exp(0.5*(t1+t2))
    
def estimate_convergence(data_set, simulation_set):
    '''convergence estimator for scalar quantities,
    based on the within and between variances of
    data and simulated sets'''
    #constants
    d_n = len(data_set)
    s_n  = len(simulation_set)
    n = d_n+s_n
    n_m = 0.5*n
    mean_d = mean(data_set)
    mean_s = mean(simulation_set)
    mean_t = 0.5*(mean_d+mean_s)
    var_d = (1./(d_n-1))*sum((data_set-mean_d)**2)
    var_s = (1./(s_n-1))*sum((simulation_set-mean_d)**2)
    
    #Between variance
    B = n_m*(((mean_d-mean_t)**2)+((mean_d-mean_t)**2))
    
    #within variance
    W = 0.5*(var_d+var_s)
    
    #marginal posterior variance estimand
    var_mp = ((n-1.)/n)*W + (1./n)*B
    
    #convergence estimate (converges to 1)
    return sqrt(var_mp/W)
