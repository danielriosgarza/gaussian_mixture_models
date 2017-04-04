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
