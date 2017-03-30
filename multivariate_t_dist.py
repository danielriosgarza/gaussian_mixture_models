# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 01:49:53 2017

@author: user
"""
from scipy.special import gammaln as gam_log
from math import pi, log
import numpy as np

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
