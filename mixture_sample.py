from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import useful_functions as uf

d = 2
n=600
k=3.
alpha_0 = 10.

pi_0 = np.array([alpha_0/k for i in xrange(int(k))])


true_pi= np.array([300., 200.,100.])
p = true_pi/sum(true_pi)

A1 = np.array([[11, 0.5*(sqrt(11*2))],[0.5*(sqrt(11*2)), 2]])
A2 = np.array([[5, 0.9*(sqrt(5*12))],[0.9*(sqrt(5*12)), 12]])
A3 = np.array([[8, 0.1*(sqrt(8*7))],[0.1*(sqrt(8*7)), 7]])

mu1 = np.random.uniform(0,1000, 2)
mu2 = np.random.uniform(0,1000, 2)
mu3 = np.random.uniform(0,1000, 2)

cvms = np.array([A1, A2, A3])
mus = np.array([mu1, mu2, mu3])

X = []

for i in xrange(n):
    ind = np.random.choice(np.arange(3), p=p)
    X.append(np.random.multivariate_normal(mean=mus[ind], cov=cvms[ind]))

X = np.array(X)
    



def store_sufficient_statistics_Gaussian_mix(X):
    n = len(X)
    d = len(X[0])
    SS_dict={}
    for i in xrange(n):
        b= X[i].copy()
        b.shape=(d,1)
        SS_dict[i]= b.dot(b.T)
    return SS_dict




