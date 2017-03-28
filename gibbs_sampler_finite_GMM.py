# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 23:08:58 2017

@author: user
"""

import numpy as np
import scipy.stats as sts




def assign_to_nk(gaussian_dict, k, data_point):
    p =np.array([gaussian_dict[i].logpdf(data_point) for i in xrange(k)])
    p = abs(p/sum(abs(p)))
    return np.arange(k)[np.random.multinomial(1, p)==1]

def build_nk_and_indicator_dict(gaussian_dict, k, data):
    nk_dict = {i:0 for i in xrange(k)}
    indicator_dict={}
    for i in xrange(len(data)):
        b = int(assign_to_nk(gaussian_dict, k, data[i]))
        nk_dict[b]+=1
        indicator_dict[i]=b
    return nk_dict, indicator_dict


def draw_pi(pi_0, nk_dict, k):
    alpha = np.array([nk_dict[i] for i in xrange(k)])
    return sts.dirichlet(alpha+pi_0).rvs()
    
def build_sk_dict(indicator_dict, mu_dict, k, N, data):
    sk_dict = {i:np.eye(N) for i in xrange(k)}
    for i in xrange(len(data)):
        b = indicator_dict[i]
        a = data[i]-mu_dict[b]
        sk_dict[b]+=np.dot(a[np.newaxis].T, a[np.newaxis])
       
    return sk_dict
    
def draw_sigma_k(sk_dict, nk_dict, k, N):
    return {i: sts.invwishart(nk_dict[i]+N-1, scale=sk_dict[i]).rvs() for i in xrange(k)}

def build_inv_vk_dict(V_0, nk_dict, sigma_dict, k):
    return {i: np.linalg.inv(V_0) + nk_dict[i]*np.linalg.inv(sigma_dict[i]) for i in xrange(k)}

def build_empirical_xk_dict(indicator_dict, data, nk_dict, k, N):
    empirical_xk_dict = {i:np.zeros(N) for i in xrange(k)}
    for i in xrange(len(data)):
        b = indicator_dict[i]
        empirical_xk_dict[b]+=data[i]/nk_dict[b]
    return empirical_xk_dict
    
def build_mk_dict(V_0, m_0, inv_vk_dict, sigma_dict, nk_dict, empirical_xk_dict, k):
    p1 = np.linalg.inv(V_0)
    p2 = {i: np.linalg.inv(inv_vk_dict[i]) for i in xrange(k)}
    p3 = {i: np.linalg.inv(sigma_dict[i]) for i in xrange(k)}
    p4 = {i: nk_dict[i]*empirical_xk_dict[i] for i in xrange(k)}
    p5 = {i: np.dot(p3[i], p4[i]) for i in xrange(k)}
    p6 = np.dot(p1, m_0)
    p7 = {i: p5[i]+p6 for i in xrange(k)}
    return {i: np.dot(p2[i], p7[i]) for i in xrange(k)}

def draw_mu_k(mk_dict, inv_vk_dict,k):
    return {i:sts.multivariate_normal(mk_dict[i], np.linalg.inv(inv_vk_dict[i])).rvs() for i in xrange(k)}

#Constants

N = 6 # dimension of the MVN
J = 200 # number of data points
K = 3 #number of MVN distributions


#Bulding test data
true_pi = sts.dirichlet([3,6,9]).rvs()
s1 = np.random.rand(6,6)*np.random.uniform(0,30)
s1 = np.dot(s1, s1.T)
s2 = np.random.rand(6,6)*np.random.uniform(0,30)
s2 = np.dot(s1, s1.T)
s3 = np.random.rand(6,6)*np.random.uniform(0,30)
s3 = np.dot(s1, s1.T)

true_g1 =sts.multivariate_normal(np.random.uniform(-100,100,6),s1)
true_g2 =sts.multivariate_normal(np.random.uniform(-100,100,6),s2)
true_g3 =sts.multivariate_normal(np.random.uniform(-100,100,6),s3)




true_z = np.random.multinomial(J, true_pi[0])
data = []

for i in xrange(len(true_z)):
    data.append(true_g1.rvs(true_z[i]))
data = np.concatenate(data)


# Getting values for t0

S_0 = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
V_0 = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
m_0 = np.zeros(N)
pi_0 = np.ones(K)

mu_dict = {i:data[np.random.choice(np.arange(len(data)))] for i in xrange(K)}

gaussian_dict = {i: sts.multivariate_normal(mu_dict[i], S_0) for i in xrange(K)}

mus = {}
pis = {}
sigmas = {}

for i in xrange(10000):
    nk_dict, indicator_dict = build_nk_and_indicator_dict(gaussian_dict, K, data)
    pi = draw_pi(pi_0, nk_dict, K)
    pis[i] = pi.copy()
    sk_dict = build_sk_dict(indicator_dict, mu_dict, K, N, data)
    sigma_dict = draw_sigma_k(sk_dict, nk_dict, K, N)
    sigmas[i] = sigma_dict.copy()
    inv_vk_dict = build_inv_vk_dict(V_0, nk_dict, sigma_dict, K)
    empirical_xk_dict = build_empirical_xk_dict(indicator_dict, data, nk_dict, K, N)
    mk_dict = build_mk_dict(V_0, m_0, inv_vk_dict, sigma_dict, nk_dict, empirical_xk_dict, K)
    mu_dict = draw_mu_k(mk_dict, inv_vk_dict,K)
    mus[i] = mu_dict.copy()
    gaussian_dict = {z:sts.multivariate_normal(mu_dict[z], sigma_dict[z]) for z in xrange(K)}
