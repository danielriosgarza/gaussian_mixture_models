# -*- coding: utf-8 -*-
"""
Created on Sat Aug 06 14:03:58 2016

@author: user
"""
from pylab import *
import numpy as np
import scipy.stats as st
import math as m
import scipy.sparse as sp

#http://chamroukhi.univ-tln.fr/courses/2012-2013/d36/robust_EM_clustering_GMM.pdf
def lognorm_pdf(x, loc_v, covm):
    nx = len(covm)
    normal_coeff = nx * m.log(2*m.pi) + np.linalg.slogdet(covm)[1]
    err = (x-loc_v)
    if (sp.issparse(covm)):
        numerator = spln.spsolve(covm, err).T.dot(err)
    else:
        numerator = np.linalg.solve(covm,err).T.dot(err)
    #return float(-0.5*(normal_coeff + numerator))
    return np.exp(float(-0.5*(normal_coeff + numerator)))


def initial_covm(data_vectors):
    dim1 = len(data_vectors)
    dim2 = len(data_vectors[0])
    covms = []
    for i in data_vectors:
        c = [np.linalg.norm(data_vectors[z]-i)**2 for z in xrange(dim1)]
        l = np.argsort(c)
        covms.append(np.dot(c[l[-1]], np.eye(dim2, dim2)))
    return np.array(covms)

def initial_conditions(data_vectors):
    epsilon = 0.001
    beta = 1.0
    k_clusters = len(data_vectors)
    priors = [1./k_clusters]*k_clusters
    covms = initial_covm(data_vectors)
    return epsilon, beta, k_clusters, priors, covms


def probability_of_distributions(mean_vectors, covms, data_points):
    k = len(mean_vectors)
    N = len(data_points)
    p = len(data_points[0])
    probabilities = []
    for i in xrange(k):
        probabilities.append(np.array([lognorm_pdf(data_points[z], mean_vectors[i], covms[i]) for z in xrange(N)]))
    return np.array(probabilities).T
    
def d_bayesian_operand(data_vector, prior):
    denominator = np.array([sum(data_vector[i]*prior) for i in xrange(len(data_vector))])
    numerator = [data_vector[i]*prior for i in xrange(len(data_vector))]
    return np.array([numerator[i]/denominator[i] for i in xrange(len(numerator))])

def d_bayesian_operand(data_vector, prior):
    denominator = np.array([sum(data_vector[i]*prior) for i in xrange(len(data_vector))])
    numerator = [data_vector[i]*prior for i in xrange(len(data_vector))]
    return np.array([numerator[i]/denominator[i] for i in xrange(len(numerator))])


def weighted_means_var(data_vector, weights):
    dim1 = len(weights)
    dim2 = len(weights[0])
    mc = [sum(weights.T[i]) for i in xrange(dim2)]
    mean_vectors = []
    cov_matrices = []
    for i in xrange(dim2):
        b = [data_vector[z]*weights.T[i][z] for z in xrange(dim1)]
        mean_vectors.append((1./mc[i])*np.sum(b, axis=0))
    for i in xrange(dim2):
        b = [weights.T[i][z]*np.outer(np.array(data_vector[z] - mean_vectors[i]).T, np.array(data_vector[z] - mean_vectors[i])) for z in xrange(dim1)]
        cov_matrices.append((1./mc[i])*np.sum(b, axis=0))

    return np.array(mean_vectors), np.array(cov_matrices) 

def prior_update(current_priors, current_beta, cluster_probabilities):
    alphak_em = np.array([sum(cluster_probabilities.T[i])/float(len(cluster_probabilities)) for i in xrange(len(cluster_probabilities.T))])
    return alphak_em, alphak_em + current_beta*(current_priors*(log(current_priors) - (np.sum(current_priors*log(current_priors)))))

def compute_beta(data_vectors, old_priors, em_priors, new_priors):
    n =len(data_vectors)
    N = len(data_vectors[0])
    c = len(new_priors)
    vn = min(1., 0.5**((N/2.)-1))
    print vn
    E = sum(old_priors*log(old_priors))
    p1 = sum(exp(-vn*n*abs(new_priors - old_priors)))/float(c)
    p2= ((1 - np.max(em_priors))/(-1.*max(old_priors)*E) )
    return min(p1, p2)

def update_k(new_priors, cluster_probabilities, N, means, covms):
    keep = new_priors[new_priors>(1./N)-(0.1/N)]
    print len(keep)
    adjusted_priors = keep*(1./sum(keep))
    keep_probability = cluster_probabilities.T[new_priors>(1./N)-(0.1/N)]
    keep_probability =keep_probability.T
    adjusted_cluster_probabilities = [keep_probability[i]/sum(keep_probability[i]) for i in xrange(len(keep_probability))]
    return np.array(adjusted_priors), np.array(adjusted_cluster_probabilities), len(keep) 


def robust_gaussian_mixture_model(data_vectors):
    epsilon, beta, k_clusters, priors, covms = initial_conditions(data_vectors)
    pd = probability_of_distributions(data_vectors, covms, data_vectors)
    cd = d_bayesian_operand(pd, priors)
    priors_a, priors_b = prior_update(priors, beta, cd)
    beta = compute_beta(data_vectors, priors, priors_a, priors_b)
    priors, cd, k_clusters = update_k(priors_b, cd, len(data_vectors), data_vectors, covms)
    means, covms = weighted_means_var(data_vectors,cd)
    n = 0
    a= 30
    clus = [k_clusters, k_clusters]
    while a>epsilon:
        print 'p',clus[-1]-clus[-2]
        if n>=60 and clus[-1]-clus[-2] ==0:
            n+=1
            print n
            beta = 0.0
            means, covms = weighted_means_var(data_vectors,cd)
            pd = probability_of_distributions(means, covms, data_vectors)
            cd = d_bayesian_operand(pd, priors)
            meansb, covmsb = weighted_means_var(data_vectors,cd)
            a = np.linalg.norm(meansb-means)
            print a
        else:
            n+=1
            print n
            priors_a, priors_b = prior_update(priors, beta, cd)
            beta = compute_beta(data_vectors, priors, priors_a, priors_b)
            print 'beta', beta
            priors, cd, k_clusters = update_k(priors_b, cd, len(data_vectors), data_vectors, covms)
            clus.append(k_clusters)
            means, covms = weighted_means_var(data_vectors,cd)
            pd = probability_of_distributions(means, covms, data_vectors)
            cd = d_bayesian_operand(pd, priors)
    hard_assignments = [list(cd[i]).index(max(cd[i])) for i in xrange(len(cd))]
    return {'means': means, 'variance':covms, 'p_assignment':cd, 'k':k_clusters, 'h_assignment': hard_assignments, 'priors':priors}


obs = np.concatenate((np.random.randn(100, 2), 10 + np.random.randn(300, 2),  500 + np.random.randn(300, 2)))
