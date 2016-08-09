from pylab import *
import numpy as np
import scipy.stats as st
import math as m
import scipy.sparse as sp

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


def mle_mean(sample_matrix):
    return (1./len(sample_matrix))*np.sum(sample_matrix, axis=0)

def mle_covm(mle_mean, sample_matrix):
    N = len(sample_matrix)
    a = [np.outer(np.array(sample_matrix[z] - mle_mean).T, np.array(sample_matrix[z] - mle_mean)) for z in xrange(N)]
    return (1./N)*np.sum(a, axis=0)
    
    

def random_initiation(data_vector, k_clusters):
    m = xrange(len(data_vector))
    c = [np.random.choice(m, size=2, replace=1) for i in xrange(k_clusters)]
    data_groups = np.array([data_vector[i] for i in c])
    rd_means = [mle_mean(z) for z in data_groups]
    rd_covm = [mle_covm(rd_means[z], c[z]) for z in xrange(len(data_groups))]
    return rd_means, rd_covm

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



def prior_update(cluster_probabilities):
    return np.array([sum(cluster_probabilities.T[i]/float(len(cluster_probabilities))) for i in xrange(len(cluster_probabilities.T))])

def generate_random_uniform_cov_matrix(size):
    A = np.random.rand(size, size)
    return np.dot(A, A.T)
    
    
def multi_dimensional_gaussian_mixture_model(data_vector, k_clusters):
    data_vector=np.array(data_vector)
    dim1 = len(data_vector)
    dim2 = len(data_vector[0])
    new_means, new_covm = random_initiation(data_vector, k_clusters)
    #gaussians = {i:st.multivariate_normal(random_means[i],generate_random_uniform_cov_matrix(dim2)) for i in xrange(k_clusters)}
    priors = np.array([1./k_clusters]*k_clusters)
    a = 10
    m=0
    while a>0.01:
        print m
        distribution_probabilities = np.array([[lognorm_pdf(data_vector[z], new_means[x], new_covm[x]) for x in xrange(k_clusters)] for z in xrange(len(data_vector))])
        cluster_probabilities_a = d_bayesian_operand(distribution_probabilities, priors)
        priors = prior_update(cluster_probabilities_a)
        new_means, new_covm  = weighted_means_var(data_vector,cluster_probabilities_a)
        distribution_probabilities = np.array([[lognorm_pdf(data_vector[z], new_means[x], new_covm[x]) for x in xrange(k_clusters)] for z in xrange(len(data_vector))])
        cluster_probabilities_b = d_bayesian_operand(distribution_probabilities, priors) 
        a = np.linalg.norm(cluster_probabilities_b-cluster_probabilities_a)
        m+=1

        #gaussians = {z:st.multivariate_normal(new_means[z], new_covm[z]) for z in xrange(k_clusters)}
        
    hard_assignments = [list(cluster_probabilities_a[i]).index(max(cluster_probabilities_a[i])) for i in xrange(len(cluster_probabilities_a))]
    return {'m':m,'hard_assignments': hard_assignments, 'cluster_probabilities':cluster_probabilities_a, 'means' :weighted_means_var(data_vector, cluster_probabilities_a)[0], 'variance': weighted_means_var(data_vector, cluster_probabilities_a)[1], 'a':a}
    #return weighted_means_var(data_vector, cluster_probabilities)



def noisy_cosine():
    x = np.random.rand(100) * np.pi * 2.0
    x.sort()
    y = np.cos(x) + 0.1 * np.random.randn(100)
    return x,y

x,y = noisy_cosine()
data = np.vstack([x,y]).transpose() 

#ps= multi_dimensional_gaussian_mixture_model(data, 2)   

obs = np.concatenate((np.random.randn(100, 2), 10 + np.random.randn(300, 2)))

ps = multi_dimensional_gaussian_mixture_model(obs, 2)
