from pylab import *
import numpy as np
import scipy.stats as st

def d_bayesian_operand(data_vector, prior):
    denominator = float(sum([data_vector[i]*prior[i] for i in xrange(len(data_vector))]))
    return np.array([(data_vector[i]*prior[i])/denominator for i in xrange(len(data_vector))])

def weighted_means_var(data_vector, weights):
    w_mean =  sum(data_vector*weights)/sum(weights)
    w_var = (sum(((data_vector-w_mean)**2)*weights)/sum(weights))**0.5
    return w_mean,w_var

def prior_update(cluster_probabilities):
    return np.array([sum(cluster_probabilities.T[i])/float(len(cluster_probabilities)) for i in xrange(len(cluster_probabilities.T))])
    
    
def one_dimensional_gaussian_mixture_model(data_vector, k_clusters, max_iter):
    data_vector=np.array(data_vector)
    random_means = np.random.uniform(min(data_vector), max(data_vector), k_clusters)
    gaussians = {i:st.norm(random_means[i],max(data_vector)-min(data_vector)) for i in xrange(k_clusters)}
    priors = np.array([0.5]*k_clusters)
    
    for i in xrange(max_iter):
        distribution_probabilities = np.array([[gaussians[x].pdf(data_vector[z]) for x in xrange(len(gaussians))] for z in xrange(len(data_vector))])
        #b = np.concatenate(distribution_probabilities)
        #distribution_probabilities = distribution_probabilities+min(b[b>0])
        cluster_probabilities = np.array([d_bayesian_operand(distribution_probabilities[z], priors) for z in xrange(len(distribution_probabilities))])
        
        priors = prior_update(cluster_probabilities)
        gaussians = {z:st.norm(weighted_means_var(data_vector, cluster_probabilities.T[z])[0], weighted_means_var(data_vector, cluster_probabilities.T[z])[1]) for z in xrange(k_clusters))}
        
    hard_assignments = [list(cluster_probabilities[i]).index(max(cluster_probabilities[i])) for i in xrange(len(cluster_probabilities))]
    return {'hard_assignments': hard_assignments, 'cluster_probabilities':cluster_probabilities, 'means' :[weighted_means_var(data_vector, cluster_probabilities.T[i])[0] for i in xrange(len(cluster_probabilities.T))], 'variance': [weighted_means_var(data_vector, cluster_probabilities.T[i])[1] for i in xrange(len(cluster_probabilities.T))]}
    #return distribution_probabilities, priors



rd_means = np.random.normal(10,1000,36)
rd_var = np.random.randint(1,6,36)

l = [np.random.normal(rd_means[i],rd_var[i], np.random.randint(20,50)) for i in xrange(36)]
dv = np.concatenate(l)
dv = np.random.permutation(dv)
dv = np.random.permutation(dv)



ps = one_dimensional_gaussian_mixture_model(dv, 36,150)
