from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import useful_functions as uf

def make_param_dict(k_0, v_0, mu_0, sigma_2_0):
    '''make a dictionary with the prior parameters for Gibbs sampling.
    updatated parameters are the 'up_...' entries of the dictionary,
    they will begin with the same value as the priors'''
    return {'prior_k_0':k_0, 'prior_v_0':v_0, 'prior_mu_0':mu_0, 'prior_sigma_2_0':sigma_2_0, 'up_k_0':k_0, 'up_v_0':v_0, 'up_mu_0':mu_0, 'up_sigma_2_0':sigma_2_0}


def update_parameters(param_dict, data_set):
    '''update the parameters for the univariate gaussian gamma model.
    param_dict --> the dict built with the function 'make_param_dict'
    data_set --> N measures of a scalar-valued variable.'''
    
    #constants  
    empirical_mean = mean(data_set)
    delta_2 = (np.array(data_set)-empirical_mean)**2
    N=len(data_set)
    
    #updated prior parameters
    k_m = param_dict['prior_k_0']+N #mean prior 'confidence'
    
    v_m=param_dict['prior_v_0']+N #variance prior 'confidence'
    
    mu_m = ((param_dict['prior_k_0']*param_dict['prior_mu_0'])+N*empirical_mean)/k_m #mean prior
    
    #variance prior. Generally 'prior variance' + 'scatter of the data' +  `deviation from the mean` 
    sigma_2_m = param_dict['prior_sigma_2_0']+ sum(delta_2)+((param_dict['prior_k_0']*N)/k_m)*((empirical_mean-param_dict['prior_mu_0'])**2) 
    
    #update param_dict        
    param_dict['up_k_0'] = k_m
    param_dict['up_v_0'] = v_m

    param_dict['up_mu_0'] = mu_m
    param_dict['up_sigma_2_0'] = sigma_2_m

    
    
def draw_sigma_2(param_dict):
    '''draw the variace from the updated prior params'''
    inv_sigma_2 = np.random.gamma(shape=param_dict['up_v_0']/2., scale=2./param_dict['up_sigma_2_0']) #faster than scipy
    return 1./inv_sigma_2 #this is used to draw the posterior and transformed to draw the mean

def draw_mu(param_dict, sigma_2):
    '''draw the mean from the updated prior params'''
    return np.random.normal(loc=param_dict['up_mu_0'],scale=sqrt( (1./param_dict['up_k_0'])*sigma_2))

def Gibbs_sampler(param_dict):
    sigma_2 = draw_sigma_2(param_dict)
    mu = draw_mu(param_dict, sigma_2)
    return np.random.normal(loc=mu, scale=sqrt(sigma_2))

def draw_from_collapsed_Gibbs_sampler(param_dict):
    return sts.t(df = param_dict['prior_v0'], loc = param_dict['prior_mu'],scale = sqrt(((param_dict['prior_k0']+1)/(param_dict['prior_k0']*param_dict['prior_v0'])) * param_dict['prior_sigma_2'])).rvs()

#example_data

data_set = np.array([np.random.poisson(60) for i in xrange(1000)])
param_dict = make_param_dict(k_0=10, v_0 =10, mu_0=0, sigma_2_0=1)


update_parameters(param_dict, data_set)

gibbs_sample = np.array([Gibbs_sampler(param_dict) for i in xrange(50000)])
#collapsed_gibbs_sample = np.array([draw_from_collapsed_Gibbs_sampler(param_dict) for i in xrange(5000)])

g2 = []


for i in xrange(10, len(gibbs_sample)):
    g2.append(uf.estimate_convergence(data_set, gibbs_sample[0:i]))
    
