
import numpy as np
from pylab import *

def rotation_matrix(a_in_rad, vec):
    '''take a 2 component vector and rotate by a given angle''' 
    transformation_matrix = np.array([[np.cos(a_in_rad), -1*np.sin(a_in_rad)], [np.sin(a_in_rad), np.cos(a_in_rad)]])
    return np.dot(transformation_matrix, vec)


def standard_normal_Gauss(n):
    '''generative implementation of a standard normal distribution'''
    #define a radius using the square root of the exponential distribution
    #where theta equals two.
    radius = np.array([[0, sqrt(np.random.exponential(2))] for i in xrange(n)])
    #rotate the radius by random uniform angle in the interval [0, 2pi],
    #the coordinates will be samples from two independent standard normal dist.
    norm_sample = np.array([rotation_matrix(np.random.uniform(0, 2*pi), i) for i in radius])
    return norm_sample

def Box_Muller():
    '''Box_Muller (1958) for a single draw from two independent standard normal distributions.'''
    u_1 = np.random.uniform()
    u_2 = np.random.uniform()
    return np.array([sqrt(-2*log(u_1))*sin(2*pi*u_2), sqrt(-2*log(u_1))*cos(2*pi*u_2)])

def Box_Muller_sample(n):
    '''draw n samples'''
    return np.array([Box_Muller() for i in xrange(n)])
    
def bivariate_gaussian_1(mean_v, covariance_matrix, n):
    '''draw n samples from a bivariate distribution,
    when a mean and covariance matrix are given'''
    var_1 = covariance_matrix[0][0]
    var_2 = covariance_matrix[1][1]
    corr = covariance_matrix[0][1]/(sqrt(var_1)*sqrt(var_2))
    phi_2 = standard_normal_Gauss(n)
    D = np.array([[sqrt(var_1)*sqrt((1-corr)/2.0), sqrt(var_1)*sqrt((1+corr)/2.0)], [-sqrt(var_2)*sqrt((1-corr)/2.0), sqrt(var_2)*sqrt((1+corr)/2.0)]])
    var = np.array([np.dot(D, i) for i in phi_2])
    return np.array([i + mean_v for i in var]) 


def bivariate_gaussian_2(mu_1, mu_2, var_1, var_2, corr,n):
    '''draw n samples from a bivariate distribution
    by specifically assingning a correlation between
    var 1 and 2.'''
    mean_v = np.array([mu_1, mu_2])
    phi_2 = standard_normal_Gauss(n)
    D = np.array([[sqrt(var_1)*sqrt((1-corr)/2.0), sqrt(var_1)*sqrt((1+corr)/2.0)], [-sqrt(var_2)*sqrt((1-corr)/2.0), sqrt(var_2)*sqrt((1+corr)/2.0)]])
    var =  np.array([np.dot(D, i) for i in phi_2])
    return np.array([i + mean_v for i in var])  

def bivariate_gaussian_3(mu_1, mu_2, var_1, var_2, corr,n):
    '''draw n samples from a bivariate distribution
    by specifically assingning a correlation between
    var 1 and 2. Alternative approach, starting from 3 independent standard normals'''
    mean_v = np.array([mu_1, mu_2])
    phi_2_a = standard_normal_Gauss(n)
    phi_2_b = standard_normal_Gauss(n)
    dirac_p = corr/abs(corr)
    return np.array([[sqrt(var_1)*(sqrt(1-abs(corr))*phi_2_a[i][1]+sqrt(abs(corr))*phi_2_a[i][0]) + mu_1, sqrt(var_2)*(sqrt(1-abs(corr))*phi_2_b[i][0]+dirac_p*sqrt(abs(corr))*phi_2_a[i][0]) + mu_2] for i in xrange(n)])

def Q1(x, mu,var):
    return ((x-mu)**2)/float(var)
     
def gaussian_function(x, mu, var):
    return (1./(sqrt(2*pi*var)))*(exp((-1./2)*Q1(x,mu,var)))

def conditional_bivariate_gaussian_mean(x1,x2,mu1, mu2, var1, var2, corr):
    mu = mu1+((sqrt(var1)/float(sqrt(var2)))*corr*(x2-mu2))
    var = var1*(1-corr**2)
    return gaussian_function(x1,mu ,var)
    
    
    

def gibbs_sampler_2d_Gaussian(sample, mu_1, mu_2,c_r):
    y2 = sample.copy()
    run=[[mu_1, mu_2]]
    conv = 5
    while conv>c_r:
        s2=np.array(run)
        mu_1 = conditional_bivariate_gaussian_mean(y2[0][0], y2[0][1], mu1, mu2, 1,1,0.8)
        y2 = bivariate_gaussian_3(mu1, mu2, 1,1,0.8,1)
        run.append(y2[0])
        mu_2 = conditional_bivariate_gaussian_mean(y2[0][1], y2[0][1], mu1, mu2, 1,1,0.8)
        y2 = bivariate_gaussian_3(mu1, mu2, 1,1,0.8,1)
        run.append(y2[0])
        s1 = np.array(run)
        m = np.array([mean(s2.T[0]), mean(s2.T[1])])
        m2 = np.array([mean(s1.T[0]), mean(s1.T[1])])
        conv = sum(abs(m-m2))
    return np.array(run)



    
