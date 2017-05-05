from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
from Cholesky import Cholesky
from Gaussian_variable import Gaussian_variable
import random
trm = linalg.get_blas_funcs('trmm')
from scipy.special import gammaln as gamlog




class Gauss_Wishart_probability_model:
    def __init__(self):
        self.prec_mu_norm_Z = None
        self.prec_norm_Z = None
        self.mu_norm_Z = None
        self.prior_lp_prec = None
        self.prior_lp_mu = None
        self.prior_lp = None
            
    def __prec_mu_norm_Z(self,S, kappa,v, d):
        if v<d:
            df = d+1
        else:
            df= v+1
        
        if kappa==0:
            kt=0
        else:
            kt = 0.5*d*math.log(kappa)
        
        self.prec_mu_norm_Z = (0.5*df*d*math.log(2))+ ((0.25*((d**2)+d))*math.log(math.pi)) - kt \
        -(((df-1)/2)*Cholesky(S).log_determinant(chol_A=S))+sum(gamlog(df-np.arange(d)))
        
        return self.prec_mu_norm_Z
    
    def __prec_norm_Z(self, S,v,d):
        '''normalizing constant for the probability that the precision matrix is Lamda, given the prior.
        This is modelled as Wi(v, S). The normalizing constant is 1/Z and 
        Z = [2**(0.5*v*d)][det(S)**(-0.5*v)][prod(gamma_func(v+1-i)) for i in xrange(d)].'''
        
        
        if v<d:
            df = d+1
        else:
            df= v+1
        
        self.prec_norm_Z = (0.5*(df-1)*d*math.log(2))-((0.5*(df-1))*Cholesky(S).log_determinant(chol_A=S)) + ((0.25*d*(d-1))*math.log(math.pi))\
        +sum(gamlog(df-np.arange(d)))
         
        return self.prec_norm_Z

    def __mu_norm_Z(self, kappa,d):
        
        if kappa==0:
            kt=0
        else:
            kt = 0.5*d*math.log(kappa)
        
        self.mu_norm_Z = (-0.5*d*(math.log(2*math.pi)))+kt
        
        return self.mu_norm_Z
        
    def prior_lp_prec_(self, prec,S, v,d):
        if v<d:
            df = d+1
        else:
            df= v+1

        self.prior_lp_prec = -self.__prec_norm_Z(S, v, d)+(0.5*(df-d-2)*Cholesky(prec).log_determinant())-(0.5*np.einsum('ij, ij', prec, S))
        
        return self.prior_lp_prec

    def prior_lp_mu_(self, prec, mu, mu_0, kappa,d):
        
        self.prior_lp_mu = self.__mu_norm_Z(kappa, d)+(0.5*Cholesky(prec).log_determinant())-(0.5*np.einsum('ij, ij', kappa*prec, np.einsum('i,j->ij', mu-mu_0, mu-mu_0)))
        
        return self.prior_lp_mu

    def prior_lp_(self, prec, mu, mu_0, S, kappa,v,d):
        
        if v<d:
            df = d+1
        else:
            df= v+1
        
        self.prior_lp = -self.__prec_mu_norm_Z(S, kappa, v,d)+((0.5*df-1-d)*Cholesky(prec).log_determinant())\
        -(0.5*np.einsum('ij, ij', prec, (kappa*np.einsum('i,j->ij', mu-mu_0, mu-mu_0))+S))
        
        return self.prior_lp
