from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
from Cholesky import Cholesky
trm = linalg.get_blas_funcs('trmm')

class Gaussian_variable:
    
    def __init__(self, d, X=None, S=None,  mu=None, method = 'cov'):
        if X is None:
            self.X = np.zeros(d)
        else:
            self.X = X
        self.d = d
        self.method = method
        self.mu = self.__mu(mu)
        self.S = self.__S(S)
    
    def __mu(self,mu=None):
        if mu is None:
            return np.zeros(self.d)
        else:
            return mu
            
    def __S(self, S=None):
        if S is None:
            return np.eye(self.d)
        else:
            return S
    
    def __cov(self):
        if self.S is None:
            return np.eye(self.d)
        elif self.method=='cov':
            return self.S
        elif self.method=='chol_cov':
            return Cholesky(self.S).mat(self.S)
        elif self.method=='prec':
            return Cholesky(self.S).inv()
        elif self.method=='chol_prec':
            return Cholesky(self.S).inv(self.S)    
    
    def __chol_cov(self):
        if self.S is None:
            return np.eye(self.d)
        elif self.method=='cov':
            return Cholesky(self.S).lower()
        elif self.method=='chol_cov':
            return self.S
        elif self.method=='prec':
            return Cholesky(self.S).chol_of_the_inv()
        elif self.method=='chol_prec':
            return Cholesky(self.S).chol_of_the_inv(self.S)    
    
    def __prec(self):
        if self.S is None:
            return np.eye(self.d)
        elif self.method=='cov':
            return Cholesky(self.S).inv()
        elif self.method=='chol_cov':
            return Cholesky(self.S).chol_of_the_inv(self.S)
        elif self.method=='prec':
            return self.S
        elif self.method=='chol_prec':
            return Cholesky(self.S).mat(self.S)
    
    def __chol_prec(self):
        if self.S is None:
            return np.eye(self.d)
        elif self.method=='cov':
            return Cholesky(self.S).chol_of_the_inv()
        elif self.method=='chol_cov':
            return Cholesky(self.S).chol_of_the_inv(self.S)
        elif self.method=='prec':
            return Cholesky(self.S).lower()
        elif self.method=='chol_prec':
            return self.S
    
    def delta(self):
        return self.X-self.mu

    def rvs(self, n=1):
    
        '''returns the a random multivariate Gaussian variable 
        from a mean and the Cholesky decomposition of the precision matrix.
        To be used in Normal-Wishart sampler or similar approaches
        where only the precision matrix is available, avoiding matrix inversions.
        It has a slightly different output than numpy or scipy
        multivariate normal rvs, but has similar statistical properties.
        The algorithm is mentioned in the book 'Handbook of Monte Carlo Methods'
        from Kroese et al.(2011) (algorithm 5.2, pag. 155)'''
        
        
        m = self.__chol_prec().T
        Z = np.random.standard_normal(size=(self.d,n))
        return self.mu + linalg.solve_triangular(m, Z, lower=0, overwrite_b=1, check_finite=0).T

    def logp(self):
        
        '''Multivariate normal probability function.    
        The equation is (Wikipedia or Kevin P. Murphy, 2007):
            (2pi)**(-0.5*d) |prec_m|**(0.5) exp[0.5(X-mu_v)' prec_m (x-mu_v)]
        Should give the same result as scipy.stats.multivariate_normal(mu_v, inv(prec_m)).pdf(X)
        
        Outputs
        --------
        logpdf estimate of X'''
        
        pm = self.__prec()
        cpm = self.__chol_prec()
        det = Cholesky(cpm).log_determinant(cpm)
        delta = self.delta()
        in_exp = delta.T.dot(pm).dot(delta)
        
        
        return (-0.5*self.d)*math.log(2*math.pi) + 0.5*det -0.5*in_exp
    
    def p(self):
        return math.exp(self.logp())
    
        
        

   
