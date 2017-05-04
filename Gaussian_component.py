from __future__ import division
import numpy as np
import math
import scipy.linalg as linalg
from Cholesky import Cholesky
from Gaussian_variable import Gaussian_variable
import random
trm = linalg.get_blas_funcs('trmm')
from scipy.special import gammaln as gamlog


class Gaussian_component:
    
    def __init__(self, d, kappa_0 = 0, v_0=0, mu_0=None, S_0=None, X=None, method='cov'):
        
        if X is None:
            self.n =0
        else:
            self.n=len(X)
        
        self.d = d
        
        self.method=method
        
        self.kappa_0 = kappa_0
        
        self.v_0 = v_0
        
        self.GI = Gaussian_variable(self.d, mu = mu_0, S = S_0, method=self.method)
        
        if mu_0 is None:
            self.mu_0 = self.GI.mu
        else:
            self.mu_0 = mu_0
        
        if S_0 is None:
            self.S_0 = self.GI.S
        else:
            self.S_0 = S_0 
        
        if X is None:
            self.X = self.GI.Xi
            self.X.shape=(1,self.d)
        else:
            self.X = X 
        
        self.sX = None
        
        self.mu= self.__mu()
        
        self.XX_T=None
        
        self.cov = None
        
        self.chol_cov=None
        
        self.prec= None
        
        self.chol_prec=None

    
        
    def __sX(self, X=None):
        if self.n == 0:
            self.sX = np.zeros(self.d)
        elif self.n == 1:
            self.sX = self.X.flatten()
        else:
            self.sX = np.einsum('ij->j', self.X) 
        return self.sX
    
    def __mu(self):
        if self.n is 0:
            self.mu = self.mu_0 
            return self.mu
        else:
            self.mu = (self.kappa_0*self.mu_0 + self.__sX(self.X))/(self.kappa_0+self.n)
            return self.mu

    def __XX_T(self):
        self.XX_T = np.einsum('ij, iz->jz', self.X, self.X)
        return self.XX_T
        
    def __cov(self):
        if self.n is 0:
            self.cov=self.GI._Gaussian_variable__cov()+self.kappa_0*np.einsum('i,j->ji', self.mu_0, self.mu_0)
            return self.cov
        else:
            
            self.cov= self.GI._Gaussian_variable__cov()+self.__XX_T()+self.kappa_0*np.einsum('i,j->ji', self.mu_0, self.mu_0)-(self.kappa_0+self.n)*np.einsum('i,j->ji', self.mu, self.mu)
            return self.cov
            
    def __chol_cov(self):
        self.chol_cov= Cholesky(self.__cov()).lower()
        return self.chol_cov
        
    def __prec(self):
        self.prec= Cholesky(self.__cov()).inv()
        return self.prec
    
    def __chol_prec(self):
        self.chol_prec = Cholesky(self.__cov()).chol_of_the_inv()
        return self.chol_prec

    
    def s_down_date(self, ind_X, cov=False, chol_cov=False, prec=False, chol_prec=False):
        
        if self.n is 0:
            print 'Failled to downdate: component is empty'
            return None
        try:
            dwnX = self.X[ind_X]
        except IndexError:
            print 'Failled to downdate: ind_X not in X'
            return None
        
        if self.n-1==0:
            print 'Component is now empty. Setting parameters to priors'
            self.n=0
            self.mu = self.GI.mu
            if self.cov is None:
                pass
            else:
                self.cov= self.GI._Gaussian_variable__cov()
            if self.chol_cov is None:
                pass
            else:
                self.chol_cov = self.GI._Gaussian_variable__chol_cov()
            if self.prec is None:
                pass
            else:
                self.prec= self.GI._Gaussian_variable__prec()
            if self.chol_prec is None:
                pass
            else:
                self.chol_prec = self.GI._Gaussian_variable__chol_prec()
            
            return None
            
        n_c = self.n        
        self.n-=1
        mu_c = self.mu.copy()

        self.mu = (((self.kappa_0+n_c)*mu_c)-dwnX)/(self.kappa_0+self.n)


        if self.sX is None:
            pass
        else:
            self.sX-=dwnX
        
                
        if self.X is None:
            pass
        else:
            ind =np.ones(len(self.X), dtype=np.bool)
            ind[ind_X]=0
            self.X = self.X[ind]
        
        if self.XX_T is None:
            pass
        else:
            self.XX_T -= np.einsum('i,j->ij', dwnX, dwnX)
        
        if self.cov is None:
            pass
        elif cov:
            a= self.__cov()
        else:
            pass
        
        if self.chol_cov is None:
            pass
        elif chol_cov:
            up1 = math.sqrt(self.kappa_0+n_c)*mu_c
            down1 = math.sqrt(self.kappa_0+self.n)*self.mu
            self.chol_cov = Cholesky(self.chol_cov).r_1_downdate(dwnX, chol_A = self.chol_cov)
            self.chol_cov = Cholesky(self.chol_cov).r_1_update(up1, chol_A = self.chol_cov)
            self.chol_cov = Cholesky(self.chol_cov).r_1_downdate(down1, chol_A=self.chol_cov)
        else:
            pass
                    
        if self.prec is None:
            pass
        elif prec:
            a= self.__prec()
        else:
            pass
        
        if self.chol_prec is None:
            pass
        elif chol_prec:
            a= self.__chol_prec()
        else:
            pass
            
    
    
    def s_up_date(self, Xi, cov=False, chol_cov=False, prec=False, chol_prec=False):
        
        Xi=Xi.flatten()         

        if self.n==0:
            self.n=1
            self.X = Xi
            self.mu = self.__mu()
            if self.XX_T is None:
                pass
            else:
                self.XX_T= self.__XX_T()
            
            if self.cov is None:
                pass
            else:
                self.cov= self.__cov()
            
            if self.chol_cov is None:
                pass
            else:
                self.cov= self.__chol_cov()
            
            if self.prec is None:
                pass
            else:
                self.prec= self.__prec()
                                    
            
            if self.chol_prec is None:
                pass
            else:
                self.chol_prec = self.__chol_prec()
            
            return None
        
        
        n_c = self.n        
        self.n+=1
        mu_c = self.mu.copy()
        
        

        self.mu = (((self.kappa_0+n_c)*mu_c)+Xi.flatten())/(self.kappa_0+self.n)

        if self.sX is None:
            pass
        else:
            self.sX+=Xi.flatten()
        
                
        if self.X is None:
            pass
        else:
            self.X = np.concatenate([self.X, [Xi]])
        
        if self.XX_T is None:
            pass
        else:
            self.XX_T += np.einsum('i,j->ij', Xi, Xi)
        
        if self.cov is None:
            pass
        elif cov:
            a= self.__cov()
        else:
            pass
        
        if self.chol_cov is None:
            pass
        elif chol_cov:
            up1 = math.sqrt(self.kappa_0+n_c)*mu_c
            down1 = math.sqrt(self.kappa_0+self.n)*self.mu
            self.chol_cov = Cholesky(self.chol_cov).r_1_update(Xi, chol_A = self.chol_cov)
            self.chol_cov = Cholesky(self.chol_cov).r_1_update(up1, chol_A = self.chol_cov)
            self.chol_cov = Cholesky(self.chol_cov).r_1_downdate(down1, chol_A=self.chol_cov)
        else:
            pass
            
        if self.prec is None:
            pass
        elif prec:
            a= self.__prec()
        else:
            pass
        
        if self.chol_prec is None:
            pass
        elif chol_prec:
            a= self.__chol_prec()
        else:
            pass
        
    
    def __normalizing_Z(self, d,kappa,v,S):
        
        if v<d:
            df = d+1
        else:
            df= v+1
        if kappa==0:
            kt=0
        else:
            kt = 0.5*d*math.log(kappa)
        return (0.5*df*d*math.log(2))+ ((0.25*((d**2)+d))*math.log(math.pi)) - kt \
        -(((df-1)/2)*Cholesky(S).log_determinant(chol_A=S))+sum(gamlog(df-np.arange(d)))
        
    def chol_prec_rvs(self):
        if self.v_0+self.n<self.d+1:
            df = self.d+1
        else:
            df = self.v_0+self.n
        
        if self.chol_prec is None:
            a = self.__chol_prec()
        ind = np.tril_indices(self.d,-1) #lower triangular non-diagonal elements
        B = np.zeros((self.d,self.d)) 
        norm = np.random.standard_normal(len(ind[0])) #normal samples for the lower triangular non-diagonal elements
        B[ind] = norm 
        chisq = [math.sqrt(random.gammavariate(0.5*(df-(i+1)+1), 2.0)) for i in xrange(self.d)]
        B = B+np.diag(chisq)
        ch_d =trm(alpha=1, a=self.chol_prec, b=B, lower=1)
        dg= np.diag(ch_d)#Assuring the result is the Cholesky decomposition that contains positive diagonals
        adj = np.tile(dg/abs(dg), (self.d,1))
        return ch_d*adj

    def mu_rvs(self):
        
        return Gaussian_variable(self.d, mu=self.mu, S = math.sqrt(self.kappa_0+self.n)*self.chol_prec_rvs(), method='chol_prec').rvs()

    def rvs(self, n):
        return Gaussian_variable(self.d, mu=self.mu_rvs(), S = self.chol_prec_rvs(), method='chol_prec').rvs(n)

    def rvs_Gibbs(self, n):
        s = {i:Gaussian_variable(self.d, mu=self.mu_rvs(), S = self.chol_prec_rvs(), method='chol_prec').rvs()[0] for i in xrange(n)}
        return np.array(s.values())

        

