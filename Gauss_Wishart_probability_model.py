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
        self.normalizing_Z = None
        
            
    def __normalizing_Z(self, d,kappa,v,S):
            
            if v<d:
                df = d+1
            else:
                df= v+1
            if kappa==0:
                kt=0
            else:
                kt = 0.5*d*math.log(kappa)
            self.normalizing_Z = (0.5*df*d*math.log(2))+ ((0.25*((d**2)+d))*math.log(math.pi)) - kt \
            -(((df-1)/2)*Cholesky(S).log_determinant(chol_A=S))+sum(gamlog(df-np.arange(d)))
            return self.normalizing_Z 
