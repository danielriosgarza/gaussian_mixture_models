from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import math
import scipy.special as spe
import scipy.linalg.lapack as lpack
import scipy.linalg as linalg
from IPython.display import display, Math, Latex
import random
trm = linalg.get_blas_funcs('trmm') #multiply triangular matrices. If lower use lower=1.

class Cholesky:
    def __init__(self, A):
        self.matrix = A
        self.d = len(A)
        self.upper = linalg.cholesky(A, lower=0, check_finite=0, overwrite_a=0)
        self.lower =  linalg.cholesky(A, lower=1, check_finite=0, overwrite_a=0)

    def  chol_of_the_inverse(self, chol_A=None):
        if chol_A is None:
            chol_A=self.lower
        self.v1 = linalg.solve_triangular(chol_A, np.eye(self.d), lower=1,trans=0, overwrite_b=1,check_finite=0)
        self.v2 = linalg.solve_triangular(chol_A.T, self.v1, lower=0,trans=0, overwrite_b=1,check_finite=0)
        return linalg.cholesky(self.v2, lower=1, check_finite=0, overwrite_a=1)

    def inverse(self, chol_A=None):
        if chol_A is None:
            chol_A=self.lower
        self.choliA = self.chol_of_the_inverse(chol_A)
        return trm(alpha=1, a=self.choliA, b=self.choliA.T,lower=1)

