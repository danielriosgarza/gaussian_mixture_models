from __future__ import division
from pylab import *
import numpy as np
import scipy.stats as sts
import math
import scipy.special as spe
import scipy.linalg.lapack as lpack
import scipy.linalg as linalg
#%alias_magic t timeit
trm = linalg.get_blas_funcs('trmm') #multiply triangular matrices. If lower use lower=1.



def multivariate_t_rvs_chol(mu, L, df, n=1):
    '''generate a random variable from multivariate t ditribution inputing directly the
    Cholesky decomposition of the covariace matrix.
    Parameters
    ----------
    mu : array_like
        mean of random variable, length determines dimension of random variable
    L : array_like
        Cholesky decomposition of the covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    d = len(mu)
    if df == np.inf:
        x = 1.
    else:
        x = np.sqrt(df/np.random.chisquare(df, n))
    return np.array([mu+(L.dot(np.random.standard_normal(d))*x[i]) for i in xrange(n)])



def multivariate_t_rvs(mu, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution

    Parameters
    ----------
    mu : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(mu))
        each row is an independent draw of a multivariate t distributed
        random variable


    '''
    mu = np.asarray(mu)
    d = len(mu)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return mu + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

def cholesky_r1_update(L, X):
    '''perform a rank 1 update of
    the cholesky decomposition of positive definite
    a matrix.
    
    L --> dxd lower triangular matrix, such that A = LL'
    X---> d dimensional vector, such that A* = A + XX'
    '''
    
    l = L.copy()
    x = X.copy()
    d = len(x)
    for k in xrange(d):
        r = math.sqrt((l[k,k]**2)+(x[k]**2))
        c = r/l[k,k]
        s = x[k]/l[k,k]
        l[k,k] = r
        for i in xrange(k+1, d):
            l[i, k] = (l[i,k]+ (s*x[i]))/c
            x[i] = (x[i]*c)-(s*l[i,k])
    return l


def gamma_pdf(x, alpha, theta, log_form=False):
    '''PDF of the gamma distribution. 
    is equivalent to scipy.stats.gamma(a=alpha, scale=theta).pdf(x)
    
    x--> scalar quantity representing an observation
    alpha, theta ---> scalar parameters'''
    
    t1 = -spe.gammaln(alpha)
    t2 = -alpha*math.log(theta)
    t3 = (alpha-1)*math.log(x)
    t4 = -x/theta
    if log_form:
        return t1+t2+t3+t4
    else:
        return math.exp(t1+t2+t3+t4)



def Gaussian_pdf(x, mu, precision, log_form = False):
    delta = (x-mu)**2
    t1 = math.log(0.5*(precision/math.pi))
    t2 = -precision*delta
    
    if log_form:
        return 0.5*(t1+t2)
    else:
        return math.exp(0.5*(t1+t2))
    
def estimate_convergence(data_set, simulation_set):
    '''convergence estimator for scalar quantities,
    based on the within and between variances of
    data and simulated sets'''
    #constants
    d_n = len(data_set)
    s_n  = len(simulation_set)
    n = d_n+s_n
    n_m = 0.5*n
    mean_d = mean(data_set)
    mean_s = mean(simulation_set)
    mean_t = 0.5*(mean_d+mean_s)
    var_d = (1./(d_n-1))*sum((data_set-mean_d)**2)
    var_s = (1./(s_n-1))*sum((simulation_set-mean_d)**2)
    
    #Between variance
    B = n_m*(((mean_d-mean_t)**2)+((mean_d-mean_t)**2))
    
    #within variance
    W = 0.5*(var_d+var_s)
    
    #marginal posterior variance estimand
    var_mp = ((n-1.)/n)*W + (1./n)*B
    
    #convergence estimate (converges to 1)
    return sqrt(var_mp/W)


def chol_determinant(L):
    '''compute the determinant of a cholesky decomposition.
    if A = LL', the function returns det(A)'''
    return np.prod(diag(L)**2)

def chol_log_determinant(L):
    '''compute the determinant of a cholesky decomposition.
    if A = LL', the function returns det(A)'''
    return np.sum(2*log(diag(L)))


        
def wishart_pdf(X, S, v, d, chol=False, log_form = False):
    '''Wishart probability density with possible use of the cholesky decomposition of S.
    Returns the same output as scipy.stats.wishart(df=v, scale=S).pdf(X).
    
    The equation is (Wikipedia or Kevin P. Murphy, 2007):
        {|X|**[0.5(v-d-1)] exp[-0.5tr(inv(S)X)]}/{2**[0.5vd] |S|**[0.5v] [multivariate_gamma_function(0.5v, d)]}
    
    Thomas Minka (1998) has a different form for the equation, but both are equivalent for the same inputs:
        {1}/{[multivariate_gamma_function(0.5v, d)] |X|**(0.5(d+1))} {|0.5X inv(S)|**(0.5v)} {exp[-0.5tr(inv(S)X)]}
        
    Parameters
    ----------
    X: array-like. 
    Positive definite dxd matrix for which the probability function is to be estimated.
    If chol, this must be the matrix L, instead. L is a lower triangular decomposition of X, such that X = LL'.
    
    S:array-like
    Positive definite dxd scale matrix
    If chol, this must be the matrix L2, instead. L2 is a lower triangular decomposition of S, such that S = L2L2'
    
    v: int or float.
    degrees of freedom for the distribution. v must be >d
    
    d: int
    dimension of each row or column of X
    
    
    Outputs
    --------
    If log_form returns the logpdf estimate of X, else it returns the pdf estimate of X
    '''
    
    #constants
    
    if chol:
        det_X = chol_log_determinant(X)
        det_S = chol_log_determinant(S)
        iS = lpack.dtrtri(S, lower=1)[0]
        trace = np.einsum('ij,ji', iS.T.dot(iS), X.dot(X.T))

           
    else:
        det_X = np.linalg.slogdet(X)[1]
        det_S = np.linalg.slogdet(S)[1]
        trace = np.trace(np.linalg.inv(S).dot(X))

    #
    
       
    p1 = 0.5*(v-d-1)*det_X
    p2 = -0.5*trace
    p3 = -0.5*(v*d)*math.log(2)
    p4 = -0.5*(v)*det_S
    p5 = -spe.multigammaln(0.5*v,d)
    
    if log_form:
        return p1+p2+p3+p4+p5
    else:
        return math.exp(p1+p2+p3+p4+p5)
    
    
def invwishart_pdf(X, S, v, d, chol=False, log_form = False):
    '''Inverse Wishart probability density with possible use of the cholesky decomposition of S and X.
    Returns the output that is comparable to scipy.stats.invwishart(df=v, scale=S).pdf(X). 
    
    The equation is (Wikipedia or Kevin P. Murphy, 2007):
        {|S|**[0.5v] |X|**[-0.5(v+d+1)] exp[-0.5tr(S inv(X))]}/{2**[0.5vd]  [multivariate_gamma_function(0.5v, d)]}
    
     
    Parameters
    ----------
    X: array-like. 
    Positive definite dxd matrix for which the probability function is to be estimated.
    If chol, this must be the matrix L, instead. L is a lower triangular decomposition of X, such that X = LL'.
    
    S:array-like
    Positive definite dxd scale matrix
    If chol, this must be the matrix L2, instead. L2 is a lower triangular decomposition of S, such that S = L2L2'
    
    v: int or float.
    degrees of freedom for the distribution. v must be >d
    
    d: int
    dimension of each row or column of X
    
    
    Outputs
    --------
    If log_form returns the logpdf estimate of X, else it returns the pdf estimate of X
    '''
    
    #constants
    
    if chol:
        det_X = chol_log_determinant(X)
        det_S = chol_log_determinant(S)
        iX = lpack.dtrtri(X, lower=1)[0]
        trace = np.einsum('ij,ji', S.dot(S.T),iX.T.dot(iX))

           
    else:
        det_X = np.linalg.slogdet(X)[1]
        det_S = np.linalg.slogdet(S)[1]
        trace = np.trace(S.dot(np.linalg.inv(X)))

    #
    
    p1 = -0.5*(v*d)*math.log(2)
    p2 = -spe.multigammaln(0.5*v,d)
    p3 = 0.5*(v)*det_S
    p4 = -0.5*(v+d+1)*det_X
    p5 = -0.5*trace
       
    
    if log_form:
        return p1+p2+p3+p4+p5
    else:
        return math.exp(p1+p2+p3+p4+p5)
 
def multivariate_normal_pdf(X, mu_v, prec_m, chol=False, log_form=False):
    

    #constants
    d =len(mu_v)
    delta = X-mu_v
    if chol:
        det = chol_log_determinant(prec_m)
        inexp = delta.dot(prec_m).dot(trm(1,prec_m.T,delta))

    else:
        det = np.linalg.slogdet(prec_m)[1]
        inexp = delta.dot(prec_m).dot(delta)
    
    p1 = (-0.5*d)*math.log(2*math.pi)
    p2 = 0.5*det
    p3 = -0.5*inexp
    
    if log_form:
        return p1+p2+p3
    else:
        return exp(p1+p2+p3)

    
