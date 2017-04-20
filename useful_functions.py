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

#https://github.com/danielriosgarza/pychud
#sudo apt-get install gfortran
#python setup.py install
try:
    import pychud
    pychud_im =True
except ImportError:
    pychud_im = False

#np.set_printoptions(precision=5)
#%alias_magic t timeit
trm = linalg.get_blas_funcs('trmm') #multiply triangular matrices. If lower use lower=1.


def fancy_inversion(A):
    d = len(A)
    m = np.fliplr(np.flipud(A))
    c =  linalg.cholesky(m, lower=1, check_finite=0,overwrite_a=1)
    inv_c = linalg.solve_triangular(c, np.eye(d), lower=1,trans=0, overwrite_b=1,check_finite=0)
    n = np.fliplr(np.flipud(inv_c))
    return trm(alpha=1, a=n.T, b=n,lower=1)

def inv_and_chol(A, chol_of_A = False, chol_of_invA=False):
    '''return the cholesky factorization and the inverse of matrix A.
    Or the inverse and the Cholesky factorization of the inverse.
    Has the same precision as np.linalg, with a possible slight gain of speed
    
    Parameters
    ---------
    A: array-like
    square positive definite matrix
    
    chol_of_A: boolean
    if the Cholesky decomposition of A should be returned.
    
    chol_of_invA: boolean
    if the Cholesky decomposition of the inverse of A should be returned.

    Output
    -------
    
    if chol_of_A=0 and chol_of_invA=1, returns only the inverse of A.
    if chol_of_A=1 and chol_of_invA=0 returns (1)Lower Cholesky dec of A, (2) inverse of A
    if chol_of_A=0 and chol_of_invA=1 returns (1)Lower Cholesky dec of inverse of A, (2) inverse of A
    if chol_of_A=1 and chol_of_invA=1 returns (1)Lower Cholesky dec of A, (2) Lower Cholesky dec of inverse of A
    (3)inverse of A'''
    
    c =  linalg.cholesky(A, lower=1, check_finite=0,overwrite_a=1)#cholesky factorization
    d = len(A)
    if chol_of_invA and not chol_of_A:
        choliA = chol_of_the_inverse(c)
        return choliA, trm(alpha=1, a=choliA, b=choliA.T,lower=1)

    elif chol_of_A and not chol_of_invA:
        icholA = linalg.solve_triangular(c, np.eye(d), lower=1,trans=0, overwrite_b=1,check_finite=0) #solve for the inverse
        return c, trm(alpha=1, a=icholA.T, b=icholA,lower=0)
         
    elif chol_of_A and chol_of_invA:
        choliA = chol_of_the_inverse(c)
        return c, choliA, trm(alpha=1, a=choliA, b=choliA.T,lower=1)    
    
    else:
        icholA = linalg.solve_triangular(c, np.eye(d), lower=1,trans=0, overwrite_b=1,check_finite=0) #solve for the inverse
        return trm(alpha=1, a=icholA.T, b=icholA,lower=0)



def chol_of_the_inverse(cholA):
    '''Take a Cholesky of matrix A and return the Cholesky
    decomposition of the inverse of A. The algorithm is based on the antidiagonal 
    matrix J and on the folowing equality:
        If JAJ = LL', then the Cholesky decomposition of 
        inv(A) is given by (J(inv(L))J)'. While inv(L) is related
        to the Cholesly decomposition of A (and not on JAJ), k by:
            inv(L) =inv(Jkk'J). Notice the JA = flipupdown(A) and
            AJ = flipleftright(A). 
    The output is the same as np.linalg.choleske(np.linealg.inv(cholA.dot(cholA.T))).

    Paramerter
    --------
    chlA: array-like
    lower triangular Cholesky decomposition of a matrix
    
    Output
    --------
    Lower triangular Choloesky decomposition of the inverse of matrix A'''
    
    d = len(cholA)
    p1 = np.flipud(cholA)
    p2 = np.fliplr(cholA.T)
    p3 = p1.dot(p2)
    p4 = linalg.cholesky(p3, lower=1, check_finite=0,overwrite_a=1)  
    p5 = linalg.solve_triangular(p4, np.eye(d), lower=1,trans=0, overwrite_b=1,check_finite=0)
    return np.fliplr(np.flipud(p5)).T


def chol_of_the_inverse2(cholA):
    '''same as above, usually the fastest'''
    d = len(cholA)
    v1 = linalg.solve_triangular(cholA, np.eye(d), lower=1,trans=0, overwrite_b=1,check_finite=0)
    v2 = linalg.solve_triangular(cholA.T, v1, lower=0,trans=0, overwrite_b=1,check_finite=0)
    return np.linalg.cholesky(v2)

def chol_of_the_inverse3(cholA):
    A = cholA.dot(cholA.T)
    iA = np.linalg.inv(A)
    return np.linalg.cholesky(iA)

def scatter_matrix(X, E_mu = False, P_mu=None, P_k=1):
    '''get the scatter matrix of multidimensional data.
    with possibility of returning the empirical mean, since it's one of the steps in the
    computation of the scatter matrix. Can be used with a prior mean (see 'store_sufficient_statistic' function).

    The scatter matrix is positive semi-definite matrix defined by:
    sum[(Xi-mean(X))' *(Xi-mean(X))].
    The approach used in this function is based on the centering matrix(Wikipedia):
        - Centering Matrix: C= I(n) - [n**(-1)]*11' 
        (I(n) is n-dimensional identity matrix 11' is nxn matrix of ones.)
        - Scatter matrix = XCX'
        - Empirical mean = X-XC
    If a prior mean is provided. The approach is based in Thomas Minka(1998, sec. 7) and mounts 
    to subtracting the prior mean from each data component and dividing by n + P_k. If used,
    The final S is sum{[Xi-(P_k*mu_0 + n*mean(X))/(P_k+n)]' *[Xi-(P_k*mu_0 + n*mean(X))/(P_k+n)]}: 
    
    Parameters
    ---------
    data_set: array-like
    multidimensional data. N columns of data-points in M dimensions. This means that the input array needs to be 
    in the shape of (M,N).
    
    Output
    ---------
    returns the scatter matrix of the data in array-like format and the empirical mean if required.'''
    
    ds = np.array(X)
    N = size(ds[0]) #number of samples
    dim = len(ds) #dimensions
    if type(P_mu) is np.ndarray:
        ds = ds-P_mu.dot(np.ones((1,N))) #subtract prior mean from X
        C = np.eye(N) - (1./(N+P_k))*np.ones(N) #centering matrix considering P_k
    else:
        C = np.eye(N) - (1./N)*np.ones(N) #centering matrix
        
    if  E_mu:
        E = ds.dot(C)#X-E is the empirical mean
        return E.dot(ds.T), (X-E).T[0]
    else :
        return ds.dot(C).dot(ds.T)

def store_sufficient_statistics_mvGaussian(X, P_mu=None, P_k=1):
    '''stores sufficient statistics to evaluate the likelihood of a dataset under a Gaussian
    multivariate probability model. Can be used with the function 'multivariate_Gaussian_likelihood'
    or other similar functions.
    It can handle a prior mean and 'pseudo-counts' , and thus store the sufficient statistics of a seperate
    subset of data. For detailes see Thomas Minka(1998, sec. 7).
    
    
    Parameters
    --------
    X: array-like
    Vector of observations assumed to be iid mvGaussian. len(X) should return the number of observations, and len(X[0]) 
    the dimensionallity. (shape = (n, d)). To assure the correct dimensionality in the dot products, this is transposed
    in the function to retrieve the scatter_matrix.
    
    P_mu: array-like. *optional
    A prior mean or a mean from previous observations. If used, the sufficient statistic will be stored taking into account
    a weighted average of prior plus sample
    
    P_k: int float
    Weight given as 'confidence' for the prior, or pseudo counts, or previous samples when evaluating the posterior
    predictive probability of a new set of observations
    
    Output
    --------
    A dictionary with the sufficient statistics to be used in the likelihood function. 'n' = P_k+n, 
    'E_mu=[(P_k*mu_0)+(n*mu)]/[P_k+n], S_m=scatter matrix'''
    
    n = len(X) #number of data_points
    if type(P_mu) is np.ndarray:
        S_m, E_mu= scatter_matrix(X.T, E_mu=1,P_mu=P_mu, P_k=P_k) #scatter matrix, empirical mean
        n+=P_k #add the prior counts
    else:
        S_m, E_mu= scatter_matrix(X.T, E_mu=1)
    
    return {'n':n, 'E_mu':E_mu, 'S_m' :S_m}


def sufficient_statistics_for_Gaussian_likelihood(X):
    n = len(X)
    d = len(X[0])
    sX = np.einsum('ij->j', X)
    C=np.einsum('ij,jk->ik',X.T,X)
    em = sX/n
    cC = np.linalg.cholesky(C)
    invcC = chol_of_the_inverse(cC)
    sm = cholesky_r1_update(cC, math.sqrt(n)*em, down=1)
    return {'n':n, 'd':d, 'sX':sX, 'C_m':C, 'em':em, 'Chol_sm':sm, 'invChol_sm':sm}

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

def cholesky_r1_update(L, X, down=False, pychud_im=pychud_im):
    '''perform a rank 1 update or downdate of the cholesky decomposition of a positive definite
    a matrix.
    If available, can use the pychud package for optimal performance. Otherwise it will still compute
    rank1 updates, bu using a double loop.
    Parameters
    --------
    L :array-like
    dxd lower triangular matrix, such that A = LL'.
    X: array-like
    d dimensional vector, such that A* = A + XX' or if down is selected A* = A - XX'
    down: Boolean
    If downgrade is desired.
    
     Output
     --------
     Lower triangular rank 1 up or downdated cholesky factorization of matrix A=LL'. 
    '''
    if pychud_im:
        if down:
            return pychud.dchdd(L.T, X, overwrite_r=False)[0].T #if L is to be stored, set overwrite_r to False.
        else:
            pychud.dchud(L.T, X, overwrite_r=False).T
    else:
        l = L.copy()
        x = X.copy()
        d = len(x)
        for k in xrange(d):
            if down:
                a=abs(l[k,k])
                b=abs(x[k])
                num = min([a,b])
                den = max([a,b])
                t = num/den
                r = den*math.sqrt(1-(t*t))
            else:
                r = math.sqrt((l[k,k]**2)+(x[k]**2))
            c = r/l[k,k]
            s = x[k]/l[k,k]
            l[k,k] = r
            for i in xrange(k+1, d):
                if down:
                    l[i, k] = (l[i,k]- (s*x[i]))/c
                else:
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




    
def estimate_convergence(data_set, simulation_set):
    '''convergence estimator for scalar quantities,
    based on the within and between variances of
    data and simulated sets

    Parameters
    --------
    
    data_set: array-like
    scalar quantity of measured data
    
    simulation: array-like
    simulation of the data-set by some Monte Carlo sampling algorithm
    
    Output
    --------
    score measuring the Between and within vairiance.
    If comparable should be close to one'''
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


def retrieve_convergence_curves(i0, data_set, simulation_set):
    '''retrieve a curve with convergence estimate of for t simulation points.
    
    Parameters
    --------
    i0: int or float
    initial cutoff to begin evaluating convergence from 0-i0.

    data_set: array-like
    scalar quantity of measured data
    
    simulation: array-like
    simulation of the data-set by some Monte Carlo sampling algorithm
    
    Output
    --------
    x-points for the course of the convergence in time points of the simulation.'''
    
    x = [estimate_convergence(data_set, simulation_set[0:i]) for i in xrange(i0,len(simulation_set))]
    return np.array(x)

def plot_multivariate_convergence(s_dict, data_set, i0):
    '''plot convergence curves for multivariate data'''
    s = s_dict.values()
    s = np.array(s)
    for i in xrange(len(s.T)):
        plot(retrieve_convergence_curves(i0, data_set.T[i], s.T[i]))
    
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
    
def Wishart_rvs(df, S, chol=0):
    '''generate random Wishart distributed variables, with possibility of using the cholesky decomposition.
    same result as sts.wishart(df=df, scale=S).rvs() with a gain of speed, particularly for large matrices. 
    Parameter
    -------
    df: integer
    degrees of freedom, assumed to be greated than the dimensions.
    S: array-like
    Positive definite scaling matrix. If chol, this matrix is assumed to be the lower triangular Cholesky 
    decomposition of S, L. such that S = LL'.
    
    Output
    --------
    A Wishart distributed positive definite random matrix. If one a assumes that the scale matrix S is a covariance matrix, 
    then the function returns a random Wishart matrix, W,  where W is a random precision matrix 
    and if chol is selected, the function returns the cholesky decomposition of W'''
    
    d = S.shape[0] #dimensions
    ind = np.tril_indices(d,-1) #lower triangular non-diagonal elements
    B = np.zeros((d,d)) 
    norm = np.random.standard_normal(len(ind[0])) #normal samples for the lower triangular non-diagonal elements
    B[ind] = norm 
    chisq = [math.sqrt(random.gammavariate(0.5*(df-(i+1)+1), 2.0)) for i in xrange(d)]
    B = B+diag(chisq)
    
    if chol:
        ch_d =trm(alpha=1, a=S, b=B, lower=1)
        #ch_d = S.dot(B)
        dg= diag(ch_d)#Assuring the result is the Cholesky decomposition that contains positive diagonals
        adj = np.tile(dg/abs(dg), (d,1))
        return ch_d*adj
    else:
        ch = linalg.cholesky(S, lower=1, check_finite=0)
        m = trm(alpha=1, a=ch, b=B, lower=1)
        return m.dot(m.T) 
    
            
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

def Gaussian_pdf(x, mu, precision, log_form = False):
    '''Univariate Gaussian probability density function. Parametrized by the precision.
    t = r'if  $x \sim \mathcal{N}(\mu, \lambda^{2})$ then $f(x)=\frac{\lambda}{\sqrt{2 \pi }}'
    t+=r'exp(- \frac{\lambda^2}{2 }(x-\mu)^2)$'
    text(0.1, 0.5,t)
    
    Returns the same result as scipy.stats.normal(loc=mu, scale=sqrt(1./precision)).pdf(x)
    Parameters
    --------
    x: float-like
    scalar for which the probability is to be estimated.
    mu: float-like
    scalar representing the mean or center of the distribution
    precision: float-like
    scalar representing the precision of the inverse of the variance
    log_form: Bolean
    
    Output
    --------
    If log_form is selected, returns the log of the probability density 
    evaluated at x, the density is returned otherwise.''' 

    delta = (x-mu)**2
    t1 = 0.5*math.log(precision)
    t2= -0.5*math.log(2*math.pi)
    t3 = -0.5*precision*delta
    
    if log_form:
        return t1+t2+t3
    else:
        return math.exp(t1+t2+t3) 

def Gaussian_likelihood(x_vec, mu, precision, log_form=False):
    '''return the likelihood of a set of scalar-valued samples, assumed to be iid univariate Gaussian.
     This function is parametrized by the precision matrix. See Gaussian_pdf for the relavant
     mathematical formula.
     
     Parameters
     ---------
     x_vec: array-like
     one dimensional array of n 'observed' data-points for which the likelihood is to be estimated.
     mu: float-like
     scalar representing the mean or center of the distribution
     precision: float-like
     scalar representing the precision of the inverse of the variance
     log_form: Bolean
    
     Output
     --------
     Joint likelihood of data-points in x_vec. If log_form, then the sum of loglikelihoods, else the product of 
     likelihoods.'''
    n = len(x_vec)
    x = np.array(x_vec)
    delta = (x-mu)**2
    s_delta = sum(delta)
    
    p1 = 0.5*n*math.log(precision)
    p2 = -0.5*n*math.log(2*math.pi)
    p3 = -0.5*precision*s_delta
    
    if log_form:
        return p1+p2+p3
    
    else:
        return math.exp(p1+p2+p3)
        
        
def multivariate_Gaussian_pdf(X, mu_v, prec_m, chol=False, log_form=False):
    '''Multivariate normal probability function with possibility of using the cholesky decomposition
    of the precision matrix. The function is paraterized by the precision instead of the covariance matrix.
    This avoids the need to take the inverse of the covariace, which gives a significant gain in speed. The 
    cholesky parametrization is aprox. 5 times faster because of the determinant. Even with the additional 
    dot product.
    
    The equation is (Wikipediaor Kevin P. Murphy, 2007):
        (2pi)**(-0.5*d) |prec_m|**(0.5) exp[0.5(X-mu_v)' prec_m (x-mu_v)]
    Should give the same result as scipy.stats.multivariate_normal(mu_v, inv(prec_m)).pdf(X)
    Parameter
    --------
    X: array-like
    d-dimensional vector of observations assumed to come from a multivariate Gaussian with the
    same number of components.
    
    mu_v: array-like
    d-dimensional vector of the mean parameter for the Gaussian.
    
    prec_m: array-like
    dXd precision matrix. matrix must be symmetric positive definite and is the inverse of
    the covariance matrix. prec_m = inv(cov_matrix). If the cholesky parametrization is
    selected, this parameter must be the lower triangular decomposition of the precision
    matrix, L. prec_matrix = LL'
    
      Outputs
    --------
    If log_form returns the logpdf estimate of X, else it returns the pdf estimate of X
    '''
   #constants
    d =len(mu_v) #dimension
    delta = X-mu_v
    if chol:
        det = chol_log_determinant(prec_m)
        inexp = delta.dot(prec_m).dot(trm(1,prec_m.T,delta, lower=1))

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

def multivariate_Gaussian_likelihood_prec(SS_dict, prec_m, chol=0, log_form = 0):
    '''Likelihood function for a Gaussian multivariate probability model computed
    from cached sufficient statistic. Possible to use the cholesky decomposition to speed
     the calculation of the determinant (At the cost of one extra dot product).
     
     Parameters
     --------
     SS_dict: python-dict
     Cached sufficient statistic for the likelihood model. Is retrieved as an output from the function
     'store_sufficient_statistics_mvGaussian';  'n' = P_k+n,  'E_mu=[(P_k*mu_0)+(n*mu)]/[P_k+n], 
     S_m=scatter matrix
     
     prec_m: array-likeprec_m = uf.Wishart_rvs(df=param_dict['up_v_0'], S=param_dict['up_Prec_0'])
     dXd precision matrix. matrix must be symmetric positive definite and is the inverse of
    the covariance matrix. prec_m = inv(cov_matrix). If the cholesky parametrization is
    selected, this parameter must be the lower triangular decomposition of the precision
    matrix, L. prec_matrix = LL
    
    output
    --------
    scalar. total likelihood of your data'
     '''
     
    
    if chol:
        det =   chol_log_determinant(prec_m) 
        trace = np.einsum('ij,ji', prec_m.dot(prec_m.T), SS_dict['S_m'])
    else:
        det  = np.linalg.slogdet(prec_m)[1]
        trace = np.einsum('ij,ji', prec_m, SS_dict['S_m'])
    
    d = len(SS_dict['S_m'])
    n = SS_dict['n']
    p1 = -0.5*n*d*math.log(2*math.pi)
    p2 = 0.5*n*det
    p3 = -0.5*trace
    if log_form:
        return p1+p2+p3
    else:
        return math.exp(p1+p2+p3)



def multivariate_Gaussian_likelihood_cov(SS_dict, log_form = 0):
    '''Likelihood function for a Gaussian multivariate probability model computed
    from cached sufficient statistic and the cholesky decomposition
    of the covariance matrix. 
     
     Parameters
     --------
     SS_dict: python-dict
     
     cov_m: array-like
     dXd covariance matrix. matrix must be symmetric positive definite. 
         
    output
    --------
    scalar. total likelihood of your data'
     '''
    prec_m = chol_of_the_inverse(SS_dict['cS_m']) 
    
    det =   chol_log_determinant(prec_m) 
    trace = np.einsum('ij,ji', prec_m.dot(prec_m.T), SS_dict['S_m'])
    
    d = len(SS_dict['S_m'])
    n = SS_dict['n']
    p1 = -0.5*n*d*math.log(2*math.pi)
    p2 = 0.5*n*det
    p3 = -0.5*trace
    if log_form:
        return p1+p2+p3
    else:
        return math.exp(p1+p2+p3)

def multivariate_Gaussian_likelihood_cached_suff_stat(ss_dict):
    intrace=ss_dict['mean_factor']-ss_dict['sum_factor']+ss_dict['C_m']
    trace = np.einsum('ij,ji', ss_dict['prec_m'].ss_dict['prec_m'], intrace)
    p1 = -0.5*ss_dict['n']*ss_dict['d']*math.log(2*pi)
    p2 = 0.5*chol_log_determinant(ss_dict['prec_m'])
    p3 = -0.5*trace
    return p1+p2+p3
    

def multivariate_Gaussian_rvs(mu, prec_m, chol=False):
    '''returns the a random multivariate Gaussian variable 
    from a mean and precision matrix and possibly a Cholesky decomposition 
    of the precision matrix.
    To be used in Normal-Wishart sampler or similar approaches
    where only the precision matrix is available, avoiding matrix inversions.
    It has a slightly different output than numpy or scipy
    multivariate normal rvs, but has similar statistical properties.
    The algorithm is mentioned in the book 'Handbook of Monte Carlo Methods'
    from Kroese et al.(2011) (algorithm 5.2, pag. 155)'''
    
    d = len(prec_m)
    Z = np.random.standard_normal(size=(d,1))
    if chol:
        V = linalg.solve_triangular(prec_m.T, Z, lower=0, overwrite_b=1, check_finite=0)
       
    else:
        cp_m = np.linalg.cholesky(prec_m)
        V = linalg.solve_triangular(cp_m.T, Z, lower=0, overwrite_b=1, check_finite=0)
        
    return np.array(mu) + V.flatten()




def mvg(mu, prec_m,a):
    d=len(mu)
    diag = np.diag(prec_m)
    D_1 = np.diag(-1./diag)
    D_1_2 = np.diag(sqrt(1./diag))
    L = prec_m.copy()
    L[np.tril_indices(d)]=0
    s0 = np.random.standard_normal(size=(d,1))
    s1 = np.random.standard_normal(size=(d,1))
    for i in xrange(a):
        z=np.random.standard_normal(size=(d,1))
        p1 = D_1.dot(L).dot(s1)
        p2 = D_1.dot(L.T).dot(s0)
        p3 = D_1_2.dot(z)
        s0=s1
        s1 = p1+p2+p3
    return s1.flatten() + mu

