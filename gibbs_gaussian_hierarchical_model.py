import numpy as np
import scipy.stats as sts
from pylab import *



def generate_intial_theta_j(data):
    tj = {}
    for i in xrange(len(data)):
        d = np.random.randint(0,len(data[i]), 10)
        tj[i] = data[i][d]
    im = np.array([mean(np.array(tj.values()).T[i]) for i in xrange(10)])
    return tj, im

def compute_mu(J, theta_dict, tau_2):
    mu_hat = mean(theta_dict.values())
    return sts.norm(mu_hat, sqrt(tau_2/J)).rvs()


def compute_tau_2_hat(J,theta_dict):
    b1 = np.array(theta_dict.values())-mean(theta_dict.values())
    return (1./(J-1))*(sum(b1**2))


def compute_sigma_2_hat(n,data, theta_dict):
    theta = np.array([theta_dict[i] for i in xrange(len(theta_dict))])
    p1 = np.array([sum((data[k]-theta[k])**2) for k in xrange(len(data))])
    return (1./n)*sum(p1)


def draw_chi_tau_sigma(tau_2_hat, sigma_2_hat, J, n_vec):
    chi_tau = 1./sts.chi2(J-1, tau_2_hat).rvs()
    chi_sigma = 1./sts.chi2(sum(n_vec), sigma_2_hat).rvs()
    return chi_tau, chi_sigma
        
def compute_single_v_j_hat_theta_j_hat(J, j, n_vec, tau_2, sigma_2, data_mean, mu, tau_2_hat, sigma_2_hat):
    tau_2, sigma_2 = draw_chi_tau_sigma(tau_2_hat, sigma_2_hat, J, n_vec)
    b1 = (float(n_vec[j])/sigma_2)
    theta_j = ((tau_2*mu)+(b1*data_mean[j]))/(tau_2+b1)
    v_j = 1./(tau_2+b1)
    return theta_j, v_j
    
def single_Gibbs_iteration(J, theta_dict, data,data_mean, n_vec):
    for i in xrange(J):
        tau_2_hat = compute_tau_2_hat(J, theta_dict)
        sigma_2_hat = compute_sigma_2_hat(sum(n_vec), data,theta_dict)
        tau_2, sigma_2 = draw_chi_tau_sigma(tau_2_hat, sigma_2_hat, J, n_vec)
        mu = compute_mu(J, theta_dict, tau_2)
        theta_j, vj = compute_single_v_j_hat_theta_j_hat(J, i, n_vec, tau_2, sigma_2, data_mean, mu,tau_2_hat, sigma_2_hat)
        theta_dict[i] = sts.norm(theta_j, sqrt(vj)).rvs()

data = np.array([np.array([62,60,63,59]),np.array([63,67,71,64,65,66]),np.array([68,66,71,67,68,68]),np.array([56,62,60,61,63,64,63,55])])

data_mean = np.array([mean(i) for i in data])

n_vec = np.array([len(i) for i in data])

theta_dict = {i:data[i][0] for i in xrange(len(data))}

J = len(data)

means = []
for i in xrange(5000):
    means.append(np.array([theta_dict[i] for i in xrange(len(theta_dict))]))
    single_Gibbs_iteration(J, theta_dict, data, data_mean, n_vec)

    
