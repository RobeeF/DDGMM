# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:52:28 2020

@author: RobF
"""

from lik_functions import log_py_zM_bin, log_py_zM_ord, binom_loglik_j, \
                            ord_loglik_j
                            
from lik_gradients import bin_grad_j, ord_grad_j
from SEM_functions import draw_z_s, fz2_z1s, draw_z2_z1s, draw_zl1_ys, fz_ys,\
    E_step_DGMM, M_step_DGMM, identifiable_estim_DGMM

import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
#from autograd.numpy.linalg import cholesky, pinv

from sklearn.preprocessing import OneHotEncoder
from scipy.stats import multivariate_normal as mvnorm
 
from copy import deepcopy
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

from utilities import compute_path_params, compute_chsi, compute_rho

def DDGMM(y, r, k, init, var_distrib, nj, M, it = 50, eps = 1E-05, maxstep = 100, seed = None): 
    ''' Fit a Generalized Linear Mixture of Latent Variables Model (GLMLVM)
    
    y (numobs x p ndarray): The observations containing categorical variables
    r (list): The dimension of latent variables through the first 2 layers
    k (1d array): The number of components of the latent Gaussian mixture layers
    it (int): The maximum number of EM iterations of the algorithm
    eps (float): If the likelihood increase by less than eps then the algorithm stops
    maxstep (int): The maximum number of optimisation step for each variable
    var_distrib (p 1darray): An array containing the types of the variables in y 
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    M (list of ints): The number of MC points to use at each layer 
    seed (int): The random state seed to set (Only for numpy generated data for the moment)
    ------------------------------------------------------------------------------------------------
    returns (dict): The predicted classes and the likelihood through the EM steps
    '''

    prev_lik = - 100000
    tol = 0.01
    
    # Initialize the parameters
    eta = deepcopy(init['eta'])
    psi = deepcopy(init['psi'])
    lambda_bin = deepcopy(init['lambda_bin'])
    lambda_ord = deepcopy(init['lambda_ord'])
    H = deepcopy(init['H'])
    w_s = deepcopy(init['w_s']) # Probability of path s' through the network for all s' in Omega
   
    numobs = len(y)
    likelihood = []
    hh = 0
    ratio = 1000
    classes = np.zeros((numobs))
    np.random.seed = seed
        
    # Dispatch variables between categories
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nb_bin = len(nj_bin)
        
    y_ord = y[:, var_distrib == 'ordinal']    
    nj_ord = nj[var_distrib == 'ordinal']
    nb_ord = len(nj_ord)
    
    L = len(k)
    k_aug = k + [1]
    S = np.array([np.prod(k_aug[l:]) for l in range(L + 1)])    

    
    assert nb_ord + nb_bin > 0 
                     
    while ((hh < it) & (ratio > eps)):
        print(hh)
        print(w_s)
        hh = hh + 1
        log_py_zl1 = np.zeros((M[0], numobs, S[0])) # l1 standing for the first layer

        #####################################################################################
        ################################# S step ############################################
        #####################################################################################

        #=====================================================================
        # Draw from f(z^{l} | s, Theta) for all s in Omega
        #=====================================================================  
        
        mu_s, sigma_s = compute_path_params(eta, H, psi)
        z_s, zc_s = draw_z_s(mu_s, sigma_s, eta, M)
  
        if np.array([np.isnan(mu_s[l]).any() for l in range(L + 1)]).any():
            raise RuntimeError('Nan in mu_s')
        if np.array([np.isnan(sigma_s[l]).any() for l in range(L + 1)]).any():
            raise RuntimeError('Nan in sigma_s')
         
        #========================================================================
        # Draw from f(z^{l+1} | z^{l}, s, Theta) for l >= 1
        #========================================================================
        
        chsi = compute_chsi(H, psi, mu_s, sigma_s)
        rho = compute_rho(eta, H, psi, mu_s, sigma_s, zc_s, chsi)

        if np.array([np.isnan(chsi[l]).any() for l in range(L)]).any():
            raise RuntimeError('Nan in chsi')
        if np.array([np.isnan(rho[l]).any() for l in range(L)]).any():
            raise RuntimeError('Nan in rho')
        
        # In the following z2 and z1 will denote z^{l+1} and z^{l} respectively
        z2_z1s = draw_z2_z1s(chsi, rho, M, r)
       
        if np.array([np.isnan(z2_z1s[l]).any() for l in range(L)]).any():
            raise RuntimeError('Nan in z2_z1s')
            
        #=======================================================================
        # Compute the p(y| z1) for all variable categories
        #=======================================================================
        
        if nb_bin: # First the Count/Binomial variables
            log_py_zl1 = log_py_zl1 + log_py_zM_bin(lambda_bin, y_bin, z_s[0], S[0], nj_bin) 
                
        if nb_ord: # Then the ordinal variables 
            log_py_zl1 = log_py_zl1 + log_py_zM_ord(lambda_ord, y_ord, z_s[0], S[0], nj_ord)[:,:,:,0] 
        
        py_zl1 = np.exp(log_py_zl1)
        py_zl1 = np.where(py_zl1 == 0, 1E-50, py_zl1)

        #========================================================================
        # Draw from p(z1 | y, s) proportional to p(y | z1) * p(z1 | s) for all s
        #========================================================================
                
        zl1_ys = draw_zl1_ys(z_s, py_zl1, M)
        
        
        #####################################################################################
        ################################# E step ############################################
        #####################################################################################
        
        #=====================================================================
        # Compute conditional probabilities used in the appendix of asta paper
        #=====================================================================
        
        pzl1_s = np.zeros((M[0], 1, S[0]))
        
        for s in range(S[0]): # Have to retake the function for DGMM to parallelize or use apply along axis
            pzl1_s[:,:, s] = mvnorm.pdf(z_s[0][:,:,s], mean = mu_s[0][s].flatten(order = 'C'), \
                                       cov = sigma_s[0][s])[..., n_axis]
            
        # Compute p(y | s_i = 1)
        pzl1_s_norm = pzl1_s / np.sum(pzl1_s, axis = 0, keepdims = True) 
        py_s = (pzl1_s_norm * py_zl1).sum(axis = 0)
        
        # Compute p(z |y, s) and normalize it
        pzl1_ys = pzl1_s * py_zl1 / py_s[n_axis]
        pzl1_ys = pzl1_ys / np.sum(pzl1_ys, axis = 0, keepdims = True) 

        # Compute unormalized (18)
        ps_y = w_s[n_axis] * py_s
        ps_y = ps_y / np.sum(ps_y, axis = 1, keepdims = True)        
        p_y = py_s @ w_s[..., n_axis]
        
        # Compute E_{y,s}(z) and E_{y,s}((z1 - eta)^T (z1 - eta))
        #Ez_ys.append(t(np.mean(zl1_ys, axis = 0), (0, 2, 1))) 
                
        # Free some memory
        del(pzl1_s_norm)
        del(pzl1_s)
 
        if np.isnan(pzl1_ys).any():
            raise RuntimeError('Nan in pzl1_ys')
            
        #=====================================================================
        # Compute p(z^{(l)}| s, y). Equation (5) of the paper
        #=====================================================================
        
        pz2_z1s = fz2_z1s(t(pzl1_ys, (1, 0, 2)), z2_z1s, chsi, rho, S)
        pz_ys = fz_ys(t(pzl1_ys, (1, 0, 2)), pz2_z1s)
                
        del(py_zl1)
        
        #=====================================================================
        # Compute MFA expectations
        #=====================================================================
        
        Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys = \
            E_step_DGMM(zl1_ys, H, z_s, zc_s, z2_z1s, pz_ys, pz2_z1s, S)
               
        ###########################################################################
        ############################ M step #######################################
        ###########################################################################
             
        #=======================================================
        # Compute MFA Parameters 
        #=======================================================

        w_s = np.mean(ps_y, axis = 0)      
        eta, H, psi = M_step_DGMM(Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys, ps_y, H, k)

        #=======================================================
        # Identifiability conditions
        #======================================================= 
        
        AT, eta, H, psi = identifiable_estim_DGMM(eta, H, psi, w_s, mu_s, sigma_s)
        
        #=======================================================
        # Compute GLLVM Parameters
        #=======================================================
        
        # We optimize each column separately as it is faster than all column jointly 
        # (and more relevant with the independence hypothesis)
        
        #****************************
        # Binomial link parameters
        #****************************       
        
        for j in range(nb_bin):
            if j < r[0] - 1: # Constrained columns
                nb_constraints = r[0] - j - 1
                lcs = np.hstack([np.zeros((nb_constraints, j + 2)), np.eye(nb_constraints)])
                linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, 0), \
                                                 np.full(nb_constraints, 0), keep_feasible = True)
            
                opt = minimize(binom_loglik_j, lambda_bin[j] , \
                        args = (y_bin[:,j], z_s[0], S[0], ps_y, pzl1_ys, nj_bin[j]), 
                               tol = tol, method='trust-constr',  jac = bin_grad_j, \
                               constraints = linear_constraint, hess = '2-point', \
                                   options = {'maxiter': maxstep})
                        
            else: # Unconstrained columns
                opt = minimize(binom_loglik_j, lambda_bin[j], \
                        args = (y_bin[:,j], z_s[0], S[0], ps_y, pzl1_ys, nj_bin[j]), \
                               tol = tol, method='BFGS', jac = bin_grad_j, 
                               options = {'maxiter': maxstep})
                    
            if not(opt.success):
                raise RuntimeError('Binomial optimization failed')
                
            lambda_bin[j, :] = deepcopy(opt.x)  

        # Last identifiability part
        if nb_bin > 0:
            lambda_bin[:,1:] = lambda_bin[:,1:] @ AT[0] 
  
        
        
        #****************************
        # Ordinal link parameters
        #****************************    
        
        for j in range(nb_ord):
            enc = OneHotEncoder(categories='auto')
            y_oh = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()                
            
            # Define the constraints such that the threshold coefficients are ordered
            nb_constraints = nj_ord[j] - 2 
            nb_params = nj_ord[j] + r[0] - 1
            
            lcs = np.full(nb_constraints, -1)
            lcs = np.diag(lcs, 1)
            np.fill_diagonal(lcs, 1)
            
            lcs = np.hstack([lcs[:nb_constraints, :], \
                    np.zeros([nb_constraints, nb_params - (nb_constraints + 1)])])
            
            linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, -np.inf), \
                                np.full(nb_constraints, 0), keep_feasible = True)
                    
            opt = minimize(ord_loglik_j, lambda_ord[j] ,\
                    args = (y_oh, z_s[0], S[0], ps_y, pzl1_ys, nj_ord[j]), 
                    tol = tol, method='trust-constr',  jac = ord_grad_j, \
                    constraints = linear_constraint, hess = '2-point',\
                        options = {'maxiter': maxstep})
            
            if not(opt.success):
                raise RuntimeError('Ordinal optimization failed')
                     
            # Ensure identifiability for Lambda_j
            new_lambda_ord_j = (opt.x[-r[0]: ].reshape(1, r[0]) @ AT[0]).flatten() 
            new_lambda_ord_j = np.hstack([deepcopy(opt.x[: nj_ord[j] - 1]), new_lambda_ord_j]) 
            lambda_ord[j] = new_lambda_ord_j
            
         
        ###########################################################################
        ################## Clustering parameters updating #########################
        ###########################################################################
          
        new_lik = np.sum(np.log(p_y))
        likelihood.append(new_lik)
        ratio = (new_lik - prev_lik)/abs(prev_lik)
        
        if (hh < 3): 
            ratio = 2 * eps
        #print(hh)
        #print(likelihood)
        
        # Refresh the classes only if they provide a better explanation of the data
        if prev_lik > new_lik:
            idx_to_sum = tuple(set(range(1, L + 1)) - set([1]))
            psl1_y = ps_y.reshape(numobs, *k, order = 'C').sum(idx_to_sum) 

            classes = np.argmax(psl1_y, axis = 1) 
            
        prev_lik = new_lik
        
        # According to the SEM by Celeux and Diebolt it is a good practice 
        # to increase the number of MC copies through the iterations
        M += 3

    out = dict(likelihood = likelihood, classes = classes)
    return(out)
