# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:52:28 2020

@author: Utilisateur
"""


from lik_functions import log_py_zM_bin, log_py_zM_ord, binom_loglik_j, ord_loglik_j
from lik_gradients import bin_grad_j, ord_grad_j

import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
from autograd.numpy.random import multivariate_normal
from autograd.numpy.linalg import cholesky, pinv

from scipy.linalg import block_diag
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import multivariate_normal as mvnorm
 
from copy import deepcopy
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

import warnings
warnings.filterwarnings("error")

def glmlvm(y, r, k, init, var_distrib, nj, M, it = 50, eps = 1E-05, maxstep = 100, seed = None): 
    ''' Fit a Generalized Linear Mixture of Latent Variables Model (GLMLVM)
    
    y (numobs x p ndarray): The observations containing categorical variables
    r (list): The dimension of latent variables through the first 2 layers
    k (int): The number of components of the latent Gaussian mixture
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
    w = deepcopy(init['w'])
    H = deepcopy(init['H'])
    
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
    
    L = 1 # One layer DGMM
    
    assert nb_ord + nb_bin > 0 
                     
    while ((hh < it) & (ratio > eps)):
        hh = hh + 1
        log_py_z1 = np.zeros((M[0], numobs, k))


        #####################################################################################
        ################################# S step ############################################
        #####################################################################################

        #=====================================================================
        # Draw from f(z1 | s1, Theta), f(z2 | Theta) and f(z1 | z2, s1, Theta)
        #=====================================================================        
        '''
        # To finish later (define cov_z1_s as sigma_s[0])
        k_aug = [k] + [1]
        mu_s = np.array([np.zeros((k_aug[i], r[i], 1)) for i in range(L + 1)])
        mu_s[-1] = np.zeros((r[-1], 1))
        sigma_s = np.array([np.zeros((k_aug[i], r[i], r[i])) for i in range(L + 1)]) # New axis to remove when real deepgmm
        sigma_s[-1] = np.eye(r[1])[n_axis]

        mu_s[0] = eta
        sigma_s[0] = H @ H.transpose((0, 2, 1)) + psi
        '''

        #chsi = np.array([np.zeros((k, r[1], r[1]))])
        #rho = np.array([np.zeros((M[0], k, r[1], 1))])
        
        # z1 drawn from f(z1 | s1, Theta) of dim (M, k, r1). WILL CHANGE IN THE DGMM
        cov_z1_s = H @ H.transpose((0, 2, 1)) + psi

        z1_s = multivariate_normal(size = (M[0], 1), mean = eta.flatten(order = 'F'), cov = block_diag(*cov_z1_s)) 
        z1_s = t(z1_s.reshape(M[0], k, r[0], order = 'F'), (0, 2, 1))
                
        #========================================================================
        # Mimicking the DGMM :
        #========================================================================
        
        mu_s = np.array([np.zeros((k, r[1], 1))])
        sigma_s = np.array([np.eye(r[1])[n_axis]]) # New axis to remove when real deepgmm
        chsi = np.array([np.zeros((k, r[1], r[1]))])
        rho = np.array([np.zeros((M[0], k, r[1], 1))])
        
        chsi[0] = pinv(sigma_s[0] + t(H, (0, 2, 1)) @ pinv(psi) @ H) # Use of pinv ?
        z1_s_c = t(z1_s - t(eta, (2, 1, 0)), (0, 2, 1))[..., n_axis]
        HxPsi_inv = t(H, (0, 2, 1)) @ pinv(psi)
        rho[0] = chsi[0][n_axis] @ (HxPsi_inv[n_axis] @ z1_s_c + sigma_s[0] @ mu_s[0][n_axis])
        
        #del(z1_s_c)
        del(HxPsi_inv)

        z2_z1s = np.zeros((M[1], M[0], r[1], k))    
        for j in range(k):
            z2_z1s_j = multivariate_normal(size = M[1], mean = rho[0][:,j].flatten(order = 'F'), \
                                cov = block_diag(*np.repeat(chsi[0][j][n_axis], M[0], axis = 0))) 
            z2_z1s[:, :, :, j] = z2_z1s_j.reshape(M[1], M[0], r[1], order = 'F') # TO CHECK !!
        
        z2_z1s = t(z2_z1s, (1, 0 , 2, 3))
        
        del(z2_z1s_j)
        #del(chsi)
        #del(rho)
                          
        #=====================================================================
        # Compute the p(y| z1) for all variable categories
        #=====================================================================
        
        if nb_bin: # First the Count/Binomial variables
            log_py_z1 = log_py_z1 + log_py_zM_bin(lambda_bin, y_bin, z1_s, k, nj_bin) 
                
        if nb_ord: # Then the ordinal variables 
            log_py_z1 = log_py_z1 + log_py_zM_ord(lambda_ord, y_ord, z1_s, k, nj_ord)[:,:,:,0] 
        
        py_z1 = np.exp(log_py_z1)
        py_z1 = np.where(py_z1 == 0, 1E-50, py_z1)

        
        #=====================================================================
        # Draw from p(z1 | y, s) proportional to p(y | z1) * p(z1 | s)
        #=====================================================================
        
        py_z1_norm = py_z1 / np.sum(py_z1, axis = 0, keepdims = True) 
        
        z1_ys = np.zeros((M[0], numobs, r[0], k))
        for i in range(k):
            qM_cum = py_z1_norm[:,:, i].T.cumsum(axis=1)
            u = np.random.rand(numobs, 1, M[0])
            
            choices = u < qM_cum[..., np.newaxis]
            idx = choices.argmax(1)
            
            z1_ys[:,:,:,i] = np.take(z1_s[:,:, i], idx.T, axis=0)
        
        del(u)
        
        #####################################################################################
        ################################# E step ############################################
        #####################################################################################
    
        #=====================================================================
        # Compute conditional probabilities used in the appendix of asta paper
        #=====================================================================
        
        pz1_s = np.zeros((M[0], 1, k))
                
        for i in range(k): # Have to retake the function for DGMM to parallelize or use apply along axis
            pz1_s[:,:, i] = mvnorm.pdf(z1_s[:,:,i], mean = eta[i].flatten(), \
                                       cov = cov_z1_s[i])[..., n_axis]
                
        # Compute (17) p(y | s_i = 1)
        pz1_s_norm = pz1_s / np.sum(pz1_s, axis = 0, keepdims = True) 
        py_s = (pz1_s_norm * py_z1).sum(axis = 0)
        
        # Compute (16) p(z |y, s) and normalize it
        pz1_ys = pz1_s * py_z1 / py_s[n_axis]
        pz1_ys = pz1_ys / np.sum(pz1_ys, axis = 0, keepdims = True) 
               
        # Compute unormalized (18)
        ps_y = w[n_axis] * py_s
        ps_y = ps_y / np.sum(ps_y, axis = 1, keepdims = True)        
        p_y = py_s @ w
        
        # Compute E_{y,s}(z) and E_{y,s}((z1 - eta)^T (z1 - eta))
        Ez1_ys = t(np.mean(z1_ys, axis = 0), (0, 2, 1)) 
        
        # z1_ys centered = z1_ys_c = z1_ys - eta
        z1_ys_c = t(z1_ys - t(eta, (2, 1, 0))[n_axis], (1, 0, 3, 2))[..., n_axis]
        z1cz1cT_ys = z1_ys_c @ t(z1_ys_c, (0, 1, 2, 4, 3))
        Ez1cz1cT_ys = z1cz1cT_ys.mean(1)
        
        # Free some memory
        del(py_z1)
        del(pz1_s_norm)
        del(pz1_s)
        del(py_z1_norm)
     
        #=====================================================================
        # Compute MFA expectations
        #=====================================================================
        
        # Mimicking DGMM
        z1_s_formated = z1_s.transpose((0, 2, 1))[..., n_axis]
        z2_z1s_formated = t(z2_z1s, (0, 1, 3, 2))

        pz1_ys_formated = pz1_ys.transpose((1, 0, 2))[..., n_axis]        
        
        ### E(z2 | z1, s) = integral_z2 [ p(z2 | z1, s) * z2 ]
        Ez2_z1s = z2_z1s_formated.mean(1)
        
        ### E(z2z2T | z1, s) = integral_z2 [ p(z2 | z1, s) * z2 @ z2T ] = sum_{m2=1}^M2 z2_m2 @ z2_m2T     
        Ez2z2T_z1s = (z2_z1s_formated[..., n_axis] @ np.expand_dims(z2_z1s_formated, 3)).mean(1) 

        #### E(z2 | y, s) = integral_z1 [ p(z1 | y, s) * E(z2 | z1, s) ] 
        Ez2_ys = (pz1_ys_formated * Ez2_z1s[n_axis]).sum(1)
        
        ### E(z1z2T | y, s) = integral_z1 [ p(z1 | y, s) * z1 @ E(z2T | z1, s) ] 
        Ez1z2T_ys = (pz1_ys_formated[..., n_axis] * \
                   (z1_s_formated @ np.expand_dims(Ez2_z1s, 2))[n_axis]).sum(1)
        
        ### E(z2z2T | y, s) = integral_z1 [ p(z1 | y, s) @ E(z2 z2T | z1, s) ] 
        Ez2z2T_ys = (pz1_ys_formated[..., n_axis] * Ez2z2T_z1s[n_z2axis]).sum(1)
        
        
        ### E[((z1 - eta) - Lambda z2)((z1 - eta) - Lambda z2)^T | y, s]       
        Ez1cz2T_ys = (pz1_ys_formated[..., n_axis] * \
                   (z1_s_c @ np.expand_dims(Ez2_z1s, 2))[n_axis]).sum(1)
        
        EeeT_ys = Ez1cz1cT_ys - H[n_axis] @ t(Ez1cz2T_ys, (0, 1, 3, 2)) - Ez1cz2T_ys @ t(H, (0, 2, 1))[n_axis] \
            + H[n_axis] @ Ez2z2T_ys @ t(H, (0, 2, 1))[n_axis]
        
        # Free some memory
        del(z1_s_formated)
        del(z2_z1s_formated)
        del(pz1_ys_formated)
        del(Ez2z2T_z1s)
        del(Ez1cz2T_ys)
        del(Ez2_z1s)
        del(Ez1cz1cT_ys)
        
        
        ###########################################################################
        ############################ M step #######################################
        ###########################################################################
             
        #=======================================================
        # Compute MFA Parameters 
        #=======================================================
        
        ps_y_formated = ps_y[..., n_axis, n_axis]
        
        w = np.mean(ps_y, axis = 0)
        
        # Compute common denominator    
        den = ps_y.sum(0, keepdims = True).T[..., n_axis]
        den = np.where(den < 1E-14, 1E-14, den)
               
        # eta estimator
        eta_num = Ez1_ys[..., n_axis] - (H[n_axis] @ Ez2_ys[..., n_axis])
        eta = (ps_y_formated * eta_num).sum(0) / den 
        
        # Lambda computation: Not triangular superior...
        H_num = Ez1z2T_ys - eta[n_axis] @ np.expand_dims(Ez2_ys,2)
        H = (ps_y_formated * H_num  @ pinv(Ez2z2T_ys)).sum(0) / den 
        
        # Psi estimator
        psi = (ps_y_formated * EeeT_ys).sum(0) / den
        
        # Free some memory
        del(eta_num)
        del(H_num)
        del(Ez2z2T_ys)
        del(ps_y_formated)
        
        
        #=======================================================
        # Identifiability conditions
        #=======================================================        
        w_formated = w[..., n_axis, n_axis]
        
        etaTeta = eta @ t(eta, (0,2,1))  
        
        E_z1z1T = (w_formated * (psi + etaTeta)).sum(0, keepdims = True)
        Ez1z1_T = (w_formated * eta).sum(0, keepdims = True)

        var_z1 = E_z1z1T - Ez1z1_T @ t(Ez1z1_T, (0,2,1)) 
        AT = cholesky(var_z1)
        inv_AT = pinv(AT) 
        
        # Identifiability 
        psi = inv_AT @ psi @ t(inv_AT, (0, 2, 1))
        H = inv_AT @ H
        eta = inv_AT @ (eta -  Ez1z1_T)
    
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
            
                opt = minimize(binom_loglik_j, lambda_bin[j] , args = (y_bin[:,j], z1_s, k, ps_y, pz1_ys, nj_bin[j]), 
                               tol = tol, method='trust-constr',  jac = bin_grad_j, \
                               constraints = linear_constraint, hess = '2-point', options = {'maxiter': maxstep})
                        
            else: # Unconstrained columns
                opt = minimize(binom_loglik_j, lambda_bin[j], args = (y_bin[:,j], z1_s, k, ps_y, pz1_ys, nj_bin[j]), 
                               tol = tol, method='BFGS', jac = bin_grad_j, options = {'maxiter': maxstep})
                    
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
            
            lcs = np.hstack([lcs[:nb_constraints, :], np.zeros([nb_constraints, nb_params - (nb_constraints + 1)])])
            
            linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, -np.inf), \
                                                 np.full(nb_constraints, 0), keep_feasible = True)
            
            warnings.filterwarnings("default")
        
            opt = minimize(ord_loglik_j, lambda_ord[j] , args = (y_oh, z1_s, k, ps_y, pz1_ys, nj_ord[j]), 
                               tol = tol, method='trust-constr',  jac = ord_grad_j, \
                               constraints = linear_constraint, hess = '2-point', options = {'maxiter': maxstep})
            
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
            classes = np.argmax(ps_y, axis = 1) 
            
        prev_lik = new_lik


    out = dict(likelihood = likelihood, classes = classes)
    return(out)
