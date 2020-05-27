# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:52:28 2020

@author: RobF
"""

from identifiability_DGMM import identifiable_estim_DGMM, compute_z_moments
                         
from SEM_DGMM import draw_z_s, fz2_z1s, draw_z2_z1s, fz_ys,\
    E_step_DGMM, M_step_DGMM

from SEM_GLLVM import draw_zl1_ys, fy_zl1, E_step_GLLVM, \
        bin_params_GLLVM, ord_params_GLLVM
  
from utilities import compute_path_params, compute_chsi, compute_rho, \
    ensure_psd, M_growth

from copy import deepcopy
import autograd.numpy as np
from autograd.numpy import transpose as t
 
import warnings 
warnings.simplefilter("error")

def DDGMM(y, n_clusters, r, k, init, var_distrib, nj, M, it = 50, eps = 1E-05, maxstep = 100, seed = None): 
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

    prev_lik = - 1E12
    best_lik = -1E12
    tol = 0.01
    max_patience = 2
    patience = 0
    
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
    
    # The clustering layer is the one used to perform the clustering 
    # i.e. the layer l such that k[l] == n_clusters
    clustering_layer = np.argmax(np.array(k) == n_clusters)

    #M = np.ones_like(r)

    
    assert nb_ord + nb_bin > 0 
                     
    while ((hh < it) & (ratio > eps)):
        print(hh)
        hh = hh + 1

        #####################################################################################
        ################################# S step ############################################
        #####################################################################################

        #=====================================================================
        # Draw from f(z^{l} | s, Theta) for all s in Omega
        #=====================================================================  
        
        mu_s, sigma_s = compute_path_params(eta, H, psi)
        sigma_s = ensure_psd(sigma_s)
        z_s, zc_s = draw_z_s(mu_s, sigma_s, eta, M)
         
        #========================================================================
        # Draw from f(z^{l+1} | z^{l}, s, Theta) for l >= 1
        #========================================================================
        
        chsi = compute_chsi(H, psi, mu_s, sigma_s)
        chsi = ensure_psd(chsi)
        rho = compute_rho(eta, H, psi, mu_s, sigma_s, zc_s, chsi)

        # In the following z2 and z1 will denote z^{l+1} and z^{l} respectively
        z2_z1s = draw_z2_z1s(chsi, rho, M, r)
                   
        #=======================================================================
        # Compute the p(y| z1) for all variable categories
        #=======================================================================
        
        py_zl1 = fy_zl1(lambda_bin, y_bin, nj_bin, lambda_ord, y_ord, nj_ord, z_s[0])
        
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
        
        pzl1_ys, ps_y, p_y = E_step_GLLVM(z_s[0], mu_s[0], sigma_s[0], w_s, py_zl1)
        #del(py_zl1)

        #=====================================================================
        # Compute p(z^{(l)}| s, y). Equation (5) of the paper
        #=====================================================================
        
        pz2_z1s = fz2_z1s(t(pzl1_ys, (1, 0, 2)), z2_z1s, chsi, rho, S)
        pz_ys = fz_ys(t(pzl1_ys, (1, 0, 2)), pz2_z1s)
                
        
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
        
        # Update mu and sigma with new eta, H and Psi values
        mu_s, sigma_s = compute_path_params(eta, H, psi)        
        Ez1, AT = compute_z_moments(w_s, mu_s, sigma_s)
        eta, H, psi = identifiable_estim_DGMM(eta, H, psi, Ez1, AT)
        
        del(Ez1)
        #=======================================================
        # Compute GLLVM Parameters
        #=======================================================
        
        # We optimize each column separately as it is faster than all column jointly 
        # (and more relevant with the independence hypothesis)
                
        lambda_bin = bin_params_GLLVM(y_bin, nj_bin, lambda_bin, ps_y, pzl1_ys, z_s[0], AT,\
                     tol = tol, maxstep = maxstep)
                 
        lambda_ord = ord_params_GLLVM(y_ord, nj_ord, lambda_ord, ps_y, pzl1_ys, z_s[0], AT,\
                     tol = tol, maxstep = maxstep)

        ###########################################################################
        ################## Clustering parameters updating #########################
        ###########################################################################
          
        new_lik = np.sum(np.log(p_y))
        likelihood.append(new_lik)
        ratio = (new_lik - prev_lik)/abs(prev_lik)
        
        # Wait for max_patience without likelihood augmentation before stopping the algo
        if patience < max_patience:
            ratio = 2 * eps
            patience += 1
            
        #if (hh < 2): 
            #ratio = 2 * eps
        
        # Refresh the classes only if they provide a better explanation of the data
        if best_lik < new_lik:
            best_lik = deepcopy(prev_lik)
            
            idx_to_sum = tuple(set(range(1, L + 1)) - set([clustering_layer + 1]))
            psl1_y = ps_y.reshape(numobs, *k, order = 'C').sum(idx_to_sum) 

            classes = np.argmax(psl1_y, axis = 1) 
        
        if prev_lik < new_lik:
            patience = 0
            # According to the SEM by Celeux and Diebolt it is a good practice 
            # to increase the number of MC copies through the iterations
            M = M_growth(hh, r)
        
        prev_lik = deepcopy(new_lik)
        print(likelihood)


    out = dict(likelihood = likelihood, classes = classes)
    return(out)

