# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:52:28 2020

@author: RobF
"""

from copy import deepcopy

from numeric_stability import ensure_psd
from parameter_selection import r_select, k_select 

from identifiability_DGMM import identifiable_estim_DGMM, compute_z_moments, \
    diagonal_cond
                         
from MCEM_DGMM import draw_z_s, fz2_z1s, draw_z2_z1s, fz_ys,\
    E_step_DGMM, M_step_DGMM

from MCEM_GLLVM import draw_zl1_ys, fy_zl1, E_step_GLLVM, \
        bin_params_GLLVM, ord_params_GLLVM
  
from hyperparameters_selection import M_growth, look_for_simpler_network
from utilities import compute_path_params, compute_chsi, compute_rho

import autograd.numpy as np
from autograd.numpy import transpose as t
from autograd.numpy import newaxis as n_axis


def DDGMM(y, n_clusters, r, k, init, var_distrib, nj, it = 50, \
          eps = 1E-05, maxstep = 100, seed = None, perform_selec = True): 
    
    ''' Fit a Generalized Linear Mixture of Latent Variables Model (GLMLVM)
    
    y (numobs x p ndarray): The observations containing categorical variables
    n_clusters (int): The number of clusters to look for in the data
    r (list): The dimension of latent variables through the first 2 layers
    k (list): The number of components of the latent Gaussian mixture layers
    init (dict): The initialisation parameters for the algorithm
    var_distrib (p 1darray): An array containing the types of the variables in y 
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    it (int): The maximum number of MCEM iterations of the algorithm
    eps (float): If the likelihood increase by less than eps then the algorithm stops
    maxstep (int): The maximum number of optimisation step for each variable
    seed (int): The random state seed to set (Only for numpy generated data for the moment)
    perform_selec (Bool): Whether to perform architecture selection or not
    ------------------------------------------------------------------------------------------------
    returns (dict): The predicted classes, the likelihood through the EM steps
                    and a continuous representation of the data
    '''

    prev_lik = - 1E16
    best_lik = -1E16
    tol = 0.01
    max_patience = 1
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
    it_num = 0
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
    M = M_growth(1, r, numobs)
   
    assert nb_ord + nb_bin > 0 
                     
    while (it_num < it) & ((ratio > eps) | (patience <= max_patience)):
        print(it_num)

        # The clustering layer is the one used to perform the clustering 
        # i.e. the layer l such that k[l] == n_clusters
        clustering_layer = np.argmax(np.array(k) == n_clusters)

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
        H = diagonal_cond(H, psi)

        #=======================================================
        # Identifiability conditions
        #======================================================= 
        
        # Update mu and sigma with new eta, H and Psi values
        mu_s, sigma_s = compute_path_params(eta, H, psi)        
        Ez1, AT = compute_z_moments(w_s, mu_s, sigma_s)
        eta, H, psi = identifiable_estim_DGMM(eta, H, psi, Ez1, AT)
        
        del(Ez1)
        '''
        print(eta[-1].mean())
        print(H[-1].mean())
        print(psi[-1].mean())
        '''

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
        print(likelihood)

        # Refresh the classes only if they provide a better explanation of the data
        if best_lik < new_lik:
            best_lik = deepcopy(prev_lik)
            
            idx_to_sum = tuple(set(range(1, L + 1)) - set([clustering_layer + 1]))
            psl_y = ps_y.reshape(numobs, *k, order = 'C').sum(idx_to_sum) 

            classes = np.argmax(psl_y, axis = 1) 
            
            z = (ps_y[..., n_axis] * Ez_ys[clustering_layer]).sum(1)
            
            '''
            fig = plt.figure(figsize=(8,8))
            plt.scatter(z[:, 0], z[:, 1])
            plt.show()
            '''
            
            best_r = deepcopy(r)
            best_k = deepcopy(k)

        
        if prev_lik < new_lik:
            patience = 0
            M = M_growth(it_num + 2, r, numobs)
        else:
            patience += 1
                          
        ###########################################################################
        ######################## Parameter selection  #############################
        ###########################################################################

        is_not_min_specif = not(np.all(np.array(k) == n_clusters) & np.array_equal(r, [2,1]))

        
        if look_for_simpler_network(it_num) & perform_selec & is_not_min_specif:
            r_to_keep = r_select(y_bin, y_ord, zl1_ys, z2_z1s, w_s)
            
            # If r_l == 0, delete the last l + 1: layers
            new_L = np.sum([len(rl) != 0 for rl in r_to_keep]) - 1 
            
            k_to_keep = k_select(w_s, k, new_L, clustering_layer)
    
            is_L_unchanged = L == new_L
            is_r_unchanged = np.all([len(r_to_keep[l]) == r[l] for l in range(new_L + 1)])
            is_k_unchanged = np.all([len(k_to_keep[l]) == k[l] for l in range(new_L)])
              
            is_selection = not(is_r_unchanged & is_k_unchanged & is_L_unchanged)
            
            assert new_L > 0
            
            if is_selection:           
                
                eta = [eta[l][k_to_keep[l]] for l in range(new_L)]
                eta = [eta[l][:, r_to_keep[l]] for l in range(new_L)]
                
                H = [H[l][k_to_keep[l]] for l in range(new_L)]
                H = [H[l][:, r_to_keep[l]] for l in range(new_L)]
                H = [H[l][:, :, r_to_keep[l + 1]] for l in range(new_L)]
                
                psi = [psi[l][k_to_keep[l]] for l in range(new_L)]
                psi = [psi[l][:, r_to_keep[l]] for l in range(new_L)]
                psi = [psi[l][:, :, r_to_keep[l]] for l in range(new_L)]
                
                if nb_bin > 0:
                    # Add the intercept:
                    bin_r_to_keep = np.concatenate([[0], np.array(r_to_keep[0]) + 1]) 
                    lambda_bin = lambda_bin[:, bin_r_to_keep]
                 
                if nb_ord > 0:
                    # Intercept coefficients handling is a little more complicated here
                    lambda_ord_intercept = [lambda_ord_j[:-r[0]] for lambda_ord_j in lambda_ord]
                    Lambda_ord_var = np.stack([lambda_ord_j[-r[0]:] for lambda_ord_j in lambda_ord])
                    Lambda_ord_var = Lambda_ord_var[:, r_to_keep[0]]
                    lambda_ord = [np.concatenate([lambda_ord_intercept[j], Lambda_ord_var[j]])\
                                  for j in range(nb_ord)]
    
                w = w_s.reshape(*k, order = 'C')
                new_k_idx_grid = np.ix_(*k_to_keep[:new_L])
                
                # If layer deletion, sum the last components of the paths
                if L > new_L: 
                    deleted_dims = tuple(range(L)[new_L:])
                    w_s = w[new_k_idx_grid].sum(deleted_dims).flatten(order = 'C')
                else:
                    w_s = w[new_k_idx_grid].flatten(order = 'C')
    
                w_s /= w_s.sum()
    
                k = [len(k_to_keep[l]) for l in range(new_L)]
                r = [len(r_to_keep[l]) for l in range(new_L + 1)]
                
                k_aug = k + [1]
                S = np.array([np.prod(k_aug[l:]) for l in range(new_L + 1)])    
                L = new_L
                
                patience = 0
            
            
            print('New architecture:')
            print('k', k)
            print('r', r)
            print('L', L)
            print('S',S)
            print("w_s", len(w_s))
            
            
        prev_lik = deepcopy(new_lik)
        it_num = it_num + 1

    out = dict(likelihood = likelihood, classes = classes, z = z, \
               best_r = best_r, best_k = best_k)
    return(out)

