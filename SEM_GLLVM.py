# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:51:26 2020

@author: rfuchs
"""

from lik_functions import ord_loglik_j, log_py_zM_ord, \
            log_py_zM_bin, binom_loglik_j
from lik_gradients import ord_grad_j, bin_grad_j

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import multivariate_normal as mvnorm

from copy import deepcopy
import autograd.numpy as np 
from autograd.numpy import newaxis as n_axis

import warnings
#=============================================================================
# S Step functions
#=============================================================================

def draw_zl1_ys(z_s, py_zl1, M):
    ''' Draw from p(z1 | y, s) proportional to p(y | z1) * p(z1 | s) for all s '''
    
    numobs = py_zl1.shape[1]
    L = len(z_s) - 1
    S = [z_s[l].shape[2] for l in range(L)]
    r = [z_s[l].shape[1] for l in range(L + 1)]

    py_zl1_norm = py_zl1 / np.sum(py_zl1, axis = 0, keepdims = True) 
        
    zl1_ys = np.zeros((M[0], numobs, r[0], S[0]))
    for s in range(S[0]):
        qM_cum = py_zl1_norm[:,:, s].T.cumsum(axis=1)
        u = np.random.rand(numobs, 1, M[0])
        
        choices = u < qM_cum[..., np.newaxis]
        idx = choices.argmax(1)
        
        zl1_ys[:,:,:,s] = np.take(z_s[0][:,:, s], idx.T, axis=0)
        
    return zl1_ys

#=============================================================================
# E Step functions
#=============================================================================

def fy_zl1(lambda_bin, y_bin, nj_bin, lambda_ord, y_ord, nj_ord, zl1_s):
    ''' Compute p(y | z^{0})  '''
    M0 = zl1_s.shape[0]
    S0 = zl1_s.shape[2] 
    numobs = len(y_bin)
    
    nb_ord = len(nj_ord)
    nb_bin = len(nj_bin)

     
    log_py_zl1 = np.zeros((M0, numobs, S0)) # l1 standing for the first layer
    
    if nb_bin: # First the Count/Binomial variables
        log_py_zl1 += log_py_zM_bin(lambda_bin, y_bin, zl1_s, S0, nj_bin) 
    
    if nb_ord: # Then the ordinal variables 
        log_py_zl1 += log_py_zM_ord(lambda_ord, y_ord, zl1_s, S0, nj_ord)[:,:,:,0] 
    
    
    py_zl1 = np.exp(log_py_zl1)
    py_zl1 = np.where(py_zl1 == 0, 1E-50, py_zl1)
    
    if np.isnan(py_zl1).any():
        py_zl1 = np.where(np.isnan(py_zl1), 1E-50, py_zl1)
        raise RuntimeError('Nan in py_zl1')      
    return py_zl1


def E_step_GLLVM(zl1_s, mu_l1_s, sigma_l1_s, w_s, py_zl1):
    
    M0 = zl1_s.shape[0]
    S0 = zl1_s.shape[2] 
    pzl1_s = np.zeros((M0, 1, S0))
        
    for s in range(S0): # Have to retake the function for DGMM to parallelize or use apply along axis
        pzl1_s[:,:, s] = mvnorm.pdf(zl1_s[:,:,s], mean = mu_l1_s[s].flatten(order = 'C'), \
                                           cov = sigma_l1_s[s])[..., n_axis]            
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
     
    return pzl1_ys, ps_y, p_y

#=============================================================================
# M Step functions
#=============================================================================

def bin_params_GLLVM(y_bin, nj_bin, lambda_bin_old, ps_y, pzl1_ys, zl1_s, AT,\
                     tol = 1E-5, maxstep = 100):
    
    r0 = zl1_s.shape[1] 
    S0 = zl1_s.shape[2] 
    nb_bin = len(nj_bin)
    
    new_lambda_bin = []    
    
    for j in range(nb_bin):
        if j < r0 - 1: # Constrained columns
            nb_constraints = r0 - j - 1
            lcs = np.hstack([np.zeros((nb_constraints, j + 2)), np.eye(nb_constraints)])
            linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, 0), \
                                             np.full(nb_constraints, 0), keep_feasible = True)
        
            opt = minimize(binom_loglik_j, lambda_bin_old[j] , \
                    args = (y_bin[:,j], zl1_s, S0, ps_y, pzl1_ys, nj_bin[j]), 
                           tol = tol, method='trust-constr',  jac = bin_grad_j, \
                           constraints = linear_constraint, hess = '2-point', \
                               options = {'maxiter': maxstep})
                    
        else: # Unconstrained columns
            opt = minimize(binom_loglik_j, lambda_bin_old[j], \
                    args = (y_bin[:,j], zl1_s, S0, ps_y, pzl1_ys, nj_bin[j]), \
                           tol = tol, method='BFGS', jac = bin_grad_j, 
                           options = {'maxiter': maxstep})

        res = opt.x                
        if not(opt.success):
            res = lambda_bin_old[j]
            warnings.warn('One of the binomial optimisations has failed', RuntimeWarning)
            
        new_lambda_bin.append(deepcopy(res))  

    # Last identifiability part
    if nb_bin > 0:
        new_lambda_bin = np.stack(new_lambda_bin)
        new_lambda_bin[:,1:] = new_lambda_bin[:,1:] @ AT[0] 
        
    return new_lambda_bin


def ord_params_GLLVM(y_ord, nj_ord, lambda_ord_old, ps_y, pzl1_ys, zl1_s, AT,\
                     tol = 1E-5, maxstep = 100):
    
    #****************************
    # Ordinal link parameters
    #****************************  
    
    r0 = zl1_s.shape[1] 
    S0 = zl1_s.shape[2] 
    nb_ord = len(nj_ord)
    
    new_lambda_ord = []
    
    for j in range(nb_ord):
        enc = OneHotEncoder(categories='auto')
        y_oh = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()                
        
        # Define the constraints such that the threshold coefficients are ordered
        nb_constraints = nj_ord[j] - 2 
        nb_params = nj_ord[j] + r0 - 1
        
        lcs = np.full(nb_constraints, -1)
        lcs = np.diag(lcs, 1)
        np.fill_diagonal(lcs, 1)
        
        lcs = np.hstack([lcs[:nb_constraints, :], \
                np.zeros([nb_constraints, nb_params - (nb_constraints + 1)])])
        
        linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, -np.inf), \
                            np.full(nb_constraints, 0), keep_feasible = True)
                
        opt = minimize(ord_loglik_j, lambda_ord_old[j] ,\
                args = (y_oh, zl1_s, S0, ps_y, pzl1_ys, nj_ord[j]), 
                tol = tol, method='trust-constr',  jac = ord_grad_j, \
                constraints = linear_constraint, hess = '2-point',\
                    options = {'maxiter': maxstep})
        
        res = opt.x
        if not(opt.success): # If the program fail, keep the old estimate as value
            res = lambda_ord_old[j]
            warnings.warn('One of the ordinal optimisations has failed', RuntimeWarning)
                 
        # Ensure identifiability for Lambda_j
        new_lambda_ord_j = (res[-r0: ].reshape(1, r0) @ AT[0]).flatten() 
        new_lambda_ord_j = np.hstack([deepcopy(res[: nj_ord[j] - 1]), new_lambda_ord_j]) 
        new_lambda_ord.append(new_lambda_ord_j)
    
    return new_lambda_ord
        