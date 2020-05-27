# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:07:58 2020

@author: rfuchs
"""

from copy import deepcopy
from utilities import ensure_psd
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
from autograd.numpy.linalg import cholesky, pinv

def compute_z_moments(w_s, mu_s, sigma_s):
    ''' Compute the first moment variance of the latent variable '''
    full_paths_proba = w_s[..., n_axis, n_axis]
    
    muTmu = mu_s[0] @ t(mu_s[0], (0, 2, 1)) 
    E_z1z1T = (full_paths_proba * (sigma_s[0] + muTmu)).sum(0, keepdims = True)
    Ez1 = (full_paths_proba * mu_s[0]).sum(0, keepdims = True)
    
    var_z1 = E_z1z1T - Ez1 @ t(Ez1, (0,2,1)) 
    var_z1 = ensure_psd([var_z1])[0] # Numeric stability check
    AT = cholesky(var_z1)

    return Ez1, AT


def identifiable_estim_DGMM(eta_old, H_old, psi_old, Ez1, AT):
    ''' Enforce identifiability conditions for DGMM estimators''' 

    eta_new = deepcopy(eta_old)
    H_new = deepcopy(H_old)
    psi_new = deepcopy(psi_old)
    
    inv_AT = pinv(AT) 
    
    # Identifiability 
    psi_new[0] = inv_AT @ psi_old[0] @ t(inv_AT, (0, 2, 1))
    H_new[0] = inv_AT @ H_old[0]
    eta_new[0] = inv_AT @ (eta_old[0] -  Ez1)    
    
    return eta_new, H_new, psi_new

