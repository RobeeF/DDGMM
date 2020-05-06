# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: Utilisateur
"""

import os
os.chdir('C:/Users/rfuchs/Documents/GitHub/GLMLVM_MFA')

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from factor_analyzer import FactorAnalyzer

import prince
import pandas as pd

# Dirty local hard copy of the Github bevel package
from bevel.linear_ordinal_regression import  OrderedLogit 

import autograd.numpy as np
from autograd.numpy.random import uniform
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
from autograd.numpy.linalg import cholesky, pinv

from glmlvm import glmlvm


####################################################################################
########################## Random initialisations ##################################
####################################################################################

def random_init(r, nj_bin, nj_ord, k, init_seed):
    ''' Generate random initialisations for the parameters
    
    r (int): The dimension of latent variables
    nj_bin (nb_bin 1darray): For binary/count data: The maximum values that the variable can take. 
    nj_ord (nb_ord 1darray): For ordinal data: the number of different existing categories for each variable
    k (int): The number of components of the latent Gaussian mixture
    init_seed (int): The random state seed to set (Only for numpy generated data for the moment)
    --------------------------------------------------------------------------------------------
    returns (dict): The initialisation parameters   
    '''
    
    # Seed for init
    np.random.seed = init_seed
    init = {}
    
    
    # Gaussian mixture params
    init['w'] = np.full(k, 1/k) 
    
    mu_init = np.repeat(np.linspace(-1.0, 1.0, num = k)[..., n_axis], axis = 1, repeats =r)
    init['mu'] = (uniform(low = -5, high = 5, size = (1,1)) * mu_init)
    init['mu'] = init['mu'][..., np.newaxis]
  
    init['sigma'] = np.zeros(shape = (k, r, r))
    for i in range(k):
        init['sigma'][i] = 0.050 * np.eye(r)
        
    # Enforcing identifiability constraints
    muTmu = init['mu'] @ t(init['mu'], (0,2,1))  
     
    E_zzT = (init['w'][..., n_axis, n_axis] * (init['sigma'] + muTmu)).sum(0, keepdims = True)
    Ezz_T = (init['w'][...,n_axis, n_axis] * init['mu']).sum(0, keepdims = True)
    
    var_z = E_zzT - Ezz_T @ t(Ezz_T, (0,2,1)) # Koenig-Huyghens Formula for Variance Computation
    sigma_z = cholesky(var_z)
     
    init['sigma'] = pinv(sigma_z) @ init['sigma'] @ t(pinv(sigma_z), (0, 2, 1))
    init['mu'] = pinv(sigma_z) @ init['mu']
    init['mu']  = init['mu']  - Ezz_T

    # GLLVM params    
    p1 = len(nj_bin)
    p2 = len(nj_ord)
    
    if p1 > 0:
        init['lambda_bin'] = uniform(low = -3, high = 3, size = (p1, r + 1))
        init['lambda_bin'][:,1:] = init['lambda_bin'][:,1:] @ sigma_z[0] 
        
        if (r > 1): 
            init['lambda_bin'] = np.tril(init['lambda_bin'], k = 1)

    else:
        init['lambda_bin'] = np.array([]) #np.full((p1, r + 1), np.nan)
  
    if p2 > 0:

        lambda_ord = []
        for j in range(p2):
            lambda0_ord = np.sort(uniform(low = -2, high = 2, size = (nj_ord[j] - 1)))
            Lambda_ord = uniform(low = -3, high = 3, size = r)
            lambda_ord.append(np.hstack([lambda0_ord, Lambda_ord]))
              
        init['lambda_ord'] = lambda_ord
        
    else:
        init['lambda_ord'] = np.array([])#np.full((p2, 1), np.nan)
    
    return(init)


def init_cv(y, var_distrib, r, nj_bin, nj_ord, k, seed):
    ''' Test 20 different inits for a few iterations and returns the best one
    
    y (numobs x p ndarray): The observations containing categorical variables
    var_distrib (p 1darray): An array containing the types of the variables in y 
    r (int): The dimension of latent variables
    nj_bin (nb_bin 1darray): For binary/count data: The maximum values that the variable can take. 
    nj_ord (nb_ord 1darray): For ordinal data: the number of different existing categories for each variable
    k (int): The number of components of the latent Gaussian mixture
    seed (int): The random state seed to set (Only for numpy generated data for the moment)
    --------------------------------------------------------------------------------------------
    returns (dict): The initialisation parameters that gave the best likelihood
    '''
      
    nb_init_tested = 10
    M = 4 * r
    best_lik = -1000000
    best_init = {}
    nb_it = 2
    maxstep = 100
    eps = 1E-5
    nj = np.concatenate([nj_bin, nj_ord])

    for i in range(nb_init_tested):
        init = random_init(r, nj_bin, nj_ord, k, None)
        try:
            out = glmlvm(y, r, k, nb_it, init, eps, maxstep, var_distrib, nj, M, seed)
        except:
            continue

        lik = out['likelihood'][-1]
    
        if (best_lik < lik):
            best_lik = lik
            best_init = init
    
    return(best_init)


####################################################################################
################### MCA GMM + Logistic Regressions initialisation ##################
####################################################################################

def bin_to_bern(Nj, yj_binom, zM_binom):
    ''' Split the binomial variable into Bernoulli. Them just recopy the corresponding zM.
    It is necessary to fit binary logistic regression
    Example: yj has support in [0,10]: Then if y_ij = 3 generate a vector with 3 ones and 7 zeros 
    (3 success among 10).
    
    Nj (int): The upper bound of the support of yj_binom
    yj_binom (numobs 1darray): The Binomial variable considered
    zM_binom (numobs x r nd-array): The continuous representation of the data
    -----------------------------------------------------------------------------------
    returns (tuple of 2 (numobs x Nj) arrays): The "Bernoullied" Binomial variable
    '''
    
    n_yk = len(yj_binom) # parameter k of the binomial
    
    # Generate Nj Bernoullis from each binomial and get a (numobsxNj, 1) table
    u = uniform(size =(n_yk,Nj))
    p = (yj_binom/Nj)[..., n_axis]
    yk_bern = (u > p).astype(int).flatten('A')#[..., n_axis] 
        
    return yk_bern, np.repeat(zM_binom, Nj, 0)



def get_MFA_params(z1, k1, r):
    ''' Adjust a MFA over z1'''
    #======================================================
    # Fit a GMM in the continuous space
    #======================================================
    numobs = z1.shape[0]
    km = KMeans(n_clusters = k1)
    km.fit(z1)
    s = km.predict(z1)
    
    psi = np.full((k1, r[0], r[0]), 0).astype(float)
    psi_inv = np.full((k1, r[0], r[0]), 0).astype(float)
    H = np.full((k1, r[0], r[1]), 0).astype(float)
    eta = np.full((k1, r[0]), 0).astype(float)
    z2 = np.full((numobs, r[1]), np.nan).astype(float)
  
    #========================================================
    # And then a MFA on each of those group
    #========================================================
    
    for j in range(k1):
        indices = (s == j)
        fa = FactorAnalyzer(rotation = None, method = 'ml', n_factors = r[1])
        fa.fit(z1[indices])

        psi[j] = np.diag(fa.get_uniquenesses())
        H[j] = fa.loadings_
        psi_inv[j] = np.diag(1/fa.get_uniquenesses())
        z2[indices] = fa.transform(z1[indices])
  
        eta[j] = np.mean(z1[indices], axis = 0)

    w = np.unique(s, return_counts = True)[1] / numobs
  
    params = {'w': w, 'H': H, 'psi': psi, 'z2': z2, 'eta': eta, 'preds': s}
    return params


def dim_reduce_init(y, k, r, nj, var_distrib, seed = None):
    ''' Perform dimension reduction into a continuous r dimensional space and determine 
    the init coefficients in that space
    
    y (numobs x p ndarray): The observations containing categorical variables
    k (int): The number of components of the latent Gaussian mixture
    r (int): The dimension of latent variables
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    var_distrib (p 1darray): An array containing the types of the variables in y 
    dim_red_method (str): Choices are 'prince' for MCA, 'umap' of 'tsne'
    seed (None): The random state seed to use for the dimension reduction
    M (int): The number of MC points to compute     
    ---------------------------------------------------------------------------------------
    returns (dict): All initialisation parameters
    '''
    
    #==============================================================
    # Dimension reduction performed with MCA
    #==============================================================
             
    if type(y) != pd.core.frame.DataFrame:
        raise TypeError('y should be a dataframe for prince')
    
    mca = prince.MCA(n_components = r[0], n_iter=3, copy=True, \
                     check_input=True, engine='auto', random_state=42)
    mca = mca.fit(y)
    z1 = mca.row_coordinates(y).values.astype(float)
    
    y = y.values.astype(int)

    #==============================================================
    # Set the shape parameters of each data type
    #==============================================================    
    
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nb_bin = len(nj_bin)
    
    y_ord = y[:, var_distrib == 'ordinal']    
    nj_ord = nj[var_distrib == 'ordinal']
    nb_ord = len(nj_ord)
     
    #=======================================================
    # Determining the Gaussian Parameters
    #=======================================================
    init = {}

    params = get_MFA_params(z1, k, r)
    eta = params['eta'][..., n_axis]
    H = params['H']
    psi = params['psi']
    w = params['w']
    z2 = params['z2']
        
    #=======================================================
    # Enforcing identifiability constraints
    #=======================================================
    w_formated = w[..., n_axis, n_axis]
        
    etaTeta = eta @ t(eta, (0,2,1))  
    
    E_z1z1T = (w_formated * (psi + etaTeta)).sum(0, keepdims = True)
    Ez1z1_T = (w_formated * eta).sum(0, keepdims = True)

    var_z1 = E_z1z1T - Ez1z1_T @ t(Ez1z1_T, (0,2,1)) 
    AT = cholesky(var_z1)
    inv_AT = pinv(AT) 
    
    psi = inv_AT @ psi @ t(inv_AT, (0, 2, 1))
    H = inv_AT @ H
    eta = inv_AT @ (eta -  Ez1z1_T)
    
    init['eta']  = eta     
    init['H'] = H
    init['psi'] = psi

    init['w'] = w
    init['z2'] = z2
    init['preds'] = params['preds']
             
    #=======================================================
    # Determining the coefficients of the GLLVM layer
    #=======================================================
    
    # Determining lambda_bin coefficients.
    
    lambda_bin = np.zeros((nb_bin, r[0] + 1))
    
    for j in range(nb_bin): 
        Nj = np.max(y_bin[:,j]) # The support of the jth binomial is [1, Nj]
        
        if Nj ==  1:  # If the variable is Bernoulli not binomial
            yj = y_bin[:,j]
            z = z1
        else: # If not, need to convert Binomial output to Bernoulli output
            yj, z = bin_to_bern(Nj, y_bin[:,j], z1)
        
        lr = LogisticRegression()
        
        if j < r[0] - 1:
            lr.fit(z[:,:j + 1], yj)
            lambda_bin[j, :j + 2] = np.concatenate([lr.intercept_, lr.coef_[0]])
        else:
            lr.fit(z, yj)
            lambda_bin[j] = np.concatenate([lr.intercept_, lr.coef_[0]])
    
    ## Identifiability of bin coefficients
    lambda_bin[:,1:] = lambda_bin[:,1:] @ AT[0] 
    
    # Determining lambda_ord coefficients
    lambda_ord = []
    
    for j in range(nb_ord):
        Nj = len(np.unique(y_ord[:,j], axis = 0))  # The support of the jth ordinal is [1, Nj]
        yj = y_ord[:,j]
        
        ol = OrderedLogit()
        ol.fit(z1, yj)
        
        ## Identifiability of ordinal coefficients
        beta_j = (ol.beta_.reshape(1, r[0]) @ AT[0]).flatten()
        lambda_ord_j = np.concatenate([ol.alpha_, beta_j])
        lambda_ord.append(lambda_ord_j)        
        
    init['lambda_bin'] = lambda_bin
    init['lambda_ord'] = lambda_ord
    
    return init

