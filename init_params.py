# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: RobF
"""

import os
os.chdir('C:/Users/rfuchs/Documents/GitHub/DDGMM')

from identifiability_DGMM import identifiable_estim_DGMM, compute_z_moments
from sklearn.linear_model import LogisticRegression
from factor_analyzer import FactorAnalyzer
from sklearn.mixture import GaussianMixture
from utilities import compute_path_params, add_missing_paths, \
    gen_categ_as_bin_dataset
    
from sklearn.preprocessing import LabelEncoder 


import prince
import pandas as pd

# Dirty local hard copy of the Github bevel package
from bevel.linear_ordinal_regression import  OrderedLogit 

import autograd.numpy as np
from autograd.numpy.random import uniform
from autograd.numpy import newaxis as n_axis

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



def get_MFA_params(zl, kl, rl_nextl):
    ''' Determine clusters with a GMM and then adjust a Factor Model over each cluster
    zl (ndarray): The lth layer latent variable 
    kl (int): The number of components of the lth layer
    rl_nextl (1darray): The dimension of the lth layer and (l+1)th layer
    -----------------------------------------------------
    returns (dict): Dict with the parameters of the MFA approximated by GMM + FA. 
    '''
    #======================================================
    # Fit a GMM in the continuous space
    #======================================================
    numobs = zl.shape[0]

    gmm = GaussianMixture(n_components = kl)
    s = gmm.fit_predict(zl)
    
    psi = np.full((kl, rl_nextl[0], rl_nextl[0]), 0).astype(float)
    psi_inv = np.full((kl, rl_nextl[0], rl_nextl[0]), 0).astype(float)
    H = np.full((kl, rl_nextl[0], rl_nextl[1]), 0).astype(float)
    eta = np.full((kl, rl_nextl[0]), 0).astype(float)
    z_nextl = np.full((numobs, rl_nextl[1]), np.nan).astype(float)
  
    #========================================================
    # And then a MFA on each of those group
    #========================================================
    
    for j in range(kl):
        indices = (s == j)
        fa = FactorAnalyzer(rotation = None, method = 'ml', n_factors = rl_nextl[1])
        fa.fit(zl[indices])

        psi[j] = np.diag(fa.get_uniquenesses())
        H[j] = fa.loadings_
        psi_inv[j] = np.diag(1/fa.get_uniquenesses())
        z_nextl[indices] = fa.transform(zl[indices])
  
        eta[j] = np.mean(zl[indices], axis = 0)
  
    params = {'H': H, 'psi': psi, 'z_nextl': z_nextl, 'eta': eta, 'classes': s}
    return params


def dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, use_famd = False, seed = None):
    ''' Perform dimension reduction into a continuous r dimensional space and determine 
    the init coefficients in that space
    
    y (numobs x p ndarray): The observations containing categorical variables
    k (1d array): The number of components of the latent Gaussian mixture layers
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
    
    L = len(k)
    numobs = len(y)
    S = np.prod(k)
    
    #==============================================================
    # Dimension reduction performed with MCA
    #==============================================================
             
    if type(y) != pd.core.frame.DataFrame:
        raise TypeError('y should be a dataframe for prince')
    
    if (var_distrib == 'ordinal').all():
        print('PCA init')

        pca = prince.PCA(n_components = r[0], n_iter=3, rescale_with_mean=True,\
            rescale_with_std=True, copy=True, check_input=True, engine='auto',\
                random_state = seed)
        z1 = pca.fit_transform(y).values

    elif use_famd:
        famd = prince.FAMD(n_components = r[0], n_iter=3, copy=True, check_input=True, \
                               engine='auto', random_state = seed)
        z1 = famd.fit_transform(y).values
            
        # Encode categorical datas
        y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)
        
        # Encode binary data
        le = LabelEncoder()
        for col_idx, colname in enumerate(y.columns):
            if var_distrib[col_idx] == 'bernoulli':
                y[colname] = le.fit_transform(y[colname])

    else:
        # Check input = False to remove
        mca = prince.MCA(n_components = r[0], n_iter=3, copy=True,\
                         check_input=False, engine='auto', random_state = seed)
        z1 = mca.fit_transform(y).values
        #z1 = mca.row_coordinates(y).values.astype(float)
        
    z = [z1]
    y = y.values.astype(int)

    #==============================================================
    # Set the shape parameters of each data type
    #==============================================================    
    
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli', var_distrib == 'binomial')]
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nb_bin = len(nj_bin)
    
    y_ord = y[:, var_distrib == 'ordinal']    
    nj_ord = nj[var_distrib == 'ordinal']
    nb_ord = len(nj_ord)
     
    #=======================================================
    # Determining the Gaussian Parameters
    #=======================================================
    init = {}

    eta = []
    H = []
    psi = []
    paths_pred = np.zeros((numobs, L))
    
    for l in range(L):
        params = get_MFA_params(z[l], k[l], r[l:])
        eta.append(params['eta'][..., n_axis])
        H.append(params['H'])
        psi.append(params['psi'])
        z.append(params['z_nextl']) 
        paths_pred[:, l] = params['classes']
    
    paths, nb_paths = np.unique(paths_pred, return_counts = True, axis = 0)
    paths, nb_paths = add_missing_paths(k, paths, nb_paths)
    
    w_s = nb_paths / numobs
    w_s = np.where(w_s == 0, 1E-16, w_s)
    
    # Check all paths have been explored
    if len(paths) != S:
        raise RuntimeError('Real path len is', S, 'while the initial number', \
                           'of path was only',  len(paths))

    w_s = w_s.reshape(*k).flatten('C') 

    #=============================================================
    # Enforcing identifiability constraints over the first layer
    #=============================================================
    
    mu_s, sigma_s = compute_path_params(eta, H, psi)
        
    Ez1, AT = compute_z_moments(w_s, mu_s, sigma_s)
    eta, H, psi = identifiable_estim_DGMM(eta, H, psi, Ez1, AT)
    
    init['eta']  = eta     
    init['H'] = H
    init['psi'] = psi

    init['w_s'] = w_s # Probabilities of each path through the network
    init['z'] = z
    
    # The clustering layer is the one used to perform the clustering 
    # i.e. the layer l such that k[l] == n_clusters
    clustering_layer = np.argmax(np.array(k) == n_clusters)
    
    init['classes'] = paths_pred[:,clustering_layer] # 0 To change with clustering_layer_idx
    
        
    #=======================================================
    # Determining the coefficients of the GLLVM layer
    #=======================================================
    
    # Determining lambda_bin coefficients.
    
    lambda_bin = np.zeros((nb_bin, r[0] + 1))
    
    for j in range(nb_bin): 
        Nj = np.max(y_bin[:,j]) # The support of the jth binomial is [1, Nj]
        
        if Nj ==  1:  # If the variable is Bernoulli not binomial
            yj = y_bin[:,j]
            z_new = z[0]
        else: # If not, need to convert Binomial output to Bernoulli output
            yj, z_new = bin_to_bern(Nj, y_bin[:,j], z[0])
        
        lr = LogisticRegression()
        
        if j < r[0] - 1:
            lr.fit(z_new[:,:j + 1], yj)
            lambda_bin[j, :j + 2] = np.concatenate([lr.intercept_, lr.coef_[0]])
        else:
            lr.fit(z_new, yj)
            lambda_bin[j] = np.concatenate([lr.intercept_, lr.coef_[0]])
    
    ## Identifiability of bin coefficients
    lambda_bin[:,1:] = lambda_bin[:,1:] @ AT[0] 
    
    # Determining lambda_ord coefficients
    lambda_ord = []
    
    for j in range(nb_ord):
        Nj = len(np.unique(y_ord[:,j], axis = 0))  # The support of the jth ordinal is [1, Nj]
        yj = y_ord[:,j]
        
        ol = OrderedLogit()
        ol.fit(z[0], yj)
        
        ## Identifiability of ordinal coefficients
        beta_j = (ol.beta_.reshape(1, r[0]) @ AT[0]).flatten()
        lambda_ord_j = np.concatenate([ol.alpha_, beta_j])
        lambda_ord.append(lambda_ord_j)        
        
    init['lambda_bin'] = lambda_bin
    init['lambda_ord'] = lambda_ord
    
    return init
