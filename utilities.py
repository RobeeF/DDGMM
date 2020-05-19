# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:26:07 2020

@author: Utilisateur
"""

from scipy import linalg
from copy import deepcopy
from itertools import permutations
from sklearn.preprocessing import OneHotEncoder

import itertools
import pandas as pd
import matplotlib as mpl
import autograd.numpy as np
import matplotlib.pyplot as plt

from autograd.numpy.linalg import pinv
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t

def sample_MC_points(zM, p_z_ys, nb_points):
    ''' Resample nb_points from zM with the highest p_z_ys probability
    
    zM (M x r x k ndarray) : The M Monte Carlo copies of z for each path k
    p_z_ys (M x k x 1 ndarray): The probability density of each point
    nb_points (int): The number of points to resample from the original M points
    --------------------------------------------------------------------------------
    returns (tuple: (nb_points x k x 1), (nb_points x r x k) ndarrays): The resampled p(z | y, s) and zM
    '''
    M = p_z_ys.shape[0]
    numobs = p_z_ys.shape[1]
    k = p_z_ys.shape[2]
    r = zM.shape[1]

    assert nb_points > 0    
    assert nb_points < M
    
    # Compute the fraction of points to keep
    rs_frac = nb_points / M
    
    # Compute the <nb_points> points that have the highest probability through the observations
    sum_p_z_ys = p_z_ys.sum(axis = 1, keepdims = True)
    
    # Masking the the points with less probabilities over all observations
    imask = sum_p_z_ys <= np.quantile(sum_p_z_ys, [1 - rs_frac], axis = 0)
    
    msp_z_ys = np.ma.masked_where(np.repeat(imask, axis = 1, repeats = numobs),\
                                  p_z_ys, copy=True)
    
    mzM = np.ma.masked_where(np.repeat(imask, axis = 1, repeats = r),\
                             zM, copy=True)

    # Need to transpose then detranspose due to compressed ordering conventions
    msp_z_ys = np.transpose(msp_z_ys, (1, 2, 0)).compressed()
    msp_z_ys = msp_z_ys.reshape(numobs, k, int(M * rs_frac))

    mzM = np.transpose(mzM, (1, 2, 0)).compressed()
    mzM = mzM.reshape(r, k, int(M * rs_frac))
    
    return np.transpose(msp_z_ys, (2, 0, 1)), np.transpose(mzM, (2, 0, 1))


def misc(true, pred, return_relabeled = False):
    ''' Compute a label invariant misclassification error and can return the relabeled predictions
    
    true (numobs 1darray): array with the true labels
    pred (numobs 1darray): array with the predicted labels
    return_relabeled (Bool): Whether or not to return the relabeled predictions
    --------------------------------------------------------
    returns (float): The misclassification error rate  
    '''
    best_misc = 0
    true_classes = np.unique(true).astype(int)
    nb_classes = len(true_classes)
    
    best_labeled_pred = pred

    best_misc = 1
    
    # Compute of the possible labelling
    all_possible_labels = [list(l) for l in list(permutations(true_classes))]
    
    # And compute the misc for each labelling
    for l in all_possible_labels:
        shift = max(true_classes) + 1
        shift_pred = pred + max(true_classes) + 1
        
        for i in range(nb_classes):
            shift_pred = np.where(shift_pred == i + shift, l[i], shift_pred)
        
        current_misc = np.mean(true != shift_pred)
        if current_misc < best_misc:
            best_misc = deepcopy(current_misc)
            best_labeled_pred = deepcopy(shift_pred)
      
    if return_relabeled:
        return best_misc, best_labeled_pred
    else:
        return best_misc
        


def gen_categ_as_bin_dataset(y, var_distrib):
    ''' Convert the categorical variables in the dataset to binary variables
    
    y (numobs x p ndarray): The observations containing categorical variables
    var_distrib (p 1darray): An array containing the types of the variables in y 
    ----------------------------------------------------------------------------
    returns ((numobs, p_new) ndarray): The new dataset where categorical variables 
    have been converted to binary variables
    '''
    new_y = deepcopy(y)
    new_y = new_y.reset_index(drop = True)
    new_var_distrib = deepcopy(var_distrib[var_distrib != 'categorical'])

    categ_idx = np.where(var_distrib == 'categorical')[0]
    oh = OneHotEncoder(drop = 'first')
        
    for idx in categ_idx:
        name = y.iloc[:, idx].name
        categ_var = pd.DataFrame(oh.fit_transform(pd.DataFrame(y.iloc[:, idx])).toarray())
        nj_var = len(categ_var.columns)
        categ_var.columns = [str(name) + '_' + str(categ_var.columns[i]) for i in range(nj_var)]
        
        # Delete old categorical variable & insert new binary variables in the dataframe
        del(new_y[name])
        new_y = new_y.join(categ_var.astype(int))
        new_var_distrib = np.concatenate([new_var_distrib, ['bernoulli'] * nj_var])
        
    return new_y, new_var_distrib

def ordinal_encoding(sequence, ord_labels, codes):
    ''' Perform label encoding, replacing ord_labels with codes
    
    sequence (numobs 1darray): The sequence to encode
    ord_labels (nj_ord_j 1darray): The labels existing in sequences 
    codes (nj_ord_j 1darray): The codes used to replace ord_labels 
    -----------------------------------------------------------------
    returns (numobs 1darray): The encoded sequence
    '''
    new_sequence = deepcopy(sequence.values)
    for i, lab in enumerate(ord_labels):
        new_sequence = np.where(new_sequence == lab, codes[i], new_sequence)

    return new_sequence
    

def plot_gmm_init(X, Y_, means, covariances, index, title):
    ''' Plot the GMM fitted in the continuous representation of the original data X.
    Code from sklearn website.
    
    X (numobs x 2 nd-array): The 2D continuous representation of the original data
    Y_ (numobs 1darray): The GMM predicted labels 
    means (k x r ndarray): The means of the Gaussian components identified
    covariances (k x r x r): The covariances of the Gaussian components identified
    index (int): Set to zero is ok
    title (str): The title displayed over the graph
    ------------------------------------------------------------------------------
    returns (void): pyplot figure
    '''
    
    color_iter = itertools.cycle(['navy', 'darkorange', 'purple', 'gold', 'red'])

    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

def compute_nj(y, var_distrib):
    ''' Compute nj for each variable y_j
    
    y (numobs x p ndarray): The original data
    var_distrib (p 1darray): The type of the variables in the data
    -------------------------------------------------------------------
    returns (tuple (p 1d array, nb_bin 1d array, nb_ord 1d array)): The number 
    of categories of all the variables, for count/bin variables only and for 
    ordinal variables only
    '''
    
    nj = []
    nj_bin = []
    nj_ord = []
    for i in range(len(y.columns)):
        if np.logical_or(var_distrib[i] == 'bernoulli',var_distrib[i] == 'binomial'): 
            max_nj = np.max(y.iloc[:,i], axis = 0)
            nj.append(max_nj)
            nj_bin.append(max_nj)
        else:
            card_nj = len(np.unique(y.iloc[:,i]))
            nj.append(card_nj)
            nj_ord.append(card_nj)
    
    nj = np.array(nj)
    nj_bin = np.array(nj_bin)
    nj_ord = np.array(nj_ord)

    return nj, nj_bin, nj_ord


def cluster_purity(cm):
    ''' Compute the cluster purity index mentioned in Chen and He (2016)'''
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm) 


##########################################################################################################
#################################### DGMM Utils ##########################################################
##########################################################################################################


def repeat_tile(x, reps, tiles):
    ''' Repeat then tile a quantity to mimic the former code logic'''
    x_rep = np.repeat(x, reps, axis = 0)
    x_tile_rep = np.tile(x_rep, (tiles, 1, 1))
    return x_tile_rep
        

def compute_path_params(eta, H, psi):
    ''' Compute the gaussian parameters for each path
    H (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda parameters for each layer
    psi (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi parameters for each layer
    eta (list of nb_layers elements of shape (K_l x r_l-1)): mu parameters for each layer
    ------------------------------------------------------------------------------------------------
    returns (tuple of len 2): The updated parameters mu_s and sigma for all s in Omega
    '''
    
    #=====================================================================
    # Retrieving model parameters
    #=====================================================================
    
    L = len(H)
    k = [len(h) for h in H]
    k_aug = k + [1] # Integrating the number of components of the last layer i.e 1
    
    r1 = H[0].shape[1]
    r2_L = [h.shape[2] for h in H] # r[2:L]
    r = [r1] + r2_L # r augmented
    
    #=====================================================================
    # Initiating the parameters for all layers
    #=====================================================================
    
    mu_s = [0 for i in range(L + 1)]
    sigma_s = [0 for i in range(L + 1)]
    
    # Initialization with the parameters of the last layer
    mu_s[-1] = np.zeros((1, r[-1], 1)) # Inverser k et r plus tard
    sigma_s[-1] = np.eye(r[-1])[n_axis]
    
    #==================================================================================
    # Compute Gaussian parameters from top to bottom for each path
    #==================================================================================

    for l in reversed(range(0, L)):
        H_repeat = np.repeat(H[l], np.prod(k_aug[l + 1: ]), axis = 0)
        eta_repeat = np.repeat(eta[l], np.prod(k_aug[l + 1: ]), axis = 0)
        psi_repeat = np.repeat(psi[l], np.prod(k_aug[l + 1: ]), axis = 0)
        
        mu_s[l] = eta_repeat + H_repeat @ np.tile(mu_s[l + 1], (k[l], 1, 1))
        
        sigma_s[l] = H_repeat @ np.tile(sigma_s[l + 1], (k[l], 1, 1)) @ t(H_repeat, (0, 2, 1)) \
            + psi_repeat
        
    return mu_s, sigma_s


def compute_chsi(H, psi, mu_s, sigma_s):
    ''' Compute chsi as defined in equation (8) of the DGMM paper '''
    L = len(H)
    k = [len(h) for h in H]
    
    #=====================================================================
    # Initiating the parameters for all layers
    #=====================================================================
    
    # Initialization with the parameters of the last layer    
    chsi = [0 for i in range(L)]
    chsi[-1] = pinv(pinv(sigma_s[-1]) + t(H[-1], (0, 2, 1)) @ pinv(psi[-1]) @ H[-1]) 

    #==================================================================================
    # Compute chsi from top to bottom 
    #==================================================================================
        
    for l in range(L - 1):
        Ht_psi_H = t(H[l], (0, 2, 1)) @ pinv(psi[l]) @ H[l]
        Ht_psi_H = np.repeat(Ht_psi_H, np.prod(k[l + 1:]), axis = 0)
        
        sigma_next_l = np.tile(sigma_s[l + 1], (k[l], 1, 1))
        chsi[l] = pinv(pinv(sigma_next_l) + Ht_psi_H)
            
    return chsi

def compute_rho(eta, H, psi, mu_s, sigma_s, z_c, chsi):
    ''' Compute rho as defined in equation (8) of the DGMM paper '''
    
    L = len(H)    
    rho = [0 for i in range(L)]
    k = [len(h) for h in H]
    k_aug = k + [1] 

    for l in range(0, L):
        sigma_next_l = np.tile(sigma_s[l + 1], (k[l], 1, 1))
        mu_next_l = np.tile(mu_s[l + 1], (k[l], 1, 1))

        HxPsi_inv = t(H[l], (0, 2, 1)) @ pinv(psi[l])
        HxPsi_inv = np.repeat(HxPsi_inv, np.prod(k_aug[l + 1: ]), axis = 0)

        rho[l] = chsi[l][n_axis] @ (HxPsi_inv[n_axis] @ z_c[l][..., n_axis] \
                                    + (pinv(sigma_next_l) @ mu_next_l)[n_axis])
                
    return rho
    