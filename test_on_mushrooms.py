# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:33:34 2020

@author: rfuchs
"""

import os 
os.chdir('C:/Users/rfuchs/Documents/GitHub/DDGMM')

#import warnings 
#warnings.simplefilter("default")

from init_params import dim_reduce_init
from ddgmm import DDGMM
from data_preprocessing import gen_categ_as_bin_dataset, \
        ordinal_encoding, compute_nj
from metrics import misc, cluster_purity

import pandas as pd
import autograd.numpy as np

from copy import deepcopy
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix


###############################################################################################
##############        Clustering on the Mushrooms dataset (UCI)          ######################
###############################################################################################

#===========================================#
# Importing and droping NaNs
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

# Importing and selecting data
mush = pd.read_csv('mushrooms/agaricus-lepiota.csv', sep = ',', header = None)
mush = mush.infer_objects()

y = mush.iloc[:,1:]

le = LabelEncoder()
labels = mush.iloc[:,0]
labels_oh = le.fit_transform(labels)

#Delete missing data
missing_idx = y.iloc[:, 10] != '?'
y = y[missing_idx]

labels = labels[missing_idx]
labels_oh = labels_oh[missing_idx]
n_clusters = len(np.unique(labels_oh))

#===========================================#
# Formating the data
#===========================================#

var_distrib = np.array(['categorical', 'categorical', 'categorical', 'bernoulli', 'categorical',\
                        'categorical', 'categorical', 'bernoulli', 'categorical', 'categorical',\
                        'categorical', 'categorical', 'categorical', 'categorical', 'categorical', \
                        'categorical', 'categorical', 'ordinal', 'categorical', 'categorical', \
                        'categorical', 'categorical'])

ord_idx = np.where(var_distrib == 'ordinal')[0]

# Extract labels for each y_j and then perform dirty manual reordering
all_labels = [np.unique(y.iloc[:,idx]) for idx in ord_idx]
all_codes = [list(range(len(lab))) for lab in all_labels]    

# Encode ordinal data
for i, idx in enumerate(ord_idx):
    y.iloc[:,idx] = ordinal_encoding(y.iloc[:,idx], all_labels[i], all_codes[i])


y_categ_non_enc = deepcopy(y)
vd_categ_non_enc = deepcopy(var_distrib)

# Encode categorical datas
y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)

# Encode binary data
le = LabelEncoder()
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'bernoulli': # Attention
        y[colname] = le.fit_transform(y[colname])

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

#===========================================#
# Running the algorithm
#===========================================# 

r = np.array([2, 1])
numobs = len(y)
k = [n_clusters]

seed = 1
init_seed = 2
    
eps = 1E-05
it = 10
maxstep = 100

# Prince init
prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
out = DDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps, maxstep, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))


#=======================================================================
# Performance measure : Finding the best specification for init and DDGMM
#=======================================================================

res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/mushrooms'


# Init
# Best one r = (2,1)
numobs = len(y)
k = [n_clusters]

nb_trials= 30
mca_res = pd.DataFrame(columns = ['it_id', 'r', 'micro', 'macro', 'purity'])

for r1 in range(2, 9):
    print(r1)
    r = np.array([r1, 1])
    for i in range(nb_trials):
        # Prince init
        try:
            prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
            m, pred = misc(labels_oh, prince_init['classes'], True) 
            cm = confusion_matrix(labels_oh, pred)
            purity = cluster_purity(cm)
                
            micro = precision_score(labels_oh, pred, average = 'micro')
            macro = precision_score(labels_oh, pred, average = 'macro')
            #print(micro)
            #print(macro)
        
            mca_res = mca_res.append({'it_id': i + 1, 'r': str(r), 'micro': micro, 'macro': macro, \
                                            'purity': purity}, ignore_index=True)
        except:
            mca_res = mca_res.append({'it_id': i + 1, 'r': str(r), 'micro': np.nan, 'macro': np.nan, \
                                            'purity': np.nan}, ignore_index=True)            
       

mca_res.groupby('r').mean()
mca_res.groupby('r').std()

mca_res.to_csv(res_folder + '/mca_res.csv')

# DDGMM
r = np.array([5, 4, 2])
numobs = len(y)
k = [4, n_clusters]
eps = 1E-05
it = 10
maxstep = 30

# First fing the best architecture 
prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
out = DDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, M, it, \
            eps, maxstep, seed = None)

r = [2,1]
numobs = len(y)
k = [2]

nb_trials= 30
ddgmm_res = pd.DataFrame(columns = ['it_id', 'micro', 'macro', 'purity'])

for i in range(nb_trials):

    print(i)
    # Prince init
    prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)

    try:
        out = DDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, \
                    nj, it, eps, maxstep, seed = None, perform_selec = False)
        m, pred = misc(labels_oh, out['classes'], True) 
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)
        
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        print(micro)
        print(macro)

        ddgmm_res = ddgmm_res.append({'it_id': i + 1, 'micro': micro, 'macro': macro, \
                                    'purity': purity}, ignore_index=True)
        ddgmm_res.to_csv(res_folder + '/ddgmm_res.csv')

    except:
        ddgmm_res = ddgmm_res.append({'it_id': i + 1, 'micro': np.nan, 'macro': np.nan, \
                                    'purity': np.nan}, ignore_index=True)



ddgmm_res.mean()
ddgmm_res.std()

ddgmm_res.to_csv(res_folder + '/ddgmm_res.csv')




#=======================================================================
# Performance measure : Finding the best specification for other algos
#=======================================================================

from gower import gower_matrix
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom   
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/mushrooms'

# Feature category (cf)
cf_non_enc = (vd_categ_non_enc != 'ordinal') & (vd_categ_non_enc != 'binomial')

# Non encoded version of the dataset:
y_nenc_typed = y_categ_non_enc.astype(np.object)
y_np_nenc = y_nenc_typed.values
y_np = y.values
numobs = len(y)

ss = StandardScaler()
y_scale = ss.fit_transform(y_np)


# Defining distances over the non encoded features
dm = gower_matrix(y_nenc_typed, cat_features = cf_non_enc) 

# <nb_trials> tries for each specification
nb_trials = 30


#****************************
# Partitional algorithm
#****************************

part_res_modes = pd.DataFrame(columns = ['it_id', 'init', 'micro', 'macro', 'purity'])

inits = ['Huang', 'Cao', 'random']

for init in inits:
    print(init)
    for i in range(nb_trials):
        km = KModes(n_clusters= n_clusters, init=init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y_np_nenc)
        m, pred = misc(labels_oh, kmo_labels, True) 
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        part_res_modes = part_res_modes.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'purity': purity}, \
                                               ignore_index=True)
  
# cao best spe
#part_res_modes = pd.read_csv(res_folder + '/part_res.csv')
part_res_modes.groupby('init').mean()
part_res_modes.groupby('init').std()

part_res_modes.to_csv(res_folder + '/part_res.csv')

#****************************
# K prototypes
#****************************

part_res_proto = pd.DataFrame(columns = ['it_id', 'init', 'micro', 'macro', 'purity'])


for init in inits:
    print(init)
    for i in range(nb_trials):
        km = KPrototypes(n_clusters = n_clusters, init = init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y_np_nenc, categorical = np.where(cf_non_enc)[0].tolist())
        m, pred = misc(labels_oh, kmo_labels, True) 
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        part_res_proto = part_res_proto.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'purity': purity}, \
                                               ignore_index=True)

# Cao is best
#part_res_proto = pd.read_csv(res_folder +  '/part_res_proto.csv')
part_res_proto.groupby('init').mean()
part_res_proto.groupby('init').std()

part_res_proto.to_csv(res_folder +  '/part_res_proto.csv')

#****************************
# Hierarchical clustering
#****************************

hierarch_res = pd.DataFrame(columns = ['it_id', 'linkage', 'micro', 'macro', 'purity'])

linkages = ['complete', 'average', 'single']

for linky in linkages: 
    print('Linkage:', linky)
    for i in range(nb_trials):  
        aglo = AgglomerativeClustering(n_clusters = n_clusters, affinity ='precomputed', linkage = linky)
        aglo_preds = aglo.fit_predict(dm)
        m, pred = misc(labels_oh, aglo_preds, True) 
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        hierarch_res = hierarch_res.append({'it_id': i + 1, 'linkage': linky, \
                            'micro': micro, 'macro': macro, 'purity': purity},\
                                           ignore_index=True)


# Average is the best
#hierarch_res = pd.read_csv(res_folder +  '/hierarch_res.csv')
hierarch_res.groupby('linkage').mean()
hierarch_res.groupby('linkage').std()

hierarch_res.to_csv(res_folder +  '/hierarch_res.csv')

#****************************
# Neural-network based
#****************************

som_res = pd.DataFrame(columns = ['it_id', 'sigma', 'lr' ,'micro', 'macro', 'purity'])

sigmas = np.linspace(0.001, 3, 5)
lrs = np.linspace(0.0001, 0.5, 10)

for sig in sigmas:
    print('Sigma:', sig)
    for lr in lrs:
        for i in range(nb_trials):
            som = MiniSom(n_clusters, 1, y_np.shape[1], sigma = sig, learning_rate = lr) # initialization of 6x6 SOM
            som.train(y_np, 100) # trains the SOM with 100 iterations
            som_labels = [som.winner(y_np[i])[0] for i in range(numobs)]
            m, pred = misc(labels_oh, som_labels, True) 
            cm = confusion_matrix(labels_oh, pred)
            micro = precision_score(labels_oh, pred, average = 'micro')
            macro = precision_score(labels_oh, pred, average = 'macro')
            purity = cluster_purity(cm)

            som_res = som_res.append({'it_id': i + 1, 'sigma': sig, 'lr': lr, \
                            'micro': micro, 'macro': macro, 'purity': purity},\
                                     ignore_index=True)

# lr = 0.166733 and sigma = 0.00100 is the best specification
#som_res = pd.read_csv(res_folder +  '/som_res.csv')
som_res.groupby(['sigma', 'lr']).mean()
som_res.groupby(['sigma', 'lr']).std()

som_res.to_csv(res_folder + '/som_res.csv')

#****************************
# Other algorithms family
#****************************

dbs_res = pd.DataFrame(columns = ['it_id', 'data' ,'leaf_size', 'eps', 'min_samples','micro', 'macro', 'purity'])

lf_size = np.arange(1,6) * 10
epss = np.linspace(0.01, 5, 5)
min_ss = np.arange(1, 5)
data_to_fit = ['scaled', 'gower']

for lfs in lf_size:
    print("Leaf size:", lfs)
    for eps in epss:
        for min_s in min_ss:
            for data in data_to_fit:
                for i in range(1):
                    if data == 'gower':
                        dbs = DBSCAN(eps = eps, min_samples = min_s, metric = 'precomputed', leaf_size = lfs).fit(dm)
                    else:
                        dbs = DBSCAN(eps = eps, min_samples = min_s, leaf_size = lfs).fit(y_scale)
                        
                    dbs_preds = dbs.labels_
                    
                    if len(np.unique(dbs_preds)) > n_clusters:
                        continue
                    
                    m, pred = misc(labels_oh, dbs_preds, True) 
                    cm = confusion_matrix(labels_oh, pred)
                    micro = precision_score(labels_oh, pred, average = 'micro')
                    macro = precision_score(labels_oh, pred, average = 'macro')
                    purity = cluster_purity(cm)
    
                    dbs_res = dbs_res.append({'it_id': i + 1, 'leaf_size': lfs, \
                                'eps': eps, 'min_samples': min_s, 'micro': micro,\
                                    'data': data, 'macro': macro, 'purity': purity},\
                                             ignore_index=True)

# Gower is the best nomatter the  other params. No error in computing mean or std ?
#dbs_res = pd.read_csv(res_folder +  '/dbs_res.csv').iloc[:, 2:]
np.unique(dbs_res[dbs_res['micro'] > 0.6180]['data'])
dbs_res[dbs_res['micro'] > 0.6180].groupby(['data','leaf_size', 'eps', 'min_samples']).std()

dbs_res.groupby(['data','leaf_size', 'eps', 'min_samples']).mean().max()
dbs_res.to_csv(res_folder + '/dbs_res.csv')

