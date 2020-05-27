# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:32:07 2020

@author: RobF
"""

import os 
os.chdir('C:/Users/rfuchs/Documents/GitHub/DDGMM')

import pandas as pd
import autograd.numpy as np

from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler


from ddgmm import DDGMM
from init_params import dim_reduce_init
from utilities import misc, ordinal_encoding, compute_nj


###############################################################################################
#################        Clustering on the car dataset (UCI)          #########################
###############################################################################################

#===========================================#
# Importing and droping NaNs
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

# Importing and selecting data
car = pd.read_csv('car/car.csv', sep = ',', header = None)
car = car.infer_objects()

y = car.iloc[:,:-1]
labels = car.iloc[:,-1]

# Rebalancing the data
samp_dict = {'unacc': 100, 'acc': 100, 'vgood': 65, 'good': 69}
rus = RandomUnderSampler(sampling_strategy = samp_dict, random_state = 2)
y, labels = rus.fit_sample(y, labels)

le = LabelEncoder()
labels_oh = le.fit_transform(labels)
n_clusters = len(np.unique(labels_oh))


#===========================================#
# Formating the data
#===========================================#

var_distrib = np.array(['ordinal'] * y.shape[1])

ord_idx = np.where(var_distrib == 'ordinal')[0]

# Extract labels for each y_j and then perform dirty manual reordering
all_labels = [np.unique(y.iloc[:,idx]) for idx in ord_idx]
all_labels[0] = ['low', 'med', 'high', 'vhigh']
all_labels[1] = ['low', 'med', 'high', 'vhigh']
all_labels[4] = ['small', 'med', 'big']
all_labels[5] = ['low', 'med', 'high']

all_codes = [list(range(len(lab))) for lab in all_labels]    

# Encode ordinal data
for i, idx in enumerate(ord_idx):
    y.iloc[:,idx] = ordinal_encoding(y.iloc[:,idx], all_labels[i], all_codes[i])


#===========================================#
# Running the algorithm
#===========================================# 

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

# Launching the algorithm
r = [3, 2, 1]
numobs = len(y)
M = np.array(r) * 1
k = [7, n_clusters]

seed = 1
init_seed = 2
    
eps = 1E-05
it = 30
maxstep = 100


# Prince init
prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib,  seed = None)
m, pred = misc(labels_oh, prince_init['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
#init = prince_init
#y = y_np
out = DDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, M, it, eps, maxstep, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))


# Short exploratory SVD 
u, s, vh = np.linalg.svd(prince_init['z'][0], full_matrices=True)

var_explained = np.round(s**2/np.sum(s**2), decimals=3)
np.cumsum(var_explained)

#==================================================================
# Performance measure : Finding the best specification
#==================================================================

from sklearn.preprocessing import StandardScaler

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

# Launching the algorithm
r = [2, 1]
numobs = len(y)
M = np.array(r) * 8
k = 2

seed = 1
init_seed = 2
    
eps = 1E-05
it = 30
maxstep = 100

ss = StandardScaler()
y_scale = ss.fit_transform(y_np)


nb_trials = 30
miscs_df = pd.DataFrame(columns = ['it_id', 'r', 'model', 'misc'])


for r1 in range(1,6):
    for r2 in range(1, 5):
        if r1 <= r2:
            continue

        print('r1=',r1, 'r2', r2)
        for i in range(nb_trials):
            # Prince init
            prince_init = dim_reduce_init(y, k, [r1, r2], nj, var_distrib, seed = None)
            miscs_df = miscs_df.append({'it_id': i + 1, 'r': str([r1, r2]), 'model': 'k-means', 'misc': misc(labels_oh, prince_init['preds'])},\
                                       ignore_index=True)
        
            try:
                out = DDGMM(y_np, [r1, r2], k, prince_init, var_distrib, nj, M, it, eps, maxstep, None)
                miscs_df = miscs_df.append({'it_id': i + 1, 'r': str([r1, r2]), 'model': 'MCA & 1L-DGMM','misc': misc(labels_oh, out['classes'])}, \
                                           ignore_index=True)
    
            except:
                miscs_df = miscs_df.append({'it_id': i + 1, 'r': str([r1, r2]), 'model': 'MCA & 1L-DGMM', 'misc': np.nan}, \
                                           ignore_index=True)
                

miscs_df.boxplot(by = ['r','model'], figsize = (20, 10))

miscs_df[(miscs_df['model'] == 'MCA & 1L-DGMM')].boxplot(by = 'r', figsize = (20, 10))

miscs_df.to_csv('car_DGMM_MFA.csv')


#=======================================================================
# Performance measure : Finding the best specification for other algos
#=======================================================================

from sklearn.metrics import precision_score
from utilities import cluster_purity
from gower import gower_matrix
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom   
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


# Feature category (cf)
cf = (var_distrib != 'ordinal') & (var_distrib != 'binomial')

# Non encoded version of the dataset:
y_typed = y.astype(np.object)
y_np = y_typed.values

ss = StandardScaler()
y_scale = ss.fit_transform(y_np)

# Defining distances over the non encoded features
dm = gower_matrix(y_typed, cat_features = cf) 

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
        km = KModes(n_clusters=k, init=init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y.values)
        m, pred = misc(labels_oh, kmo_labels, True) 
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        part_res_modes = part_res_modes.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'purity': purity}, \
                                               ignore_index=True)
            
part_res_modes.groupby('init').mean()
part_res_modes.to_csv('part_res_modes_car.csv')

#****************************
# K prototypes
#****************************

part_res_proto = pd.DataFrame(columns = ['it_id', 'init', 'micro', 'macro', 'purity'])


for init in inits:
    for i in range(nb_trials):
        print(init)
        km = KPrototypes(n_clusters = k, init = init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y_np, categorical = np.where(cf)[0].tolist())
        m, pred = misc(labels_oh, kmo_labels, True) 
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        part_res_proto = part_res_proto.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'purity': purity}, \
                                               ignore_index=True)

part_res_proto.groupby('init').mean()
part_res_proto.to_csv('part_res_proto_car.csv')

#****************************
# KMeans
#****************************

part_res_km = pd.DataFrame(columns = ['it_id', 'micro', 'macro', 'purity'])

for i in range(nb_trials):
    kmeans_labels = KMeans(n_clusters=2).fit_predict(y_np)
    m, pred = misc(labels_oh, kmeans_labels, True) 
    micro = precision_score(labels_oh, pred, average = 'micro')
    macro = precision_score(labels_oh, pred, average = 'macro')
    cm = confusion_matrix(labels_oh, pred)
    purity = cluster_purity(cm)

    part_res_km = part_res_km.append({'it_id': i + 1, \
                        'micro': micro, 'macro': macro, 'purity': purity}, \
                                           ignore_index=True)

part_res_km.mean()
part_res_km.to_csv('part_res_proto_car.csv')




#****************************
# Hierarchical clustering
#****************************

hierarch_res = pd.DataFrame(columns = ['it_id', 'linkage', 'micro', 'macro', 'purity'])

linkages = ['complete', 'average', 'single']

for linky in linkages: 
    for i in range(nb_trials):  
        aglo = AgglomerativeClustering(n_clusters=k, affinity ='precomputed', linkage = linky)
        aglo_preds = aglo.fit_predict(dm)
        m, pred = misc(labels_oh, aglo_preds, True) 
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        hierarch_res = hierarch_res.append({'it_id': i + 1, 'linkage': linky, \
                            'micro': micro, 'macro': macro, 'purity': purity},\
                                           ignore_index=True)

 
hierarch_res.groupby('linkage').mean()

hierarch_res.to_csv('hierarch_res_car.csv')

#****************************
# Neural-network based
#****************************

som_res = pd.DataFrame(columns = ['it_id', 'sigma', 'lr' ,'micro', 'macro', 'purity'])

sigmas = np.linspace(0.001, 3, 5)
lrs = np.linspace(0.0001, 0.5, 10)

for sig in sigmas:
    for lr in lrs:
        for i in range(nb_trials):
            som = MiniSom(k, 1, y_np.shape[1], sigma = sig, learning_rate = lr) # initialization of 6x6 SOM
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
            
som_res.groupby(['sigma', 'lr']).mean()

som_res.to_csv('som_res_car.csv')


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
                    
                    if len(np.unique(dbs_preds)) > k:
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

dbs_res.groupby(['data','leaf_size', 'eps', 'min_samples']).mean()
dbs_res.to_csv('dbs_res_car.csv')


# https://github.com/arranger1044/pam
# https://www.researchgate.net/post/What_is_the_best_way_for_cluster_analysis_when_you_have_mixed_type_of_data_categorical_and_scale
