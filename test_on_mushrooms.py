# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:33:34 2020

@author: rfuchs
"""

import os 
os.chdir('C:/Users/rfuchs/Documents/GitHub/GLMLVM_MFA')

import warnings 
warnings.filterwarnings("ignore") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!!

from init_params import dim_reduce_init
from glmlvm import glmlvm
from utils import misc, gen_categ_as_bin_dataset, \
        ordinal_encoding, compute_nj

import pandas as pd
import autograd.numpy as np


from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("error") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!

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
k = len(np.unique(labels_oh))

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

# Encode categorical datas
y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)

# Encode binary data
le = LabelEncoder()
for colname in y.columns:
    if y[colname].dtype != np.int64:
        y[colname] = le.fit_transform(y[colname])

#===========================================#
# Running the algorithm
#===========================================# 

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

# Launching the algorithm
r = np.array([3,1])
numobs = len(y)
M = r * 8
k = 2

seed = 1
init_seed = 2
    
eps = 1E-05
it = 30
maxstep = 100

# Prince init
prince_init = dim_reduce_init(y, k, r, nj, var_distrib, seed = None)
out = glmlvm(y_np, r, k, prince_init, var_distrib, nj, M, it, eps, maxstep, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))


#==================================================================
# Performance measure : Finding the best specification
#==================================================================

from sklearn.preprocessing import StandardScaler

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

# Launching the algorithm
k = 2

seed = 1
init_seed = 2
    
eps = 1E-05
it = 30
maxstep = 100

ss = StandardScaler()
y_scale = ss.fit_transform(y_np)


nb_trials = 5
miscs_df = pd.DataFrame(columns = ['it_id', 'r', 'model', 'misc'])

for r1 in range(1,4):
    for r2 in range(1, 3):
        if r1 <= r2:
            continue

        M = np.array([r1,r2]) * 4
        M[1] = M[1]* 2

        print('r1=',r1, 'r2', r2)
        for i in range(nb_trials):
            # Prince init
            prince_init = dim_reduce_init(y, k, [r1, r2], nj, var_distrib, seed = None)
            miscs_df = miscs_df.append({'it_id': i + 1, 'r': str([r1, r2]), 'model': 'k-means', 'misc': misc(labels_oh, prince_init['preds'])},\
                                       ignore_index=True)
        
            try:
                out = glmlvm(y_np, [r1, r2], k, prince_init, var_distrib, nj, M, it, eps, maxstep, None)
                miscs_df = miscs_df.append({'it_id': i + 1, 'r': str([r1, r2]), 'model': 'MCA & 1L-DGMM','misc': misc(labels_oh, out['classes'])}, \
                                           ignore_index=True)
    
            except:
                miscs_df = miscs_df.append({'it_id': i + 1, 'r': str([r1, r2]), 'model': 'MCA & 1L-DGMM', 'misc': np.nan}, \
                                           ignore_index=True)
                

miscs_df.boxplot(by = ['r','model'], figsize = (20, 10))

miscs_df[(miscs_df['model'] == 'MCA & 1L-DGMM')].boxplot(by = 'r', figsize = (20, 10))

miscs_df.to_csv('mush_DGMM_MFA.csv')
