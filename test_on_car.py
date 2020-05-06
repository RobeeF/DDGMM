# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:32:07 2020

@author: rfuchs
"""

import os 
os.chdir('C:/Users/rfuchs/Documents/GitHub/GLMLVM_MFA')

import warnings 
warnings.filterwarnings("ignore") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!!

import pandas as pd
import autograd.numpy as np

from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler


from glmlvm import glmlvm
from init_params import dim_reduce_init
from utils import misc, ordinal_encoding, compute_nj

warnings.filterwarnings("error") # Attention..!!!!!!!!!!!!!!!!!!!!!!!!


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
rus = RandomUnderSampler(sampling_strategy = samp_dict, random_state = 0)
y, labels = rus.fit_sample(y, labels)

le = LabelEncoder()
labels_oh = le.fit_transform(labels)
k = len(np.unique(labels_oh))


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
r = [3,2]
numobs = len(y)
M = np.array(r) * 8

seed = 1
init_seed = 2
    
eps = 1E-05
it = 30
maxstep = 100

# Prince init
prince_init = dim_reduce_init(y, k, r, nj, var_distrib,  seed = None)
#init = prince_init
#y = y_np
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
                out = glmlvm(y_np, [r1, r2], k, prince_init, var_distrib, nj, M, it, eps, maxstep, None)
                miscs_df = miscs_df.append({'it_id': i + 1, 'r': str([r1, r2]), 'model': 'MCA & 1L-DGMM','misc': misc(labels_oh, out['classes'])}, \
                                           ignore_index=True)
    
            except:
                miscs_df = miscs_df.append({'it_id': i + 1, 'r': str([r1, r2]), 'model': 'MCA & 1L-DGMM', 'misc': np.nan}, \
                                           ignore_index=True)
                

miscs_df.boxplot(by = ['r','model'], figsize = (20, 10))

miscs_df[(miscs_df['model'] == 'MCA & 1L-DGMM')].boxplot(by = 'r', figsize = (20, 10))

miscs_df.to_csv('car_DGMM_MFA.csv')