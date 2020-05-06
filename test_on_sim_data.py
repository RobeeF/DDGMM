# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:45:45 2020

@author: Utilisateur
"""

import autograd.numpy as np
import os

os.chdir('C:/Users/rfuchs/Documents/GitHub/GLMLVM_MFA')

from init_params import random_init, init_cv
from utils import misc
from glmlvm import glmlvm
from datagen_functions import gen_mvdata, gen_mv_z, gen_z, gen_data

numobs = 100
seed = None

    
##############################################################################
# Debugging part
##############################################################################

r = 4
p1 = 3
numobs = 2000
M = 200
k = 3

seed = 1
init_seed = 2

nj= np.array([1,1,10,5])
nj_bin = np.array([1,1,10])
nj_ord = np.array([5])
init = init_params(r, nj_bin, nj_ord, k, init_seed)
var_distrib = np.array(["bernoulli","bernoulli","binomial","ordinal"])


eps = 1E-05
it = 30
maxstep = 100

z, labels = gen_mv_z(numobs, r, init, seed)
y = gen_mvdata(z, init, seed = None)

nb_trials = 30
miscs = np.zeros(nb_trials)

for i in range(nb_trials):
    
    #random_init = init_params(r, nj_bin, nj_ord, k, init_seed)
    random_init = init_cv(y, var_distrib, r, nj_bin, nj_ord, k, seed)
    out = glmlvm(y, r, k, random_init, var_distrib, nj, M, it, eps, maxstep, seed)
    
    miscs[i] = misc(labels, out['classes'])
    print(miscs[i])
    
    
miscsM10 = miscs

miscs.mean()
miscsM10.mean()

out['mu']
init['mu']

out['sigma']
init['sigma']

out["lambda_bin"]
init["lambda_bin"]

out["lambda_ord"]
init["lambda_ord"]


tsne = manifold.TSNE(n_components = 2, init='pca', random_state=0, perplexity=20)
y_tsne = tsne.fit_transform(y)

df = pd.DataFrame(y_tsne, columns = ["tsne-2d-one", 'tsne-2d-two'])
df['classes'] = labels

flatui = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="classes",
    palette=sns.color_palette(sns.xkcd_palette(flatui), len(np.unique(labels))),
    data=df,
    legend="full",
    alpha=0.3
)


# Representing the learned clustering with UMAP   
reducer = umap.UMAP()
embedding = reducer.fit_transform(y)
df_umap = pd.DataFrame(embedding, columns = ["umap-2d-one", 'umap-2d-two'])
df_umap['classes'] = labels

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="umap-2d-one", y="umap-2d-two",
    hue="classes",
    palette=sns.color_palette(sns.xkcd_palette(flatui), len(np.unique(labels))),
    data=df_umap,
    legend="full",
    alpha=0.3
)





# Check identifiability
'''
print((mu.flatten() * w.flatten()).sum())
print((w[..., n_axis, n_axis] * (sigma + mu @ t(mu, (0,2,1)))  - \
      t((w[..., n_axis, n_axis] * mu).sum(0, keepdims = True)) @ (w[..., n_axis, n_axis] * mu).sum(0, keepdims= True)).sum())


plt.scatter(Ez_y, z) 
'''

####################################################################################
# Last (?) debug 
####################################################################################

numobs = 500
M = 50

z, labels = gen_z(numobs, seed)
y = gen_data(z, 1)

y = np.genfromtxt('C:/Users/rfuchs/Documents/These/Stats/binary_dgmm/asta/asta_Rcode_compMCGH/y.csv', delimiter = ',', skip_header = True)[:,1:]

p1=3
p2=1
p=p1+p2
szo=5
o = 5
k = 3
r = 1
var_distrib = np.array(["bernoulli","bernoulli","binomial","ordinal"])
nj= np.array([1,1,10,5])
lik = - np.inf
eps = 1E-05
it = 3
maxstep = 10
seed = 1

## Veri values
r = 1
w = np.array([0.3,0.4,0.3])
mu = np.array([-1.26,0,1.26]).reshape(k,r)
sigma = np.array([0.055,0.0361,0.055]).reshape(k, r, r)

lambda_ord = np.array([[-1.00, -0.30, 0.40,  0.90, 1.50]])
lambda_bin = np.array([[0.50, 2.00], [0.60, 2.50], [0.70, 2.00]]) 


init = {}
init['mu'] = mu 
init['sigma'] = sigma
init['lambda_bin'] = lambda_bin
init['lambda_ord'] = lambda_ord
init['w'] = w



####################################################################################
# On text data
####################################################################################
os.chdir('C:/Users/rfuchs/Documents/These/Stats/20news_binary_dataset')
y_text = np.genfromtxt('20news_count.csv', delimiter = ',', skip_header = True)
labels = np.genfromtxt('20news_targets.csv', delimiter = ',', skip_header = True)


r = 1
numobs = len(y_text)
M = 10
k = 4

seed = 1
init_seed = 2

len(nj)

nj= y_text.max(0)
nj_bin = np.array([1,1,10])
nj_ord = np.array([5])
init = init_params(r, nj_bin, nj_ord, k, init_seed)
var_distrib = np.array(["bernoulli","bernoulli","binomial","ordinal"])
