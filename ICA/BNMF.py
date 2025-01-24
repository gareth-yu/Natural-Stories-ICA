import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import nimfa
import pandas as pd


##################
# Multiple Final #
##################

# Import csv file
data = pd.read_csv('/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/BNMF_data_1_4.csv')
data = data.drop(data.columns[0], axis=1)

total_components = 50
iterations = 50

# Change data to a numpy
V = data.to_numpy()
V = (V - V.min(0))

save_dir = "/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/BNMF Files/"

for i in range(total_components):
  n_components = i + 1
  RSS = np.empty((iterations, 1))

  for r_ in range(iterations):
    bdnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
              beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
              n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
    bdnmf_fit = bdnmf()
        
    W = bdnmf_fit.basis()
    H = bdnmf_fit.coef()
  
    RSS[r_][0] = bdnmf_fit.fit.rss()
    
    data_transformed = np.asarray(H.T).copy()
    
    np.savetxt('%sBNMF_1_4_ncomp_%d_run%d.csv' % (save_dir, n_components, r_), W, delimiter=",")
    np.savetxt('%sBNMF_1_4_ncomp_%d_run%d_coef.csv' % (save_dir, n_components, r_), H, delimiter=",")
    np.save('%sBNMF_1_4_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), {'data_transformed': data_transformed, 'W': np.asarray(W), 'fit': bdnmf_fit.fit})
  
  np.savetxt('%sBNMF_1_4_ncomp_%d_RSS.csv' % (save_dir, n_components), RSS, delimiter=",")

##########################################################################################################
########################################################################################################## FOR SINGLE COMPONENT
##########################################################################################################

################
# Single Final #
################

# Import csv file
data = pd.read_csv('/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/BNMF_data_1_4.csv')
data = data.drop(data.columns[0], axis=1)

n_components = 
iterations = 200

# Change data to a numpy
V = data.to_numpy()
V = (V - V.min(0))

save_dir = "/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/BNMF Files/"

for r_ in range(iterations):
  bdnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
            beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
            n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
  bdnmf_fit = bdnmf()
      
  W = bdnmf_fit.basis()
  H = bdnmf_fit.coef()
  
  data_transformed = np.asarray(H.T).copy()
  
  np.savetxt('%sBNMF_1_4_ncomp_%d_run%d.csv' % (save_dir, n_components, r_), W, delimiter=",")
  np.save('%sBNMF_1_4_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), {'data_transformed': data_transformed, 'W': np.asarray(W), 'fit': bdnmf_fit.fit})











########################################################################################################## FOR TESTING FAKE DATA
# Import csv file
data = pd.read_csv('/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF/fake_data.csv')
data = data.drop(data.columns[0], axis=1)

total_components = 3
iterations = 50

# Change data to a numpy
V = data.to_numpy()
V = (V - V.min(0))

save_dir = "/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF/"


############
# Multiple #
############

for i in range(total_components):
  n_components = i + 1
  RSS = np.empty((iterations, 1))

  for r_ in range(iterations):
    bdnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
              beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
              n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
    bdnmf_fit = bdnmf()
        
    W = bdnmf_fit.basis()
    H = bdnmf_fit.coef()
  
    RSS[r_][0] = bdnmf_fit.fit.rss()
    
    data_transformed = np.asarray(H.T).copy()
    
    np.savetxt('%sfake_data_ncomp_%d_run%d.csv' % (save_dir, n_components, r_), W, delimiter=",")
    np.savetxt('%sfake_data_ncomp_%d_run%d_coef.csv' % (save_dir, n_components, r_), H, delimiter=",")
    np.save('%sfake_data_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), {'data_transformed': data_transformed, 'W': np.asarray(W), 'fit': bdnmf_fit.fit})
  
  np.savetxt('%sfake_data_ncomp_%d_RSS.csv' % (save_dir, n_components), RSS, delimiter=",")



##########
# Single #
##########

n_components = 4
RSS = np.empty((iterations, 1))

for r_ in range(iterations):
  bdnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
            beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
            n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
  bdnmf_fit = bdnmf()
      
  W = bdnmf_fit.basis()
  H = bdnmf_fit.coef()

  RSS[r_][0] = bdnmf_fit.fit.rss()
  
  data_transformed = np.asarray(H.T).copy()
  
  np.savetxt('%sfake_data_ncomp_%d_run%d.csv' % (save_dir, n_components, r_), W, delimiter=",")
  np.save('%sfake_data_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), {'data_transformed': data_transformed, 'W': np.asarray(W), 'fit': bdnmf_fit.fit})

np.savetxt('%sfake_data_ncomp_%d_RSS.csv' % (save_dir, n_components), RSS, delimiter=",")




############
## Actual ##
############

# Import csv file
data = pd.read_csv('/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/cleaned_story_1.csv')
data = data.drop(data.columns[0], axis=1)
data = data.drop(data.columns[0], axis=1)

total_components = 2
iterations = 50

# Change data to a numpy
V = data.to_numpy()
V = (V - V.min(0))

save_dir = "/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/"


n_components = 1
RSS = np.empty((iterations, 1))

for r_ in range(iterations):
  bdnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
            beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
            n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
  bdnmf_fit = bdnmf()
      
  W = bdnmf_fit.basis()
  H = bdnmf_fit.coef()

  RSS[r_][0] = bdnmf_fit.fit.rss()
  
  data_transformed = np.asarray(H.T).copy()
  
  np.savetxt('%sfake_data_ncomp_%d_run%d.csv' % (save_dir, n_components, r_), W, delimiter=",")
  np.save('%sfake_data_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), {'data_transformed': data_transformed, 'W': np.asarray(W), 'fit': bdnmf_fit.fit})

np.savetxt('%sfake_data_ncomp_%d_RSS.csv' % (save_dir, n_components), RSS, delimiter=",")







for i in range(total_components):
  n_components = i + 1
  RSS = np.empty((iterations, 1))

  for r_ in range(iterations):
    bdnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
              beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
              n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
    bdnmf_fit = bdnmf()
        
    W = bdnmf_fit.basis()
    H = bdnmf_fit.coef()
  
    RSS[r_][0] = bdnmf_fit.fit.rss()
    
    data_transformed = np.asarray(H.T).copy()
    
    np.savetxt('%sSPR_ncomp_%d_run%d.csv' % (save_dir, n_components, r_), W, delimiter=",")
    np.savetxt('%sSPR_ncomp_%d_run%d_coef.csv' % (save_dir, n_components, r_), H, delimiter=",")
    np.save('%sSPR_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), {'data_transformed': data_transformed, 'W': np.asarray(W), 'fit': bdnmf_fit.fit})
  
  np.savetxt('%sSPR_ncomp_%d_RSS.csv' % (save_dir, n_components), RSS, delimiter=",")



