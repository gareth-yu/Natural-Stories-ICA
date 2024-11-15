import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import nimfa
import pandas as pd

# Import csv file
data = pd.read_csv('/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF/fake_data.csv')
data = data.drop(data.columns[0], axis=1)

n_components = 6

# Change data to a numpy
V = data.to_numpy()
V = (V - V.min(0))

save_dir = "/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF/"

for r_ in range(50):
  bdnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
            beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
            n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
  bdnmf_fit = bdnmf()
      
  W = bdnmf_fit.basis()
  H = bdnmf_fit.coef()
  
  data_transformed = np.asarray(H.T).copy()
  
  np.savetxt('%sfake_data_ncomp_%d_run%d.csv' % (save_dir, n_components, r_), W, delimiter=",")
  np.save('%sfake_data_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), {'data_transformed': data_transformed, 'W': np.asarray(W), 'fit': bdnmf_fit.fit})




