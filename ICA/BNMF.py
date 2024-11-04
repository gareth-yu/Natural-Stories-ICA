import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import nimfa
import pandas as pd

# Import csv file
data = pd.read_csv('fake_data.csv')
data = data.drop(data.columns[0], axis=1)
data = data.transpose()

n_components = 2

# Change data to a numpy
V = data.to_numpy()
V = (V - V.min(0)).T

bdnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
          beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
          n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
bdnmf_fit = bdnmf()
    
W = bdnmf_fit.basis()
np.savetxt("basis.csv", W, delimiter=",")
H = bdnmf_fit.coef()
np.savetxt("coef.csv", H, delimiter=",")
