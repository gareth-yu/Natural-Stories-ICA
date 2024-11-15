import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

n_components = 6
save_dir = '/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF/'
nruns = 50

resp = []
    
for r_ in range(nruns):
  try:
    data = np.load('%sfake_data_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), allow_pickle = True).item()     
    resp.append(data['W'].T)
    del data
  except: 
    nruns -= 1

resp = np.asarray(resp)
  resp = resp.reshape(-1, resp.shape[-1])

  combined_spectra = pd.DataFrame(resp, columns=range(resp.shape[-1]), index=['run%d' % i for i in range(n_components * nruns)])        
  combined_spectra.to_pickle('%s/spectra.pkl' % (save_dir))
  merged_spectra = pd.read_pickle('%s/spectra.pkl' % (save_dir)) 

  density_threshold = 0.5
  k = n_components
  local_neighborhood_size = 0.3

  n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0]/k)

  # Rescale topics such to length of 1.
  l2_spectra = (merged_spectra.T/np.sqrt((merged_spectra**2).sum(axis=1))).T


            #   first find the full distance matrix
  topics_dist = euclidean_distances(l2_spectra.values)
  #   partition based on the first n neighbors
  partitioning_order  = np.argpartition(topics_dist, n_neighbors+1)[:, :n_neighbors+1]
  #   find the mean over those n_neighbors (excluding self, which has a distance of 0)
  distance_to_nearest_neighbors = topics_dist[np.arange(topics_dist.shape[0])[:, None], partitioning_order]
  local_density = pd.DataFrame(distance_to_nearest_neighbors.sum(1)/(n_neighbors),
                               columns=['local_density'],
                               index=l2_spectra.index)

  density_filter = local_density.iloc[:, 0] < density_threshold
  l2_spectra = l2_spectra.loc[density_filter, :]

  kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
  kmeans_model.fit(l2_spectra)
  kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)

  # Find median usage for each gene across cluster
  median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()

  # Normalize median spectra to probability distributions.
  median_spectra = (median_spectra.T/median_spectra.sum(1)).T

  # Compute the silhouette score
  #stability = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')
    
  #### Final response profile matrix 
    
  data_transformed = median_spectra.values.T
  np.savetxt(os.path.join(save_dir, 'fake_data_consensus_ncomp_%d_nruns%d.csv' % (n_components, nruns)), data_transformed, delimiter=",")
