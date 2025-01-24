import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

# For one number of components
n_components = 
save_dir = "/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/"
nruns = 200
resp = []

for r_ in range(nruns):
  try:
    data = np.load('%sBNMF_1_4_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), allow_pickle = True).item()     
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

  # Find median usage for each component across cluster
  median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()

  # Normalize median spectra to probability distributions.
  median_spectra = (median_spectra.T/median_spectra.sum(1)).T

  # Compute the silhouette score
  #stability = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')
    
  #### Final response profile matrix 
    
  data_transformed = median_spectra.values.T
  np.savetxt(os.path.join(save_dir, 'BNMF_1_4_consensus_ncomp_%d_nruns%d.csv' % (n_components, nruns)), data_transformed, delimiter=",")





##########################################################################################################
########################################################################################################## FOR FAKE DATA
##########################################################################################################

total_components = 4
save_dir = '/Users/garethyu/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF/'
nruns = 100


# For Multiple Runs
for n_components in range(1, total_components + 1):
    resp = []
    current_nruns = nruns  # Keep a copy of the initial number of runs
    
    # Load and process the data for each run
    for r_ in range(current_nruns):
        try:
            data = np.load('%sfake_data_ncomp_%d_run%d.npy' % (save_dir, n_components, r_), allow_pickle=True).item()
            resp.append(data['W'].T)
            del data
        except FileNotFoundError:
            current_nruns -= 1  # Reduce the count if file not found

    resp = np.asarray(resp)
    resp = resp.reshape(-1, resp.shape[-1])

    combined_spectra = pd.DataFrame(
        resp, 
        columns=range(resp.shape[-1]), 
        index=['run%d' % i for i in range(n_components * current_nruns)]
    )
    combined_spectra.to_pickle('%s/spectra.pkl' % save_dir)
    merged_spectra = pd.read_pickle('%s/spectra.pkl' % save_dir)

    # Parameters for clustering
    density_threshold = 0.5
    k = n_components
    local_neighborhood_size = 0.3
    n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0] / k)

    # Rescale topics to length of 1
    l2_spectra = (merged_spectra.T / np.sqrt((merged_spectra**2).sum(axis=1))).T

    # Compute local density
    topics_dist = euclidean_distances(l2_spectra.values)
    partitioning_order = np.argpartition(topics_dist, n_neighbors + 1)[:, :n_neighbors + 1]
    distance_to_nearest_neighbors = topics_dist[np.arange(topics_dist.shape[0])[:, None], partitioning_order]
    local_density = pd.DataFrame(
        distance_to_nearest_neighbors.sum(1) / n_neighbors,
        columns=['local_density'],
        index=l2_spectra.index
    )

    density_filter = local_density.iloc[:, 0] < density_threshold
    l2_spectra = l2_spectra.loc[density_filter, :]

    # K-means clustering
    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans_model.fit(l2_spectra)
    kmeans_cluster_labels = pd.Series(kmeans_model.labels_ + 1, index=l2_spectra.index)

    # Compute median spectra and normalize
    median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()
    median_spectra = (median_spectra.T / median_spectra.sum(1)).T

    # Save the final transformed data
    data_transformed = median_spectra.values.T
    np.savetxt(os.path.join(save_dir, 'fake_data_consensus_ncomp_%d_nruns%d.csv' % (n_components, current_nruns)), data_transformed, delimiter=",")




# For one number of components
n_components = 4
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

  # Find median usage for each component across cluster
  median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()

  # Normalize median spectra to probability distributions.
  median_spectra = (median_spectra.T/median_spectra.sum(1)).T

  # Compute the silhouette score
  #stability = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')
    
  #### Final response profile matrix 
    
  data_transformed = median_spectra.values.T
  np.savetxt(os.path.join(save_dir, 'fake_data_consensus_ncomp_%d_nruns%d.csv' % (n_components, nruns)), data_transformed, delimiter=",")



new_values = np.array(r.clusters, dtype=int)
kmeans_cluster_labels[:] = new_values

median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()
median_spectra = (median_spectra.T/median_spectra.sum(1)).T
data_transformed = median_spectra.values.T

'''
n_components = 7
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
'''
