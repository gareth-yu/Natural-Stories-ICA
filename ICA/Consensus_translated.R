library(dplyr)
library(cluster)
library(FNN)  # For finding nearest neighbors
library(Matrix)

# Define initial variables
n_components <- 2
save_dir <- '~/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF'
nruns <- 50

# Initialize an empty list to store responses
resp <- list()

for (r_ in 0:(nruns-1)) {
    # Load data from CSV, assuming each CSV file has bases_ matrix with samples as rows
    file_name <- paste0("~/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF/fake_data_ncomp_", n_components, "_run", r_, ".csv")
    H <- read.csv(file_name, header = FALSE)
    resp[[r_+1]] <- t(H)
}

# Combine all responses into a single matrix
resp <- do.call(rbind, resp)
combined_spectra <- as.data.frame(resp)

# Save combined spectra as CSV
write.csv(combined_spectra, sprintf('%s/spectra.csv', save_dir), row.names = TRUE)

# Load the merged spectra
merged_spectra <- as.data.frame(read.csv(sprintf('%s/spectra.csv', save_dir), row.names = 1))

density_threshold <- 0.5
k <- n_components
local_neighborhood_size <- 0.30

n_neighbors <- as.integer(local_neighborhood_size * nrow(merged_spectra) / k)

# Rescale topics to unit length (L2 normalization)
l2_spectra <- t(apply(merged_spectra, 1, function(x) x / sqrt(sum(x^2))))

# Calculate Euclidean distance matrix
topics_dist <- as.matrix(dist(l2_spectra, method = "euclidean"))

# Find partitioning order based on nearest neighbors
nearest_neighbors <- apply(topics_dist, 1, function(row) order(row)[2:(n_neighbors + 1)])

# Calculate local density based on nearest neighbors
distance_to_nearest_neighbors <- sapply(1:nrow(topics_dist), function(i) mean(topics_dist[i, nearest_neighbors[, i]]))
local_density <- data.frame(local_density = distance_to_nearest_neighbors)
rownames(local_density) <- rownames(l2_spectra)

# Apply density filter
density_filter <- local_density$local_density < density_threshold
l2_spectra_filtered <- l2_spectra[density_filter, ]

# K-means clustering
kmeans_model <- kmeans(l2_spectra_filtered, centers = k, nstart = 10, iter.max = 100)
kmeans_cluster_labels <- kmeans_model$cluster

# Calculate median usage for each cluster
median_spectra <- aggregate(as.data.frame(l2_spectra_filtered), by = list(cluster = kmeans_cluster_labels), median)
median_spectra <- median_spectra[, -1]  # Remove cluster label column

# Normalize median spectra to probability distributions
median_spectra <- t(apply(median_spectra, 1, function(row) row / sum(row)))

# Save the final response profile matrix as CSV
data_transformed <- t(median_spectra)
write.csv(data_transformed, sprintf('%s/_ncomp_%d_nruns%d.csv', save_dir, n_components, nruns), row.names = FALSE)

