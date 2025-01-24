# Fake Data Creation

library(mvtnorm)
library(reticulate)


setwd("~/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF/")

set.seed(10)

numWords <- 1000
numPeople <- 100
component1 <- rnorm(numWords, mean = 0, sd = 1)
component2 <- rnorm(numWords, mean = 0, sd = 1)
component3 <- rnorm(numWords, mean = 0, sd = 1)
component4 <- rnorm(numWords, mean = 0, sd = 1)
individual <- rnorm(numPeople, mean = 0, sd = 1)
error <- matrix(rnorm(numWords*numPeople, mean = 0, sd = 0.8), nrow = numWords, ncol = numPeople)
beta0 <- 1
betaInd <- 1
betaC1 <- 1
betaC2 <- 2
betaC3 <- 3
betaC4 <- 4
component1 <- component1 - min(component1)
component2 <- component2 - min(component2)
component3 <- component3 - min(component3)
component4 <- component4 - min(component4)

covariance_matrix <- diag(4)
multipliers <- rmvnorm(numPeople, sigma = covariance_matrix, mean = c(betaC1, betaC2, betaC3, betaC4))

fake_data <- matrix(0, nrow = numWords, ncol = numPeople)
for (i in seq_len(numPeople)) {
  for (j in seq_len(numWords)) {
    fake_data[j, i] <- beta0 + multipliers[i, 1]*component1[j] + multipliers[i, 2]*component2[j] +
      multipliers[i, 3]*component3[j] + multipliers[i, 4]*component4[j] + 
      betaInd*individual[i]
  }
}
fake_data <- fake_data + error
write.csv(fake_data, "fake_data.csv")
write.csv(multipliers, "multipliers.csv")
write.csv(cbind(component1, component2, component3, component4), "components.csv")
components <- read.csv("components.csv")[,-1]


# Compare retrieved components
components <- read.csv("components.csv")[,-1]
Basis <- read.csv("fake_data_ncomp_4_run1.csv", header = FALSE)
Coef <- read.csv("coef1.csv", header = FALSE)
Coef <- t(Coef)
time_points = 50

plot(scale(component1[1:50]), type = 'l', col = 'blue', ylim = c(-3,3))
lines(scale(Basis[1:50, 4]), type = 'l', col = 'red')

plot(scale(component2[1:50]), type = 'l', col = 'blue', ylim = c(-3,3))
lines(scale(Basis[1:50, 1]), type = 'l', col = 'red')

plot(scale(component3[1:50]), type = 'l', col = 'blue', ylim = c(-3,3))
lines(scale(Basis[1:50, 4]), type = 'l', col = 'red')

plot(scale(component4[1:50]), type = 'l', col = 'blue', ylim = c(-3,3))
lines(scale(Basis[1:50, 1]), type = 'l', col = 'red')

plot(scale(component2[1:time_points]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(Basis[1:time_points, 2]), type = 'l', col = 'red')

plot(scale(multipliers[, 2]), type = 'l', col = 'blue')
lines(scale(Coef[,2]), type = 'l', col = 'red')

cor(components, Basis)
cor(Basis)

# Compare consensus components
time_points = 50
consensus <- read.csv("fake_data_consensus_ncomp_4_nruns50.csv", header = FALSE)

plot(scale(component1[1:time_points]), type = 'l', col = 'blue', ylim = c(-3,3))
lines(scale(consensus[1:time_points, 3]), type = 'l', col = 'darkgreen')

plot(scale(component2[1:time_points]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(-consensus[1:time_points, 1]), type = 'l', col = 'darkgreen')

plot(scale(component3[1:time_points]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(consensus[1:time_points, 4]), type = 'l', col = 'darkgreen')

plot(scale(component4[1:time_points]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(consensus[1:time_points, 2]), type = 'l', col = 'darkgreen')

plot(scale(component1[1:time_points]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(consensus[1:time_points, 3]), type = 'l', col = 'darkgreen')

test_actual_sum <- apply(components, 1, sum)
test_retreived_sum <- apply(consensus, 1, sum)

plot(scale(test_actual_sum[1:time_points]), type = 'l', col = 'blue', ylim = c(-5, 4))
lines(scale(test_retreived_sum[1:time_points]), type = 'l', col = 'darkgreen')

cor(components, consensus)
cor(consensus)

cor(multipliers, t(py$H))


# Calculate BIC
n = numWords
n = numWords * numPeople

consensus_comp <- 8
consensus_data <- list()
RSS <- matrix(0, nrow = 100, ncol = consensus_comp)


# Import Data
for (i in 1:consensus_comp) {
  consensus_file <- paste0("fake_data_consensus_ncomp_", i, "_nruns50.csv")
  consensus_data[[i]] <- read.csv(consensus_file, header = FALSE)
  RSS_file <- paste0("fake_data_ncomp_", i, "_RSS.csv")
  RSS[,i] <- as.vector(read.csv(RSS_file, header = FALSE))$V1
}

# BIC function
BIC <- function(n, RSS, n_comp) {
  mean_RSS <- mean(RSS)
  bic <- n * log(mean_RSS / n) + n_comp * log(n)
  return(bic)
} 

BICs <- numeric(consensus_comp)

for (i in 1:consensus_comp) {
  BICs[i] <- BIC(n, RSS[,i], i)
}

BICs
plot(BICs)

mean_RSSs <- apply(RSS, 2, mean)
mean_RSSs
plot(mean_RSSs)



# T-Sne
library(reticulate)
library(Rtsne)
library(ggplot2)
library(ggpubr)

sum(py$density_filter)
cluster_assignments <- as.vector(py$kmeans_cluster_labels)
input <- py$l2_spectra
sum(cluster_assignments == 1)


# Plot Clusters on Tsne
set.seed(10) 
tsne <- list(4)
for (i in 1:4) {
  name <- paste0("tsne", i)
  tsne_results <- Rtsne(input, dim = 2, perplexity = 50, max_iter = 10000, verbose = TRUE)
  tsne_data <- data.frame(
    X = tsne_results$Y[, 1],
    Y = tsne_results$Y[, 2],
    Cluster = as.factor(clusters) # Convert clusters to a factor for coloring
  )
  
  tsne[[i]] <- ggplot(tsne_data, aes(x = X, y = Y, color = Cluster)) +
    geom_point(size = 2) +
    theme_minimal() +
    labs(title = "t-SNE Plot", x = "Dimension 1", y = "Dimension 2") +
    scale_color_brewer(palette = "Set1")
}

ggarrange(tsne[[1]], tsne[[2]], tsne[[3]], tsne[[4]])


# Plot with original components
components_temp <- t(components)
colnames(components_temp) <- colnames(input)
set.seed(10) 
tsne <- list(4)
for (i in 1:4) {
  name <- paste0("tsne", i)
  tsne_results <- Rtsne(rbind(input, components_temp), dim = 2, perplexity = 50, max_iter = 1000, verbose = TRUE)
  tsne_data <- data.frame(
    X = tsne_results$Y[, 1],
    Y = tsne_results$Y[, 2],
    Cluster = as.factor(c(cluster_assignments, 5, 5, 5, 5)), # Convert clusters to a factor for coloring
    Type = c(rep("Retrieved", length(cluster_assignments)), "Real", "Real", "Real", "Real")
  )
  
  custom_shapes <- c("Retrieved" = 16, "Real" = 23)
  
  tsne[[i]] <- ggplot(tsne_data, aes(x = X, y = Y, color = Cluster, shape = Type)) +
    geom_point(size = 2) +
    scale_shape_manual(values = custom_shapes) +
    theme_minimal() +
    labs(title = "t-SNE Plot for Original Consensus", x = "Dimension 1", y = "Dimension 2") +
    scale_color_brewer(palette = "Set1")
}

ggarrange(tsne[[1]], tsne[[2]], tsne[[3]], tsne[[4]])




tsne_results <- Rtsne(input, dim = 2, perplexity = 10, max_iter = 100000)
tsne_data <- data.frame(
  X = tsne_results$Y[, 1],
  Y = tsne_results$Y[, 2],
  Cluster = as.factor(cluster_assignments) # Convert clusters to a factor for coloring
)

par(mfrow=c(2,2))
ggplot(tsne_data, aes(x = X, y = Y, color = Cluster)) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "t-SNE Plot", x = "Dimension 1", y = "Dimension 2") +
  scale_color_brewer(palette = "Set1")



reassigned <- logical(50)
reassigned[1:50] <- TRUE

clusters <- cluster_assignments
counter <- 0

while (any(reassigned)) {
  counter <- counter + 1
  print(counter)
  for (i in 1:50) { 
      r1 <- (i-1)*4 + 1
      rn <- r1 + 4 - 1
      indices <- r1:rn
      
      components_i <- input[indices,]
      components_rest <- input[-indices,]
      
      clusters_i <- clusters[indices]
      clusters_rest <- clusters[-indices]
      
      medians <- matrix(0, nrow = 4, ncol = 1000)
      
      for (j in 1:4) {
        medians[j,] <- apply(components_rest[clusters_rest == j,], 2, median)
      }
      
      correlations <- cor(t(components_i), t(medians))
      
      
      for (j in 1:4) {
        max_cor <- which.max(correlations)
        max_index <- arrayInd(max_cor, .dim = dim(correlations))
        clusters_i[max_index[1]] <- max_index[2]
        correlations[max_index[1],] <- -1
        correlations[,max_index[2]] <- -1
      }
      
      if (all(clusters_i == clusters[indices])) {
        reassigned[i] = FALSE
      }
      
      clusters[indices] <- clusters_i
      
  }

}

final <- cbind(cluster_assignments, clusters)

for (i in 1:4) {
  median(input[which(clusters == 1), ])
}


consensus <- read.csv("fake_data_consensus_ncomp_4_nruns50.csv", header = FALSE)
new1 <- apply(input[which(clusters == 1), ], 2, median)
new2 <- apply(input[which(clusters == 2), ], 2, median)
new3 <- apply(input[which(clusters == 3), ], 2, median)
new4 <- apply(input[which(clusters == 4), ], 2, median)

apply(input[which(cluster_assignments == 1), ], 2, median)

test1 <- rbind(woop, apply(input[which(cluster_assignments == 1), ], 2, median))

new_comp <- py$data_transformed

new_all <- rbind(new1, new2, new3, new4)

cor(components, t(new_all))
cor(components, new_comp)
cor(components, consensus)




## Testing Residual Error
set.seed(10)
consensus_comp <- 8
A <- fake_data
Aperm <- A
for (j in 1:ncol(A)) {  
  Aperm[, j] <- sample(A[, j])
}
write.csv(Aperm, "fake_data_perm.csv")

W <- list()
H <- list()
Wperm <- list()
Hperm <- list()

# Import Weights and Bases for data and permuted data
for (i in 1:consensus_comp) {
  W_file <- paste0("fake_data_ncomp_", i, "_run1.csv")
  W[[i]] <- as.matrix(read.csv(W_file, header = FALSE))
  H_file <- paste0("fake_data_ncomp_", i, "_run1_coef.csv")
  H[[i]] <- as.matrix(read.csv(H_file, header = FALSE))
  Wperm_file <- paste0("fake_data_ncomp_", i, "_run0_perm.csv")
  Wperm[[i]] <- as.matrix(read.csv(Wperm_file, header = FALSE))
  Hperm_file <- paste0("fake_data_ncomp_", i, "_run0_coef_perm.csv")
  Hperm[[i]] <- as.matrix(read.csv(Hperm_file, header = FALSE))
}


# Calculate REA and REAperm
REA <- numeric(consensus_comp)
REAperm <- numeric(consensus_comp)
for (i in 1:consensus_comp) {
  REA[i] <- sum(abs(A - (W[[i]] %*% H[[i]])))
  REAperm[i] <- sum(abs(Aperm - (Wperm[[i]] %*% Hperm[[i]])))
}


plot(REA, type = 'l', ylim = c(1863939, 1871441), col = 'blue')
lines(REAperm, type = 'l', color = 'red')



# Other Exploratory Stuff
plot(fake_data[1:50,2], type = 'l', col = 'blue')
abline(h=0)
lines(component1[1:50], type = 'l')
lines(component2[1:50], type = 'l', col = 'red')
