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
components <- read.csv("components.csv")


# Compare retrieved components
components <- read.csv("components.csv")
Basis <- read.csv("basis1.csv", header = FALSE)
Coef <- read.csv("coef1.csv", header = FALSE)
Coef <- t(Coef)
nn_component1 <- component1 - min(component1)
nn_component2 <- component2 - min(component2)

plot(scale(nn_component1[1:50]), type = 'l', col = 'blue', ylim = c(-3,3))
lines(scale(Basis[1:50, 1]), type = 'l', col = 'red')

plot(scale(multipliers[, 1]), type = 'l', col = 'blue')
lines(scale(Coef[,1]), type = 'l', col = 'red')

plot(scale(nn_component2[1:time_points]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(Basis[1:time_points, 2]), type = 'l', col = 'red')

plot(scale(multipliers[, 2]), type = 'l', col = 'blue')
lines(scale(Coef[,2]), type = 'l', col = 'red')


# Compare consensus components
time_points = 50
consensus <- read.csv("fake_data_consensus_ncomp_4_nruns50.csv", header = FALSE)

plot(scale(component1[1:time_points]), type = 'l', col = 'blue', ylim = c(-3,3))
lines(scale(consensus[1:time_points, 3]), type = 'l', col = 'darkgreen')

plot(scale(component2[1:time_points]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(consensus[1:time_points, 2]), type = 'l', col = 'darkgreen')

plot(scale(component3[1:time_points]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(consensus[1:time_points, 1]), type = 'l', col = 'darkgreen')

plot(scale(component4[1:time_points]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(consensus[1:time_points, 4]), type = 'l', col = 'darkgreen')


# Calculate BIC
n = numWords * numPeople
n = numWords
nn_fake_data = fake_data - min(fake_data)
consensus1 <- read.csv("fake_data_consensus_ncomp_1_nruns50.csv", header = FALSE)
consensus2 <- read.csv("fake_data_consensus_ncomp_2_nruns50.csv", header = FALSE)
consensus3 <- read.csv("fake_data_consensus_ncomp_3_nruns50.csv", header = FALSE)
consensus4 <- read.csv("fake_data_consensus_ncomp_4_nruns50.csv", header = FALSE)
consensus5 <- read.csv("fake_data_consensus_ncomp_5_nruns50.csv", header = FALSE)
consensus6 <- read.csv("fake_data_consensus_ncomp_6_nruns50.csv", header = FALSE)

BIC <- function(n, data, components) {
  sum_components <- apply(scale(components), 1, sum)
  comp_matrix <- matrix(rep(sum_components, times = ncol(data)), ncol = ncol(data))
  comp_matrix <- scale(comp_matrix)
  scale_data <- scale(data)
  RSS <- sum(scale_data - comp_matrix)^2
  n * log(RSS/n) + ncol(components) * log(n)
} 
sum_components <- apply(consensus4, 1, sum)
comp_matrix <- matrix(rep(sum_components, times = 100), ncol = 100)
comp_matrix <- scale(comp_matrix)
scale_data <- scale(nn_fake_data)

BIC(n, nn_fake_data, consensus1)
BIC(n, nn_fake_data, consensus2)
BIC(n, nn_fake_data, consensus3)
BIC(n, nn_fake_data, consensus4)
BIC(n, nn_fake_data, consensus5)
BIC(n, nn_fake_data, consensus6)




# Other Exploratory Stuff
plot(fake_data[1:50,2], type = 'l', col = 'blue')
abline(h=0)
lines(component1[1:50], type = 'l')
lines(component2[1:50], type = 'l', col = 'red')
