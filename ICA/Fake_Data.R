# Fake Data Creation

library(mvtnorm)

numWords <- 1000
numPeople <- 100
component1 <- rnorm(numWords, mean = 0, sd = 1)
component2 <- rnorm(numWords, mean = 0, sd = 1)
individual <- rnorm(numPeople, mean = 0, sd = 1)
error <- matrix(rnorm(numWords*numPeople, mean = 0, sd = 1), nrow = numWords, ncol = numPeople)
beta0 <- 1
betaInd <- 1

covariance_matrix <- matrix(c(1, 0.3, 0.3, 1), nrow = 2, ncol = 2)
multipliers <- rmvnorm(numPeople, sigma = covariance_matrix)

fake_data <- matrix(0, nrow = numWords, ncol = numPeople)
for (i in seq_len(numPeople)) {
  for (j in seq_len(numWords)) {
    fake_data[j, i] <- beta0 + multipliers[i, 1]*component1[j] + multipliers[i, 2]*component2[j] + betaInd*individual[i]
  }
}
fake_data <- fake_data + error

write.csv(fake_data, "fake_data.csv")