# Fake Data Creation

library(mvtnorm)

setwd("~/Documents/GitHub/Natural-Stories-ICA/ICA")

set.seed(10)

numWords <- 1000
numPeople <- 100
component1 <- rnorm(numWords, mean = 0, sd = 1)
component2 <- rnorm(numWords, mean = 0, sd = 1)
individual <- rnorm(numPeople, mean = 0, sd = 1)
error <- matrix(rnorm(numWords*numPeople, mean = 0, sd = 0.66), nrow = numWords, ncol = numPeople)
beta0 <- 1
betaInd <- 1
betaC1 <- 2
betaC2 <- 3

covariance_matrix <- matrix(c(1, 0, 0, 1), nrow = 2, ncol = 2)
multipliers <- rmvnorm(numPeople, sigma = covariance_matrix, mean = c(betaC1, betaC2))

fake_data <- matrix(0, nrow = numWords, ncol = numPeople)
for (i in seq_len(numPeople)) {
  for (j in seq_len(numWords)) {
    fake_data[j, i] <- beta0 + multipliers[i, 1]*component1[j] + multipliers[i, 2]*component2[j] + betaInd*individual[i]
  }
}
fake_data <- fake_data + error

write.csv(fake_data, "~/Documents/GitHub/Natural-Stories-ICA/ICA/fake_data.csv")
write.csv(multipliers, "~/Documents/GitHub/Natural-Stories-ICA/ICA/multipliers.csv")
write.csv(cbind(component1, component2), "~/Documents/GitHub/Natural-Stories-ICA/ICA/components.csv")



# Compare retrieved components
Components <- read.csv("components.csv")
Basis <- read.csv("basis.csv", header = FALSE)
Coef <- read.csv("coef.csv", header = FALSE)
Coef <- t(Coef)
nn_component1 <- component1 - min(component1)
nn_component2 <- component2 - min(component2)

plot(scale(nn_component1[1:50]), type = 'l', col = 'blue', ylim = c(-3,3))
lines(scale(Basis[1:50, 1]), type = 'l', col = 'red')

plot(scale(multipliers[, 1]), type = 'l', col = 'blue')
lines(scale(Coef[,1]), type = 'l', col = 'red')

plot(scale(nn_component2[1:100]), type = 'l', col = 'blue', ylim = c(-3, 3))
lines(scale(Basis[1:100, 2]), type = 'l', col = 'red')

plot(scale(multipliers[, 2]), type = 'l', col = 'blue')
lines(scale(Coef[,2]), type = 'l', col = 'red')


# Other Exploratory Stuff
plot(fake_data[1:50,2], type = 'l', col = 'blue')
abline(h=0)
lines(component1[1:50], type = 'l')
lines(component2[1:50], type = 'l', col = 'red')
