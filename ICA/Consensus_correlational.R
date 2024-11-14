setwd("~/Documents/GitHub/Natural-Stories-ICA/ICA")

nruns = 50
n_components = 2
n_samples = 1000
component_list <- list()  # List to store components from each run
consensus_matrix <- matrix(0, nrow = n_components, ncol = n_samples)
component_count <- rep(0, n_components)
similarity_threshold = 0.4

for (run in 0:(nruns-1)) {
  file_name <- paste0("~/Documents/GitHub/Natural-Stories-ICA/ICA/Fake Data BNMF/fake_data_ncomp_", n_components, "_run", run, ".csv")
  H <- read.csv(file_name, header = FALSE)
  H = t(H)
  
  if (run == 0) {
    consensus_matrix <- H  # Use the first run's components as the initial consensus
  } else {
  
  # For each component, compare with the cumulative consensus
  for (comp in 1:n_components) {
    # Calculate correlation with each consensus component
    max_corr <- 0
    best_match <- 0
    for (consensus_comp in 1:n_components) {
      
      # Check if the component has zero variance before calculating correlation
      if (sd(consensus_matrix[consensus_comp, ]) > 0 && sd(H[comp, ]) > 0) {
        corr <- cor(consensus_matrix[consensus_comp, ], H[comp, ])
        
        if (!is.na(corr) && abs(corr) > max_corr) {
          max_corr <- abs(corr)
          best_match <- consensus_comp
        }
      }
    }
    
    # Update consensus if correlation is above threshold
    if (max_corr >= similarity_threshold) {
      consensus_matrix[best_match, ] <- (consensus_matrix[best_match, ] * component_count[best_match] + H[comp, ]) / (component_count[best_match] + 1)
      component_count[best_match] <- component_count[best_match] + 1
    }
  }
  }
}

result <- consensus_matrix