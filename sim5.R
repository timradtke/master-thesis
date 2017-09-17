########################################################################

# this script runs the simulations used in the fifth simulation
# discussed in the master thesis, using Poisson distributions.

########################################################################

# Here we test how the algorithms can adapt to distributions other than
# the Bernoulli distributions in simulations 1-3. We don't consider
# the Bayes-UCB algorithm in this experiment.

########################################################################
# load functions (adjust paths if necessary)

current_path <- "/code/functions/"
source(paste0(current_path, "functions.R"))
source(paste0(current_path, "functions_parallelized.R"))
current_path <- "/code/simulations/sim5/"

########################################################################
# Here we create the data set. 

# Fix the means of the arms in a vector
set.seed(3795969)
tau_pois2 <- 0.4
epsilon_pois2 <- 0
mu_pois2 <- rexp(20, 1/0.7)

# Create a data set 5000 times with 4000 rounds each from the parameters
data_list_pois2 <- list()
for(j in 1:5000) {
  curr_data <- data.frame(rep(NA, times = 10000))
  for(i in 1:length(mu_pois2)) {
    curr_data[[i]] <- as.numeric(rpois(10000, mu_pois2[i]))
  }
  names(curr_data) <- paste0("V", rep(1:length(mu_pois2)))
  data_list_pois2[[j]] <- curr_data
}
gc()

########################################################################
# UNIFORM SAMPLING #####################################################
# Run the uniform sampling strategy on the data set

system.time(pois2_UNIFORM <- para_bandit_sim_uniform(data = data_list_pois2, 
                                                     rounds = 10000))
#save(pois2_UNIFORM, file = paste0(current_path, "pois2_UNIFORM.Rda"))

# Get the average error over all 5000 simulations at each round
pois2_comp_UNIFORM <- compare_to_ground_truth(mu_pois2, 
                                              pois2_UNIFORM, 
                                              tau_pois2, 
                                              epsilon_pois2)$mean
save(pois2_comp_UNIFORM, file = paste0(current_path, "pois2_comp_UNIFORM.Rda"))

########################################################################
# ANYTIME PARAMETER FREE THRESHOLDING ALGORITHM ########################
# Locatelli et al. (2016) ##############################################

# run the APT algorithm on the data set
pois2_APT <- para_bandit_sim_APT(data = data_list_pois2, 
                                 rounds = 10000, 
                                 tau = tau_pois2, 
                                 epsilon = epsilon_pois2)

#save(pois2_APT, file = paste0(current_path, "pois2_APT.Rda"))

# Get the average error over all 5000 simulations at each round
pois2_comp_APT <- compare_to_ground_truth(mu_pois2, pois2_APT, 
                                          tau_pois2, epsilon_pois2)$mean
save(pois2_comp_APT, file = paste0(current_path, "pois2_comp_APT.Rda"))
rm(pois2_APT)
gc()

########################################################################
# SIMPLE LIKELIHOOD RATIO ALGORITHM ####################################

# Run the SLR algorithm on the data set with Poisson likelihoods
pois2_LR <- para_bandit_sim_LR_poisson(data = data_list_pois2, 
                                       rounds = 10000, 
                                       tau = tau_pois2, 
                                       epsilon = epsilon_pois2)

#save(pois2_LR, file = paste0(current_path, "pois2_LR.Rda"))

# Get the average error over all 5000 simulations at each round
pois2_comp_LR <- compare_to_ground_truth(mu_pois2, pois2_LR, 
                                         tau_pois2, epsilon_pois2)$mean
save(pois2_comp_LR, file = paste0(current_path, "pois2_comp_LR.Rda"))
rm(pois2_LR)
gc()

########################################################################
# EMPIRICAL VARIANCE GUIDED THRESHOLDING ALGORITHM #####################
# Zhong et al. (2017) ##################################################

# run the EVT algorithm on the data set
pois2_EVT <- para_bandit_sim_EVT(data = data_list_pois2, 
                                 rounds = 10000, 
                                 tau = tau_pois2, 
                                 epsilon = epsilon_pois2)
#save(pois2_EVT, file = paste0(current_path, "pois2_EVT.Rda"))

# Get the average error over all 5000 simulations at each round
pois2_comp_EVT <- compare_to_ground_truth(mu_pois2, pois2_EVT, 
                                          tau_pois2, epsilon_pois2)$mean
save(pois2_comp_EVT, file = paste0(current_path, "pois2_comp_EVT.Rda"))
gc()

########################################################################
# AUGMENTED UCB ALGORITHM ##############################################
# Mukherjee et al. (2017) ##############################################

# Run the AugUCB algorithm with the standard parameter choice of rho = 1/3
pois2_AugUCB <- para_bandit_sim_AugUCB(data = data_list_pois2, 
                                       rounds = 10000, 
                                       tau = tau_pois2)
#save(pois2_AugUCB, file = paste0(current_path, "pois2_AugUCB.Rda"))

# Get the average error over all 5000 simulations at each round
pois2_comp_AugUCB <- compare_to_ground_truth(mu_pois2, pois2_AugUCB, 
                                             tau_pois2, epsilon_pois2)$mean
save(pois2_comp_AugUCB, file = paste0(current_path, "pois2_comp_AugUCB.Rda"))

########################################################################

# Plot the performance results quickly

load(paste0(current_path, "pois2_comp_AugUCB.Rda"))
load(paste0(current_path, "pois2_comp_LR.Rda"))
load(paste0(current_path, "pois2_comp_APT.Rda"))
load(paste0(current_path, "pois2_comp_EVT.Rda"))
load(paste0(current_path, "pois2_comp_UNIFORM.Rda"))

plot(c(0,10000), c(0, -7), type = "n")
abline(h=log(0.1), lty = 2)
abline(h=log(0.01), lty = 2)
lines(log(pois2_comp_UNIFORM), col = "black")
lines(log(pois2_comp_APT), col = "darkgreen")
lines(log(pois2_comp_LR), col = "blue")
lines(log(pois2_comp_EVT), col = "red")
lines(log(pois2_comp_AugUCB), col = "red")