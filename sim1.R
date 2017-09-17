########################################################################

# this script runs the simulations used in the first simulation
# discussed in the master thesis. This simulation corresponds to
# simulation 3 in Locatelli et al. (2016).

########################################################################

# Given the threshold of 0.5 for Bernoulli distributions, this is
# a problem on which the APT algorithms should perform well.

########################################################################
# load functions (adjust paths if necessary)

current_path <- "/code/functions/"
source(paste0(current_path, "functions.R"))
source(paste0(current_path, "functions_parallelized.R"))
current_path <- "/code/simulations/sim1/"

########################################################################
# Here we create the data set

# Fix the means of the arms in a vector
# All parameters as in Locatelli et al. (2016)
mean_loc2 <- c(0.4-0.2^(1:4), 0.45, 0.55, 0.6+0.2^(5-(1:4)))
tau_loc2 <- 0.5
epsilon_loc2 <- 0.1

# Create a data set 5000 times with each 4000 rounds from the parameters
data_list2 <- list()
set.seed(1024)
for(j in 1:5000) {
  curr_data <- data.frame(rep(NA, times = 4000))
  for(i in 1:length(mean_loc2)) {
    curr_data[[i]] <- as.numeric(purrr::rbernoulli(4000, p  = mean_loc2[i]))
  }
  names(curr_data) <- paste0("V", rep(1:length(mean_loc2)))
  data_list2[[j]] <- curr_data
}

########################################################################
# UNIFORM SAMPLING #####################################################
# Run the uniform sampling strategy on the data set
system.time(loc2_UNIFORM_2000 <- para_bandit_sim_uniform(data = data_list2, 
                                                         rounds = 2000))
#   user  system elapsed 
# 14.440   8.049 582.563 
# -> Takes about 10 minutes -> 0.1164 seconds per simulation

#save(loc2_UNIFORM_2000, file = paste0(current_path, "loc2_UNIFORM_2000.Rda"))
#load(file = paste0(current_path, "loc2_UNIFORM_2000.Rda"))

# Get the average error over all 5000 simulations at each round
loc2_comp_UNIFORM_2000 <- compare_to_ground_truth(mean_loc2, 
                                                  loc2_UNIFORM_2000, 
                                                  tau_loc2, 
                                                  epsilon_loc2)$mean
save(loc2_comp_UNIFORM_2000, file = paste0(current_path, 
                                           "loc2_comp_UNIFORM_2000.Rda"))
rm(loc2_UNIFORM_2000)
gc()

########################################################################
# ANYTIME PARAMETER FREE THRESHOLDING ALGORITHM ########################
# Locatelli et al. (2016) ##############################################

# run the APT algorithm on the data set
loc2_APT_2000 <- para_bandit_sim_APT(data = data_list2, 
                                     rounds = 2000, 
                                     tau = tau_loc2, 
                                     epsilon = epsilon_loc2)
#save(loc2_APT_2000, file = paste0(current_path, "loc2_APT_2000.Rda"))
#load(file = paste0(current_path, "loc2_APT.Rda"))

# Get the average error over all 5000 simulations at each round
loc2_comp_APT_2000 <- compare_to_ground_truth(mean_loc2, loc2_APT_2000, tau_loc2, 
                                              epsilon_loc2)$mean
save(loc2_comp_APT_2000, file = paste0(current_path, "loc2_comp_APT_2000.Rda"))
rm(loc2_APT)
gc()

########################################################################
# EMPIRICAL VARIANCE GUIDED THRESHOLDING ALGORITHM #####################
# Zhong et al. (2017) ##################################################

# run the EVT algorithm on the data set
loc2_EVT <- para_bandit_sim_EVT(data = data_list2, 
                                rounds = 2000, 
                                tau = tau_loc2, 
                                epsilon = epsilon_loc2)
#save(loc2_EVT, file = paste0(current_path, "loc2_EVT.Rda"))

# Get the average error over all 5000 simulations at each round
loc2_comp_EVT <- compare_to_ground_truth(mean_loc2, loc2_EVT, 
                                         tau_loc2, epsilon_loc2)$mean
save(loc2_comp_EVT, file = paste0(current_path, "loc2_comp_EVT.Rda"))
rm(loc2_EVT)
gc()

########################################################################
# BAYES-UCB ALGORITHM ##################################################
# Kaufmann et al. (2012) ###############################################

# Run the Bayes-UCB algorithm on the data set with prior set to have
# a mean equal to the fixed threshold
loc2_BUCB_horizon_4000 <- para_bandit_sim_bucb(data = data_list2, rounds = 4000, 
                                          rate = "inverse_horizon",
                                          tau = tau_loc2, epsilon = epsilon_loc2, 
                                          alpha = tau_loc2, beta = 1-tau_loc2)
#save(loc2_BUCB_horizon_4000, 
#     file = paste0(current_path, "loc2_BUCB_horizon_4000.Rda"))

# Get the average error over all 5000 simulations at each round
loc2_comp_BUCB_horizon_4000 <- compare_to_ground_truth(mean_loc2, 
                                                       loc2_BUCB_horizon_4000,
                                                       tau_loc2,
                                                       epsilon_loc2)$mean
save(loc2_comp_BUCB_horizon_4000, file = paste0(current_path, 
                                                "loc2_comp_BUCB_horizon_4000.Rda"))
rm(loc2_BUCB_horizon_4000)
gc()

########################################################################
# SIMPLE LIKELIHOOD RATIO ALGORITHM ####################################

# Run the SLR algorithm on the data set with Bernoulli likelihoods
loc2_LR_2000 <- para_bandit_sim_LR(data = data_list2, 
                                   rounds = 2000, 
                                   tau = tau_loc2, 
                                   epsilon = epsilon_loc2)
#save(loc2_LR_2000, file = paste0(current_path, "loc2_LR_2000.Rda"))

# Get the average error over all 5000 simulations at each round
loc2_comp_LR_2000 <- compare_to_ground_truth(mean_loc2, loc2_LR_2000, 
                                             tau_loc2, epsilon_loc2)$mean
save(loc2_comp_LR_2000, file = paste0(current_path, "loc2_comp_LR_2000.Rda"))
rm(loc2_LR)
gc()

########################################################################
# AUGMENTED UCB ALGORITHM ##############################################
# Mukherjee et al. (2017) ##############################################

# Run the AugUCB algorithm with the standard parameter choice of rho = 1/3
loc2_AugUCB_2000 <- para_bandit_sim_AugUCB(data = data_list2, 
                                           rounds = 2000, 
                                           tau = tau_loc2)
# user   system  elapsed 
# 13.972   10.042 2284.875
save(loc2_AugUCB_2000, file = paste0(current_path, "loc2_AugUCB_2000.Rda"))
#load(file = paste0(current_path, "loc2_AugUCB.Rda"))
loc2_comp_AugUCB_2000 <- compare_to_ground_truth(mean_loc2, loc2_AugUCB_2000,
                                                 tau_loc2, 
                                                 epsilon_loc2)$mean
save(loc2_comp_AugUCB_2000, file = paste0(current_path,
                                          "loc2_comp_AugUCB_2000.Rda"))
rm(loc2_AugUCB_2000)
gc()

########################################################################
# Plot the results #####################################################

load(paste0(current_path, "/loc2_comp_BUCB_horizon_4000.Rda"))
load(paste0(current_path, "/loc2_comp_APT_2000.Rda"))
load(paste0(current_path, "/loc2_comp_LR_2000.Rda"))
load(paste0(current_path, "/loc2_comp_EVT.Rda"))
load(paste0(current_path, "/loc2_comp_AugUCB_2000.Rda"))
load(paste0(current_path, "/loc2_comp_UNIFORM_2000.Rda"))

plot(c(0,2000), c(0, -9), type = "n")
lines(log(loc2_comp_UNIFORM_2000), col = "black")
lines(log(loc2_comp_BUCB_horizon_4000), col = "blue")
lines(log(loc2_comp_APT_2000), col = "red")
lines(log(loc2_comp_LR_2000), col = "orange")
lines(log(loc2_comp_AugUCB_2000), col = "grey")
lines(log(loc2_comp_EVT), col = "green")