########################################################################

# this script runs the simulations used in the third simulation
# discussed in the master thesis.

########################################################################

# Given the threshold of 0.01 for Bernoulli distributions, this is
# a problem on which the SLR should outperform the APT.

########################################################################
# load functions (adjust paths if necessary)

current_path <- "/code/functions/"
source(paste0(current_path, "functions.R"))
source(paste0(current_path, "functions_parallelized.R"))
current_path <- "/code/simulations/sim3/"

########################################################################
# Here we create the data set

# Fix the means of the arms in a vector
mean_loc7nt <- c(10^-4, 10^-3.5, 10^-3.25, 10^-3, 
                 10^-2.75, 10^-2.5, 10^-2.25, 10^-1.85, 10^-1.75,
                 10^-1.5, 10^-1)
tau_loc7nt <- 10^-2
epsilon_loc7nt <- 0

# Create a data set 5000 times with 10000 rounds each from the parameters
data_list7_10000 <- list()
set.seed(386303469)
for(j in 1:5000) {
  curr_data <- data.frame(rep(NA, times = 10000))
  for(i in 1:length(mean_loc7nt)) {
    curr_data[[i]] <- as.numeric(purrr::rbernoulli(10000, p  = mean_loc7nt[i]))
  }
  names(curr_data) <- paste0("V", rep(1:length(mean_loc7nt)))
  data_list7_10000[[j]] <- curr_data
}
gc()

########################################################################
# UNIFORM SAMPLING #####################################################
# Run the uniform sampling strategy on the data set

system.time(loc7nt_UNIFORM_10000 <- para_bandit_sim_uniform(data = data_list7_10000, 
                                                            rounds = 10000))
#save(loc7nt_UNIFORM_10000, file = paste0(current_path, "loc7nt_UNIFORM_10000.Rda"))

# Get the average error over all 5000 simulations at each round
loc7nt_comp_UNIFORM_10000 <- compare_to_ground_truth(mean_loc7nt, loc7nt_UNIFORM_10000, 
                                                     tau_loc7nt, 
                                                     epsilon_loc7nt)$mean
save(loc7nt_comp_UNIFORM_10000, file = paste0(current_path,
                                              "loc7nt_comp_UNIFORM_10000.Rda"))
rm(loc7nt_UNIFORM_10000, loc7nt_comp_UNIFORM_10000)
gc()

########################################################################
# ANYTIME PARAMETER FREE THRESHOLDING ALGORITHM ########################
# Locatelli et al. (2016) ##############################################

# run the APT algorithm on the data set
system.time(loc7nt_APT_10000 <- para_bandit_sim_APT(data = data_list7_10000, 
                                                    rounds = 10000, 
                                                    tau = tau_loc7nt, 
                                                    epsilon = epsilon_loc7nt))
#save(loc7nt_APT_10000, file = paste0(current_path, "loc7nt_APT_10000.Rda"))

# Get the average error over all 5000 simulations at each round
loc7nt_comp_APT_10000 <- compare_to_ground_truth(mean_loc7nt, loc7nt_APT_10000, 
                                                 tau_loc7nt, 
                                                 epsilon_loc7nt)$mean
save(loc7nt_comp_APT_10000, file = paste0(current_path, 
                                          "loc7nt_comp_APT_10000.Rda"))

########################################################################
# SIMPLE LIKELIHOOD RATIO ALGORITHM ####################################

# Run the SLR algorithm on the data set with Bernoulli likelihoods
system.time(loc7nt_LR_10000 <- para_bandit_sim_LR(data = data_list7_10000, 
                                                  rounds = 10000, 
                                                  tau = tau_loc7nt, 
                                                  epsilon = epsilon_loc7nt))

#save(loc7nt_LR_10000, file = paste0(current_path, "loc7nt_LR_10000.Rda"))

# Get the average error over all 5000 simulations at each round
loc7nt_comp_LR_10000 <- compare_to_ground_truth(mean_loc7nt, loc7nt_LR_10000, 
                                                tau_loc7nt, epsilon_loc7nt)$mean
save(loc7nt_comp_LR_10000, file = paste0(current_path, 
                                         "loc7nt_comp_LR_10000.Rda"))
rm(loc7nt_LR_10000, loc7nt_comp_LR_10000)
gc()

########################################################################
# SIMPLE LIKELIHOOD RATIO ALGORITHM ####################################
# PLUS D-TRACKING RULE (from Garivier and Kaufmann, 2016) ##############

# Run the SLR algorithm on the data set with Bernoulli likelihoods
system.time(loc7nt_LRD_10000 <- para_bandit_sim_LRD(data = data_list7_10000, 
                                                    rounds = 10000, 
                                                    tau = tau_loc7nt, 
                                                    epsilon = epsilon_loc7nt))

#save(loc7nt_LRD_10000, file = paste0(current_path, "loc7nt_LRD_10000.Rda"))

# Get the average error over all 5000 simulations at each round
loc7nt_comp_LRD_10000 <- compare_to_ground_truth(mean_loc7nt, loc7nt_LRD_10000, 
                                                 tau_loc7nt, epsilon_loc7nt)$mean
save(loc7nt_comp_LRD_10000, file = paste0(current_path, 
                                          "loc7nt_comp_LRD_10000.Rda"))
rm(loc7nt_LRD_10000)
gc()

########################################################################
# EMPIRICAL VARIANCE GUIDED THRESHOLDING ALGORITHM #####################
# Zhong et al. (2017) ##################################################

# run the EVT algorithm on the data set
loc7nt_EVT <- para_bandit_sim_EVT(data = data_list7_10000, 
                                  rounds = 10000, 
                                  tau = tau_loc7nt, 
                                  epsilon = epsilon_loc7nt)
#save(loc7nt_EVT, file = paste0(current_path, "loc7nt_EVT.Rda"))

# Get the average error over all 5000 simulations at each round
loc7nt_comp_EVT <- compare_to_ground_truth(mean_loc7nt, loc7nt_EVT, 
                                           tau_loc7nt, epsilon_loc7nt)$mean
save(loc7nt_comp_EVT, file = paste0(current_path, "loc7nt_comp_EVT.Rda"))
rm(loc7nt_EVT)
gc()

########################################################################
# AUGMENTED UCB ALGORITHM ##############################################
# Mukherjee et al. (2017) ##############################################

# Run the AugUCB algorithm with the standard parameter choice of rho = 1/3
loc7nt_AugUCB_10000 <- para_bandit_sim_AugUCB(data = data_list7_10000, 
                                              rounds = 10000, 
                                              tau = tau_loc7nt)
#save(loc7nt_AugUCB_10000, file = paste0(current_path, "loc7nt_AugUCB_10000.Rda"))
#load(file = paste0(current_path, "loc7nt_AugUCB_10000.Rda"))

# Get the average error over all 5000 simulations at each round
loc7nt_comp_AugUCB_10000 <- compare_to_ground_truth(mean_loc7nt, loc7nt_AugUCB_10000, 
                                                    tau_loc7nt, epsilon_loc7nt)$mean
save(loc7nt_comp_AugUCB_10000, file = paste0(current_path, 
                                             "loc7nt_comp_AugUCB_10000.Rda"))

########################################################################
# BAYES-UCB ALGORITHM ##################################################
# Kaufmann et al. (2012) ###############################################

# Run the Bayes-UCB algorithm on the data set with prior set to have
# a mean equal to the fixed threshold
loc7nt_BUCB_horizon_10000 <- para_bandit_sim_bucb(data = data_list7_10000, 
                                                  rounds = 10000, 
                                                  rate = "inverse_horizon",
                                                  tau = tau_loc7nt, 
                                                  epsilon = epsilon_loc7nt, 
                                                  alpha = tau_loc7nt, 
                                                  beta = 1-tau_loc7nt)
#save(loc7nt_BUCB_horizon_10000, 
#     file = paste0(current_path, "loc7nt_BUCB_horizon_10000.Rda"))

# Get the average error over all 5000 simulations at each round
loc7nt_comp_BUCB_horizon_10000 <- compare_to_ground_truth(mean_loc7nt, 
                                                       loc7nt_BUCB_horizon_10000,
                                                       tau_loc7nt,
                                                       epsilon_loc7nt)$mean
save(loc7nt_comp_BUCB_horizon_10000,
     file = paste0(current_path, "loc7nt_comp_BUCB_horizon_10000.Rda"))
rm(loc7nt_BUCB_horizon_10000)
gc()

########################################################################

load(file = paste0(current_path, "loc7nt_comp_UNIFORM_10000.Rda"))
load(file = paste0(current_path, "loc7nt_comp_APT_10000.Rda"))
load(file = paste0(current_path, "loc7nt_comp_LR_10000.Rda"))
load(file = paste0(current_path, "loc7nt_comp_LRD_10000.Rda"))
load(file = paste0(current_path, "loc7nt_comp_BUCB_horizon_10000.Rda"))
load(file = paste0(current_path, "loc7nt_comp_AugUCB_10000.Rda"))
load(file = paste0(current_path, "loc7nt_comp_EVT.Rda"))

plot(c(0,10000), c(0, -5), type = "n")
lines(log(loc7nt_comp_BUCB_horizon_10000), col = "green")
lines(log(loc7nt_comp_UNIFORM_10000), col = "darkblue")
lines(log(loc7nt_comp_APT_10000), col = "red")
lines(log(loc7nt_comp_LR_10000), col = "darkred")
lines(log(loc7nt_comp_LRD_10000), col = "darkgreen")
lines(log(loc7nt_comp_AugUCB_10000), col = "darkgreen")
lines(log(loc7nt_comp_EVT), col = "blue")
