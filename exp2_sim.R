####################################################################

# this script runs the simulations for the synthetic data used in
# comparison to the real data in the second experiment reported
# in the master thesis.

# The synthetic data is created by computing the sample mean in the
# real data, and using those as the means of the distributions that
# we sample the random values from.

# We do this to observe how strong the time dependency is, and what
# effect it has on the results.

####################################################################
# load functions (adjust paths if necessary)

current_path <- "/code/functions/"
source(paste0(current_path, "functions.R"))
source(paste0(current_path, "functions_parallelized.R"))
current_path <- "/code/experiments/exp2/"

# load the sample means from the real data.
load(paste0(current_path, "data_amo4_mean_firsthalf.Rda"))

# Set the same threshold as in the experiment
tau_amo4sim <- 5/60
epsilon_amo4sim <- 0

# Create the synthetic data set with the same size of data
data_amo4sim <- list()
set.seed(458693)
for(j in 1:5000) {
  curr_data <- data.frame(rep(NA, times = 10080))
  for(i in 1:length(data_amo4_mean_firsthalf)) {
    curr_data[[i]] <- as.numeric(purrr::rbernoulli(10080, p  = data_amo4_mean_firsthalf[i]))
  }
  names(curr_data) <- paste0("V", rep(1:length(data_amo4_mean_firsthalf)))
  data_amo4sim[[j]] <- curr_data
}
gc()

########################################################################
# ANYTIME PARAMETER FREE THRESHOLDING ALGORITHM ########################
# Locatelli et al. (2016) ##############################################

# run the APT algorithm on the data set
system.time(amo4sim_APT <- para_bandit_sim_APT(data = data_amo4sim, rounds = 10080, 
                                               tau = tau_amo4sim, epsilon = epsilon_amo4sim))
#save(amo4sim_APT, file = paste0(current_path, "amo4sim_APT.Rda"))
#load(paste0(current_path, "amo4sim_APT.Rda"))


amo4sim_comp_APT <- compare_to_ground_truth(data_amo4_mean_firsthalf, amo4sim_APT, 
                                            tau_amo4sim, epsilon_amo4sim)$mean

save(amo4sim_comp_APT, file = paste0(current_path, "amo4sim_comp_APT.Rda"))
rm(amo4sim_APT)
gc()

########################################################################
# UNIFORM SAMPLING #####################################################
# Run the uniform sampling strategy on the data set
system.time(amo4sim_UNIFORM <- para_bandit_sim_uniform(data = data_amo4sim, 
                                                       rounds = 10080))
#save(amo4sim_UNIFORM, file = paste0(current_path, "amo4sim_UNIFORM.Rda"))
#load(file = paste0(current_path, "amo4sim_UNIFORM.Rda"))

amo4sim_comp_UNIFORM <- compare_to_ground_truth(data_amo4_mean_firsthalf, 
                                                amo4sim_UNIFORM, 
                                                tau_amo4sim, epsilon_amo4sim)$mean
save(amo4sim_comp_UNIFORM, file = paste0(current_path, "amo4sim_comp_UNIFORM.Rda"))
rm(amo4sim_UNIFORM)
gc()

########################################################################
# SIMPLE LIKELIHOOD RATIO ALGORITHM ####################################

# Run the SLR algorithm on the data set with Bernoulli likelihoods
system.time(amo4sim_LR <- para_bandit_sim_LR(data = data_amo4sim, 
                                          rounds = 10080, 
                                          tau = tau_amo4sim, 
                                          epsilon = epsilon_amo4sim))
#save(amo4sim_LR, file = paste0(current_path, "amo4sim_LR.Rda"))
#load(file = paste0(current_path, "amo4sim_LR.Rda"))

amo4sim_comp_LR <- compare_to_ground_truth(data_amo4_mean_firsthalf, 
                                           amo4sim_LR, 
                                           tau_amo4sim, epsilon_amo4sim)$mean
save(amo4sim_comp_LR, file = paste0(current_path, "amo4sim_comp_LR.Rda"))
rm(amo4sim_LR, amo4sim_comp_LR)
gc()

########################################################################
# EMPIRICAL VARIANCE GUIDED THRESHOLDING ALGORITHM #####################
# Zhong et al. (2017) ##################################################

# run the EVT algorithm on the data set
amo4sim_EVT <- para_bandit_sim_EVT(data = data_amo4sim, 
                                    rounds = 10080, 
                                    tau = tau_amo4sim, 
                                    epsilon = epsilon_amo4sim)
#save(amo4sim_EVT, file = paste0(current_path, "amo4sim_EVT.Rda"))

amo4sim_comp_EVT <- compare_to_ground_truth(data_amo4_mean_firsthalf, 
                                            amo4sim_EVT, 
                                            tau_amo4sim, epsilon_amo4sim)$mean
save(amo4sim_comp_EVT, file = paste0(current_path, "amo4sim_comp_EVT.Rda"))
rm(amo4sim_EVT, amo4sim_comp_EVT)
gc()

########################################################################
# BAYES-UCB ALGORITHM ##################################################
# Kaufmann et al. (2012) ###############################################

# Run the Bayes-UCB algorithm on the data set with prior set to have
# a mean equal to the fixed threshold
amo4sim_BUCB <- para_bandit_sim_bucb(data = data_amo4sim, rounds = 10080, 
                                     rate = "inverse_horizon",
                                     tau = tau_amo4sim, epsilon = epsilon_amo4sim, 
                                     alpha = tau_amo4sim, beta = 1-tau_amo4sim)
#save(amo4sim_BUCB, 
#     file = paste0(current_path, "amo4sim_BUCB.Rda"))

amo4sim_comp_BUCB <- compare_to_ground_truth(data_amo4_mean_firsthalf, amo4sim_BUCB, 
                                             tau_amo4sim, epsilon_amo4sim)$mean
save(amo4sim_comp_BUCB, file = paste0(current_path, "amo4sim_comp_BUCB.Rda"))
rm(amo4sim_BUCB)
gc()

########################################################################
# AUGMENTED UCB ALGORITHM ##############################################
# Mukherjee et al. (2017) ##############################################

# Run the AugUCB algorithm with the standard parameter choice of rho = 1/3
system.time(amo4sim_AugUCB <- para_bandit_sim_AugUCB(data = data_amo4sim, 
                                                  rounds = 10080, 
                                                  tau = tau_amo4sim))
#save(amo4sim_AugUCB, file = paste0(current_path, "amo4sim_AugUCB.Rda"))
#load(file = paste0(current_path, "amo4sim_AugUCB.Rda"))

amo4sim_comp_AugUCB <- compare_to_ground_truth(data_amo4_mean_firsthalf, 
                                            amo4sim_AugUCB, 
                                            tau_amo4sim, epsilon_amo4sim)$mean
save(amo4sim_comp_AugUCB, file = paste0(current_path, "amo4sim_comp_AugUCB.Rda"))
rm(amo4sim_AugUCB, amo4sim_comp_AugUCB)
gc()

########################################################################

# Plot the results quickly

load(paste0(current_path, "amo4sim_comp_UNIFORM.Rda"))
load(paste0(current_path, "amo4sim_comp_APT.Rda"))
load(paste0(current_path, "amo4sim_comp_LR.Rda"))
load(paste0(current_path, "amo4sim_comp_AugUCB.Rda"))
load(paste0(current_path, "amo4sim_comp_EVT.Rda"))
load(paste0(current_path, "amo4sim_comp_BUCB.Rda"))

plot(c(0,10080), c(0, -5), type = "n")
lines(log(amo4sim_comp_UNIFORM), col = "black")
lines(log(amo4sim_comp_APT), col = "blue")
lines(log(amo4sim_comp_AugUCB), col = "green")
lines(log(amo4sim_comp_EVT), col = "darkgreen")
lines(log(amo4sim_comp_LR), col = "red")
lines(log(amo4sim_comp_BUCB), col = "violet")
abline(h = log(0.1), lty = 2)
