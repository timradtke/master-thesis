########################################################################

# this script runs the experiment used in the first experiment
# discussed in the master thesis.

####################################################################

# This script loads real data for 10 products out of 197 products.
# We look at 10080 rounds per iteration so as to cover a whole week.
# This is useful to account for seasonality effects in the data (if
# looking at uniformly sampled data).

####################################################################

sessions_threshold <- 0 # binarize data using this threshold
set.seed(1068)
selected_products <- sample(1:197, 10) # randomly select 10 products
niter <- 5000   # Create 5000 data sets
nrounds <- 10080 # These two values give the maximum size data 
nahead <- 10080  # possible given the data set size we have
# Import the data set (adjust path if necessary):
source("/code/experiments/data/get_dataset.R")

####################################################################
# load functions (adjust paths if necessary)

current_path <- "/code/functions/"
source(paste0(current_path, "functions.R"))
source(paste0(current_path, "functions_parallelized.R"))
current_path <- "/code/experiments/exp1/"

####################################################################
# Store the loaded data that is specific for this experiment
# and only contains information on 10 products

data_amo3 <- pv_list # main data set
data_amo3_wide <- pv_products_wide
# The means we want to use to evaluate our classification:
data_amo3_mean_secondhalf <- colMeans(data_amo3_wide[15081:25160,])
data_amo3_mean_firsthalf <- colMeans(data_amo3_wide[1:15080,])
#save(data_amo3, file = paste0(current_path, "data_amo3.Rda"))
#save(data_amo3_wide, file = paste0(current_path, "data_amo3_wide.Rda"))
#save(data_amo3_mean_firsthalf, file = paste0(current_path, "data_amo3_mean_firsthalf.Rda"))
#save(data_amo3_mean_secondhalf, file = paste0(current_path, "data_amo3_mean_secondhalf.Rda"))
rm(pv_list, pv_own_means, pv_next_means, pv_products_wide)
gc()

#load(paste0(current_path, "data_amo3_mean_firsthalf.Rda"))
#load(paste0(current_path, "data_amo3_mean_secondhalf.Rda"))
#load(paste0(current_path, "data_amo3.Rda"))

# Set the parameters of the problem

tau_amo3 <- 2/60
epsilon_amo3 <- 0

########################################################################
# ANYTIME PARAMETER FREE THRESHOLDING ALGORITHM ########################
# Locatelli et al. (2016) ##############################################

# run the APT algorithm on the data set
system.time(amo3_APT <- para_bandit_sim_APT(data = data_amo3, rounds = 10080, 
                                            tau = tau_amo3, epsilon = epsilon_amo3))
#save(amo3_APT, file = paste0(current_path, "amo3_APT.Rda"))
#load(paste0(current_path, "amo3_APT.Rda"))


amo3_comp_APT <- compare_to_ground_truth(data_amo3_mean_firsthalf, amo3_APT, 
                                         tau_amo3, epsilon_amo3)$mean
amo3_compholdout_APT <- compare_to_ground_truth(data_amo3_mean_secondhalf, amo3_APT, 
                                                tau_amo3, epsilon_amo3)$mean

save(amo3_comp_APT, file = paste0(current_path, "amo3_comp_APT.Rda"))
save(amo3_compholdout_APT, file = paste0(current_path, "amo3_compholdout_APT.Rda"))
rm(amo3_APT)
gc()

########################################################################
# UNIFORM SAMPLING #####################################################
# Run the uniform sampling strategy on the data set
system.time(amo3_UNIFORM <- para_bandit_sim_uniform(data = data_amo3, 
                                                    rounds = 10080))
save(amo3_UNIFORM, file = paste0(current_path, "amo3_UNIFORM.Rda"))
#load(file = paste0(current_path, "amo3_UNIFORM.Rda"))


amo3_comp_UNIFORM <- compare_to_ground_truth(data_amo3_mean_firsthalf, 
                                             amo3_UNIFORM, 
                                             tau_amo3, epsilon_amo3)$mean
amo3_compholdout_UNIFORM <- compare_to_ground_truth(data_amo3_mean_secondhalf, 
                                                    amo3_UNIFORM, 
                                                    tau_amo3, epsilon_amo3)$mean
save(amo3_comp_UNIFORM, file = paste0(current_path, "amo3_comp_UNIFORM.Rda"))
save(amo3_compholdout_UNIFORM, file = paste0(current_path, 
                                             "amo3_compholdout_UNIFORM.Rda"))
rm(amo3_UNIFORM)
gc()

########################################################################
# SIMPLE LIKELIHOOD RATIO ALGORITHM ####################################

# Run the SLR algorithm on the data set with Bernoulli likelihoods
system.time(amo3_LR <- para_bandit_sim_LR(data = data_amo3, 
                                          rounds = 10080, 
                                          tau = tau_amo3, 
                                          epsilon = epsilon_amo3))

save(amo3_LR, file = paste0(current_path, "amo3_LR.Rda"))
#load(file = paste0(current_path, "amo3_LR.Rda"))


amo3_comp_LR <- compare_to_ground_truth(data_amo3_mean_firsthalf, 
                                             amo3_LR, 
                                             tau_amo3, epsilon_amo3)$mean
amo3_compholdout_LR <- compare_to_ground_truth(data_amo3_mean_secondhalf, 
                                                    amo3_LR, 
                                                    tau_amo3, epsilon_amo3)$mean
save(amo3_comp_LR, file = paste0(current_path, "amo3_comp_LR.Rda"))
save(amo3_compholdout_LR, file = paste0(current_path, 
                                        "amo3_compholdout_LR.Rda"))
rm(amo3_LR, amo3_comp_LR)
gc()

########################################################################
# EMPIRICAL VARIANCE GUIDED THRESHOLDING ALGORITHM #####################
# Zhong et al. (2017) ##################################################

# run the EVT algorithm on the data set
system.time(amo3_EVT <- para_bandit_sim_EVT(data = data_amo3, 
                                            rounds = 10080, 
                                            tau = tau_amo3, 
                                            epsilon = epsilon_amo3))
save(amo3_EVT, file = paste0(current_path, "amo3_EVT.Rda"))


amo3_comp_EVT <- compare_to_ground_truth(data_amo3_mean_firsthalf, 
                                        amo3_EVT, 
                                        tau_amo3, epsilon_amo3)$mean
amo3_compholdout_EVT <- compare_to_ground_truth(data_amo3_mean_secondhalf, 
                                               amo3_EVT, 
                                               tau_amo3, epsilon_amo3)$mean
save(amo3_comp_EVT, file = paste0(current_path, "amo3_comp_EVT.Rda"))
save(amo3_compholdout_EVT, file = paste0(current_path, 
                                        "amo3_compholdout_EVT.Rda"))
rm(amo3_EVT, amo3_comp_EVT, amo3_compholdout_EVT)
gc()

########################################################################
# AUGMENTED UCB ALGORITHM ##############################################
# Mukherjee et al. (2017) ##############################################

# Run the AugUCB algorithm with the standard parameter choice of rho = 1/3
system.time(amo3_AugUCB <- para_bandit_sim_AugUCB(data = data_amo3, 
                                                  rounds = 10080, 
                                                  tau = tau_amo3))
save(amo3_AugUCB, file = paste0(current_path, "amo3_AugUCB.Rda"))
#load(file = paste0(current_path, "amo3_AugUCB.Rda"))


amo3_comp_AugUCB <- compare_to_ground_truth(data_amo3_mean_firsthalf, 
                                            amo3_AugUCB, 
                                            tau_amo3, epsilon_amo3)$mean
amo3_compholdout_AugUCB <- compare_to_ground_truth(data_amo3_mean_secondhalf, 
                                                   amo3_AugUCB, 
                                                   tau_amo3, epsilon_amo3)$mean
save(amo3_comp_AugUCB, file = paste0(current_path, "amo3_comp_AugUCB.Rda"))
save(amo3_compholdout_AugUCB, file = paste0(current_path, 
                                            "amo3_compholdout_AugUCB.Rda"))
rm(amo3_AugUCB, amo3_comp_AugUCB, amo3_compholdout_AugUCB)
gc()

########################################################################
# BAYES-UCB ALGORITHM ##################################################
# Kaufmann et al. (2012) ###############################################

# Run the Bayes-UCB algorithm on the data set with prior set to have
# a mean equal to the fixed threshold
amo3_BUCB <- para_bandit_sim_bucb(data = data_amo3, rounds = 10080, 
                                  rate = "inverse_horizon",
                                  tau = tau_amo3, epsilon = epsilon_amo3, 
                                  alpha = tau_amo3, beta = 1-tau_amo3)
save(amo3_BUCB, 
     file = paste0(current_path, "amo3_BUCB.Rda"))


amo3_comp_BUCB <- compare_to_ground_truth(data_amo3_mean_firsthalf, amo3_BUCB, 
                                          tau_amo3, epsilon_amo3)$mean
amo3_compholdout_BUCB <- compare_to_ground_truth(data_amo3_mean_secondhalf, 
                                                 amo3_BUCB, 
                                                 tau_amo3, epsilon_amo3)$mean
save(amo3_comp_BUCB, file = paste0(current_path, "amo3_comp_BUCB.Rda"))
save(amo3_compholdout_BUCB, file = paste0(current_path, "amo3_compholdout_BUCB.Rda"))
rm(amo3_BUCB)
gc()

########################################################################

# Plot the performance results quickly

load(paste0(current_path, "amo3_comp_UNIFORM.Rda"))
load(paste0(current_path, "amo3_comp_APT.Rda"))
load(paste0(current_path, "amo3_comp_LR.Rda"))
load(paste0(current_path, "amo3_comp_AugUCB.Rda"))
load(paste0(current_path, "amo3_comp_EVT.Rda"))
load(paste0(current_path, "amo3_comp_BUCB.Rda"))

plot(c(0,10080), c(0, -3), type = "n")
lines(log(amo3_comp_UNIFORM), col = "black")
lines(log(amo3_comp_APT), col = "blue")
lines(log(amo3_comp_AugUCB), col = "green")
lines(log(amo3_comp_EVT), col = "darkgreen")
lines(log(amo3_comp_LR), col = "red")
abline(h = log(0.1), lty = 2)
