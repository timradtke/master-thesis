########################################################################

# this script runs the simulations used in the second simulation
# discussed in the master thesis.

########################################################################

# Given the threshold of 0.05 for Bernoulli distributions, this is
# a problem on which the SLR should outperform the APT.

########################################################################
# load functions (adjust paths if necessary)

current_path <- "/code/functions/"
source(paste0(current_path, "functions.R"))
source(paste0(current_path, "functions_parallelized.R"))
current_path <- "/code/simulations/sim2/"

########################################################################
# Here we create the data set

# Fix the means of the arms in a vector
mean_loc6nt <- c(0.001, 0.005, 0.01, 0.015,
                 0.04, 0.06,
                 0.085, 0.09, 0.095, 0.099)
tau_loc6nt <- 0.05
epsilon_loc6nt <- 0

# Create a data set 5000 times with each 7000 rounds from the parameters
data_list6 <- list()
set.seed(8247502)
for(j in 1:5000) {
  curr_data <- data.frame(rep(NA, times = 7000))
  for(i in 1:length(mean_loc6nt)) {
    curr_data[[i]] <- as.numeric(purrr::rbernoulli(7000, p  = mean_loc6nt[i]))
  }
  names(curr_data) <- paste0("V", rep(1:length(mean_loc6nt)))
  data_list6[[j]] <- curr_data
}

########################################################################
# ANYTIME PARAMETER FREE THRESHOLDING ALGORITHM ########################
# Locatelli et al. (2016) ##############################################

# run the APT algorithm on the data set
system.time(loc6nt_APT_7000 <- para_bandit_sim_APT(data = data_list6, 
                                                   rounds = 7000, 
                                                   tau = tau_loc6nt, 
                                                   epsilon = epsilon_loc6nt))

#save(loc6nt_APT_7000, file = paste0(current_path, "loc6nt_APT_7000.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_APT_7000 <- compare_to_ground_truth(mean_loc6nt, loc6nt_APT_7000, 
                                                tau_loc6nt, epsilon_loc6nt)$mean
save(loc6nt_comp_APT_7000, file = paste0(current_path, 
                                         "loc6nt_comp_APT_7000.Rda"))
rm(loc6nt_APT_7000)
gc()

########################################################################
# SIMPLE LIKELIHOOD RATIO ALGORITHM ####################################

# Run the SLR algorithm on the data set with Bernoulli likelihoods
system.time(loc6nt_LR_7000 <- para_bandit_sim_LR(data = data_list6, 
                                                 rounds = 7000, 
                                                 tau = tau_loc6nt, 
                                                 epsilon = epsilon_loc6nt))

#save(loc6nt_LR_7000, file = paste0(current_path, "loc6nt_LR_7000.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_LR_7000 <- compare_to_ground_truth(mean_loc6nt, loc6nt_LR_7000, 
                                               tau_loc6nt, epsilon_loc6nt)$mean
save(loc6nt_comp_LR_7000, file = paste0(current_path, 
                                        "loc6nt_comp_LR_7000.Rda"))

########################################################################
# SIMPLE LIKELIHOOD RATIO ALGORITHM ####################################
# PLUS D-TRACKING RULE (from Garivier and Kaufmann, 2016) ##############

# Run the SLR algorithm on the data set with Bernoulli likelihoods
system.time(loc6nt_LRD <- para_bandit_sim_LRD(data = data_list6, 
                                              rounds = 7000, 
                                              tau = tau_loc6nt, 
                                              epsilon = epsilon_loc6nt,
                                              do_verbose = TRUE))

#save(loc6nt_LRD, file = paste0(current_path, "loc6nt_LRD.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_LRD <- compare_to_ground_truth(mean_loc6nt, loc6nt_LRD, 
                                           tau_loc6nt, epsilon_loc6nt)$mean
save(loc6nt_comp_LRD, file = paste0(current_path, 
                                    "loc6nt_comp_LRD.Rda"))

########################################################################
# UNIFORM SAMPLING #####################################################
# Run the uniform sampling strategy on the data set

system.time(loc6nt_UNIFORM_7000 <- para_bandit_sim_uniform(data = data_list6, 
                                                           rounds = 7000))
#save(loc6nt_UNIFORM_7000, file = paste0(current_path, "loc6nt_UNIFORM_7000.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_UNIFORM_7000 <- compare_to_ground_truth(mean_loc6nt, 
                                                    loc6nt_UNIFORM_7000, 
                                                    tau_loc6nt, 
                                                    epsilon_loc6nt)$mean
save(loc6nt_comp_UNIFORM_7000, file = paste0(current_path, 
                                             "loc6nt_comp_UNIFORM_7000.Rda"))

########################################################################
# EMPIRICAL VARIANCE GUIDED THRESHOLDING ALGORITHM #####################
# Zhong et al. (2017) ##################################################

# run the EVT algorithm on the data set
loc6nt_EVT <- para_bandit_sim_EVT(data = data_list6, 
                                  rounds = 7000, 
                                  tau = tau_loc6nt, 
                                  epsilon = epsilon_loc6nt)
#save(loc6nt_EVT, file = paste0(current_path, "loc6nt_EVT.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_EVT <- compare_to_ground_truth(mean_loc6nt, loc6nt_EVT, 
                                           tau_loc6nt, epsilon_loc6nt)$mean
save(loc6nt_comp_EVT, file = paste0(current_path, "loc6nt_comp_EVT.Rda"))
rm(loc6nt_EVT)
gc()

########################################################################
# AUGMENTED UCB ALGORITHM ##############################################
# Mukherjee et al. (2017) ##############################################

# Run the AugUCB algorithm with the standard parameter choice of rho = 1/3
loc6nt_AugUCB_7000 <- para_bandit_sim_AugUCB(data = data_list6, 
                                             rounds = 7000, 
                                             tau = tau_loc6nt)
#save(loc6nt_AugUCB_7000, file = paste0(current_path, "loc6nt_AugUCB_7000.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_AugUCB_7000 <- compare_to_ground_truth(mean_loc6nt, loc6nt_AugUCB_7000, 
                                                   tau_loc6nt, epsilon_loc6nt)$mean
save(loc6nt_comp_AugUCB_7000, file = paste0(current_path, 
                                            "loc6nt_comp_AugUCB_7000.Rda"))

########################################################################
# BAYES-UCB ALGORITHM ##################################################
# Kaufmann et al. (2012) ###############################################

# Here, we use the "rate" used in the other experiments as well

# Run the Bayes-UCB algorithm on the data set with prior set to have
# a mean equal to the fixed threshold
loc6nt_BUCB_horizon_7000 <- para_bandit_sim_bucb(data = data_list6, 
                                                 rounds = 7000, 
                                                 rate = "inverse_horizon",
                                                 tau = tau_loc6nt, 
                                                 epsilon = epsilon_loc6nt, 
                                                 alpha = tau_loc6nt, 
                                                 beta = 1-tau_loc6nt)
#save(loc6nt_BUCB_horizon_7000, 
#     file = paste0(current_path, "loc6nt_BUCB_horizon_7000.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_BUCB_horizon_7000 <- compare_to_ground_truth(mean_loc6nt, 
                                                         loc6nt_BUCB_horizon_7000,
                                                         tau_loc6nt,
                                                         epsilon_loc6nt)$mean
save(loc6nt_comp_BUCB_horizon_7000, file = paste0(current_path, 
                                                  "loc6nt_comp_BUCB_horizon_7000.Rda"))

########################################################################
# BAYES-UCB ALGORITHM ##################################################
# Kaufmann et al. (2012) ###############################################

# Here, we use a different rate to explore its influence on performance
# (reported as performing well in Kaufmann et al. 2012)

# Run the Bayes-UCB algorithm on the data set with prior set to have
# a mean equal to the fixed threshold
loc6nt_BUCB_7000 <- para_bandit_sim_bucb(data = data_list6, 
                                         rounds = 7000, 
                                         rate = "inverse",
                                         tau = tau_loc6nt, 
                                         epsilon = epsilon_loc6nt, 
                                         alpha = tau_loc6nt, 
                                         beta = 1-tau_loc6nt)
#save(loc6nt_BUCB_7000,
#     file = paste0(current_path, "loc6nt_BUCB_7000.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_BUCB_7000 <- compare_to_ground_truth(mean_loc6nt, 
                                                 loc6nt_BUCB_7000,
                                                 tau_loc6nt,
                                                 epsilon_loc6nt)$mean
save(loc6nt_comp_BUCB_7000, file = paste0(current_path, 
                                          "loc6nt_comp_BUCB_7000.Rda"))
rm(loc6nt_BUCB_7000)

########################################################################
# BAYES-UCB ALGORITHM ##################################################
# Kaufmann et al. (2012) ###############################################

# Here, we use a different rate to explore its influence on performance
# (optimal rate for bounds given in Kaufmann et al. 2012)

# Run the Bayes-UCB algorithm on the data set with prior set to have
# a mean equal to the fixed threshold
loc6nt_BUCB_horizon_c5_7000 <- para_bandit_sim_bucb(data = data_list6, 
                                                    rounds = 7000, 
                                                    rate = "inverse_horizon_c",
                                                    tau = tau_loc6nt, 
                                                    epsilon = epsilon_loc6nt, 
                                                    alpha = tau_loc6nt, 
                                                    beta = 1-tau_loc6nt,
                                                    const = 5)
#save(loc6nt_BUCB_horizon_c5_7000, 
#     file = paste0(current_path, "loc6nt_BUCB_horizon_c5_7000.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_BUCB_horizon_c5_7000 <- compare_to_ground_truth(mean_loc6nt, 
                                                            loc6nt_BUCB_horizon_c5_7000,
                                                            tau_loc6nt,
                                                            epsilon_loc6nt)$mean
save(loc6nt_comp_BUCB_horizon_c5_7000, file = paste0(current_path, 
                                                     "loc6nt_comp_BUCB_horizon_c5_7000.Rda"))
rm(loc6nt_BUCB_horizon_c5_7000)

########################################################################
# BAYES-UCB ALGORITHM ##################################################
# Kaufmann et al. (2012) ###############################################

# Here, we use a different rate to explore its influence on performance

# Run the Bayes-UCB algorithm on the data set with prior set to have
# a mean equal to the fixed threshold
loc6nt_BUCB_horizon_c15_7000 <- para_bandit_sim_bucb(data = data_list6, 
                                                     rounds = 7000, 
                                                     rate = "inverse_horizon_c",
                                                     tau = tau_loc6nt, 
                                                     epsilon = epsilon_loc6nt, 
                                                     alpha = tau_loc6nt, 
                                                     beta = 1-tau_loc6nt,
                                                     const = 1/5)
#save(loc6nt_BUCB_horizon_c15_7000, 
#     file = paste0(current_path, "loc6nt_BUCB_horizon_c15_7000.Rda"))

# Get the average error over all 5000 simulations at each round
loc6nt_comp_BUCB_horizon_c15_7000 <- compare_to_ground_truth(mean_loc6nt, 
                                                             loc6nt_BUCB_horizon_c15_7000,
                                                             tau_loc6nt,
                                                             epsilon_loc6nt)$mean
save(loc6nt_comp_BUCB_horizon_c15_7000, file = paste0(current_path, 
                                                      "loc6nt_comp_BUCB_horizon_c15_7000.Rda"))
rm(loc6nt_BUCB_horizon_c15_7000)

########################################################################

# Repeatedly extract specific data from the results to prepare for
# visualization used in the thesis.

# The graphs at the very end give a distribution of the pulls of a
# specific arm for a specific algorithm using the sample of 5000 observations
# from the 5000 simulations.

load(paste0(current_path, "/loc6nt_BUCB_horizon_7000.Rda"))
load(paste0(current_path, "/loc6nt_APT_7000.Rda"))
load(paste0(current_path, "/loc6nt_LR_7000.Rda"))
load(paste0(current_path, "/loc6nt_LRD.Rda"))

library(ggplot2)
library(ggjoy)
library(dplyr)
library(tidyr)

pulls_of_arm_6_LR <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_6_LR[i] <- table(loc6nt_LR_7000[[i]]$arm_sequence)[6]
}
summary(pulls_of_arm_6_LR)
quantile(pulls_of_arm_6_LR, 0.05)

pulls_of_arm_6_APT <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_6_APT[i] <- table(loc6nt_APT_7000[[i]]$arm_sequence)[6]
}
summary(pulls_of_arm_6_APT)
quantile(pulls_of_arm_6_APT, 0.05)

pulls_of_arm_6_BUCB <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_6_BUCB[i] <- table(loc6nt_BUCB_horizon_7000[[i]]$arm_sequence)[6]
}
summary(pulls_of_arm_6_BUCB)
quantile(pulls_of_arm_6_BUCB, 0.05)

pulls_of_arm_6_LRD <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_6_LRD[i] <- table(loc6nt_LRD[[i]]$arm_sequence)[6]
}

pulls_arm_6 <- data.frame(APT = pulls_of_arm_6_APT,
                          SLR = pulls_of_arm_6_LR,
                          SLRD = pulls_of_arm_6_LRD,
                          BUCB = pulls_of_arm_6_BUCB)

pulls_arm_6 %>% gather(key = "Algorithm", value = "Pulls") %>%
  ggplot(aes(x = Pulls, group = Algorithm)) +
  geom_histogram() +
  facet_grid(Algorithm~.)

pulls_of_arm_1_LR <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_1_LR[i] <- table(loc6nt_LR_7000[[i]]$arm_sequence)[1]
}

pulls_of_arm_1_LRD <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_1_LRD[i] <- table(loc6nt_LRD[[i]]$arm_sequence)[1]
}

pulls_of_arm_1_APT <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_1_APT[i] <- table(loc6nt_APT_7000[[i]]$arm_sequence)[1]
}

pulls_of_arm_1_BUCB <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_1_BUCB[i] <- table(loc6nt_BUCB_horizon_7000[[i]]$arm_sequence)[1]
}

pulls_arm_1 <- data.frame(APT = pulls_of_arm_1_APT,
                          SLR = pulls_of_arm_1_LR,
                          SLRD = pulls_of_arm_1_LRD,
                          BUCB = pulls_of_arm_1_BUCB)

pulls_arm_1 %>% gather(key = "Algorithm", value = "Pulls") %>%
  ggplot(aes(x = Pulls, group = Algorithm)) +
  geom_histogram() +
  facet_grid(Algorithm~.)

pulls_of_arm_10_LR <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_10_LR[i] <- table(loc6nt_LR_7000[[i]]$arm_sequence)[10]
}

pulls_of_arm_10_LRD <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_10_LRD[i] <- table(loc6nt_LRD[[i]]$arm_sequence)[10]
}

pulls_of_arm_10_APT <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_10_APT[i] <- table(loc6nt_APT_7000[[i]]$arm_sequence)[10]
}

pulls_of_arm_10_BUCB <- vector(length = 5000)
for(i in 1:5000){
  pulls_of_arm_10_BUCB[i] <- table(loc6nt_BUCB_horizon_7000[[i]]$arm_sequence)[10]
}

pulls_arm_10 <- data.frame(APT = pulls_of_arm_10_APT,
                           SLR = pulls_of_arm_10_LR,
                           SLRD = pulls_of_arm_10_LRD,
                           BUCB = pulls_of_arm_10_BUCB)

pulls_arm_10 %>% gather(key = "Algorithm", value = "Pulls") %>%
  ggplot(aes(x = Pulls, group = Algorithm)) +
  geom_histogram(binwidth = 50) +
  facet_grid(Algorithm~.)

pulls_arm_1_long <- pulls_arm_1 %>% gather(key = "Algorithm", value = "Pulls")
pulls_arm_6_long <- pulls_arm_6 %>% gather(key = "Algorithm", value = "Pulls")
pulls_arm_10_long <- pulls_arm_10 %>% gather(key = "Algorithm", value = "Pulls")

save(pulls_arm_1_long, pulls_arm_6_long, pulls_arm_10_long,
     file = paste0(current_path, "loc6nt_pulls_of_arms_vis.Rda"))
load(file = paste0(current_path, "loc6nt_pulls_of_arms_vis.Rda"))

pulls_arm_6_long %>%
  ggplot(aes(x = Pulls)) +
  geom_histogram(binwidth = 100) +
  facet_grid(.~Algorithm, scales = "free_y") +
  geom_vline(aes(xintercept = 7000/10), linetype = 2) +
  labs(x = "Pulls", y = "Count", 
       title = "Distribution of Pulls of Arm 6 Across Simulations",
       subtitle = "Binwidth = 100. Budget = 7000. 5000 Simulations.
       Vertical line indicates pulls by uniform sampling given 10 arms in experiment.") +
  theme_bw()

cbind.data.frame(
  Arm = rep(c("01", "10"), each = 20000),
  rbind.data.frame(pulls_arm_1_long, pulls_arm_10_long),
  stringsAsFactors = FALSE
) %>% 
  ggplot(aes(x = Pulls, group = Arm)) +
  geom_histogram(binwidth = 50) +
  facet_grid(Arm~Algorithm, scales = "free_y")+
  geom_vline(aes(xintercept = 7000/10), linetype = 2) +
  labs(x = "Pulls", y = "Count", 
       title = "Distribution of Pulls of Arms 1 and 10 Across Simulations",
       subtitle = "Binwidth = 50. Budget = 7000. 5000 Simulations.
       Vertical line indicates pulls by uniform sampling given 10 arms in experiment.") +
  theme_bw()