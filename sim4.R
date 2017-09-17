########################################################################

# this script runs the simulations used in the fourth simulation
# discussed in the master thesis, using Exponential distributions.

########################################################################

# Here we test how the algorithms can adapt to distributions other than
# the Bernoulli distributions in simulations 1-3. We don't consider
# the Bayes-UCB algorithm in this experiment.

########################################################################
# load functions (adjust paths if necessary)

current_path <- "/code/functions/"
source(paste0(current_path, "functions.R"))
source(paste0(current_path, "functions_parallelized.R"))
current_path <- "/code/simulations/sim4/"

########################################################################
# Here we create the data set. 

# Fix the means of the arms in a vector
set.seed(93468734)
tau_exp2 <- 1
epsilon_exp2 <- 0
mu_exp2 <- rexp(20, 1) # Pull means randomly from an exponential distribution

# display the exponential distributions
plot(c(0,6), c(0,3), type = "n")
for(i in order(mu_exp2)) {
  lines(seq(0,6,0.001), dexp(seq(0,6,0.001), 1/mu_exp2[i]), col = rainbow(length(mu_exp2))[i])
}

# Create a data set 5000 times with 4000 rounds each from the parameters
data_list_exp2 <- list()
for(j in 1:5000) {
  curr_data <- data.frame(rep(NA, times = 4000))
  for(i in 1:length(mu_exp2)) {
    curr_data[[i]] <- as.numeric(rexp(4000, 1/mu_exp2[i]))
  }
  names(curr_data) <- paste0("V", rep(1:length(mu_exp2)))
  data_list_exp2[[j]] <- curr_data
}
gc()

########################################################################
# UNIFORM SAMPLING #####################################################
# Run the uniform sampling strategy on the data set

system.time(exp2_UNIFORM <- para_bandit_sim_uniform(data = data_list_exp2, 
                                                    rounds = 4000))
#save(exp2_UNIFORM, file = paste0(current_path, "exp2_UNIFORM.Rda"))

# Get the average error over all 5000 simulations at each round
exp2_comp_UNIFORM <- compare_to_ground_truth(mu_exp2, 
                                             exp2_UNIFORM, 
                                             tau_exp2, 
                                             epsilon_exp2)$mean
save(exp2_comp_UNIFORM, file = paste0(current_path, "exp2_comp_UNIFORM.Rda"))
rm(exp2_UNIFORM)
gc()

########################################################################
# SIMPLE LIKELIHOOD RATIO ALGORITHM ####################################

# Run the SLR algorithm on the data set with Exponential likelihoods
system.time(exp2_LR <- para_bandit_sim_LR_exponential(data = data_list_exp2, 
                                                      rounds = 4000, 
                                                      tau = tau_exp2, 
                                                      epsilon = epsilon_exp2))

#save(exp2_LR, file = paste0(current_path, "exp2_LR.Rda"))

# Get the average error over all 5000 simulations at each round
exp2_comp_LR <- compare_to_ground_truth(mu_exp2, exp2_LR, 
                                           tau_exp2, epsilon_exp2)$mean
save(exp2_comp_LR, file = paste0(current_path, "exp2_comp_LR.Rda"))
gc()

########################################################################
# ANYTIME PARAMETER FREE THRESHOLDING ALGORITHM ########################
# Locatelli et al. (2016) ##############################################

# run the APT algorithm on the data set
system.time(exp2_APT <- para_bandit_sim_APT(data = data_list_exp2, 
                                               rounds = 4000, 
                                               tau = tau_exp2, 
                                               epsilon = epsilon_exp2))
#save(exp2_APT, file = paste0(current_path, "exp2_APT.Rda"))

# Get the average error over all 5000 simulations at each round
exp2_comp_APT <- compare_to_ground_truth(mu_exp2, exp2_APT, 
                                            tau_exp2, epsilon_exp2)$mean
save(exp2_comp_APT, file = paste0(current_path, "exp2_comp_APT.Rda"))
rm(exp2_APT)
gc()

########################################################################
# EMPIRICAL VARIANCE GUIDED THRESHOLDING ALGORITHM #####################
# Zhong et al. (2017) ##################################################

# run the EVT algorithm on the data set
system.time(exp2_EVT <- para_bandit_sim_EVT(data = data_list_exp2, 
                                             rounds = 4000, 
                                             tau = tau_exp2, 
                                             epsilon = epsilon_exp2))
#save(exp2_EVT, file = paste0(current_path, "exp2_EVT.Rda"))

# Get the average error over all 5000 simulations at each round
exp2_comp_EVT <- compare_to_ground_truth(mu_exp2, exp2_EVT, 
                                            tau_exp2, epsilon_exp2)$mean
save(exp2_comp_EVT, file = paste0(current_path, "exp2_comp_EVT.Rda"))
rm(exp2_EVT)
gc()

########################################################################
# AUGMENTED UCB ALGORITHM ##############################################
# Mukherjee et al. (2017) ##############################################

# Run the AugUCB algorithm with the standard parameter choice of rho = 1/3
exp2_AugUCB <- para_bandit_sim_AugUCB(data = data_list_exp2, 
                                      rounds = 4000, 
                                      tau = tau_exp2)
#save(exp2_AugUCB, file = paste0(current_path, "exp2_AugUCB.Rda"))

# Get the average error over all 5000 simulations at each round
exp2_comp_AugUCB <- compare_to_ground_truth(mu_exp2, exp2_AugUCB, 
                                            tau_exp2, epsilon_exp2)$mean
save(exp2_comp_AugUCB, file = paste0(current_path, "exp2_comp_AugUCB.Rda"))

# Try to find out whether the AugUCB algorithm really drops all arms very
# early into the sampling procedure
load(paste0(current_path, "exp2_AugUCB.Rda"))
length(exp2_AugUCB[[1]]$arm_sequence)
length(exp2_AugUCB[[2]]$arm_sequence)
length(exp2_AugUCB[[3]]$arm_sequence)
length(exp2_AugUCB[[4]]$arm_sequence)
length(exp2_AugUCB[[5]]$arm_sequence)
length(exp2_AugUCB[[6]]$arm_sequence)
augucb_lengths <- vector(length = 5000)
for(i in 1:5000) {
  augucb_lengths[i] <- length(exp2_AugUCB[[i]]$arm_sequence)
}
summary(augucb_lengths)

rm(exp2_AugUCB)
gc()

########################################################################
load(paste0(current_path, "exp2_comp_LR.Rda"))
load(paste0(current_path, "exp2_comp_APT.Rda"))
load(paste0(current_path, "exp2_comp_EVT.Rda"))
load(paste0(current_path, "exp2_comp_UNIFORM.Rda"))
load(paste0(current_path, "exp2_comp_AugUCB.Rda"))

plot(c(0,4000), c(0, -8), type = "n")
abline(h=log(0.01), lty = 2)
lines(log(exp2_comp_UNIFORM), col = "black")
lines(log(exp2_comp_LR), col = rainbow(4)[1])
lines(log(exp2_comp_APT), col = rainbow(4)[2])
lines(log(exp2_comp_EVT), col = rainbow(4)[3])
lines(log(exp2_comp_AugUCB), col = rainbow(4)[4])
