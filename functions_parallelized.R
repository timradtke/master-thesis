#################################################################################

# this script takes functions from the functions.R script and
# uses them in new functions that can be used for repeated
# simulations in a parallelized fashion.

#################################################################################
# this function parallelizes the APT_from_tsdata() function for repeated
# simulations.

# input has to be a list of different data frames
# each data frame is a run of the algorithm
# in the end, return a list of lists
# each object of the first list should be a list that contains 
# the mean_storage data frame and the arm_sequence vector

para_bandit_sim_APT <- function(data, seed = NA, do_verbose = FALSE, ...) {
  require(foreach)
  require(doParallel)
  
  # assume that data is a list of data frames
  reps <- length(data)
  
  gc()
  cl <- makeCluster(max(1,detectCores()-1))
  registerDoParallel(cl)
  res <- foreach(j = 1:reps, #.errorhandling = 'remove',
                 .export = c("APT_from_tsdata", "get_next_arm_apt",
                             "get_min"), 
                 .verbose = do_verbose, .inorder = TRUE) %dopar% {
                   alg_res <- APT_from_tsdata(data = data[[j]], 
                                              seed = 512+j, ...)
                   list(mean_storage = alg_res$mean_storage,
                        arm_sequence = alg_res$arm_sequence,
                        input_data = data[[j]])
                 }
  stopCluster(cl)
  return(res)
}

#################################################################################
# this function parallelizes the EVT_from_tsdata() function for repeated
# simulations.

# input has to be a list of different data frames
# each data frame is a run of the algorithm
# in the end, return a list of lists
# each object of the first list should be a list that contains 
# the mean_storage data frame and the arm_sequence vector

para_bandit_sim_EVT <- function(data, seed = NA, do_verbose = FALSE, ...) {
  require(foreach)
  require(doParallel)
  
  # assume that data is a list of data frames
  reps <- length(data)
  
  gc()
  cl <- makeCluster(max(1,detectCores()-1))
  registerDoParallel(cl)
  res <- foreach(j = 1:reps, #.errorhandling = 'remove',
                 .export = c("EVT_from_tsdata", "get_next_arm_evt",
                             "get_min"), 
                 .verbose = do_verbose, .inorder = TRUE) %dopar% {
                   alg_res <- EVT_from_tsdata(data = data[[j]], 
                                              seed = 512+j, ...)
                   list(mean_storage = alg_res$mean_storage,
                        arm_sequence = alg_res$arm_sequence,
                        input_data = data[[j]])
                 }
  stopCluster(cl)
  return(res)
}

#################################################################################
# this function parallelizes the LR_bandit_from_tsdata() function for repeated
# simulations.

# input has to be a list of different data frames
# each data frame is a run of the algorithm
# in the end, return a list of lists
# each object of the first list should be a list that contains 
# the mean_storage data frame and the arm_sequence vector

para_bandit_sim_LR <- function(data, seed = NA, do_verbose = FALSE, ...) {
  require(foreach)
  require(doParallel)
  
  # assume that data is a list of data frames
  reps <- length(data)
  
  gc()
  cl <- makeCluster(max(1,detectCores()-1))
  registerDoParallel(cl)
  res <- foreach(j = 1:reps, #.errorhandling = 'remove',
                 .export = c("LR_bandit_from_tsdata", "get_next_arm_lr",
                             "get_min", "lr_ber"), 
                 .verbose = do_verbose, .inorder = TRUE) %dopar% {
                   alg_res <- LR_bandit_from_tsdata(data = data[[j]], 
                                                    seed = 512+j, ...)
                   list(mean_storage = alg_res$mean_storage,
                        arm_sequence = alg_res$arm_sequence,
                        input_data = data[[j]])
                 }
  stopCluster(cl)
  return(res)
}

#################################################################################
# this function parallelizes the LRD_bandit_from_tsdata() function for repeated
# simulations.

# input has to be a list of different data frames
# each data frame is a run of the algorithm
# in the end, return a list of lists
# each object of the first list should be a list that contains 
# the mean_storage data frame and the arm_sequence vector

para_bandit_sim_LRD <- function(data, seed = NA, do_verbose = FALSE, ...) {
  require(foreach)
  require(doParallel)
  
  # assume that data is a list of data frames
  reps <- length(data)
  
  gc()
  cl <- makeCluster(max(1,detectCores()-1))
  registerDoParallel(cl)
  res <- foreach(j = 1:reps, #.errorhandling = 'remove',
                 .export = c("LRD_bandit_from_tsdata", "get_next_arm_lr",
                             "get_min", "lr_ber"), 
                 .verbose = do_verbose, .inorder = TRUE) %dopar% {
                   alg_res <- LRD_bandit_from_tsdata(data = data[[j]], 
                                                     seed = 512+j, ...)
                   list(mean_storage = alg_res$mean_storage,
                        arm_sequence = alg_res$arm_sequence,
                        input_data = data[[j]])
                 }
  stopCluster(cl)
  return(res)
}

#################################################################################
# this function parallelizes the LR_bandit_from_tsdata_exponential() function 
# for repeated simulations.

# input has to be a list of different data frames
# each data frame is a run of the algorithm
# in the end, return a list of lists
# each object of the first list should be a list that contains 
# the mean_storage data frame and the arm_sequence vector

para_bandit_sim_LR_exponential <- function(data, seed = NA, do_verbose = FALSE, ...) {
  require(foreach)
  require(doParallel)
  
  # assume that data is a list of data frames
  reps <- length(data)
  
  gc()
  cl <- makeCluster(max(1,detectCores()-1))
  registerDoParallel(cl)
  res <- foreach(j = 1:reps, #.errorhandling = 'remove',
                 .export = c("LR_bandit_from_tsdata_exponential", "get_min", 
                             "kl_exponential", "get_next_arm_kl_exponential"), 
                 .verbose = do_verbose, .inorder = TRUE) %dopar% {
                   alg_res <- LR_bandit_from_tsdata_exponential(data = data[[j]], 
                                                                seed = 512+j, ...)
                   list(mean_storage = alg_res$mean_storage,
                        arm_sequence = alg_res$arm_sequence,
                        input_data = data[[j]])
                 }
  stopCluster(cl)
  return(res)
}

#################################################################################
# this function parallelizes the LR_bandit_from_tsdata_poisson() function 
# for repeated simulations.

# input has to be a list of different data frames
# each data frame is a run of the algorithm
# in the end, return a list of lists
# each object of the first list should be a list that contains 
# the mean_storage data frame and the arm_sequence vector

para_bandit_sim_LR_poisson <- function(data, seed = NA, do_verbose = FALSE, ...) {
  require(foreach)
  require(doParallel)
  
  # assume that data is a list of data frames
  reps <- length(data)
  
  gc()
  cl <- makeCluster(max(1,detectCores()-1))
  registerDoParallel(cl)
  res <- foreach(j = 1:reps, #.errorhandling = 'remove',
                 .export = c("LR_bandit_from_tsdata_poisson", "get_min", 
                             "kl_poisson", "get_next_arm_kl_poisson"), 
                 .verbose = do_verbose, .inorder = TRUE) %dopar% {
                   alg_res <- LR_bandit_from_tsdata_poisson(data = data[[j]], 
                                                            seed = 512+j, ...)
                   list(mean_storage = alg_res$mean_storage,
                        arm_sequence = alg_res$arm_sequence,
                        input_data = data[[j]])
                 }
  stopCluster(cl)
  return(res)
}

#################################################################################
# this function parallelizes the AugUCB_from_tsdata() function 
# for repeated simulations.

# input has to be a list of different data frames
# each data frame is a run of the algorithm
# in the end, return a list of lists
# each object of the first list should be a list that contains 
# the mean_storage data frame and the arm_sequence vector

para_bandit_sim_AugUCB <- function(data, seed = NA, do_verbose = FALSE, ...) {
  require(foreach)
  require(doParallel)
  
  # assume that data is a list of data frames
  reps <- length(data)
  
  gc()
  cl <- makeCluster(max(1,detectCores()-1))
  registerDoParallel(cl)
  res <- foreach(j = 1:reps, #.errorhandling = 'remove',
                 .export = c("AugUCB_from_tsdata", "get_next_arm_augucb",
                             "get_min", "delete_arms"), 
                 .verbose = do_verbose, .inorder = TRUE) %dopar% {
                   alg_res <- AugUCB_from_tsdata(data = data[[j]], 
                                                 seed = 512+j, ...)
                   list(mean_storage = alg_res$mean_storage,
                        arm_sequence = alg_res$arm_sequence,
                        input_data = data[[j]])
                 }
  stopCluster(cl)
  return(res)
}

#################################################################################
# this function parallelizes the uniform_bandit_from_tsdata() function 
# for repeated simulations.

# input has to be a list of different data frames
# each data frame is a run of the algorithm
# in the end, return a list of lists
# each object of the first list should be a list that contains 
# the mean_storage data frame and the arm_sequence vector

para_bandit_sim_uniform <- function(data, seed = NA, do_verbose = FALSE, ...) {
  require(foreach)
  require(doParallel)
  
  # assume that data is a list of data frames
  reps <- length(data)
  
  gc()
  cl <- makeCluster(max(1,detectCores()-1))
  registerDoParallel(cl)
  res <- foreach(j = 1:reps, .errorhandling = 'remove',
                 .export = c("uniform_bandit_from_tsdata",
                             "get_next_arm_uniform", "get_min"), 
                 .verbose = do_verbose, .inorder = TRUE) %dopar% {
                   alg_res <- uniform_bandit_from_tsdata(data = data[[j]], 
                                                         #seed = 512+j, 
                                                         ...)
                   list(mean_storage = alg_res$mean_storage,
                        arm_sequence = alg_res$arm_sequence,
                        input_data = data[[j]],
                        iter = j)
                 }
  stopCluster(cl)
  return(res)
}

#################################################################################
# this function parallelizes the BayesUCB_from_tsdata() function 
# for repeated simulations.

# input has to be a list of different data frames
# each data frame is a run of the algorithm
# in the end, return a list of lists
# each object of the first list should be a list that contains 
# the mean_storage data frame and the arm_sequence vector

para_bandit_sim_bucb <- function(data, seed = NA, do_verbose = FALSE, ...) {
  require(foreach)
  require(doParallel)
  
  # assume that data is a list of data frames
  reps <- length(data)
  
  gc()
  cl <- makeCluster(max(1,detectCores()-1))
  registerDoParallel(cl)
  res <- foreach(j = 1:reps, .errorhandling = 'remove',
                 .export = c("BayesUCB_from_tsdata",
                             "get_BayesUCB_metric", "get_posterior_mean", 
                             "get_min"),
                 .verbose = do_verbose, .inorder = TRUE) %dopar% {
                   alg_res <- BayesUCB_from_tsdata(data = data[[j]], 
                                                   seed = 512+j, ...)
                   list(mean_storage = alg_res$mean_storage,
                        arm_sequence = alg_res$arm_sequence,
                        input_data = data[[j]])
                 }
  stopCluster(cl)
  return(res)
}

#############################################################################

# Get the average performance over all simulations
# Gives an estimate for the error probability of an algorithm on a given problem

# The function takes with the sim_res argument the output of the previous
# para_ family of functions.

# expect a list of matrices of size (rounds x K) as input from the simulations,
# as well as a vector with the true means of length K to compare against

compare_to_ground_truth <- function(true_means, sim_res, tau, epsilon) {
  message(paste0("Comparing ", length(sim_res), " simulations."))
  # first define the two true sets of arms classified against the threshold
  true_classification_up <- which(true_means > tau+epsilon)
  true_classification_down <- which(true_means < tau-epsilon)
  
  # define a function that calculates the loss for a given round
  get_iter_error <- function(x, true_class_up, true_class_down) {
    ifelse(sum(true_class_up %in% which(x >= tau)) == length(true_class_up) &
             sum(true_class_down %in% which(x < tau)) == length(true_class_down),
           0, 1)
  }
  
  # Calculate the loss function over all rows (rounds) in a data frame
  # using this function
  get_error <- function(res_df, ...) {
    apply(res_df$mean_storage, 1, get_iter_error, ...)
  }
  
  # apply the error calculation now over all data frames in the sim_res list
  # (that is, for all repititions of the simulation)
  comp_list <- lapply(sim_res, get_error, true_class_up = true_classification_up, 
                      true_class_down = true_classification_down)
  
  # Calculate the average loss over all simulations
  comp_mean <- rowMeans(as.data.frame(comp_list, 
                                      col.names = 1:length(comp_list)))
  
  # In particular, return the average performance over all simulations at each
  # round in the 'mean' argument
  return(list(mean = comp_mean, full = comp_list))
}

#############################################################################

## example of a parallelized bandit simulation

## 1) Get a data set
#mean_loc2 <- c(0.4-0.2^(1:4),
#               0.45, 0.55,
#               0.6+0.2^(5-(1:4)))
#tau_loc2 <- 0.5
#epsilon_loc2 <- 0.1
#
#data_list2 <- list()
#set.seed(1024)
#for(j in 1:100) {
#  curr_data <- data.frame(rep(NA, times = 2000))
#  for(i in 1:length(mean_loc2)) {
#    curr_data[[i]] <- as.numeric(purrr::rbernoulli(2000, p  = mean_loc2[i]))
#  }
#  names(curr_data) <- paste0("V", rep(1:length(mean_loc2)))
#  data_list2[[j]] <- curr_data
#}
#
## Run the bandit on 100 data sets
#para_test_res <- para_bandit_sim_APT(data = data_list2, rounds = 2000, 
#                                     tau = tau_loc2, epsilon = epsilon_loc2)
#
## get the arm sequence of the first simulation
#para_test_res[[1]]$arm_sequence
#
# Calculate the average error over the 100 simulations
#loc2_comp_APT <- compare_to_ground_truth(mean_loc2, para_test_res, 
#                                         tau_loc2, epsilon_loc2)$mean
#plot(c(0,2000), c(0, -2), type = "n")
#lines(log10(loc2_comp_APT), col = "red") 
# no values between 0.01 and 0 because only 100 simulations
