##############################################################

# this script contains the main functions that define the
# thresholding bandit algorithms.
# these functions can run the bandit algorithms on a single
# data frame with K columns (arms) and T rows (observations).
# to run these functions in simulations on a list of multiple
# data frames, consider the 
#       functions_parallelized.R
# script. This allows for a minimum of parallelization that
# makes excessive repititions feasible.

##############################################################

# EXAMPLE: ###################################################

#mean_002 <- c(0.3, 0.55)
#tau_002 <- 0.5
#epsilon_002 <- 0.01

#set.seed(512)
#testt <- data.frame(rep(NA, times = 500))
#for(i in 1:length(mean_002)) {
#  testt[[i]] <- as.numeric(purrr::rbernoulli(500, p  = mean_002[i]))
#}
#names(testt) <- paste0("V", 1:2)

#apt_test <- APT_from_tsdata(testt, rounds = 500, tau = tau_002, 
#                            epsilon = epsilon_002, seed = 7543)
#table(apt_test$arm_sequence)

##############################################################
##############################################################
# General functions ##########################################
##############################################################

# given a TxK data frame as described above, this function
# takes essentially a row (iter) and column (k) index as
# arguments and returns the corresponding observation from
# the data frame. Actually quite unnecessary.

pull_arm_from_tsdata <- function(k, data, iter) {
  data[iter, k]
}

# given that the bandit algorithms decide which arm to pull
# by minimizing (or maximizing) the indexes of the arms,
# we will frequently be asking for the minimum of a vector.
# we need to add randomization when several arms have the
# same index.

get_min <- function(x) {
  mini <- min(x)
  ifelse(sum(x==mini) == 1, which.min(x),
         sample(which(x==mini)))
}

##############################################################

# this is the algorithm used for the Uniform Sampling strategy
# this implementation of Uniform Sampling is very inefficient,
# since the data could be pulled deterministically. Here we
# run the same function architecture as for the adaptive
# sampling strategies.

# this function takes a list of length K as input where
# each element of the list contains the observations so far
# for a given arm. Thus, by applying length() with lapply,
# we get a vector with the number of pulls so far for each arm.
# This is exactly the index used in the uniform sampling strategy.
# The function returns a length K vector with the indexes of the
# arms.
get_next_arm_uniform <- function(armls) {
  lapply(armls, length)
}

# this algorithm runs Uniform Sampling on the TxK data frame
# mentioned above. The argument rounds defines the overall budget,
# so that one can run the algorithm on less than the T samples of
# the data frame. The seed argument allows to set a seed to allow
# for repeating the exact same results.
# Returns a list with:
# means = K length vector with final sample means
# arm_list = K length vector; each object contains the samples drawn
#             for a given arm
# arm_sequence = numeric vector of length 'rounds' giving the sequence 
#                 in which the arms were pulled
# mean_storage = matrix of dimensions (rounds,K) containing for each arm
#                 the sample mean at each round
uniform_bandit_from_tsdata <- function(data, rounds = 5000,
                                       verbose = FALSE, seed = NA) {
  if(!is.na(seed)) set.seed(seed)
  K <- dim(data)[2] # How many arms does the data have?
  
  # initialize by pulling each arm once
  arm_list <- diag(as.matrix(data[1:K,]))
  arm_list <- as.list(arm_list)
  
  # initialize mean storage and the sequence in which the arms were pulled
  mean_storage <- matrix(unlist(lapply(arm_list, mean)), nrow = 1) # TxK matrix
  arm_sequence <- 1:K
  
  # for all subsequent rounds, get indexes, pull arm minimizing the index
  for(i in (K+1):rounds) {
    if(verbose) message(paste("this is round", i))
    # get minimizer of indexes
    next_arm <- get_min(unlist(get_next_arm_uniform(arm_list)))
    arm_sequence <- c(arm_sequence, next_arm)
    
    if(verbose) message("arm selecting done")
    arm_list[[next_arm]] <- c(arm_list[[next_arm]], data[i, next_arm])
    mean_storage <- rbind(mean_storage, unlist(lapply(arm_list, mean)))
    
    if(verbose) message("arm pulling done")
    
  }
  return(list(means = unlist(lapply(arm_list, mean)),
              arm_list = arm_list,
              arm_sequence = arm_sequence,
              mean_storage = mean_storage))
}

##############################################################

# Anytime-Parameter Free Thresholding Algorithm (APT)
# Locatelli et al., 2016
# https://arxiv.org/pdf/1605.08671

# these functions implements the APT algorithm.

# this function calculates for each arm the index used in the
# thresholding bandit problem.
# Takes as input:
# armls = list of length K with the current observations for each arm
# tau = threshold of bandit problem
# rounds = deprecated
# epsilon = epsilon parameter around the threshold
# Returns a K length vector with the indexes
get_next_arm_apt <- function(armls, tau, rounds, epsilon) {
  get_metric <- function(x) {
    xhat <- mean(x)
    ni <- length(x)
    return(sqrt(ni) * (abs(xhat-tau)+epsilon))
  }
  lapply(armls, get_metric)
}

# Main Function for APT Algorithm
# Inputs: See uniform_bandit_from_tsdata()
# additionally: threshold 'tau' and interval 'epsilon'
# Returns a list with:
# means = K length vector with final sample means
# arm_list = K length vector; each object contains the samples drawn
#             for a given arm
# arm_sequence = numeric vector of length 'rounds' giving the sequence 
#                 in which the arms were pulled
# mean_storage = matrix of dimensions (rounds,K) containing for each arm
#                 the sample mean at each round
APT_from_tsdata <- function(data, rounds = 5000, tau, epsilon,
                            verbose = FALSE, seed = NA) {
  
  if(!is.na(seed)) set.seed(seed)
  K <- dim(data)[2]
  
  # initialize by pulling each arm once
  arm_list <- diag(as.matrix(data[1:K,]))
  arm_list <- as.list(arm_list)
  
  # initialize mean storage, arm sequence
  mean_storage <- matrix(unlist(lapply(arm_list, mean)), nrow = 1)
  arm_sequence <- 1:K
  
  # run for each subsequent round
  for(i in (K+1):rounds) {
    if(verbose) message(paste("this is round", i))
    # get minimizer of APT index at current round i
    next_arm <- get_min(unlist(get_next_arm_apt(arm_list, tau = tau, 
                                                rounds = rounds, 
                                                epsilon = epsilon)))
    arm_sequence <- c(arm_sequence, next_arm)
    
    if(verbose) message("arm selecting done")
    arm_list[[next_arm]] <- c(arm_list[[next_arm]], data[i, next_arm])
    mean_storage <- rbind(mean_storage, unlist(lapply(arm_list, mean)))
    
    if(verbose) message("arm pulling done")
    
  }
  return(list(means = unlist(lapply(arm_list, mean)),
              arm_list = arm_list,
              input = data,
              arm_sequence = arm_sequence,
              mean_storage = mean_storage))
}

##############################################################

# These functions implement the Simple Likelihood Ratio 
# algorithm. First, we define the algorithm for Bernoulli
# distributions, then also for Exponential distributions and
# Poisson distributions.

# Calculate the likelihood ratio between two Bernoulli
# likelihood functions
lr_ber <- function(x,S,N) {
  # here, S/N is the empirical mean (successes over trials)
  # while x represents some null hypothesis mean we test against
  (x/(S/N))^(S) * ((1-x)/(1-S/N))^(N-S)
}

# Calculates the indexes for every arm
# Takes as input:
# armls = list of length K with the current observations for each arm
# tau = threshold of bandit problem
# epsilon = epsilon parameter around the threshold
# Returns a K length vector with the indexes
get_next_arm_lr <- function(armls, tau, epsilon) {
  get_metric <- function(x) {
    successes <- sum(x)
    trials <- length(x)
    LRhat <- ifelse(successes/trials >= tau,
                    lr_ber(tau-epsilon, successes, trials),
                    lr_ber(tau+epsilon, successes, trials))
    return(LRhat)
  }
  lapply(armls, get_metric)
}

# The main Simple Likelihood Ratio function for Bernoulli distributions
# Inputs: See APT_from_tsdata()
# Returns a list with:
# means = K length vector with final sample means
# arm_list = K length vector; each object contains the samples drawn
#             for a given arm
# arm_sequence = numeric vector of length 'rounds' giving the sequence 
#                 in which the arms were pulled
# mean_storage = matrix of dimensions (rounds,K) containing for each arm
#                 the sample mean at each round
LR_bandit_from_tsdata <- function(data, rounds = 5000, tau, epsilon,
                                  verbose = FALSE, seed = NA) {
  
  if(!is.na(seed)) set.seed(seed)
  K <- dim(data)[2]
  
  # initialize by pulling each arm once
  arm_list <- diag(as.matrix(data[1:K,]))
  arm_list <- as.list(arm_list)
  
  # initialize mean storage and the counter
  mean_storage <- matrix(unlist(lapply(arm_list, mean)), nrow = 1)
  arm_sequence <- 1:K
  
  # for each subsequent round, do...
  for(i in (K+1):rounds) {
    if(verbose) message(paste("this is round", i))
    
    # get minimizer of arms at round i
    next_arm <- get_min(-unlist(get_next_arm_lr(arm_list, tau = tau,
                                                epsilon = epsilon)))
    
    arm_sequence <- c(arm_sequence, next_arm)
    
    if(verbose) message("arm selecting done")
    if(verbose) message(next_arm)
    arm_list[[next_arm]] <- c(arm_list[[next_arm]], data[i, next_arm])
    
    mean_storage <- rbind(mean_storage, unlist(lapply(arm_list, mean)))
    
    if(verbose) message("arm pulling done")
    
  }
  return(list(means = unlist(lapply(arm_list, mean)),
              arm_list = arm_list,
              arm_sequence = arm_sequence,
              mean_storage = mean_storage))
}

##############################################################

# Likelihood Ratio based Algorithm
# for Bernoulli distribution
# with additional D-tracking rule from
# Optimal Best Arm Identification with Fixed Confidence
# Garivier, Kaufmann (2016)

# see LR_bandit_from_tsdata() for input and output
LRD_bandit_from_tsdata <- function(data, rounds = 5000, tau, epsilon,
                                   verbose = FALSE, seed = NA) {
  
  if(!is.na(seed)) set.seed(seed)
  K <- dim(data)[2]
  
  # initialize by pulling each arm once
  arm_list <- diag(as.matrix(data[1:K,]))
  arm_list <- as.list(arm_list)
  
  # initialize mean storage and the counter
  mean_storage <- matrix(unlist(lapply(arm_list, mean)), nrow = 1)
  arm_sequence <- 1:K
  
  for(i in (K+1):rounds) {
    if(verbose) message(paste("this is round", i))
    
    # D-Tracking Rule
    Uset <- which(table(arm_sequence) < sqrt(i) - K/2)
    if(length(Uset) > 0){
      # Do not pull based on index if some arm lacks behind
      next_arm <- get_min(table(arm_sequence))
    } else {
      next_arm <- get_min(-unlist(get_next_arm_lr(arm_list, tau = tau,
                                                  epsilon = epsilon)))
    }
    
    arm_sequence <- c(arm_sequence, next_arm)
    
    if(verbose) message("arm selecting done")
    if(verbose) message(next_arm)
    arm_list[[next_arm]] <- c(arm_list[[next_arm]], data[i, next_arm])
    
    mean_storage <- rbind(mean_storage, unlist(lapply(arm_list, mean)))
    
    if(verbose) message("arm pulling done")
    
  }
  return(list(means = unlist(lapply(arm_list, mean)),
              arm_list = arm_list,
              arm_sequence = arm_sequence,
              mean_storage = mean_storage))
}

##############################################################

# Simple Likelihood Ratio algorithm for Exponential distributions

# Use Empirical Kullback-Leibler between two Exponential distributions
kl_exponential <- function(n, mu, tau) {
  n*(log(tau/mu)+mu/tau-1)
}

# get the index for each arm based on KL-divergence
get_next_arm_kl_exponential <- function(armls, tau, epsilon) {
  get_metric <- function(x) {
    LRhat <- ifelse(mean(x) >= tau,
                    kl_exponential(length(x), mean(x), tau-epsilon),
                    kl_exponential(length(x), mean(x), tau+epsilon))
    return(LRhat)
  }
  lapply(armls, get_metric)
}

# For input, output, see LR_bandit_from_tsdata()
LR_bandit_from_tsdata_exponential <- function(data, rounds = 5000, tau, epsilon,
                                              verbose = FALSE, seed = NA) {
  
  if(!is.na(seed)) set.seed(seed)
  K <- dim(data)[2]
  
  # initialize by pulling each arm once
  arm_list <- diag(as.matrix(data[1:K,]))
  arm_list <- as.list(arm_list)
  
  # initialize mean storage and the counter
  mean_storage <- matrix(unlist(lapply(arm_list, mean)), nrow = 1)
  arm_sequence <- 1:K
  
  # for each subsequent round, do...
  for(i in (K+1):rounds) {
    if(verbose) message(paste("this is round", i))
    
    # get the minimizer of the indexes at round i
    next_arm <- get_min(unlist(get_next_arm_kl_exponential(arm_list, tau = tau,
                                                           epsilon = epsilon)))
    
    arm_sequence <- c(arm_sequence, next_arm)
    
    if(verbose) message("arm selecting done")
    if(verbose) message(next_arm)
    arm_list[[next_arm]] <- c(arm_list[[next_arm]], data[i, next_arm])
    
    mean_storage <- rbind(mean_storage, unlist(lapply(arm_list, mean)))
    
    if(verbose) message("arm pulling done")
    
  }
  return(list(means = unlist(lapply(arm_list, mean)),
              arm_list = arm_list,
              arm_sequence = arm_sequence,
              mean_storage = mean_storage))
}

##############################################################

# Simple Likelihood Ratio algorithm for Poisson distributions

# Use Empirical Kullback-Leibler between two Poisson distributions
kl_poisson <- function(n, mu, tau) {
  mu_noise <- ifelse(mu == 0, 0.0001/n, mu)
  n * (tau - mu + log(mu_noise/tau)*mu)
}

# Get indexes based on the KL-divergences
get_next_arm_kl_poisson <- function(armls, tau, epsilon) {
  get_metric <- function(x) {
    LRhat <- ifelse(mean(x) >= tau,
                    kl_poisson(length(x), mean(x), tau-epsilon),
                    kl_poisson(length(x), mean(x), tau+epsilon))
    return(LRhat)
  }
  lapply(armls, get_metric)
}

# For input, output, see LR_bandit_from_tsdata()
LR_bandit_from_tsdata_poisson <- function(data, rounds = 5000, tau, epsilon,
                                          verbose = FALSE, seed = NA) {
  
  if(!is.na(seed)) set.seed(seed)
  K <- dim(data)[2]
  
  # initialize by pulling each arm once
  arm_list <- diag(as.matrix(data[1:K,]))
  arm_list <- as.list(arm_list)
  
  # initialize mean storage and the counter
  mean_storage <- matrix(unlist(lapply(arm_list, mean)), nrow = 1)
  arm_sequence <- 1:K
  
  for(i in (K+1):rounds) {
    if(verbose) message(paste("this is round", i))
    
    next_arm <- get_min(unlist(get_next_arm_kl_poisson(arm_list, tau = tau,
                                                       epsilon = epsilon)))
    
    arm_sequence <- c(arm_sequence, next_arm)
    
    if(verbose) message("arm selecting done")
    if(verbose) message(next_arm)
    arm_list[[next_arm]] <- c(arm_list[[next_arm]], data[i, next_arm])
    
    mean_storage <- rbind(mean_storage, unlist(lapply(arm_list, mean)))
    
    if(verbose) message("arm pulling done")
    
  }
  return(list(means = unlist(lapply(arm_list, mean)),
              arm_list = arm_list,
              arm_sequence = arm_sequence,
              mean_storage = mean_storage))
}

##############################################################

# Empirical Variance Guided Thresholding Algorithm
# Implements the parameter free version of the EVT algorithm
# Given in:
# Asynchronous Parallel Empirical Variance Guided Algorithms for the Thresholding Bandit Problem
# Zhong et al. (2017)
# https://arxiv.org/pdf/1704.04567

# Compute index for each arm
get_next_arm_evt <- function(armls, tau, rounds, epsilon) {
  get_metric <- function(x) {
    xhat <- mean(x)
    ni <- length(x)
    varhat <- (ni-1)/ni*var(x)
    return(sqrt(ni) * (sqrt(varhat+(abs(xhat-tau)+epsilon))-sqrt(varhat)))
  }
  lapply(armls, get_metric)
}

# For input, output, see LR_bandit_from_tsdata()
EVT_from_tsdata <- function(data, rounds = 5000, tau, epsilon,
                            verbose = FALSE, seed = NA) {
  
  if(!is.na(seed)) set.seed(seed)
  K <- dim(data)[2]
  
  # initialize by pulling each arm TWICE! to get a variance if not bernoulli
  arm_list_init <- rbind(diag(as.matrix(data[1:K,])), 
                         diag(as.matrix(data[(K+1):(2*K),])))
  arm_list <- list()
  for(i in 1:K) {
    arm_list[[i]] <- arm_list_init[,i]
  }
  
  # initialize mean storage, arm sequence
  mean_storage <- matrix(unlist(lapply(arm_list, mean)), nrow = 1)
  arm_sequence <- c(1:K,1:K)
  
  # for each subsequent round, do...
  for(i in (2*K+1):rounds) {
    if(verbose) message(paste("this is round", i))
    # get minimizer of EVT indexes
    next_arm <- get_min(unlist(get_next_arm_evt(arm_list, tau = tau, 
                                                rounds = rounds, 
                                                epsilon = epsilon)))
    arm_sequence <- c(arm_sequence, next_arm)
    
    if(verbose) message("arm selecting done")
    arm_list[[next_arm]] <- c(arm_list[[next_arm]], data[i, next_arm])
    mean_storage <- rbind(mean_storage, unlist(lapply(arm_list, mean)))
    
    if(verbose) message("arm pulling done")
    
  }
  return(list(means = unlist(lapply(arm_list, mean)),
              arm_list = arm_list,
              input = data,
              arm_sequence = arm_sequence,
              mean_storage = mean_storage))
}

##############################################################

# Augmented-UCB Algorithm as given in:
# Thresholding Bandits with Augmented UCB
# Mukherjee et al. (2017)
# https://arxiv.org/abs/1704.02281

# Calculate the indexes with parameters given as in paper
get_next_arm_augucb <- function(armls, tau, rho, psi, rounds, 
                                epsilon, active_set) {
  get_metric <- function(x) {
    xhat <- mean(x)
    ni <- length(x)
    vhat <- ifelse(is.na(var(x)),0,var(x))*(ni-1)/ni
    si <- sqrt(rho * psi * (vhat+1) * log(rounds * epsilon)/4/ni)
    return(abs(xhat - tau)-2*si)
  }
  res <- lapply(armls, get_metric)
  res[-active_set] <- Inf
  return(res)
}

# Arm deleting procedure as in paper
delete_arms <- function(armls, tau = tau, rho = rho,
                        psi = psi, rounds = rounds, epsilon = epsilon,
                        active_set = B) {
  eliminate <- function(x) {
    xhat <- mean(x)
    ni <- length(x)
    vhat <- ifelse(is.na(var(x)),0,var(x))*(ni-1)/ni
    si <- sqrt(rho * psi * (vhat+1) * log(rounds * epsilon)/4/ni)
    return((xhat + si < tau - si) || (xhat - si > tau + si))
  }
  return(lapply(armls[active_set], eliminate))
}

# For input, output, see LR_bandit_from_tsdata()
# Additionally, rho with the default 1/3 from the paper
AugUCB_from_tsdata <- function(data, rounds = 5000, rho = 1/3, tau = NA,
                               verbose = FALSE, seed = NA) {
  if(!is.na(seed)) set.seed(seed)
  K <- dim(data)[2]
  
  # initialize by pulling each arm once
  arm_list <- diag(as.matrix(data[1:K,]))
  arm_list <- as.list(arm_list)
  
  # initialize mean storage, arm sequence, and the counter
  mean_storage <- matrix(unlist(lapply(arm_list, mean)), nrow = 1)
  arm_sequence <- 1:K
  
  # Initialize the remaining parameters as in the paper
  B <- 1:K # the active set of arms
  m <- 0
  epsilon <- 1
  e <- exp(1) # ????????
  M <- floor(0.5*log2(rounds/e))
  psi <- rounds*epsilon/(128 * (log(3/16 * K * log(K)))^2)
  l <- ceiling(2*psi*log(rounds * epsilon)/epsilon)
  N <- K * l
  
  if(verbose) counter <- K
  
  for(i in (K+1):rounds) {
    if(verbose) message(paste("this is round", i))
    if(length(B) > 0) {
      next_arm <- get_min(unlist(get_next_arm_augucb(arm_list, tau = tau, rho = rho,
                                                     psi = psi, rounds = rounds, 
                                                     epsilon = epsilon,
                                                     active_set = B)))
      arm_sequence <- c(arm_sequence, next_arm)
      if(verbose) message("arm selecting done")
      
      arm_list[[next_arm]] <- c(arm_list[[next_arm]], data[i, next_arm])
    }
    
    # Continue adding the same mean if the active set is empty
    mean_storage <- rbind(mean_storage, unlist(lapply(arm_list, mean)))
    if(verbose) message("arm pulling done")
    
    if(verbose) counter <- counter + 1
    if(verbose) message("Counter: ", counter)
    
    # delete arms should return a logical index for each arm in active set
    if(length(B) > 0) { # stop updating the active set if already empty
      B <- B[!unlist(delete_arms(arm_list, tau = tau, rho = rho,
                                 psi = psi, rounds = rounds, epsilon = epsilon,
                                 active_set = B))]
    }
    
    if(verbose) message("B done")
    if(verbose) message(B)
    
    if((i >= N) & (m <= M) & (length(B)>0)) {
      # reset parameters
      epsilon <- epsilon/2
      psi <- rounds * epsilon / (128*(log(3/16*K*log(K)))^2)
      l <- ceiling( (2 * psi * log(rounds*epsilon))/epsilon )
      N <- i + length(B) * l
      m <- m+1
    }
  }
  return(list(means = unlist(lapply(arm_list, mean)),
              arm_list = arm_list,
              #active_set = B,
              mean_storage = mean_storage,
              arm_sequence = arm_sequence))
}

##############################################################

# Bayes-UCB Algorithm adapted to the thresholding bandit from:
# On Bayesian Upper Confidence Bounds for Bandit Problems
# Kaufmann et al. (2012)
# http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf

# For input, output, see LR_bandit_from_tsdata()
# Additionally:
# alpha, beta as parameters of Beta prior distribution
# with_epsilon: Should the algorithm consider the epsilon interval?
# const, rate: Parameters used by index function
BayesUCB_from_tsdata <- function(data, rounds = 5000, tau, epsilon,
                                 alpha = 1, beta = 1, 
                                 with_epsilon = TRUE, const = 5,
                                 rate, verbose = FALSE, seed = NA) {
  
  if(!is.na(seed)) set.seed(seed)
  K <- dim(data)[2]
  
  # initialize by pulling each arm once
  arm_list <- diag(as.matrix(data[1:K,]))
  arm_list <- as.list(arm_list)
  
  # initialize mean storage and the counter
  mean_storage <- matrix(unlist(lapply(arm_list, mean)), nrow = 1)
  arm_sequence <- 1:K
  
  for(i in (K+1):rounds) {
    if(verbose) message(paste("this is round", i))
    
    next_arm <- get_min(-unlist(lapply(arm_list, get_BayesUCB_metric, rate = rate, 
                                       tau = tau, epsilon = epsilon,
                                       alpha = alpha, beta = beta,
                                       rounds = rounds, current_round = i,
                                       with_epsilon = with_epsilon,
                                       const = const)))
    arm_sequence <- c(arm_sequence, next_arm)
    
    if(verbose) message("arm selecting done")
    arm_list[[next_arm]] <- c(arm_list[[next_arm]], data[i, next_arm])
    mean_storage <- rbind(mean_storage, unlist(lapply(arm_list,
                                                      get_posterior_mean,
                                                      alpha = alpha, 
                                                      beta = beta)))
    if(verbose) message("arm pulling done")
  }
  return(list(means = unlist(lapply(arm_list, mean)),
              arm_list = arm_list,
              arm_sequence = arm_sequence,
              mean_storage = mean_storage))
}

# Calculate the index for each arm
# Since we wanted to try different exploration rates, specify the preferred
# exploration rate using the 'rate' argument
get_BayesUCB_metric <- function(x, rate, tau, epsilon, 
                                alpha, beta, rounds, current_round,
                                with_epsilon, const) {
  alpha_prime <- sum(x)+alpha
  beta_prime <- length(x)+beta-sum(x)
  posterior_mean <- alpha_prime/(alpha_prime+beta_prime)
  #ni <- length(x)
  # depending on current mean estimate, give probability of error
  
  if(rate == "inverse") {
    upper_quantile <- qbeta(1-1/current_round, alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/current_round, alpha_prime, beta_prime)
  }
  #if(rate == "klucb") {
  #  upper_quantile <- qbeta(1-c/ni, alpha_prime, beta_prime)
  #  lower_quantile <- qbeta(c/ni, alpha_prime, beta_prime)
  #}
  if(rate == "inverse_horizon_linear") {
    upper_quantile <- qbeta(1-1/(rounds), alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/(rounds), alpha_prime, beta_prime)
  }
  if(rate == "inverse_horizon_linear_c") {
    upper_quantile <- qbeta(1-1/(const*rounds), alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/(const*rounds), alpha_prime, beta_prime)
  }
  if(rate == "inverse_horizon") {
    upper_quantile <- qbeta(1-1/(current_round*log(rounds)), alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/(current_round*log(rounds)), alpha_prime, beta_prime)
  }
  if(rate == "inverse_horizon_c") {
    upper_quantile <- qbeta(1-1/(current_round*log(rounds)^const), alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/(current_round*log(rounds)^const), alpha_prime, beta_prime)
  }
  if(rate == "inverse_squared") {
    upper_quantile <- qbeta(1-1/current_round^2, alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/current_round^2, alpha_prime, beta_prime)
  }
  if(rate == "inverse_squared_horizon_c") {
    upper_quantile <- qbeta(1-1/(current_round^2*log(rounds)^const), alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/(current_round^2*log(rounds)^const), alpha_prime, beta_prime)
  }
  if(rate == "inverse_cubic") {
    upper_quantile <- qbeta(1-1/current_round^3, alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/current_round^3, alpha_prime, beta_prime)
  }
  if(rate == "inverse_power5") {
    upper_quantile <- qbeta(1-1/current_round^5, alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/current_round^5, alpha_prime, beta_prime)
  }
  if(rate == "inverse_log") {
    upper_quantile <- qbeta(1-1/log(current_round), alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/log(current_round), alpha_prime, beta_prime)
  }
  if(rate == "inverse_sqrt") {
    upper_quantile <- qbeta(1-1/sqrt(current_round), alpha_prime, beta_prime)
    lower_quantile <- qbeta(1/sqrt(current_round), alpha_prime, beta_prime)
  }
  
  if(with_epsilon) {
    ifelse(posterior_mean >= tau,
           (tau - epsilon - lower_quantile),
           (upper_quantile - tau - epsilon))
  } else { # this alternative doesn't change anything because we always just substract epsilon from everything...
    ifelse(posterior_mean >= tau,
           (tau - lower_quantile),
           (upper_quantile - tau))
  }
}