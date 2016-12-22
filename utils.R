# ------------------------------------------------------------------------------
#
#             Utils functions of the modelling part of palab
#
# ------------------------------------------------------------------------------

get_numerical_variables <- function(input, var_config) {
  library(dplyr)  
  # Keeping only those variables in var_config that are in input and
  # are numerical
  
  var_config_numerical <- var_config %>%
    dplyr::filter_(~Column %in% colnames(input)) %>%
    dplyr::filter_(~Type == "numerical")
  
  # Keeping only these variables from the input and returning the dataframe
  output <- input %>%
    select_(.dots = var_config_numerical$Column)
  
  output
}

get_results <- function(results, detailed=F, all_measures=F){
  # This function calculates a bunch of summary statistics 
  
  
  # Define return list
  to_return <- list(0)
  
  # Extract results from nested CV results
  o_train <- results$measures.test
  o_train$iter <- NULL
  o_test <- results$measures.train
  o_test$iter <- NULL
  i_test <- as.data.frame(t(extractSubList(results$extract, "y")))
  
  # Unify the columns
  colnames(o_train) <- colnames(i_test)
  colnames(o_test) <- colnames(i_test)
  
  # Save whole table if user wants them
  if (all_measures){
    to_return <- c(to_return, list("outer_train" = o_train,
                                   "outer_test" = o_test,
                                   "inner_test" = i_test))
  }
  
  # Discard columns with aggregated SD values
  no_sd_ix <- !grepl("sd", colnames(o_train))
  o_train <- o_train[,no_sd_ix]
  o_test <- o_test[,no_sd_ix]
  i_test <- i_test[,no_sd_ix]
  
  # Difine difference matrices
  o_train_minus_o_test <- o_train - o_test
  i_test_minus_o_test <- i_test - o_test
  
  # Collect summary stats
  o_train_mean <- unlist(lapply(o_train, mean))
  o_train_sd <- unlist(lapply(o_train, sd))
  o_test_mean <- unlist(lapply(o_test, mean))
  o_test_sd <- unlist(lapply(o_test, sd))
  o_train_minus_o_test_mean <- unlist(lapply(o_train_minus_o_test, mean))
  o_train_minus_o_test_sd <- unlist(lapply(o_train_minus_o_test, sd))
  i_test_minus_o_test_mean <- unlist(lapply(i_test_minus_o_test, mean))
  i_test_minus_o_test_sd <- unlist(lapply(i_test_minus_o_test, sd))
  
  if (detailed){
    summary = rbind(o_train_mean, o_train_sd, o_test_mean, o_test_sd,
                    o_train_minus_o_test_mean, o_train_minus_o_test_sd,
                    i_test_minus_o_test_mean, i_test_minus_o_test_sd)
    rownames(summary) <- c("Outer train mean", "Outer train std",
                           "Outer test mean", "Outer test std",
                           "Outer train minus outer test mean",
                           "Outer train minus outer test std",
                           "Inner test minus outer test mean",
                           "Inner test minus outer test std")
  }else{
    summary <- rbind(o_test_mean, o_test_sd, o_train_minus_o_test_mean, 
                     i_test_minus_o_test_mean)
    rownames(summary) <- c("Outer test mean", "Outer test std",
                           "Outer train minus outer test mean",
                           "Inner test minus outer test mean")
  }
  to_return <- c(to_return, list("summary"=summary))
  to_return <- c(to_return, list("best_params"=getNestedTuneResultsX(results)))
  to_return
}

get_opt_path <- function(result){
  # Returns all the hyper-parameters combinations that were tested in the
  # nested  CV.
  
  opt_paths <- getNestedTuneResultsOptPathDf(result)
  # Drop unnecessary columns
  opt_paths$dob <- NULL
  opt_paths$eol <- NULL
  
  # Check if we have any error messages
  error_ix <- !is.na(opt_paths$error.message)
  
  if (any(error_ix)){
    warning("Some of the hyper-param combinations threw errors, check the
            error.message column!")
  }else{
    opt_paths$error.message <- NULL
  }
  opt_paths
}