# ------------------------------------------------------------------------------
#
#     Functions to access results, params, models of tuned mlR models
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(ggplot2)

# ------------------------------------------------------------------------------
# Get path to output csv
# ------------------------------------------------------------------------------

get_result_csv_path <- function(output_folder){
  # Get time stamp and make a (Windows safe) file name out of it
  t <- as.character(Sys.time())
  t <- gsub(" ", "_", t)
  t <- gsub(":", "-", t)
  output_csv <- paste("results_", t, '.csv', sep='')
  
  # Check output folder, create it if needed
  create_output_folder(output_folder)
  output_csv <- file.path(output_folder, output_csv)
  output_csv
}

# ------------------------------------------------------------------------------
# Extract results from tuned nested CV model
# ------------------------------------------------------------------------------

get_results <- function(results, detailed=F, all_measures=F, write_csv=F, 
                        output_folder=""){
  # This function calculates a bunch of summary statistics from a learner that
  # was tuned using nested CV. The level of detail that's returned could be 
  # set with detailed=T and all_measure=T. Also if write_csv=T, all data.frames
  # will be saved to a csv with the current timestamp as its name.
  
  # Define return list
  info_col_name <- ""
  info_df <- data.frame(info_col_name, results$task.id, results$learner.id)
  colnames(info_df) <- c("Info", "Dataset", "Learner")
  to_write <- list("info"= info_df)
  to_return <- list("info" = data.frame(results$task.id, results$learner.id))
  
  # Extract results from nested CV results
  o_train <- results$measures.test
  o_train$iter <- NULL
  o_test <- results$measures.train
  o_test$iter <- NULL
  i_test <- as.data.frame(t(extractSubList(results$extract, "y")))
  
  # Unify the columns
  colnames(o_train) <- colnames(i_test)
  colnames(o_test) <- colnames(i_test)
  
  # Discard columns with aggregated SD values in the outer loop
  no_sd_ix <- !grepl("sd", colnames(o_train))
  o_train <- o_train[,no_sd_ix]
  o_test <- o_test[,no_sd_ix]
  
  # Add table name
  o_train_w <- cbind("Outer train"=1:nrow(o_train), o_train)
  o_test_w <- cbind("Outer test"=1:nrow(o_train), o_test)
  i_test_w <- cbind("Inner test"=1:nrow(o_train), i_test)
  
  # Save whole table if user wants them
  if (all_measures){
    to_write <- c(to_write, list("outer_train" = o_train_w,
                                 "outer_test" = o_test_w,
                                 "inner_test" = i_test_w))
    to_return <- c(to_return, list("outer_train" = o_train,
                                   "outer_test" = o_test,
                                   "inner_test" = i_test))
  }
  
  # Now that we saved the outer tables, let's discard SD from the inner one too
  i_test <- i_test[,no_sd_ix]
  
  # Difine difference matrices
  o_train_minus_o_test <- o_train - o_test
  i_test_minus_o_test <- i_test - o_test
  
  # Collect summary stats
  o_train_mean <- unlist(lapply(o_train, mean))
  o_train_sd <- unlist(lapply(o_train, sd))
  o_test_mean <- unlist(lapply(o_test, mean))
  o_test_sd <- unlist(lapply(o_test, sd))
  i_test_mean <- unlist(lapply(i_test, mean))
  i_test_sd <- unlist(lapply(i_test, sd))
  o_train_minus_o_test_mean <- unlist(lapply(o_train_minus_o_test, mean))
  o_train_minus_o_test_sd <- unlist(lapply(o_train_minus_o_test, sd))
  i_test_minus_o_test_mean <- unlist(lapply(i_test_minus_o_test, mean))
  i_test_minus_o_test_sd <- unlist(lapply(i_test_minus_o_test, sd))
  
  if (detailed){
    summary = rbind(o_train_mean, o_train_sd, o_test_mean, o_test_sd,
                    i_test_mean, i_test_sd,
                    o_train_minus_o_test_mean, o_train_minus_o_test_sd,
                    i_test_minus_o_test_mean, i_test_minus_o_test_sd)
    # add row names as a new column
    row_names <- c("Outer train mean", "Outer train std",
                   "Outer test mean", "Outer test std",
                   "Inner test mean", "Inner test std",
                   "(outer train - outer test) mean",
                   "(outer train - outer test) std",
                   "(inner test - outer test) mean",
                   "(inner test - outer test) std")
  }else{
    summary <- rbind(o_test_mean, o_test_sd, o_train_minus_o_test_mean, 
                     i_test_minus_o_test_mean)
    # add row names as a new column
    row_names <- c("Outer test mean", "Outer test std",
                   "(outer train - outer test) mean",
                   "(inner test - outer test) mean")
  }
  summary_w <- cbind("Summary"=row_names, summary)
  rownames(summary) <- row_names
  to_write <- c(to_write, list("summary"=summary_w))
  to_return <- c(to_return, list("summary"=summary))
  
  # extract best params for each outer fold
  params <- getNestedTuneResultsX(results)
  to_return <- c(to_return, list("best_params"=params))
  params_w <- cbind("Best params"=1:nrow(o_train), params)
  to_write <- c(to_write, list("best_params"=params_w))
  
  # write results to csv if needed
  if (write_csv){
    output_csv <- get_result_csv_path(output_folder)
    write_df <- function(df){
      write.table(as.data.frame(df), output_csv, append=T, sep=',', row.names=F)
    }
    suppressWarnings(lapply(to_write, write_df))
  }
  
  #return results object
  to_return
}

# ------------------------------------------------------------------------------
# Extract results from tuned NON-nested CV model
# ------------------------------------------------------------------------------

get_non_nested_results <- function(results, detailed=F, all_measures=F, 
                                   write_csv=F, output_folder=""){
  
  # Define return list
  info_col_name <- ""
  info_df <- data.frame(info_col_name, results$task.id, results$learner.id)
  colnames(info_df) <- c("Info", "Dataset", "Learner")
  to_write <- list("info"= info_df)
  to_return <- list("info" = data.frame(results$task.id, results$learner.id))
  
  # Extract results from nested CV results
  o_train <- results$measures.test
  o_train$iter <- NULL
  o_test <- results$measures.train
  o_test$iter <- NULL
  
  # Discard columns with aggregated SD values in the outer loop
  no_sd_ix <- !grepl("sd", colnames(o_train))
  o_train <- o_train[,no_sd_ix]
  o_test <- o_test[,no_sd_ix]
  
  # Add table name
  o_train_w <- cbind("Outer train"=1:nrow(o_train), o_train)
  o_test_w <- cbind("Outer test"=1:nrow(o_train), o_test)
  
  # Save whole table if user wants them
  if (all_measures){
    to_write <- c(to_write, list("outer_train" = o_train_w,
                                 "outer_test" = o_test_w))
    to_return <- c(to_return, list("outer_train" = o_train,
                                   "outer_test" = o_test))
  }
  
  # Difine difference matrices
  o_train_minus_o_test <- o_train - o_test
  
  # Collect summary stats
  o_train_mean <- unlist(lapply(o_train, mean))
  o_train_sd <- unlist(lapply(o_train, sd))
  o_test_mean <- unlist(lapply(o_test, mean))
  o_test_sd <- unlist(lapply(o_test, sd))
  o_train_minus_o_test_mean <- unlist(lapply(o_train_minus_o_test, mean))
  o_train_minus_o_test_sd <- unlist(lapply(o_train_minus_o_test, sd))
  
  if (detailed){
    summary = rbind(o_train_mean, o_train_sd, o_test_mean, o_test_sd,
                    o_train_minus_o_test_mean, o_train_minus_o_test_sd)
    # add row names as a new column
    row_names <- c("Outer train mean", "Outer train std",
                   "Outer test mean", "Outer test std",
                   "(outer train - outer test) mean",
                   "(outer train - outer test) std")
  }else{
    summary <- rbind(o_test_mean, o_test_sd, o_train_minus_o_test_mean)
    # add row names as a new column
    row_names <- c("Outer test mean", "Outer test std",
                   "(outer train - outer test) mean")
  }
  summary_w <- cbind("Summary"=row_names, summary)
  rownames(summary) <- row_names
  to_write <- c(to_write, list("summary"=summary_w))
  to_return <- c(to_return, list("summary"=summary))
  
  # write results to csv if needed
  if (write_csv){
    output_csv <- get_result_csv_path(output_folder)
    write_df <- function(df){
      write.table(as.data.frame(df), output_csv, append=T, sep=',', row.names=F)
    }
    suppressWarnings(lapply(to_write, write_df))
  }
  
  #return results object
  to_return
}


# ------------------------------------------------------------------------------
# Get predictions from outer CV models, use original ids if provided
# ------------------------------------------------------------------------------

get_outer_preds <- function(results, ids=NULL){
  all_preds <- as.data.frame(results$pred)
  o_test_preds <- all_preds[all_preds$set=="test",]
  o_test_preds <- sortByCol(o_test_preds, col="id")
  if(!is.null(ids)){
    o_test_preds$id <- ids
  }
  o_test_preds
}

# ------------------------------------------------------------------------------
# Get paths of the optimized hyper parameters, models and best mean params
# ------------------------------------------------------------------------------

get_opt_paths <- function(result){
  # Returns all the hyper-parameters combinations that were tested in the
  # nested  CV.
  
  opt_paths <- getNestedTuneResultsOptPathDf(result, trafo=T)
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

get_models <- function(results){
  lapply(res$models, function(x) getLearnerModel(x, more.unwrap=T))
}

get_best_mean_param <- function(results, int=F){
  if (int){
    bmp <- lapply(results$best_params, function(x) round(mean(x)))
  }else{
    bmp <- lapply(results$best_params, mean)
  }
  bmp
}