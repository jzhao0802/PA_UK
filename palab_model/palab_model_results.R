# ------------------------------------------------------------------------------
#
#     Functions to access results, params, models of tuned mlR models
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(BBmisc)
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
  if (output_folder != ""){
    output_csv <- file.path(output_folder, output_csv)
  }
  output_csv
}

# ------------------------------------------------------------------------------
# Extract results from tuned nested CV model
# ------------------------------------------------------------------------------

get_results <- function(results, grid_ps, detailed=F, all_measures=F, 
                        write_csv=F, output_folder="", extra=NULL, shorten=T, 
                        decimal=5){
  # This function calculates a bunch of summary statistics from a learner that
  # was tuned using nested CV. The level of detail that's returned could be 
  # set with detailed=T and all_measure=T. Also if write_csv=T, all data.frames
  # will be saved to a csv with the current timestamp as its name.
  
  # Get info part of results
  info_df <- get_info(results, extra)
  
  # Define return lists that hold the various parts of the results
  to_return <- list("info" = info_df$to_return)
  to_write <- list("info"= info_df$to_write)
  
  # Add parameter grid
  ps_df <- get_paramset_df(grid_ps)
  to_return <- c(to_return, list("param_set"=ps_df))
  to_write <- c(to_write, list("param_set"=ps_df))
  
  # Extract best params for each outer fold
  outer_fold_n <- length(results$models)
  params <- getNestedTuneResultsX(results)
  to_return <- c(to_return, list("best_params"=params))
  params_w <- cbind("Best params"=1:outer_fold_n, params)
  to_write <- c(to_write, list("best_params"=params_w))
  
  # Extract results from nested CV results
  o_train <- results$measures.train
  o_train$iter <- NULL
  o_test <- results$measures.test
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
  o_train_w <- cbind("Outer train"=1:outer_fold_n, o_train)
  o_test_w <- cbind("Outer test"=1:outer_fold_n, o_test)
  i_test_w <- cbind("Inner test"=1:outer_fold_n, i_test)
  
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

  # Discard test.mean and other attached strings in colnames
  if (shorten){
    first_part <- function(x){
      strsplit(x, "\\.")[[1]][1]
    }
    colnames(o_train) <- unlist(lapply(colnames(o_train), first_part))
    colnames(o_test) <- unlist(lapply(colnames(o_test), first_part))
    colnames(i_test) <- unlist(lapply(colnames(i_test), first_part))
  }
  
  # Difine difference matrices
  o_train_minus_o_test <- o_train - o_test
  i_test_minus_o_test <- i_test - o_test
  
  # Define col names for folds
  measures_n <- ncol(o_train)
  fold_colnames <- paste(rep("Fold"), seq(outer_fold_n), sep = "")
  
  # Get summary table
  if (detailed){
    cols <- c(o_train, o_test, o_train_minus_o_test, i_test, i_test_minus_o_test)
    stats <- c("Outer train", "Outer test", "Outer train - outer test", 
                 "Inner test", "Inner test - outer test")
    summary <- get_summary_table(cols, fold_colnames, stats, measures_n, decimal)
  }else{
    cols <- c(o_train, o_test, o_train_minus_o_test)
    stats <- c("Outer train", "Outer test", "Outer train - outer test")
    summary <- get_summary_table(cols, fold_colnames, stats, measures_n, decimal)
  }
  
  to_write <- c(to_write, list("summary"=summary))
  to_return <- c(to_return, list("summary"=summary))
  
  # Write results to csv if needed
  if (write_csv){
    output_csv <- get_result_csv_path(output_folder)
    write_df <- function(df){
      write.table(as.data.frame(df), output_csv, append=T, sep=',', row.names=F)
      write("", output_csv, append=T, sep=',')
    }
    suppressWarnings(lapply(to_write, write_df))
  }
  
  # Return results object
  to_return
}

# ------------------------------------------------------------------------------
# Extract results from tuned NON-nested CV model
# ------------------------------------------------------------------------------

get_non_nested_results <- function(results, all_measures=F, write_csv=F, 
                                   output_folder="", extra=NULL, shorten=T, 
                                   decimal=5){
  
  # Get info part of results
  info_df <- get_info(results, extra)
  
  # Define return lists that hold the various parts of the results
  to_return <- list("info" = info_df$to_return)
  to_write <- list("info"= info_df$to_write)
  
  # Extract results from nested CV results
  o_train <- results$measures.train
  o_train$iter <- NULL
  o_test <- results$measures.test
  o_test$iter <- NULL
  
  # Add table name
  outer_fold_n <- length(results$models)
  o_train_w <- cbind("Outer train"=1:outer_fold_n, o_train)
  o_test_w <- cbind("Outer test"=1:outer_fold_n, o_test)
  
  # Save whole table if user wants them
  if (all_measures){
    to_write <- c(to_write, list("outer_train" = o_train_w,
                                 "outer_test" = o_test_w))
    to_return <- c(to_return, list("outer_train" = o_train,
                                   "outer_test" = o_test))
  }
  
  # Difine difference matrices
  o_train_minus_o_test <- o_train - o_test
  
  # Discard test.mean and other attached strings in colnames
  if (shorten){
    first_part <- function(x){
      strsplit(x, "\\.")[[1]][1]
    }
    colnames(o_train) <- unlist(lapply(colnames(o_train), first_part))
    colnames(o_test) <- unlist(lapply(colnames(o_test), first_part))
  }
  
  # Define col names for folds
  measures_n <- ncol(o_train)
  fold_colnames <- paste(rep("Fold"), seq(outer_fold_n), sep = "")
  
  # Get summary table
  cols <- c(o_train, o_test, o_train_minus_o_test)
  stats <- c("Outer train", "Outer test", "Outer train - outer test")
  summary <- get_summary_table(cols, fold_colnames, stats, measures_n, decimal)
  to_write <- c(to_write, list("summary"=summary))
  to_return <- c(to_return, list("summary"=summary))
  
  # write results to csv if needed
  if (write_csv){
    output_csv <- get_result_csv_path(output_folder)
    write_df <- function(df){
      write.table(as.data.frame(df), output_csv, append=T, sep=',', row.names=F)
      write("", output_csv, append=T, sep=',')
    }
    suppressWarnings(lapply(to_write, write_df))
  }
  
  #return results object
  to_return
}

# ------------------------------------------------------------------------------
# Helper function, returns the info part of the results
# ------------------------------------------------------------------------------

get_info <- function(results, extra){
  info_col_name <- ""
  info_df <- data.frame(info_col_name, results$task.id, results$learner.id)
  colnames(info_df) <- c("Info", "Dataset", "Learner")
  if (!is.null(extra)){
    for (name in names(extra)) {
      info_df[name] <- extra[[name]]
    }
  }
  to_return <- info_df
  
  # Make it pretty for printing it to csv
  info_df <- as.data.frame(t(info_df))
  info_df["Cols"] <- rownames(info_df)
  info_df <- info_df[, c("Cols", "V1")]
  colnames(info_df) <- NULL
  rownames(info_df) <- NULL
  to_write <- info_df
  
  return(list("to_return"=to_return, "to_write"=to_write))
}

# ------------------------------------------------------------------------------
# Helper function, returns the parameter set as a data.frame
# ------------------------------------------------------------------------------

get_paramset_df <- function(grid_ps){
  # This is painstaking but couldn't find a quicker way that we can actually use
  # to write to a CSV. Welcome to the wonderful world of R datatypes.
  pars <- grid_ps$pars
  par_names <- names(pars)
  ParamSet <- c()
  Type <- c()
  Length <- c()
  Lower <- c()
  Upper <- c()
  Value <- c()
  Trafo <- c()
  for (par_name in par_names){
    par <- pars[[par_name]]
    ParamSet <- c(ParamSet, par$id)
    Type <- c(Type, par$type)
    Length <- c(Length, par$len)
    Lower <- c(Lower, if (is.null(par$lower)) "-" else unlist(par$lower))
    Upper <- c(Upper, if (is.null(par$upper)) "-" else unlist(par$upper))
    vals <- paste(unlist(par$values), collapse=",")
    Value <- c(Value, if (is.null(par$values)) "-" else vals)
    Trafo <- c(Trafo, if (is.null(par$trafo)) "-" else "Y")
  }
  data.frame(ParamSet, Type, Length, Lower, Upper, Value, Trafo)
}

# ------------------------------------------------------------------------------
# Helper function, returns summary of results in a nicely formatted way
# ------------------------------------------------------------------------------

get_summary_table <- function(cols, fold_colnames, stats, measures_n, decimal){
  # Replicate the metric description row names
  out <- c()
  for (metric in stats){
    out <- c(out, rep(metric, measures_n))
  }
  folds <- dplyr::bind_cols(cols)
  # Before transposing calculate mean and std across columns
  Mean <- lapply(folds, mean)
  Std <- lapply(folds, sd)
  # Transpose folds data and add column names: Fold1, Fold2, ...
  folds <- t(folds)
  colnames(folds) <- fold_colnames
  # Assemble final summary table, the order of the following is curcial, 
  # don't touch it EVER!
  summary <- cbind(Mean, Std, folds)
  measures <- rownames(folds)
  rownames(summary) <- NULL
  summary <- as.data.frame(summary)
  summary$Measure <- measures
  summary$Statistic <- out
  # Reorder columns
  summary <- summary[,c("Measure", "Statistic", "Mean", "Std", fold_colnames)]
  # Order by Measure
  summary <- BBmisc::sortByCol(summary, "Measure")
  # Convert all columns to non-list types so we can write it out
  summary <- data.frame(lapply(summary, unlist))
  decimal_rounder(summary, decimal=decimal)
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
  lapply(results$models, function(x) getLearnerModel(x, more.unwrap=T))
}

get_best_mean_param <- function(results, int=F){
  if (int){
    bmp <- lapply(results$best_params, function(x) round(mean(x)))
  }else{
    bmp <- lapply(results$best_params, mean)
  }
  bmp
}