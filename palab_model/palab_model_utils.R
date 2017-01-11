# ------------------------------------------------------------------------------
#
#             Utils functions of the modelling part of PAlab
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(ggplot2)

# ------------------------------------------------------------------------------
# Get numeric (or categorical) variables from input dataframe
# ------------------------------------------------------------------------------

get_variables <- function(input, var_config, categorical=F) {
  if (categorical){
    accepted_types <- c("numerical", "categorical")
  }else{
    accepted_types <- c("numerical")
  }
  var_config_accepted <- var_config %>%
    dplyr::filter_(~Column %in% colnames(input)) %>%
    dplyr::filter_(~Type %in% accepted_types)
  
  # Keeping only these variables from the input and returning the dataframe
  output <- input %>%
    select_(.dots = var_config_accepted$Column)
  
  # Turning all categorical into factors
  if (categorical){
    categoricals <- var_config_accepted %>% filter_(~Type %in% accepted_types)
    output[categoricals$Column] <- lapply(output[categoricals$Column], as.factor)
  }
  output
}

# ------------------------------------------------------------------------------
# Get IDs from the data.frame
# ------------------------------------------------------------------------------

get_ids <- function(input, var_config){
  var_config_accepted <- var_config %>% dplyr::filter_(~Type %in% "key")
  
  # Keeping only the key column
  output <- input %>%
    select_(.dots = var_config_accepted$Column)
  output
}

# ------------------------------------------------------------------------------
# Return frequency of classes in classification task
# ------------------------------------------------------------------------------

get_class_freqs <- function(dataset){
  library(mlr)
  target <- table(getTaskTargets(dataset))
  target/sum(target)
}

# ------------------------------------------------------------------------------
# Impute missing data
# ------------------------------------------------------------------------------

impute_col_median <- function(x){
  # Retruns a single column where NA is replaced with median
  x <- as.numeric(as.character(x))
  x[is.na(x)] <- median(x, na.rm=TRUE)
  x 
}

impute_col_mean <- function(x){
  # Retruns a single column where NA is replaced with mean
  x <- as.numeric(as.character(x))
  x[is.na(x)] <- mean(x, na.rm=TRUE)
  x 
}

data_without_target <- function(df, target){
  # Return df without the target column
  df[,-which(colnames(df)==target)]
}

impute_data <- function(df, target, method="median"){
  # Impute missing values in cols with mean or median except in target
  df_imp <- data_without_target(df, target)
  if (method=="median"){
    df_imp <- as.data.frame(sapply(df_imp, impute_col_median))
  }else{
    df_imp <- as.data.frame(sapply(df_imp, impute_col_mean))
  }
  df[,-which(colnames(df)==target)] <- df_imp
  df
}

# ------------------------------------------------------------------------------
# Round all numeric values in a dataframe to a given decimal digits
# ------------------------------------------------------------------------------

decimal_rounder <- function(df, decimal=5){
  is.num <- sapply(df, is.numeric)
  df[is.num] <- lapply(df[is.num], round, decimal)
  df
}

# ------------------------------------------------------------------------------
# Timing functions
# ------------------------------------------------------------------------------

tic <- function(gcFirst = TRUE, type=c("elapsed", "user.self", "sys.self")){
  type <- match.arg(type)
  assign(".type", type, envir=baseenv())
  if(gcFirst) gc(FALSE)
  tic <- proc.time()[type]         
  assign(".tic", tic, envir=baseenv())
  invisible(tic)
}

toc <- function(){
  type <- get(".type", envir=baseenv())
  toc <- proc.time()[type]
  tic <- get(".tic", envir=baseenv())
  print(toc - tic)
  invisible(toc)
}

# ------------------------------------------------------------------------------
# Function to create custom precision at x% recall in mlR
# ------------------------------------------------------------------------------

make_custom_pr_measure <- function(recall_perc=5, name_str="pr5"){
  
  find_prec_at_recall <- function(pred, recall_perc=5){
    library(PRROC)
    # This function takes in a prediction output from a trained mlR learner. It
    # extracts the predicitons, and finds the highest precision at a given
    # percentage of recall. 
    
    # see docs here: https://cran.r-project.org/web/packages/PRROC/PRROC.pdf
    # the positive class has to be 1, and the negative has to be 0.
    positive_class <- pred$task.desc$positive
    prob = getPredictionProbabilities(pred, cl=positive_class)
    
    # get truth and turn it into ones (positive) and zeros (negative)
    truth <- getPredictionTruth(pred)
    if (is.factor(truth)) {
      pos_ix <- as.integer(truth) == which(levels(truth) == positive_class)
    } else {
      pos_ix <- truth == positive_class
    }
    truth <- as.integer(pos_ix)
    pr <- pr.curve(scores.class0=prob, weights.class0=truth, curve = T)
    
    # extract recall and precision from the curve of PRROC package's result
    recall <- pr$curve[,1]
    prec <- pr$curve[,2]
    
    # find closes recall value(s)
    target_recall <- recall_perc/100
    recall_diff <- abs(recall - target_recall)
    # find indice for this (these)
    recall_min_ix <- which(recall_diff == min(recall_diff))
    # find corresponding highest precision
    max(prec[recall_min_ix])
    
  }
  
  name <- paste("Precision at ", as.character(recall_perc),"%"," recall", sep='')
  
  custom_measure = makeMeasure(
    id = name_str, 
    name = name,
    properties = c("classif", "req.prob", "req.truth"),
    minimize = FALSE, best = 1, worst = 0,
    extra.args = list("threshold" = recall_perc),
    fun = function(task, model, pred, feats, extra.args) {
      find_prec_at_recall(pred, extra.args$threshold)
    }
  )
  custom_measure
}

# ------------------------------------------------------------------------------
# Plot precision recall and ROC curve
# ------------------------------------------------------------------------------

get_truth_pred <- function(pred){
  # This helper function could not have been used in make_custom_pr_measure()
  # because it was not in scope. 
  positive_class <- pred$task.desc$positive
  prob = getPredictionProbabilities(pred, cl=positive_class)
  
  # get truth and turn it into ones (positive) and zeros (negative)
  truth <- getPredictionTruth(pred)
  if (is.factor(truth)) {
    pos_ix <- as.integer(truth) == which(levels(truth) == positive_class)
  } else {
    pos_ix <- truth == positive_class
  }
  truth <- as.integer(pos_ix)
  results <- list("truth"=truth, "prob"=prob)
  results
}

plot_pr_curve <- function(results, roc=TRUE){
  # Get probabilities and truth
  tb <- get_truth_pred(results$pred)
  
  # Retain only the predictions on the test set
  df <- as.data.frame(results$pred)
  truth <- tb$truth[df$set=="test"]
  prob <- tb$prob[df$set=="test"]
  
  # Plot ROC curve first so it's the 2nd plot once this function is run
  if (roc){
    roc <- roc.curve(scores.class0=prob, weights.class0=truth, curve = T)
    plot(roc)
  }
  
  # Plot PR curve
  pr <- pr.curve(scores.class0=prob, weights.class0=truth, curve = T)
  plot(pr)
}

# ------------------------------------------------------------------------------
# Create output folder if it doesn't exist
# ------------------------------------------------------------------------------

create_output_folder <- function(output_folder){
  if (output_folder == ""){
    output_folder = getwd()
  }else{
    output_folder = file.path(getwd(), output_folder)
    if (!file.exists(output_folder)){
      dir.create(output_folder)
    }
  }
}