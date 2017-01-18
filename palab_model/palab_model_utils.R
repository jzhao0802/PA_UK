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
    
    # Create desc sorted table of probs and truth
    df <- data.frame(truth, prob)
    df_pos <- df[which(df$truth == 1),]
    df_pos <- BBmisc::sortByCol(df_pos, "prob", asc=F)
    pos_N <- nrow(df_pos)
    
    # Find the right threshold for x% recall, by walking through the probs in
    # the df_pos table and using each as a thrsh to calculate recall
    recall_tmp <- 0
    thrsh_tmp <- 0
    ix <- 1
    recall_target <- recall_perc/100
    while (recall_tmp < recall_target){
      # To make sure we pick the thrsh_tmp that leads us the closest to the 
      # desired recall level
      recall_tmp2 <- recall_tmp
      thrsh_tmp2 <- thrsh_tmp
      # Threshold we'll try
      thrsh_tmp <- df_pos$prob[ix]
      # Predictions that this threshold translates to
      pred_tmp <- as.numeric(df$prob >= thrsh_tmp)
      # Calculate true positive rate = recall
      recall_tmp <- sum(pred_tmp)/pos_N
      ix <- ix + 1
    }
    
    # Two closest recall levels
    recall_tmps <- c(recall_tmp, recall_tmp2)
    # Two corresponding thresholds
    thrsh_tmps <- c(thrsh_tmp, thrsh_tmp2)
    recall_diff <- abs(recall_tmps - recall_target)
    # Threshold to use in precision calculation
    thrsh <- thrsh_tmps[which(recall_diff == min(recall_diff))]
    # Find precision at this threshold
    tp <- sum(df$truth[df$prob >= thrsh])
    pred_n <- sum(df$prob >= thrsh)
    tp/pred_n
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