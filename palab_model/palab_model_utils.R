# ------------------------------------------------------------------------------
#
#             Utils functions of the modelling part of PAlab
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(ROCR)

# ------------------------------------------------------------------------------
# Get numeric (or categorical) variables from input dataframe
# ------------------------------------------------------------------------------

get_variables <- function(input, var_config, categorical=F) {
  if (categorical) {
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
  if (categorical) {
    output <- output %>%
      mutate_each_(funs(as.factor), vars = var_config_accepted %>% 
                     filter_(~Type == "categorical") %>%
                     .$Column)
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

# ------------------------------------------------------------------------------
# Function to create custom precision at x% recall in mlR
# ------------------------------------------------------------------------------

make_custom_pr_measure <- function(recall_perc=5, name_str="pr5"){
  
  find_prec_at_recall <- function(pred, recall_perc=5){
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
    recall_target <- recall_perc/100
    
    # this part is adopted Hui's code 
    aucobj <- ROCR::prediction(prob, truth)
    
    # generate the recall and ppv and threshold
    prec_rec <- ROCR::performance(aucobj, 'prec', 'rec')
    rec <- prec_rec@x.values[[1]]
    prec <- prec_rec@y.values[[1]]
    
    # ignore nans
    non_nan <- !is.nan(prec) & !is.nan(rec)
    rec <- rec[non_nan]
    prec <- prec[non_nan]
    
    # find closest recall value to target and return corresponding prec value
    recall_diff <- abs(rec - recall_target)
    # Return prec that corresponds to the threshold closest to recall target
    prec[which.min(recall_diff)]
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