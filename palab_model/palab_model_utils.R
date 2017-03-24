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
    # if not absolute path add working dir to it
    if (!grepl(":/", output_folder)){
      output_folder = file.path(getwd(), output_folder)
    }
    # make the folder if it doesn't exist
    if (!file.exists(output_folder)){
      dir.create(output_folder)
    }
  }
}
