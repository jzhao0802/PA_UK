# ------------------------------------------------------------------------------
#
#                Clean breast cancer breast cancer dataset
#
# ------------------------------------------------------------------------------

library(readr)

# load Breast Cancer dataset, impute missing
data(BreastCancer, package="mlbench")
df <- BreastCancer
target <- "Class"
df$Id <- NULL

# make sure that the negative class is 0 and the positive is 1, otherwise the
# custom prec@recal perf metric will not work
df[[target]] <- as.factor(as.numeric(factor(df[[target]]))-1)

# impute missing values and define classification task
impute_col_median <- function(x){
  # Retruns a single column where NA is replaced with median
  x <- as.numeric(as.character(x))
  x[is.na(x)] <- median(x, na.rm=TRUE)
  x 
}

data_without_target <- function(df, target){
  # Return df without the target column
  df[,-which(colnames(df)==target)]
}

impute_data <- function(df, target){
  # Impute missing values in cols with median except in target
  df_imp <- data_without_target(df, target)
  df_imp <- as.data.frame(sapply(df_imp, impute_col_median))
  df[,-which(colnames(df)==target)] <- df_imp
  df
}

df <- impute_data(df, target)
df=dplyr::mutate(df, ID=replicate(dim(df)[1], paste(sample(letters, 5), collapse='')))
readr::write_csv(df, "~/PAlab/palab_model/data/breast_cancer.csv")