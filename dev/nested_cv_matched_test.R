# ------------------------------------------------------------------
# NESTED CV WITH MATCHED CLASSIFICATION DATA
# ------------------------------------------------------------------

library(mlr)
library(readr)
library(parallelMap)
library(ggplot2)

test_nested_cv_matched <- function(){
  # This function demonstrates nested CV with matching on a real classification
  # dataset. First we load a breast cancer dataset, then define arbitrary
  # matching of the first 27 rows which have 9 positive samples, then we
  # generate mlR resampling object, overwrite it with the predifined CV indices.
  # Finally everything is printed in a nice way so we can check it.
  
  # load Breast Cancer dataset, impute missing
  data(BreastCancer, package="mlbench")
  df <- BreastCancer
  target <- "Class"
  df$Id <- NULL
  
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
    df[,-which(colnames(df) == target)] <- df_imp
    df
  }
  
  df <- impute_data(df, target)

  # to make life easier, let's just use the first 27 rows with 9 positives
  df <- df[1:27,]
  classif.task <- makeClassifTask(id="BreastCancer", data=df, target=target)
  
  # make random linkage between malignant and benign samples, each pos has 2 neg
  id <- 1:nrow(df)
  match <- id
  target_col <- df[[target]]
  pos_ix <- which(target_col == "malignant")
  neg_ix <- base::setdiff(id, pos_ix)
  match[neg_ix] = rep(match[pos_ix], 10)[1:length(neg_ix)]
  match_df = data.frame(id, match)

  print_break_line <- function(){
    cat("-------------------------------------------------------------------\n")
  }
  
  print_small_break_line <- function(){
    cat("---------------------------------------\n")
  } 
  
  add_positive_mark <- function(df){
    pos_ix <- which(df$id==df$match)
    pos = rep('', nrow(df))
    pos[pos_ix] = '*'
    df = dplyr::mutate(df, pos=pos)
    sortByCol(df, c("outer_fold","inner_fold"))
  }
  
  # check if these matchings make sense
  print_break_line()
  cat("match_df dataframe describes the matching of positives with negatives\n")
  print(data.frame(id, match, target_col))
  
  # load matching cv creator function: outer 3-fold, inner 3-fold
  source("palab_matching.R")
  ncv <- nested_cv_matched_ix(match_df, outer_fold_n=3, inner_fold_n=3, 
                              shuffle=F)
  print_break_line()
  cat("ncv dataframe holds the indices for nested cv with matching\n")
  print(ncv)
  
  # for nested CV with matching we will perform the outer CV manually
  print_break_line()
  cat("nested_cv_matched creates mlR resampling object with the correct
      indices, specified by ncv.\n")
  nested_cv_matched <- function(ncv){
    outer_fold_n <- max(ncv$outer_fold)
    for (i in 1:outer_fold_n){
      # define test and train datasets in the outer fold and print them
      test_fold_ncv <- ncv[ncv$outer_fold==i,]
      print_break_line()
      cat("Outer fold number, i.e. test fold:", i, "\n")
      print(add_positive_mark(test_fold_ncv))
      print_break_line()
      cat("Outer-train <- i.e. all other outer folds, inner is run on this\n")
      train_fold_ncv <- ncv[ncv$outer_fold!=i,]
      pos_ix <- which(train_fold_ncv$id==train_fold_ncv$match)
      cat("positive samples", train_fold_ncv$id[pos_ix], "\n")
      print(add_positive_mark(train_fold_ncv))
      
      # make resampling object for inner fold
      inner_fold_n <- max(train_fold_ncv$inner_fold)
      inner <- makeResampleDesc("CV", iter=inner_fold_n)
      inner_sampling <- makeResampleInstance(inner, size=nrow(train_fold_ncv))
      
      # overwrite the predefined mlR resampling indices, with ncv indices
      for (j in 1:inner_fold_n){
        test_ix = which(train_fold_ncv$inner_fold == j)
        train_ix = which(train_fold_ncv$inner_fold != j) 
        test_ids = train_fold_ncv$id[test_ix]
        train_ids = train_fold_ncv$id[train_ix]
        inner_sampling$test.inds[[j]] = test_ids
        inner_sampling$train.inds[[j]] = train_ids
        print_small_break_line()
        cat("Inner fold number:", j, "\n")
        cat("Inner-test\n")
        print(inner_sampling$test.inds[[j]])
        cat("Inner-train\n")
        print(inner_sampling$train.inds[[j]])
      }
    }
  }
  
  # check if the test and train folds are truly the ones we need
  nested_cv_matched(ncv)
}

test_nested_cv_matched()
