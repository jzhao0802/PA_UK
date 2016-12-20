# ------------------------------------------------------------------
# NESTED CV WITH MATCHED CLASSIFICATION DATA
# ------------------------------------------------------------------

library(mlr)
library(readr)
library(parallel)
library(parallelMap)
library(ggplot2)

# set seed and ensure results are reproducible even with parallelization, see 
# here: https://github.com/mlr-org/mlr/issues/938
set.seed(123, "L'Ecuyer")

# load Breast Cancer dataset, impute missing
data(BreastCancer, package="mlbench")
df <- BreastCancer
target <- "Class"

# make sure that the negative class is 0 and the positive is 1, otherwise the
# custom prec@recal perf metric will not work
df[[target]] <- as.factor(as.numeric(factor(df[[target]]))-1)
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

# fit elastic net with nested matched CV on a random grid
ps <- makeParamSet(
  makeNumericParam("alpha", lower=0, upper=1),
  makeNumericParam("lambda", lower=-3, upper=3, trafo=function(x) 10^x)
)

# define Random search grid
ctrl <- makeTuneControlRandom(maxit=50L)

# define learner
lrn <- makeLearner("classif.glmnet", predict.type="prob")

# to make life easier, let's just use the first 27 rows with 9 positives
df <- df[1:27,]

# ------------------------------------------------------------------------------

## Generate the Measure object
if (!exists("make_custom_pr_measure", mode="function")) source("prec_at_recall.R")
pr5 = make_custom_pr_measure(5, "pr5")
pr10 = make_custom_pr_measure(10, "pr10")
m = list(pr5, pr10, auc)

# ------------------------------------------------------------------------------
# make random linkage between malignant and benign samples, each pos has 2 neg
id <- 1:nrow(df)
match <- id
target_col <- df[[target]]
pos_ix <- which(target_col == "malignant")
neg_ix <- base::setdiff(id, pos_ix)
match[neg_ix] = rep(match[pos_ix], 10)[1:length(neg_ix)]
match_df = data.frame(id, match)

# load matching cv creator function: outer 3-fold, inner 3-fold
if (!exists("nested_cv_matched_ix", mode="function")) source("matching.R")
ncv <- nested_cv_matched_ix(match_df, outer_fold_n=3, inner_fold_n=3, 
                            shuffle=F)

tune_outer_fold <- function(ncv, data, target, i){
  
  get_inner_folds <- function(train_fold_ncv){
    # This function takes in a nested cv matched dataframe, and generates an mlR
    # CV resampling instance, then overwrites the indices with the predefined
    # indices in train_fold_ncv. 
    
    # make resampling object for inner fold
    inner_fold_n <- max(train_fold_ncv$inner_fold)
    inner <- makeResampleDesc("CV", iter=inner_fold_n)
    inner_sampling <- makeResampleInstance(inner, size=nrow(train_fold_ncv))
    
    # mlR uses indices and not rownames to define the CV folds. Therefore When 
    # we index subset the full ncv dataframe to get the samples for a particular 
    # outer train fold, the ids don't correspond to the subsetted data anymore 
    # (the first line is not necessarily id=1, but id=4 for example). We correct 
    # this by overwriting the id col with a 1:nrow(train_foldg_ncv). This is 
    # fine because the matching is already done and preserved across inner folds
    train_fold_ncv$id = 1:nrow(train_fold_ncv)
    
    # overwrite the predefined mlR resampling indices, with ncv indices
    for (i in 1:inner_fold_n){
      test_ix = which(train_fold_ncv$inner_fold == i)
      train_ix = which(train_fold_ncv$inner_fold != i) 
      test_ids = train_fold_ncv$id[test_ix]
      train_ids = train_fold_ncv$id[train_ix]
      inner_sampling$test.inds[[i]] = test_ids
      inner_sampling$train.inds[[i]] = train_ids
    }
    inner_sampling
  }
  
  # define test and train datasets in the outer fold and print them
  test_fold_ncv <- ncv[ncv$outer_fold == i,]
  test_fold_ids <- test_fold_ncv$id
  test_data <- makeClassifTask(id="tes", data=data[test_fold_ids,], 
                               target=target)
  train_fold_ncv <- ncv[ncv$outer_fold != i,]
  train_fold_ids <- train_fold_ncv$id
  train_data <- makeClassifTask(id="tr", data=data[train_fold_ids,], 
                                target=target)
  
  # get mlR resampling instance overwritten with predefined indices
  inner_sampling <- get_inner_folds(train_fold_ncv)
  
  # tune parameters with nested CV, preserving matching
  lrn_inner <- tuneParams(lrn, train_data, resampling=inner_sampling, 
                          par.set=ps, control=ctrl, show.info=FALSE, 
                          measures=m)
  
  # make learner with best params and predict test data
  lrn_outer = setHyperPars(makeLearner("classif.glmnet", predict.type="prob"), 
                           par.vals=lrn_inner$x)
  
  # train on the whole outer train set
  lrn_outer_trained = train(lrn_outer, train_data)
  test_pred = predict(lrn_outer_trained, task = test_data)
  
  return(list("lrn_inner" = lrn_inner, "test_pred" = test_pred))
  
}

# for nested CV with matching we will perform the outer CV manually
parallelStartSocket(detectCores(), level="mlr.tuneParams")
outer_fold_n <- max(ncv$outer_fold)
res <-  list(0)
for (i in 1:outer_fold_n){
  res = list(res, tune_outer_fold(ncv, df, target, i))
  #lrn = setHyperPars(makeLearner("classif.ksvm"), par.vals = res$x)
}


