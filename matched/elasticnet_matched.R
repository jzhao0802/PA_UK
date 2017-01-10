# ------------------------------------------------------------------------------
#
#     Nested matched CV with elastic net penalized logistic regression
#
# ------------------------------------------------------------------------------

library(mlr)
library(readr)
library(parallel)
library(parallelMap)
library(ggplot2)
library(plotmo)

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

# Set seed and ensure results are reproducible even with parallelization, see 
# here: https://github.com/mlr-org/mlr/issues/938
set.seed(123, "L'Ecuyer")

# Load breast cancer dataset and var_config
df <- readr::read_csv("data/breast_cancer.csv")
var_config <- readr::read_csv("data/breast_cancer_var_config.csv")

# To make life easier, let's just use the first 27 rows with 9 positives
df <- df[1:27,]

# Make sure to only retain the numerical columns
source("palab_model/palab_model.R")
ids <- get_ids(df, var_config)
df <- get_variables(df, var_config)

# If missing values are present, impute them with median or method="mean"
# df <- impute_data(df, target, method="median")

# Define target variable column
target = "Class"


# ------------------------------------------------------------------------------
# Make random linkage between malignant and benign samples, each pos has 2 neg
# ------------------------------------------------------------------------------

id <- 1:nrow(df)
match <- id
target_col <- df[[target]]
pos_ix <- which(target_col == 1)
neg_ix <- which(target_col == 0)
match[neg_ix] = rep(match[pos_ix], 10)[1:length(neg_ix)]
match_df = data.frame(id, match)

# Load matching cv creator function: outer 3-fold, inner 3-fold
source("palab_model/palab_model_matching.R")
ncv <- nested_cv_matched_ix(match_df, outer_fold_n=3, inner_fold_n=3, shuffle=F)

# Check matching and nested CV indicies
match_df
ncv

# ------------------------------------------------------------------------------
# Setup modelling in mlR
# ------------------------------------------------------------------------------

# Setup the classification task in mlR
dataset <- makeClassifTask(id="BreastCancer", data=df, target=target)

# Downsample number of observations to 50%, preserving the class imbalance
# dataset <- downsample(dataset, perc = .5, stratify=T)

# Check summary of dataset and frequency of classes
dataset
get_class_freqs(dataset)

# Define logistic regression with elasticnet penalty
lrn <- makeLearner("classif.glmnet", predict.type="prob", predict.threshold=0.5)

# Find max lambda as suggested here: 
# https://github.com/mlr-org/mlr/issues/1030#issuecomment-233677172
tmp_model <- train(lrn, dataset)
max_lambda <- max(tmp_model$learner.model$lambda)

# Define hyper parameters
ps <- makeParamSet(
  # for lasso delete the alpha from the search space and set it (see below)
  makeNumericParam("alpha", lower=0, upper=1),
  makeNumericParam("s", lower=0, upper=max_lambda*2)
)

# For Lasso penalty do:
# lrn <- setHyperPars(lrn, alpha=1)

# Define random grid search with 100 interation per outer fold.
ctrl <- makeTuneControlRandom(maxit=50L, tune.threshold=F)

# Define outer and inner resampling strategies
outer <- makeResampleDesc("CV", iters=3, stratify=T, predict = "both")

# The inner could be "Subsample" if we don't have enough positive samples
inner <- makeResampleDesc("CV", iters=3, stratify=T)

# Define performane metrics
pr10 <- make_custom_pr_measure(10, "pr10")
m2 <- auc
m3 <- setAggregation(pr10, test.sd)
m4 <- setAggregation(auc, test.sd)
# It's always the first in the list that's used to rank hyper-params in tuning.
m_all <- list(pr10, m2, m3, m4)

# ------------------------------------------------------------------------------
# Tune outer folds
# ------------------------------------------------------------------------------

tune_outer_fold <- function(ncv, data, target, i, positive=1){
  
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
    
    # Overwrite the predefined mlR resampling indices, with ncv indices
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
  test_data <- makeClassifTask(id="test", data=data[test_fold_ids,], 
                               target=target, positive=positive)
  train_fold_ncv <- ncv[ncv$outer_fold != i,]
  train_fold_ids <- train_fold_ncv$id
  train_data <- makeClassifTask(id="train", data=data[train_fold_ids,], 
                                target=target, positive=positive)
  
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


