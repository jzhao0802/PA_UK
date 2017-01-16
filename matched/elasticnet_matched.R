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

# Setup the classification task in mlR, explicitely define positive class
dataset <- makeClassifTask(id="BreastCancer", data=df, target=target, positive=1)

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

# Define performane metrics
pr10 <- make_custom_pr_measure(10, "pr10")
m2 <- auc
m3 <- setAggregation(pr10, test.sd)
m4 <- setAggregation(auc, test.sd)
# It's always the first in the list that's used to rank hyper-params in tuning.
m_all <- list(pr10, m2, m3, m4)

res <- palab_resample(lrn, dataset, ncv, ps, ctrl, m_all, show_info=F)

