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

# Define nested CV with maching: outer 3-fold, inner 3-fold
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

# ------------------------------------------------------------------------------
# Run training with nested CV
# ------------------------------------------------------------------------------

# Setup parallelization - if you run this on cluster, use at most 2 instead of 
# detectCores() so you don't take up all CPU resources on the server.

parallelStartSocket(detectCores(), level="mlr.tuneParams")
res <- palab_resample(lrn, dataset, ncv, ps, ctrl, m_all, show_info=F)
parallelStop()

# ------------------------------------------------------------------------------
# Get results summary and all tried parameter combinations
# ------------------------------------------------------------------------------

extra <- list("ElapsedTime(secs)"=res$runtime, "RandomSeed"=123)
results <- get_results(res, grid_ps=ps, extra=extra, decimal=5)

# Get detailed results
# results <- get_results(res, detailed=T)

# Get detailed results with the actual tables
# results <- get_results(res, detailed=T, all_measures=T)

# Save all these results into a csv
# results <- get_results(res, detailed=T, all_measures=T, write_csv=T)

# For each outer fold show all parameter combinations with their perf metric
opt_paths <- get_opt_paths(res)

# ------------------------------------------------------------------------------
# Get predictions
# ------------------------------------------------------------------------------

# Get all predicted scores and ground truth for the outer train and test folds
all_preds <- as.data.frame(res$pred)

# Get all predicted scores and ground truth for only the outer test folds
o_test_preds <- get_outer_preds(res, ids=ids)

# ------------------------------------------------------------------------------
# Plot precision-recall curve. Note, it's coming from 3 models
# ------------------------------------------------------------------------------

# If you don't need the ROC curve just set it to FALSE.
plot_pr_curve(res, roc=T)

# ------------------------------------------------------------------------------
# Get models from outer folds and their params and predictions
# ------------------------------------------------------------------------------

# Get tuned models for each outer fold
o_models <- get_models(res)

# Print betas/coefs at the tuned value of lambda for the first model
best_lambda <- results$best_params$s[1]
coef(o_models[[1]], s=best_lambda)

# Plot the regularsation paths for all models, note how different the 3 s
plot_reg_path_glmnet(res, n_feat="all")

# If the labels of variables are too crowded change to n_feat=5
plot_reg_path_glmnet(res, n_feat=5)

# This is how to predict with the first model
predict(res$models[[1]], dataset)

# ------------------------------------------------------------------------------
# Fit model on all data with average of best params
# ------------------------------------------------------------------------------

best_mean_params <- get_best_mean_param(results)
lrn_outer <- setHyperPars(lrn, par.vals=best_mean_params)

# Train on the whole dataset and extract model
lrn_outer_trained <- train(lrn_outer, dataset)
lrn_outer_model <-getLearnerModel(lrn_outer_trained, more.unwrap=T)

# Plot regularisation path of the averaged model.
plotmo::plot_glmnet(lrn_outer_model, s=best_mean_params$s, main="Average model")

# Accessing params just like above, note that the mean s is over-regularising
coef(lrn_outer_model, s=best_mean_params$s)

# ------------------------------------------------------------------------------
# Check how varying the threshold of the classifier changes performance
# ------------------------------------------------------------------------------

# Setup ggplot theme, increase base_size for larger fonts
theme_set(theme_minimal(base_size=10))
# If you like the default grey theme, then
# theme_set(theme_gray(base_size=10))

# Define performance metrics we want to plot, ppv=precision, tpr=recall
perf_to_plot <- list(fpr, tpr, ppv, mmce)
# Generate the data for the plots, do aggregate=T if you want the mean
thr_perf <- generateThreshVsPerfData(res$pred, perf_to_plot, aggregate=F)
plotThreshVsPerf(thr_perf)

# ------------------------------------------------------------------------------
# Partial dependence plots
# ------------------------------------------------------------------------------

# Columns that are not the target
all_cols <- colnames(df)[colnames(df) != target]

# Plot median of the curve of each patient for 1st outer models and average model
par_dep_data <- generatePartialDependenceData(res$models[[1]], dataset,
                                              all_cols, fun=median)
plotPartialDependence(par_dep_data)

# Note only 3 predictors remain which are completely linear and not as above
par_dep_data2 <- generatePartialDependenceData(lrn_outer_trained, dataset,
                                               all_cols, fun=median)
plotPartialDependence(par_dep_data2)

# Plot partial dependence plot for all patients
# par_dep_data <- generatePartialDependenceData(res$models[[1]], dataset,
#                                                 all_cols, individual=T)
# plotPartialDependence(par_dep_data)

# Alternatively if you have many columns use this to plot into a multipage pdf
# plot_partial_deps(lrn_outer_trained, dataset, cols=all_cols, individual=F, 
#                  output_folder="elasticnet")

# Fit linear model to each plot and return the beta, i.e. slope
get_par_dep_plot_slopes(par_dep_data, decimal=5)

# Plot them to easily see the influence of each variable
plot_par_dep_plot_slopes(par_dep_data, decimal=5)

# ------------------------------------------------------------------------------
# Generate hyper parameter plots for every pair of hyper parameters
# ------------------------------------------------------------------------------

# Plot a performance metric for each pair of hyper parameter, generates .pdf
plot_hyperpar_pairs(res, "auc.test.mean", output_folder="elasticnet")