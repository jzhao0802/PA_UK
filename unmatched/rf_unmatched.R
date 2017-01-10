# ------------------------------------------------------------------------------
#
#                Nested unmatched CV with random forest
#
# ------------------------------------------------------------------------------

library(mlr)
library(readr)
library(parallel)
library(parallelMap)
library(ggplot2)

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

# Set seed and ensure results are reproducible even with parallelization, see 
# here: https://github.com/mlr-org/mlr/issues/938
set.seed(123, "L'Ecuyer")

# Load breast cancer dataset and var_config
df <- readr::read_csv("data/breast_cancer.csv")
var_config <- readr::read_csv("data/breast_cancer_var_config2.csv")

# RF can handle categorical variables, so we'll keep those as well
source("palab_model/palab_model.R")
ids <- get_ids(df, var_config)
df <- get_variables(df, var_config, categorical=T)

# If missing values are present, impute them with median or method="mean"
# df <- impute_data(df, target, method="median")

# Define target variable column
target = "Class"

# ------------------------------------------------------------------------------
# Setup dataset and randomForest in mlR
# ------------------------------------------------------------------------------

# Setup the classification task in mlR, explicitely define positive class
dataset <- makeClassifTask(id="BreastCancer", data=df, target=target, positive=1)

# Downsample number of observations to 50%, preserving the class imbalance
# dataset <- downsample(dataset, perc = .5, stratify=T)

# Check summary of dataset and frequency of classes
dataset
get_class_freqs(dataset)

# Define random Forest, we use the fastes available implementation see:
# https://arxiv.org/pdf/1508.04409.pdf
lrn <- makeLearner("classif.ranger", predict.type="prob", predict.threshold=0.5)

# Cheaper than OOB/permutation estimation of feature importance
lrn <- setHyperPars(lrn, importance="impurity")

# Make sure we sample according to inverse class frequence
target_f <- as.factor(df[[target]])
target_f_sum <- table(target_f)
inverse_weights <- 1/(target_f_sum/sum(target_f_sum))
# Replace each class with its inverse weight
target_iw <- unlist(lapply(target_f, function(x) inverse_weights[x]))
# Not yet implemented by mlR, but hopefully soon will be, raised issue on github
# lrn <- setHyperPars(lrn, case.weights=target_iw)

# Define range of mtry we will search over
features_n <- sum(dataset$task.desc$n.feat) 
mtry_default <- floor(sqrt(features_n))
# +/-10% from the default value
mtry_range <- .25
mtry_lower <- max(1, round(mtry_default * (1 - mtry_range)))
mtry_upper <- min(features_n, round(mtry_default * (1 + mtry_range)))

# A lot of good advice from here: https://goo.gl/avkcBV
ps <- makeParamSet(
  makeNumericParam("num.trees", lower=100L, upper=2000L, trafo=round),
  makeNumericParam("mtry", lower=mtry_lower, upper=mtry_upper, trafo=round),
  # this depends on the dataset and the size of the positive class
  makeNumericParam("min.node.size", lower=10, upper=100, trafo=round)
)

# ------------------------------------------------------------------------------
# Setup rest of the nested CV in mlR
# ------------------------------------------------------------------------------

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

# Define wrapped learner: this is mlR's way of doing nested CV on a learner
lrn_wrap <- makeTuneWrapper(lrn, resampling=inner, par.set=ps, control=ctrl,
                            show.info=F, measures=m_all)

# ------------------------------------------------------------------------------
# Run training with nested CV
# ------------------------------------------------------------------------------

# Setup parallelization
parallelStartSocket(detectCores(), level="mlr.tuneParams")

# Run nested CV
res <- resample(lrn_wrap, dataset, resampling=outer, models=T,
                extract=getTuneResult, show.info=F, measures=m_all)
parallelStop()

# ------------------------------------------------------------------------------
# Get results summary and all tried parameter combinations
# ------------------------------------------------------------------------------

# Get summary of results with main stats, and best parameters
results <- get_results(res)

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

# Get variable importance for 1st model
o_models[[1]]$variable.importance

# Plot variable importance across outer fold moldes
plot_all_rf_vi(res)

# This is how to predict with the first model
predict(res$models[[1]], dataset)

# ------------------------------------------------------------------------------
# Fit model on all data with average of best params
# ------------------------------------------------------------------------------

best_mean_params <- get_best_mean_param(results, int=T)
lrn_outer <- setHyperPars(lrn, par.vals=best_mean_params)

# Train on the whole dataset and extract model
lrn_outer_trained <- train(lrn_outer, dataset)
lrn_outer_model <-getLearnerModel(lrn_outer_trained, more.unwrap=T)

# Plot regularisation path of the averaged model.
plot_rf_vi(lrn_outer_model, title="Average model")

# Accessing params just like above
lrn_outer_model$variable.importance

# ------------------------------------------------------------------------------
# Check how varying the threshold of the classifier changes performance
# ------------------------------------------------------------------------------

# Setup ggplot theme, increase base_size for larger fonts
theme_set(theme_minimal(base_size=10))
# If you like the default grey theme, then
# theme_set(theme_gray(base_size=10))

# Define performance metrics we want to plot, ppv=precision, tpr=recall
perf_to_plot <- list(fpr, tpr, ppv, mmce)
# Generate the data for the plots
thr_perf <- generateThreshVsPerfData(res$pred, perf_to_plot, aggregate=F)
plotThreshVsPerf(thr_perf)

# ------------------------------------------------------------------------------
# Partial dependence plots
# ------------------------------------------------------------------------------

# Columns that are not the target
all_cols <- colnames(df)[colnames(df) != target]

# Plot the median of the curve of each patient
par_dep_data <- generatePartialDependenceData(lrn_outer_trained, dataset,
                                              all_cols, fun=median)
plotPartialDependence(par_dep_data)

# Plot partial dependence plot for all patients
# par_dep_data <- generatePartialDependenceData(lrn_outer_trained, dataset,
#                                               all_cols, individual=T)
# plotPartialDependence(par_dep_data)

# Alternatively if you have many columns use this to plot into a multipage pdf
# plot_partial_deps(lrn_outer_trained, dataset, cols=all_cols, individual=F, 
#                  output_folder="elasticnet")

# ------------------------------------------------------------------------------
# Generate hyper parameter plots for every pair of hyper parameters
# ------------------------------------------------------------------------------

# Plot a performance metric for each pair of hyper parameter, generates .pdf
plot_hyperpar_pairs(res, "auc.test.mean", trafo=T, output_folder="rf_hypers")