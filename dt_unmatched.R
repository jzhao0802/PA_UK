# ------------------------------------------------------------------------------
#
#       Nested unmatched CV with elastic net penalized elastic net
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
var_config <- readr::read_csv("data/breast_cancer_var_config2.csv")

# RF can handle categorical variables, so we'll keep those as well
source("palab_model.R")
df <- get_numerical_variables(df, var_config, categorical=T)

# Define target variable column
target = "Class"

# ------------------------------------------------------------------------------
# Setup dataset and decision tree in mlR
# ------------------------------------------------------------------------------

# Setup the classification task in mlR
dataset <- makeClassifTask(id="BreastCancer", data=df, target=target)

# Define decision tree, using the rpart package
lrn <- makeLearner("classif.rpart", predict.type="prob")

ps <- makeParamSet(
  # this depends on the dataset and the size of the positive class
  makeNumericParam("minsplit", lower=10L, upper=100L, trafo=round),
  makeNumericParam("maxdepth", lower=2, upper=10, trafo=round)
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
pr5 <- make_custom_pr_measure(5, "pr5")
m2 <- auc
m3 <- setAggregation(pr5, test.sd)
m4 <- setAggregation(auc, test.sd)
# It's always the first in the list that's used to rank hyper-params in tuning.
# We tune auc here, because partial dependence plots are non-sensical with pr5.
m_all <- list(m2, pr5, m3, m4)

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
o_test_preds <- all_preds[all_preds$set=="test",]

# ------------------------------------------------------------------------------
# Get models from outer folds and their params and predictions
# ------------------------------------------------------------------------------

# Get tuned models for each outer fold
o_models <- get_models(res)

# Get variable importance for 1st model
o_models[[1]]$variable.importance

# Plot variable importance across outer fold moldes
plot_all_rf_vi(res)

# Find out where the actual splits happened
o_models[[1]]$splits

# Plot the decision tree of the first outer fold, very basic
plot_dt(o_models[[1]], pretty=F)

# Plot fancy decision tree from 1st outer fold, requires a few packages: rattle
plot_dt(o_models[[1]], pretty=T)

# Get rules from tree of the first outer fold
library(rpart.utils)
dt_rules1 <- rpart.rules.table(o_models[[1]])


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
thr_perf <- generateThreshVsPerfData(res$pred, perf_to_plot)
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
# par_dep_data <- generatePartialDependenceData(res$models[[1]], dataset,
#                                               all_cols, individual=T)
# plotPartialDependence(par_dep_data)

# ------------------------------------------------------------------------------
# Generate hyper parameter plots for every pair of hyper parameters
# ------------------------------------------------------------------------------

# Plot a performance metric for each pair of hyper parameter, generates .pdf
plot_hyperpar_pairs(res, "auc.test.mean", trafo=T, output_folder="dt_hypers")