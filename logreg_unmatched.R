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
var_config <- readr::read_csv("data/breast_cancer_var_config.csv")

# Make sure to only retain the numerical columns
source("palab_model.R")
df <- get_variables(df, var_config)

# If missing values are present, impute them with median or method="mean"
# df <- impute_data(df, target, method="median")

# Define target variable column
target = "Class"

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
lrn <- makeLearner("classif.logreg", predict.type="prob", predict.threshold=0.5)

# Define outer and inner resampling strategies
outer <- makeResampleDesc("CV", iters=3, stratify=T, predict = "both")

# Define performane metrics
pr5 <- make_custom_pr_measure(5, "pr5")
m2 <- auc
m3 <- setAggregation(pr5, test.sd)
m4 <- setAggregation(auc, test.sd)
# It's always the first in the list that's used to rank hyper-params in tuning.
# We tune auc here, because partial dependence plots are non-sensical with pr5.
m_all <- list(m2, pr5, m3, m4)

# ------------------------------------------------------------------------------
# Run training with nested CV
# ------------------------------------------------------------------------------

# Setup parallelization
parallelStartSocket(detectCores(), level="mlr.tuneParams")

# Run nested CV
res <- resample(lrn, dataset, resampling=outer, models=T, show.info=F, 
                measures=m_all)
parallelStop()

# ------------------------------------------------------------------------------
# Get results summary and all tried parameter combinations
# ------------------------------------------------------------------------------

# Get summary of results with main stats, and best parameters
results <- get_non_nested_results(res)

# Get detailed results
# results <- get_non_nested_results(res, detailed=T)

# Get detailed results with the actual tables
# results <- get_non_nested_results(res, detailed=T, all_measures=T)

# Save all these results into a csv
# results <- get_non_nested_results(res, detailed=T, all_measures=T, write_csv=T)

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

# Print model output for the first outer fold model
summary(o_models[[1]])

# Print odds ratios and CIs
get_odds_ratios(o_models[[1]])

# Plot model (residuals, fitted, leverage)
plot(o_models[[1]])

# This is how to predict with the first model
predict(res$models[[1]], dataset)

# ------------------------------------------------------------------------------
# Fit model on all data with average of best params
# ------------------------------------------------------------------------------

lrn_outer <- lrn

# Train on the whole dataset and extract model
lrn_outer_trained <- train(lrn_outer, dataset)
lrn_outer_model <-getLearnerModel(lrn_outer_trained, more.unwrap=T)

# Plot residuals etc for model fitted on all data
plot(lrn_outer_model)

# Accessing params just like above
summary(lrn_outer_model)

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
# par_dep_data <- generatePartialDependenceData(lrn_outer_trained, dataset,
#                                               all_cols, individual=T)
# plotPartialDependence(par_dep_data)