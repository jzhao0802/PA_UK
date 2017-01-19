# ------------------------------------------------------------------------------
#
#                   Nested CV with logistic regression
#
# ------------------------------------------------------------------------------

library(mlr)
library(readr)
library(parallel)
library(parallelMap)
library(ggplot2)

# ------------------------------------------------------------------------------
# Define main varaibles
# ------------------------------------------------------------------------------

# Matching or no matching
matching = FALSE

# Define dataset and var_config paths
if (matching){
  data_file = "data/breast_cancer_matched.csv"
}else{
  data_file = "data/breast_cancer.csv"
}
var_config_file = "data/breast_cancer_var_config.csv"

# Important variables that will make it to the result file
random_seed <- 123
recall_thrs <- 10

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

# Set seed and ensure results are reproducible even with parallelization, see 
# here: https://github.com/mlr-org/mlr/issues/938
set.seed(random_seed, "L'Ecuyer")

# Load breast cancer dataset and var_config
df <- readr::read_csv(data_file)  
var_config <- readr::read_csv(var_config_file)

if (matching){
  # Build dataframe that holds fold membership of each sample
  ncv <- data.frame(id=1:nrow(df), outer_fold=df$outer_fold, 
                    inner_fold=df$inner_fold)
}

# Make sure to only retain the numerical columns
source("palab_model/palab_model.R")
ids <- get_ids(df, var_config)
df <- get_variables(df, var_config)

# If missing values are present, impute them with median or method="mean"
# df <- impute_data(df, target, method="median")

# Define target variable column
target = "Class"

# ------------------------------------------------------------------------------
# Setup modelling in mlR
# ------------------------------------------------------------------------------

# Setup the classification task in mlR, explicitely define positive class
dataset <- makeClassifTask(id="BC", data=df, target=target, positive=1)

# Downsample number of observations to 50%, preserving the class imbalance
# dataset <- downsample(dataset, perc = .5, stratify=T)

# Check summary of dataset and frequency of classes
dataset
get_class_freqs(dataset)

# Define logistic regression with elasticnet penalty
lrn <- makeLearner("classif.logreg", predict.type="prob", predict.threshold=0.5)

# Make sure glmnet uses the class weights
pos_class_w <- get_class_freqs(dataset)["1"]
lrn <- makeWeightedClassesWrapper(lrn, wcw.weight=pos_class_w)

# Define outer resampling strategies: if matched, use ncv
if (matching){
  outer <- get_matched_cv_folds(ncv, "outer_fold")
}else{
  outer <- makeResampleDesc("CV", iters=3, stratify=T, predict = "both")
}

# Define performane metrics
pr10 <- make_custom_pr_measure(recall_thrs, "pr10")
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

res <- resample(lrn, dataset, resampling=outer, models=T, show.info=F, 
                measures=m_all)

parallelStop()

# ------------------------------------------------------------------------------
# Get results summary and all tried parameter combinations
# ------------------------------------------------------------------------------

# Get summary of results with main stats, and best parameters
# Define extra parameters that we want to save in the results
extra <- list("Matching"=as.character(matching),
              "NumSamples"=dataset$task.desc$size, 
              "NumFeatures"=sum(dataset$task.desc$n.feat),
              "ElapsedTime(secs)"=res$runtime, 
              "RandomSeed"=random_seed, 
              "Recall"=recall_thrs)

# Get summary of results with main stats, and best parameters
results <- get_non_nested_results(res, extra=extra, decimal=5)

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
thr_perf <- generateThreshVsPerfData(res$pred, perf_to_plot, aggregate=F)
plotThreshVsPerf(thr_perf)

# Find out at which threshold we maximise a given perf metric
tuneThreshold(pred=res$pred, measure=pr10)

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

# Fit linear model to each plot and return the beta, i.e. slope
get_par_dep_plot_slopes(par_dep_data, decimal=5)

# Plot them to easily see the influence of each variable
plot_par_dep_plot_slopes(par_dep_data, decimal=5)
