# ------------------------------------------------------------------------------
#
#                           Nested CV with SVM
#
# ------------------------------------------------------------------------------

library(mlr)
library(readr)
library(parallel)
library(parallelMap)
library(ggplot2)
source("palab_model/palab_model.R")

# ------------------------------------------------------------------------------
# Define main varaibles
# ------------------------------------------------------------------------------

# Matching or no matching
matching = FALSE

# Define dataset and var_config paths
if (matching){
  data_file = "data/breast_cancer_matched.csv"
  # data_file = "data/breast_cancer_matched_clustering.csv"
}else{
  data_file = "data/breast_cancer.csv"
  # data_file = "data/breast_cancer_clustering.csv"
}
var_config_file = "data/breast_cancer_var_config.csv"

# Important variables that will make it to the result file
random_seed <- 123
recall_thrs <- 10
random_search_iter <- 50L

# Define output folder and create it - if it doesn't exist
output_folder = "svm"
create_output_folder(output_folder)

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

# Set seed and ensure results are reproducible even with parallelization, see 
# here: https://github.com/mlr-org/mlr/issues/938
set.seed(random_seed, "L'Ecuyer")

# Load breast cancer dataset and var_config
df <- readr::read_csv(data_file)  
var_config <- readr::read_csv(var_config_file)

# Get the matching information from the df
if (matching){
  matches <- as.factor(df$match)
}

# Make sure to only retain the numerical columns
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
if(matching){
  dataset <- makeClassifTask(id="BC", data=df, target=target, positive=1, 
                             blocking=matches)
}else{
  dataset <- makeClassifTask(id="BC", data=df, target=target, positive=1)
}

# Downsample number of observations to 50%, preserving the class imbalance
# dataset <- downsample(dataset, perc = .5, stratify=T)

# Check summary of dataset and frequency of classes
dataset
get_class_freqs(dataset)

# Define SVM with RBF kernel. 
lrn <- makeLearner("classif.ksvm", predict.type="prob", predict.threshold=0.5, 
                   par.vals = list(kernel = "rbfdot"))

# Wrap our learner so it will randomly downsample the majority class
lrn <- makeUndersampleWrapper(lrn)

# Define hyper parameters
ps <- makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -5, upper = 5, trafo = function(x) 2^x),
  # add downsampling ratio to the hyper-param grid
  makeNumericParam("usw.rate", lower=.5, upper=1)
)

# Define SVM with polynomial kernel. 
# lrn <- makeLearner("classif.ksvm", predict.type="prob", predict.threshold=0.5, 
#                    par.vals = list(kernel = "ploydot"))

# Define hyper parameters
# ps <- makeParamSet(
#   makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x),
#   makeDiscreteParam("degree", values = 2:5)
# )

# Define random grid search with 100 interation per outer fold. Tune.threshold=T
# tunes the classifier's decision threshold in inner folds but it takes forever
ctrl <- makeTuneControlRandom(maxit=random_search_iter, tune.threshold=F)

# Define performane metrics
pr10 <- make_custom_pr_measure(recall_thrs, "pr10")
m2 <- auc
m3 <- setAggregation(pr10, test.sd)
m4 <- setAggregation(auc, test.sd)
# It's always the first in the list that's used to rank hyper-params in tuning.
m_all <- list(pr10, m2, m3, m4)

# Define outer and inner resampling strategies
outer <- makeResampleDesc("CV", iters=3, stratify=T, predict = "both")

# The inner could be "Subsample" if we don't have enough positive samples
inner <- makeResampleDesc("CV", iters=3, stratify=T)

# If we have matching then stratification is done implicitely through matching
if (matching){
  outer$stratify <- FALSE
  inner$stratify <- FALSE
}

# Define wrapped learner: this is mlR's way of doing nested CV on a learner
lrn_wrap <- makeTuneWrapper(lrn, resampling=inner, par.set=ps, control=ctrl,
                            show.info=F, measures=m_all)

# ------------------------------------------------------------------------------
# Run training with nested CV
# ------------------------------------------------------------------------------

# Setup parallelization - if you run this on cluster, use at most 2 instead of 
# detectCores() so you don't take up all CPU resources on the server.
parallelStartSocket(detectCores(), level="mlr.tuneParams")

res <- resample(lrn_wrap, dataset, resampling=outer, models=T,
                extract=getTuneResult, show.info=F, measures=m_all)  

parallelStop()

# Save results, models and everything as one .rds
readr::write_rds(res, file.path(output_folder, "all_results.rds"))

# ------------------------------------------------------------------------------
# Get results summary and all tried parameter combinations
# ------------------------------------------------------------------------------

# Define extra parameters that we want to save in the results
extra <- list("Matching"=as.character(matching),
              "NumSamples"=dataset$task.desc$size, 
              "NumFeatures"=sum(dataset$task.desc$n.feat),
              "ElapsedTime(secs)"=res$runtime, 
              "RandomSeed"=random_seed, 
              "Recall"=recall_thrs, 
              "IterationsPerFold"=random_search_iter)

# Save all these results into a csv. If output_csv="", current timestamp is used
results <- get_results(res, grid_ps=ps, extra=extra, detailed=T, write_csv=T, 
                       output_folder=output_folder, output_csv="results.csv")

# Get detailed results, with more decimal places
# results <- get_results(res, grid_ps=ps, extra=extra, detailed=T, decimal=10)

# Get detailed results with the all the available metric tables
# results <- get_results(res, grid_ps=ps, extra=extra, detailed=T,
#                        all_measures=T)

# For each outer fold show all parameter combinations with their perf metric
opt_paths <- get_opt_paths(res)
readr::write_csv(opt_paths, file.path(output_folder, "opt_paths.csv"))

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
plot_pr_curve(res$pred)
plot_roc_curve(res$pred)

# Plot any performance curve - we plot the inverse roc in this example
plot_perf_curve(res$pred, x_metric="tpr", y_metric="fpr", bin_num=1000)

# Get a summary of any perf curve as a table - here we get 20 points of the PR
pr <- binned_perf_curve(res$pred, x_metric="rec", y_metric="prec", bin_num=20)
readr::write_csv(pr$curve, file.path(output_folder, "binned_pr.csv"))

# ------------------------------------------------------------------------------
# Get models from outer folds and their params and predictions
# ------------------------------------------------------------------------------

# Get tuned models for each outer fold
o_models <- get_models(res)

# Get support vectors, note the @ notation, this is because ksvm uses S4 objects
o_models[[1]]@alpha

# Get index of resulting support vectors
o_models[[1]]@alphaindex

# Get coefficients, note: unless we use linear kernel these aren't really usful,
# and cannot be interpretted in a straighforward way.
o_models[[1]]@coef

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

# Plot a PR and ROC curve for this new model
pred_outer <- predict(lrn_outer_trained, dataset)
plot_pr_curve(pred_outer)
plot_roc_curve(pred_outer)

# ------------------------------------------------------------------------------
# Check how varying the threshold of the classifier changes performance
# ------------------------------------------------------------------------------

# Setup ggplot theme, increase base_size for larger fonts
theme_set(theme_minimal(base_size=10))
# If you like the default grey theme, then
# theme_set(theme_gray(base_size=10))

# Let's give 5 times the weight to FN compared to FP, rows=truth, cols=pred
costs = matrix(c(0, 5, 1, 0), 2)
colnames(costs) = rownames(costs) = getTaskClassLevels(dataset)
cost_measure = makeCostMeasure(id="cost_measure", name="5FN=1FP", costs=costs, 
                               best=0, worst=5)

# Define performance metrics we want to plot, ppv=precision, tpr=recall
perf_to_plot <- list(fpr, tpr, ppv, cost_measure)

# Generate the data for the plots, do aggregate=T if you want the mean
thr_perf <- generateThreshVsPerfData(res$pred, perf_to_plot, aggregate=F)
plotThreshVsPerf(thr_perf)

# Find out at which threshold we maximise a given perf metric
tuneThreshold(pred=res$pred, measure=pr10)

# ------------------------------------------------------------------------------
# Partial dependence plots
# ------------------------------------------------------------------------------

# Columns that are not the target
all_cols <- colnames(df)[colnames(df) != target]

# Plot median of the curve of each patient for 1st outer model and average model
par_dep_data <- generatePartialDependenceData(res$models[[1]], dataset,
                                              all_cols, fun=median)
plotPartialDependence(par_dep_data)

# Plot partial dependence plot for all patients
# par_dep_data <- generatePartialDependenceData(res$models[[1]], dataset,
#                                                 all_cols, individual=T)
# plotPartialDependence(par_dep_data)

# Alternatively if you have many columns use this to plot into a multipage pdf
# plot_partial_deps(lrn_outer_trained, dataset, cols=all_cols, individual=F, 
#                  output_folder="elasticnet")

# Fit linear model to each plot and return the beta, i.e. slope
get_par_dep_plot_slopes(par_dep_data, decimal=5)

# Plot them to easily see the influence of each variable, p-vals are on the bars
plot_par_dep_plot_slopes(par_dep_data, decimal=5)

# ------------------------------------------------------------------------------
# Generate hyper parameter plots for every pair of hyper parameters
# ------------------------------------------------------------------------------

# Plot a performance metric for two hyper parameters, generates .pdf
plot_hyperpar_pairs(res, ps, "pr10.test.mean", output_folder=output_folder, 
                    indiv_pars=F)