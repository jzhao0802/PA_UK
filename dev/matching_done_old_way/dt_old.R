# ------------------------------------------------------------------------------
#
#                  Nested unmatched CV with decision trees
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
var_config_file = "data/breast_cancer_var_config2.csv"

# Important variables that will make it to the result file
random_seed <- 123
recall_thrs <- 10
random_search_iter <- 50L

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

# RF can handle categorical variables, so we'll keep those as well
source("palab_model/palab_model.R")
ids <- get_ids(df, var_config)
df <- get_variables(df, var_config, categorical=T)

# If missing values are present, impute them with median or method="mean"
# df <- impute_data(df, target, method="median")

# Define target variable column
target = "Class"

# ------------------------------------------------------------------------------
# Setup dataset and decision tree in mlR
# ------------------------------------------------------------------------------

# Setup the classification task in mlR, explicitely define positive class
dataset <- makeClassifTask(id="BC", data=df, target=target, positive=1)

# Downsample number of observations to 50%, preserving the class imbalance
# dataset <- downsample(dataset, perc = .5, stratify=T)

# Check summary of dataset and frequency of classes
dataset
get_class_freqs(dataset)

# Define decision tree, using the rpart package
lrn <- makeLearner("classif.rpart", predict.type="prob", predict.threshold=0.5)

# Make sure we sample according to inverse class frequency 
pos_class_w <- get_class_freqs(dataset)
iw <- unlist(lapply(getTaskTargets(dataset), function(x) 1/pos_class_w[x]))
dataset$weights <- as.numeric(iw)

ps <- makeParamSet(
  # this depends on the dataset and the size of the positive class
  makeNumericParam("minsplit", lower=10L, upper=100L, trafo=round),
  makeNumericParam("maxdepth", lower=2, upper=10, trafo=round)
)

# ------------------------------------------------------------------------------
# Setup rest of the nested CV in mlR
# ------------------------------------------------------------------------------

# Define random grid search with 100 interation per outer fold. Tune.threshold=T
# tunes the classifier's decision threshold but it takes forever -> downsample.
ctrl <- makeTuneControlRandom(maxit=random_search_iter, tune.threshold=F)

# Define performane metrics
pr10 <- make_custom_pr_measure(recall_thrs, "pr10")
m2 <- auc
m3 <- setAggregation(pr10, test.sd)
m4 <- setAggregation(auc, test.sd)
# It's always the first in the list that's used to rank hyper-params in tuning.
m_all <- list(pr10, m2, m3, m4)

if (!matching){
  # Define outer and inner resampling strategies
  outer <- makeResampleDesc("CV", iters=3, stratify=T, predict = "both")
  
  # The inner could be "Subsample" if we don't have enough positive samples
  inner <- makeResampleDesc("CV", iters=3, stratify=T)
  
  # Define wrapped learner: this is mlR's way of doing nested CV on a learner
  lrn_wrap <- makeTuneWrapper(lrn, resampling=inner, par.set=ps, control=ctrl,
                              show.info=F, measures=m_all)
}

# ------------------------------------------------------------------------------
# Run training with nested CV
# ------------------------------------------------------------------------------

# Setup parallelization - if you run this on cluster, use at most 2 instead of 
# detectCores() so you don't take up all CPU resources on the server.
parallelStartSocket(detectCores(), level="mlr.tuneParams")

if (matching){
  res <- palab_resample(lrn, dataset, ncv, ps, ctrl, m_all, show_info=T)
}else{
  res <- resample(lrn_wrap, dataset, resampling=outer, models=T,
                  extract=getTuneResult, show.info=F, measures=m_all)  
}

parallelStop()

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

# Get summary of results with main stats, and best parameters
results <- get_results(res, grid_ps=ps, extra=extra, decimal=5)

# Get detailed results
# results <- get_results(res, grid_ps=ps, extra=extra, detailed=T)

# Get detailed results with the actual tables
# results <- get_results(res, grid_ps=ps, extra=extra, detailed=T, 
#                        all_measures=T)

# Save all these results into a csv
# results <- get_results(res, grid_ps=ps, extra=extra, detailed=T, 
#                        all_measures=T, write_csv=T)

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

plot_pr_curve(res)

# ------------------------------------------------------------------------------
# Get models from outer folds and plot their variable importances
# ------------------------------------------------------------------------------

# Get tuned models for each outer fold
o_models <- get_models(res)

# Get percentage VI table with direction of association as correlation
get_vi_table(o_models[[1]], dataset)

# Plot variable importance across outer fold moldes, for mean do aggregate=T
plot_all_rf_vi(res, aggregate=F)

# Alternatively here's a simpler plot
plot_all_rf_vi_simple(res)

# Find out where the actual splits happened
o_models[[1]]$splits

# Plot the decision tree of the first outer fold, very basic
plot_dt(o_models[[1]], pretty=F)

# Plot fancy decision tree from 1st outer fold, requires a few packages: rattle,
# rpart.plot, GTK for windows, etc, but it's worth it
plot_dt(o_models[[1]], pretty=T)

# Get rules from tree of the first outer fold
library(rattle)
get_dt_rules(o_models[[1]])

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

# Get percentage VI table with direction of association as correlation
get_vi_table(lrn_outer_model, dataset)

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

# Plot them to easily see the influence of each variable, p-vals are on the bars
plot_par_dep_plot_slopes(par_dep_data, decimal=5)

# ------------------------------------------------------------------------------
# Generate hyper parameter plots for every pair of hyper parameters
# ------------------------------------------------------------------------------

# Plot a performance metric for two hyper parameters, generates .pdf
plot_hyperpar_pairs(res, ps, "pr10.test.mean", output_folder="dt_hypers")