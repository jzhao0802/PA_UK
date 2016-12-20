# ------------------------------------------------------------------------------
#
#                Nested unmatched CV with classification data
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
var_config <- readr::read_csv("data/breast_cancer_var_config.csv")

# Make sure to only retain the numerical columns
source("utils.R")
df <- get_numerical_variables(df, var_config)

# Define target variable column
target = "Class"

# ------------------------------------------------------------------------------
# Setup modelling in mlR
# ------------------------------------------------------------------------------

# Setup the classification task in mlR
classif.task <- makeClassifTask(id="BreastCancer", data=df, target=target)

# Define hyper parameters
ps <- makeParamSet(
  makeNumericParam("alpha", lower=0, upper=1),
  makeNumericParam("s", lower=0, upper=2)
)

# Define random grid search with 100 interation per outer fold.
ctrl <- makeTuneControlRandom(maxit=100L, tune.threshold=T)

# Define outer and inner resampling strategies
outer <- makeResampleDesc("CV", iters=3)

# The inner could be "Subsample" if we don't have enough positive samples
inner <- makeResampleDesc("CV", iters=3)

# Define performane metrics
m1 <- make_custom_pr_measure(5, "pr5")
m2 <- auc
m3 <- setAggregation(m1, test.sd)
m4 <- setAggregation(auc, test.sd)
m_all <- list(m1, m2, m3, m4)

# Define logistic regression with elasticnet penalty
lrn <- makeLearner("classif.glmnet", predict.type="prob")

# Define wrapped learner: this is mlR's way of doing nested CV on a learner
lrn_wrap <- makeTuneWrapper(lrn, resampling=inner, par.set=ps, control=ctrl,
                            show.info=FALSE, measures=m_all)

# ------------------------------------------------------------------------------
# Run training with nested CV
# ------------------------------------------------------------------------------

# Setup parallelization
parallelStartSocket(detectCores(), level="mlr.tuneParams")

# Run nested CV
r <- resample(lrn_wrap, classif.task, resampling=outer, models=TRUE,
              extract=getTuneResult, show.info=FALSE, measures=m_all)
parallelStop()

# ------------------------------------------------------------------------------
# Get results
# ------------------------------------------------------------------------------

# print mse on the outer test fold
r$measures.test

# print mean mse on the inner folds of the best params
r$extract

# for each outerfold show all parameter combination with mean 
# and sd mse over the inner folds
opt_paths <- getNestedTuneResultsOptPathDf(r)

# get best parameters for each outer fold
getNestedTuneResultsX(r)

# get predicted scores
pred_scores <- as.data.frame(r$pred)

# get models
mods=lapply(r$models, function(x) getLearnerModel(x,more.unwrap=T))

# ------------------------------------------------------------------------------
# Generate hyper parameter plots
# ------------------------------------------------------------------------------

# predict data with the first model
df_no_target <-  df[,-which(colnames(df)==target)]
mlr::predictLearner(lrn_wrap, r$models[[1]], df_no_target)

# plot the two params with a random search
resdata = generateHyperParsEffectData(r, trafo = F, include.diagnostics = FALSE)
plt = plotHyperParsEffect(resdata, x = "alpha", y = "lambda", 
                          z = "auc.test.mean",plot.type = "heatmap", 
                          interpolate = "regr.earth", show.experiments = T, 
                          nested.agg = mean, facet = "nested_cv_run")
min_plt = min(resdata$data$auc.test.mean, na.rm = TRUE)
max_plt = max(resdata$data$auc.test.mean, na.rm = TRUE)
med_plt = mean(c(min_plt, max_plt))
plt + scale_fill_gradient2(breaks = seq(min_plt, max_plt, length.out = 5),
                           low = "blue", mid = "white", high = "red", 
                           midpoint = med_plt)

# plot partial dependece plots - only works with the development branch of mlR
resdata = generateHyperParsEffectData(r, partial.dep = TRUE)
plotHyperParsEffect(resdata, x = "alpha", y = "auc.test.mean", 
                    plot.type = "line", partial.dep.learn = "regr.randomForest")

# ------------------------------------------------------------------------------
# Generate variable importance plots
# ------------------------------------------------------------------------------