# ------------------------------------------------------------------
# NESTED CV WITH UNMATCHED REGRESSION DATA
# ------------------------------------------------------------------

library(mlr)
library(readr)
library(parallelMap)
library(ggplot2)

# load subgroup data, exclude ID from data
# input <- "data/subgroup_data_cleared.csv"
# target <- "Diff_12monVA_IdxVA"
# data <- readr::read_csv(input, guess_max=8000)
# data <- data[-"ID",]
# define regression task from out dataset
# regr.task <- makeRegrTask(id="subgroup", data=data, target=target)

# load Boston Housing dataset from mlbench
data(BostonHousing, package="mlbench")
target="medv"
regr.task <- makeRegrTask(id="bh", data=BostonHousing, target=target)

# we'll fit an elasticnet regression. 
# to find out about the params we can tune do:
# makeLearner("regr.glmnet")$par.set
# define parameter grid we want to search over
ps <- makeParamSet(
  # alpha=0 is ridge, alpha=1 is lasso
  makeDiscreteParam("alpha", values=seq(0, 1, by=.25)),
  # lambda is the penalty for the sum of L1 and L2 norms
  makeDiscreteParam("lambda", values=10^(-3:3))
)

# define grid-search strategy, for random search use makeTuneControlRandom()
# there are very good practical (comp cost) and theoretical reasons to why
# prefer RS: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

ctrl <- makeTuneControlRandom()

# define inner CV strategy, for resampling replace "CV" with "Subsample"
inner <- makeResampleDesc("CV", iters=5)

# define the measure we want to collect (mean test mse and its sd), for a full 
# list see:
# https://mlr-org.github.io/mlr-tutorial/release/html/measures/index.html
m1 <- mse
# we calculate the SD of the mse by setting up an aggregation, for a full list 
# see: https://rdrr.io/cran/mlr/man/aggregations.html
m2 <- setAggregation(mse, test.sd)
m_all <- list(m1, m2)

# define wrapped learner (mlR way of saying nested CV learner). In this eaxample 
# we use a gradient boosting trees in this example,  for all learners see:
# https://mlr-org.github.io/mlr-tutorial/release/html/integrated_learners/index.html
lrn <- makeTuneWrapper("regr.glmnet", resampling=inner, par.set=ps, 
                       control=ctrl, show.info=FALSE, measures=m_all)

# define outer CV strategy
outer <- makeResampleDesc("CV", iters=3)

# make a parallel environment for gridsearch
parallelStartSocket(8, level="mlr.tuneParams")

# run nested CV
r <- resample(lrn, regr.task, resampling=outer, models=TRUE,
             extract=getTuneResult, show.info=FALSE)
parallelStop()

# print mse on the outer test fold
r$measures.test

# print mean mse on the inner folds of the best params
r$extract

# for each outerfold show all parameter combination with mean 
# and sd mse over the inner folds
opt_paths <- getNestedTuneResultsOptPathDf(r)

# visualise the search paths
g <- ggplot(opt_paths, aes(x=alpha, y=lambda, fill=mse.test.mean))
g + geom_tile() + facet_wrap(~ iter)

# restrict the same plot to the lowest 50% of the values
low_c <- as.list(quantile(opt_paths$mse.test.mean, probs=c(0, .5)))[[1]]
high_c <- as.list(quantile(opt_paths$mse.test.mean, probs=c(0, .5)))[[2]]
g <- ggplot(opt_paths, aes(x=alpha, y=lambda, fill=mse.test.mean))
g + geom_tile() + facet_wrap(~ iter) + 
  scale_fill_gradient(limits=c(low_c, high_c), low="red", high="grey")

# get best parameters for each outer fold
getNestedTuneResultsX(r)

# get predicted scores
pred_scores <- as.data.frame(r$pred)

# predict with one of the returned models
# mlr::predictLearner(lrn, r$models[[1]], data[,-which(colnames(data)==target)])
