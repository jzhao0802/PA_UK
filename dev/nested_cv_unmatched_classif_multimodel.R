# ------------------------------------------------------------------
# NESTED CV WITH UNMATCHED CLASSIFICATION DATA
# ------------------------------------------------------------------

library(mlr)
library(readr)
library(parallelMap)
library(ggplot2)

# load Breast Cancer dataset, impute missing
data(BreastCancer, package="mlbench")
df <- BreastCancer
target <- "Class"
df$Id <- NULL

# impute missing values and define classification task
impute_col_median <- function(x){
  # Retruns a single column where NA is replaced with median
  x <- as.numeric(as.character(x))
  x[is.na(x)] <- median(x, na.rm=TRUE)
  x 
}

data_without_target <- function(df, target){
  # Return df without the target column
  df[,-which(colnames(df)==target)]
}

impute_data <- function(df, target){
  # Impute missing values in cols with median except in target
  df_imp <- data_without_target(df, target)
  df_imp <- as.data.frame(sapply(df_imp, impute_col_median))
  df[,-which(colnames(df)==target)] <- df_imp
  df
}

df <- impute_data(df, target)
classif.task <- makeClassifTask(id="BreastCancer", data=df, target=target)

# define Random search grid
ctrl <- makeTuneControlRandom(maxit=100L, tune.threshold=TRUE)
inner <- makeResampleDesc("CV", iters=3)

# define perf metrics
m1 <- auc
m2 <- setAggregation(auc, test.sd)
if (!exists("make_custom_pr_measure", mode="function")) source("prec_at_recall.R")
pr5 = make_custom_pr_measure(5, "pr5")
pr10 = make_custom_pr_measure(10, "pr10")
m_all <- list(m1, m2, pr5, pr10)

# define learners
lrn <- makeLearner("classif.glmnet", predict.type="prob")

base.learners = list(
  makeLearner("classif.glmnet", predict.type="prob"),
  makeLearner("classif.ksvm", predict.type="prob"),
  makeLearner("classif.randomForest", predict.type="prob")
)
lrn = makeModelMultiplexer(base.learners)

ps = makeModelMultiplexerParamSet(lrn,
  makeNumericParam("sigma", lower = -12, upper = 12, trafo = function(x) 2^x),
  makeNumericParam("alpha", lower=0, upper=1),
  makeNumericParam("lambda", lower=-3, upper=3, trafo=function(x) 10^x),
  makeIntegerParam("ntree", lower = 100L, upper = 1000L)
)

lrn_wrap <- makeTuneWrapper(lrn, resampling=inner, par.set=ps, control=ctrl,
                            show.info=FALSE, measures=m_all)

# run nested cv
outer <- makeResampleDesc("CV", iters=3)

parallelStartSocket(8, level="mlr.tuneParams")

r <- resample(lrn_wrap, classif.task, resampling=outer, models=TRUE,
              extract=getTuneResult, show.info=FALSE)
parallelStop()

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