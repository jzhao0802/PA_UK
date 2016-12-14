# ------------------------------------------------------------------
# NESTED CV WITH MATCHED DATA
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

# make random linkage

# fit elastic net with nested matched CV on a random grid
ps <- makeParamSet(
  makeNumericParam("alpha", lower=0, upper=1),
  makeNumericParam("lambda", lower=-3, upper=3, trafo=function(x) 10^x)
)

ctrl <- makeTuneControlRandom(maxit=50L)
inner <- makeResampleDesc("CV", iters=3)

m1 <- auc
m2 <- setAggregation(auc, test.sd)
m3 <- setAggregation(auc, train.sd)
m_all <- list(m1, m2, m3)

lrn <- makeLearner("classif.glmnet", predict.type="prob")

lrn_wrap <- makeTuneWrapper(lrn, resampling=inner, par.set=ps, control=ctrl,
                            show.info=FALSE, measures=m_all)

outer <- makeResampleDesc("CV", iters=3)

parallelStartSocket(8, level="mlr.resample")

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

# visualise the search paths
g <- ggplot(opt_paths, aes(x=alpha, y=lambda, fill=auc.test.mean))
g + geom_tile() + facet_wrap("iter")

# restrict the same plot to the lowest 50% of the values
#low_c <- as.list(quantile(opt_paths$mse.test.mean, probs=c(0, .5)))[[1]]
#high_c <- as.list(quantile(opt_paths$mse.test.mean, probs=c(0, .5)))[[2]]
#g <- ggplot(opt_paths, aes(x=alpha, y=lambda, fill=auc.test.mean))
#g + geom_tile() + facet_wrap(~ iter) + 
#  scale_fill_gradient(limits=c(low_c, high_c), low="red", high="grey")

# get best parameters for each outer fold
getNestedTuneResultsX(r)

# get predicted scores
pred_scores <- as.data.frame(r$pred)

# get models
#mods=lapply(r$models, function(x) getLearnerModel(x,more.unwrap=T))
#mlr::predictLearner(lrn_wrap, r$models[[1]], df[,-which(colnames(df)==target)])



# plot the two params with a random search
resdata = generateHyperParsEffectData(r, trafo = F, include.diagnostics = FALSE)
plt = plotHyperParsEffect(resdata, x = "alpha", y = "lambda", z = "auc.test.mean",
                          plot.type = "heatmap", interpolate = "regr.earth",
                          show.experiments = T, nested.agg = mean, 
                          facet = "nested_cv_run")
min_plt = min(resdata$data$auc.test.mean, na.rm = TRUE)
max_plt = max(resdata$data$auc.test.mean, na.rm = TRUE)
med_plt = mean(c(min_plt, max_plt))
plt + scale_fill_gradient2(breaks = seq(min_plt, max_plt, length.out = 5),
                           low = "blue", mid = "white", high = "red", midpoint = med_plt)

# plot partial dependece plots
resdata = generateHyperParsEffectData(r, partial.dep = TRUE)
plotHyperParsEffect(resdata, x = "alpha", y = "auc.test.mean", plot.type = "line",
                    partial.dep.learn = "regr.randomForest")

# make random linkage for this dataset, with first 100 rows being positives
pos_n = 100
id = 1:nrow(df)
match = c(1:pos_n, sample(c(1:pos_n), nrow(df)-pos_n, replace=T))
match_df = data.frame(id, match)
ncv <- nested_cv_matched(match_df, outer_fold_n=5, inner_fold_n=5, shuffle=F)

# make resampling instance
inner <- makeResampleDesc("CV", iter=5)
inner_sampling <- makeResampleInstance(inner, classif.task)

# override it with custom

# run non-nested cv

# concat results