# ------------------------------------------------------------------
# NESTED CV WITH MATCHED DATA
# ------------------------------------------------------------------

library(mlr)
library(readr)
library(parallelMap)
library(ggplot2)

# load Breast Cancer dataset, impute missing
data(BreastCancer, package = "mlbench")
df = BreastCancer
df$Id = NULL
impute_median=function(x){
  x <- as.numeric(as.character(x))
  x[is.na(x)] <- median(x, na.rm=TRU)E
  x 
}
df[,-which(colnames(df)=="Class")] <- as.data.frame(sapply(df[,-which(colnames(df)=="Class")], impute_median))
classif.task = makeClassifTask(id = "BreastCancer", data = df, target = "Class")

# make random linkage

# fit elastic net with nested matched CV
ps <- makeParamSet(
  makeDiscreteParam("alpha", values=seq(0, 1, by=.25)),
  makeDiscreteParam("lambda", values=10^(-3:3))
)

ctrl <- makeTuneControlRandom(maxit = 50)
inner <- makeResampleDesc("CV", iters=3)

m1 <- auc
m2 <- setAggregation(auc, test.sd)
m_all <- list(m1, m2)

lrn <- makeLearner("classif.glmnet", predict.type = "prob")
lrn_wrap <- makeTuneWrapper(lrn, resampling=inner, par.set=ps, control=ctrl,
                            show.info=FALSE, measures=m_all)

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

# visualise the search paths
g <- ggplot(opt_paths, aes(x=alpha, y=lambda, fill=auc.test.mean))
g + geom_tile() + facet_wrap(~ iter)

# restrict the same plot to the lowest 50% of the values
low_c <- as.list(quantile(opt_paths$mse.test.mean, probs=c(0, .5)))[[1]]
high_c <- as.list(quantile(opt_paths$mse.test.mean, probs=c(0, .5)))[[2]]
g <- ggplot(opt_paths, aes(x=alpha, y=lambda, fill=auc.test.mean))
g + geom_tile() + facet_wrap(~ iter) + 
  scale_fill_gradient(limits=c(low_c, high_c), low="red", high="grey")

# get best parameters for each outer fold
getNestedTuneResultsX(r)

# get predicted scores
pred_scores <- as.data.frame(r$pred)

# get models
mods = lapply(r$models, function(x) getLearnerModel(x,more.unwrap = T))


