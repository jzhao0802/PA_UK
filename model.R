library("dplyr")
library("tidyr")
library("readr")
library("caret")

# ------------------------------------------------------------------
# LOAD AND CLEAN SUBGROUP DATA AND SAVE IT
# ------------------------------------------------------------------

input = "data/subgroup_data.csv"
data = readr::read_csv(input, guess_max = 8000)

# add ID to rownames, and delete cols that make up the unique ID
rownames(data) = data$ID
data[["ID"]] = NULL
data[["Eye_Laterality"]] = NULL
data[["Patient_ID"]] = NULL

# the target is the difference of two measurements, remove the 2nd
data[["VA_12mon"]] = NULL
data[["Perc_12monVA_IdxVA"]] = NULL

# convert sex to numbers
data$Gender = as.numeric(factor(data$Gender))

# convert dates to UNIX timestamps
data$Idx_Dt = as.numeric(as.POSIXct(data$Idx_Dt, format="%d/%m/%Y"))

# let's have a look at the data
dplyr::glimpse(data)

# check the target
target = "Diff_12monVA_IdxVA"
hist(data[[target]])

# check if all input vars are numeric
if(sum(sapply(data, is.numeric)) < dim(data)[2]){
  warning("There are non-numeric columns. These will be removed now.")
  to_remove = -which(!(sapply(data, is.numeric)))
  data = data[,to_remove]
}

#check if there are missing values in the target
if(sum(is.na(data[[target]])) > 0){
  warning("There are missing values in the target column These will be removed now.")
  to_remove = -which(is.na(data[[target]]))
  data = data[to_remove,]
}

# replace NA's with median in each column
impute_median=function(x){
  x = as.numeric(as.character(x))
  x[is.na(x)] = median(x, na.rm=TRUE)
  x 
}
data = dplyr::tbl_df(sapply(data, impute_median))
readr::write_csv(data, "data/subgroup_data_cleared.csv")

# ------------------------------------------------------------------
# SIMPLE LIN REG MODEL
# ------------------------------------------------------------------

# simple linear model
lin_reg = lm(Diff_12monVA_IdxVA ~ ., data)
sink("summary.txt")
summary(lin_reg)
sink()
lin_reg_sum = summary(lin_reg)
write.csv(lin_reg_sum$coefficients, file="summary_coef.csv") 

# ------------------------------------------------------------------
# MLR
# Gradient Boosting Regression Trees with nested CV
# ------------------------------------------------------------------

library(mlr)
library("parallelMap")
# make a parallel environment for gridsearch
parallelStartSocket(8, level = "mlr.resample")

data(BostonHousing, package = "mlbench")
regr.task = makeRegrTask(id = "bh", data = BostonHousing, target = "medv")

ps = makeParamSet(
  makeDiscreteParam("n.trees", values = seq(400,501, by=100)),
  makeDiscreteParam("interaction.depth", values = (2:3))
)
ctrl = makeTuneControlGrid()
inner = makeResampleDesc("Subsample", iters = 10)
lrn = makeTuneWrapper("regr.gbm", resampling = inner, par.set = ps, control = ctrl, show.info = FALSE, measures = list(m1, m2))

## Outer resampling loop
# define the measure we want to collect (mean test mse and it's sd)
m1 = mse
m2 = setAggregation(mse, test.sd)
outer = makeResampleDesc("CV", iters = 3)
r = resample(lrn, regr.task, resampling = outer, extract = getTuneResult, show.info = FALSE)

# print mse on outer test fold
r$measures.test
# print average mse on the inner folds of the best params
r$extract
# for each outerfold show all parameter combination with mean and sd mse
opt.paths = getNestedTuneResultsOptPathDf(r)
# get best parametrs for outerfolds


parallelStop()










