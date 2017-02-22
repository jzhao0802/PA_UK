# ------------------------------------------------------------------------------
#
#  Function incorporating clustering between inner and outer loop of nested cv
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Hidden helper functions from mlr copied in here
# ------------------------------------------------------------------------------

getTaskFactorLevels <-  function(task) {
  cols <- vlapply(task$env$data, is.factor)
  lapply(task$env$data[cols], levels)
}

getTaskWeights = function(task) {
  task$weights
}

perfsToString <- function(y, sep="=") {
  stri_paste(stri_paste(names(y), "=", formatC(y, digits=3L), sep=""),
             collapse=",", sep=" ")
}

measureAggrName <- function(measure) {
  stri_paste(measure$id, measure$aggr$id, sep=".")
}

makeResamplePrediction <- function(instance, preds.test, preds.train) {
  library(data.table)
  tenull <- sapply(preds.test, is.null)
  trnull <- sapply(preds.train, is.null)
  if (any(tenull)) pr.te <- preds.test[!tenull] else pr.te=preds.test
  if (any(trnull)) pr.tr <- preds.train[!trnull] else pr.tr=preds.train
  
  data <- setDF(rbind(
    rbindlist(lapply(seq_along(pr.te), function(X) 
      cbind(pr.te[[X]]$data, iter=X, set="test"))),
    rbindlist(lapply(seq_along(pr.tr), function(X) 
      cbind(pr.tr[[X]]$data, iter=X, set="train")))
  ))
  
  if (!any(tenull) && instance$desc$predict %in% c("test", "both")) {
    p1 <- preds.test[[1L]]
    pall <- preds.test
  } else if (!any(trnull) && instance$desc$predict == "train") {
    p1 <- preds.train[[1L]]
    pall <- preds.train
  }
  
  makeS3Obj(c("ResamplePrediction", class(p1)),
            instance=instance,
            predict.type=p1$predict.type,
            data=data,
            threshold=p1$threshold,
            task.desc=p1$task.desc,
            time=extractSubList(pall, "time")
  )
}

exportMlrOptions=function(level) {
  .mlr.slave.options=getMlrOptions()
  parallelExport(".mlr.slave.options", level=level, master=F, show.info=F)
}

setSlaveOptions = function() {
  if (getOption("parallelMap.on.slave", FALSE)) {
    if (exists(".mlr.slave.options", envir = .GlobalEnv)) {
      opts = get(".mlr.slave.options", envir = .GlobalEnv)
      Map(setMlrOption, names(opts), opts)
    }
  }
}

# ------------------------------------------------------------------------------
# Tune outer fold with matching and predefined cv
# ------------------------------------------------------------------------------

tune_outer_fold <- function(learner, task, rin, i, weights, measures, 
                            show_info=F, cluster="negatives", cluster_test=F, 
                            ...){
  # This function is a modified replicate of mlR's doResampleIteration function
  # in the resample.R. We do this basically so we end up with the results having
  # the same classes/structure/etc so we can use everything else from the other
  # non-matchinh scripts and the rest of mlR. For implementation details check:
  # https://github.com/mlr-org/mlr/blob/master/R/resample.R
  
  library(stringi)
  setSlaveOptions()
  if (show_info)
    messagef("[Resample] %s iter %i: ", rin$desc$id, i)
  
  # Define test and train datasets in the outer fold 
  train_i = rin$train.inds[[i]]
  test_i = rin$test.inds[[i]]
  test_data <- subsetTask(task, subset=test_i)
  train_data <- subsetTask(task, subset=train_i)
  
  # Downsample train and test data using the user-defined strategy
  if (cluster == "negatives"){
    train_data <- cluster_negatives(train_data, ...)
    if (cluster_test)
      test_data <- cluster_negatives(test_data, ...)
  }else{
    train_data <- cluster_positives(train_data, ...)
    if (cluster_test)
      test_data <- cluster_negatives(test_data, ...)
  }
  
  # Tune parameters with nested CV, preserving matching
  err_msgs = c(NA_character_, NA_character_)
  m = train(learner, task, subset = train_i, weights = weights[train_i])
  if (isFailureModel(m))
    err_msgs[1L] = getFailureModelMsg(m)
  
  ms_train = rep(NA, length(measures))
  ms_test = rep(NA, length(measures))
  pred_train = NULL
  pred_test = NULL
  pp = rin$desc$predict
  train_task = task
  
  lm = getLearnerModel(m)
  if ("BaseWrapper" %in% class(learner) && !is.null(lm$train.task)) {
    # the learner was wrapped in a sampling wrapper
    train_task = lm$train.task
    train_i = lm$subset
  }
  
  # Predict train
  pred_train = predict(m, train_task, subset = train_i)
  if (!is.na(pred_train$error)) err_msgs[2L] = pred_train$error
  ms_train = performance(task=task, model=m, pred=pred_train, measures=measures)
  names(ms_train) = vcapply(measures, measureAggrName)
  
  # Predict test
  pred_test = predict(m, task, subset = test_i)
  if (!is.na(pred_test$error)) err_msgs[2L] = paste(err_msgs[2L], pred_test$error)
  ms_test = performance(task=task, model=m, pred=pred_test, measures=measures)
  names(ms_test) = vcapply(measures, measureAggrName)
  
  # Show info
  if (show_info) {
    idx_train <- which(vlapply(measures, 
                              function(x) "req.train" %in% x$aggr$properties))
    idx_test <- which(vlapply(measures, 
                             function(x) "req.test" %in% x$aggr$properties))
    x <- c(ms_train[idx_train], ms_test[idx_test])
    messagef(perfsToString(x))
  }
  
  # Compile results
  list(
    measures_test=ms_test,
    measures_train=ms_train,
    model=m,
    pred_test=pred_test,
    pred_train=pred_train,
    err_msgs=err_msgs,
    extract=getTuneResult(m)
  )
}

# ------------------------------------------------------------------------------
# Merge results from tune outer models
# ------------------------------------------------------------------------------

merge_outer_models <- function(learner, task, results, measures, rin, 
                               runtime) {
  # This is a simplified version of mergeResampleResult function in mlR's 
  # resample.R. It merges results from nested cv models using matching. It only
  # work with results from tune_outer_fold.
  
  # Collect measures
  iters = length(results)
  mids = vcapply(measures, function(m) m$id)
  ms_train = as.data.frame(extractSubList(results, "measures_train", 
                                          simplify="rows"))
  ms_test = extractSubList(results, "measures_test", simplify = FALSE)
  ms_test = as.data.frame(do.call(rbind, ms_test))
  
  # Get aggregated predictions
  pred_test = extractSubList(results, "pred_test", simplify = FALSE)
  pred_train = extractSubList(results, "pred_train", simplify = FALSE)
  pred = makeResamplePrediction(instance = rin, preds.test = pred_test,
                                preds.train = pred_train)
  
  # Aggregate measures
  aggr = vnapply(seq_along(measures), function(i) {
    m = measures[[i]]
    m$aggr$fun(task, ms_test[, i], ms_train[, i], m, rin$group, pred)
  })
  names(aggr) = vcapply(measures, measureAggrName)
  
  # Rename ms.* rows and cols
  colnames(ms_test) = mids
  rownames(ms_test) = NULL
  ms_test = cbind(iter = seq_len(iters), ms_test)
  colnames(ms_train) = mids
  rownames(ms_train) = NULL
  ms_train = cbind(iter = seq_len(iters), ms_train)
  
  err_msgs = as.data.frame(extractSubList(results, "err_msgs", 
                                          simplify="rows"))
  rownames(err_msgs) = NULL
  colnames(err_msgs) = c("train", "predict")
  err_msgs = cbind(iter = seq_len(iters), err_msgs)
  
  # Compile results - in the format mlR would return from nested CV results
  list(
    learner.id = learner$id,
    task.id = getTaskId(task),
    task.desc = getTaskDescription(task),
    measures.train = ms_train,
    measures.test = ms_test,
    aggr = aggr,
    pred = pred,
    models = lapply(results, function(x) x$model),
    err.msgs = err_msgs,
    extract = extractSubList(results, "extract", simplify = FALSE),
    runtime = runtime
  )
}

# ------------------------------------------------------------------------------
# Replicate mlr's nested resampling function WITH MATCHING
# ------------------------------------------------------------------------------

palab_downsample_clustering_resample = function(learner, task, resampling, measures, 
                                     weights=NULL, show_info=T, 
                                     cluster="negatives", ...) {
  
  # This is an altered version of the mlr function that does resampling. Here we
  # downsample the negatives by either clustering the positives or negatives.

  library(checkmate)
  
  n = getTaskSize(task)
  # instantiate resampling
  if (inherits(resampling, "ResampleDesc"))
    resampling = makeResampleInstance(resampling, task = task)
  assertClass(resampling, classes = "ResampleInstance")
  if (!is.null(weights)) {
    assertNumeric(weights, len = n, any.missing = FALSE, lower = 0)
  }
  
  r = resampling$size
  if (n != r)
    stop(stri_paste("Size of data set:", n, "and resampling instance:", r, 
                    "differ!", sep = " "))
  
  rin = resampling
  more.args = list(learner=learner, task=task, rin=rin, weights=NULL,
                   measures=measures, show_info=show_info, ...)
  if (!is.null(weights)) {
    more.args$weights = weights
  } else if (!is.null(getTaskWeights(task))) {
    more.args$weights = getTaskWeights(task)
  }
  
  #parallelLibrary("mlr", master=F, level="mlr.resample", show.inf=F)
  #exportMlrOptions(level="mlr.resample")
  time1 = Sys.time()
  results = parallelMap(tune_outer_fold, seq_len(rin$desc$iters), 
                        level = "mlr.resample", more.args=more.args)
  time2=Sys.time()
  runtime=as.numeric(difftime(time2, time1, units="secs"))
  
  merged_results <- merge_outer_models(learner, task, results, measures, rin, 
                                        runtime)
  addClasses(merged_results, "ResampleResult")
}