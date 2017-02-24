# ------------------------------------------------------------------------------
#
#  Function incorporating clustering between inner and outer loop of nested cv
#
# ------------------------------------------------------------------------------


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
  mlr:::setSlaveOptions()
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
    print(train_data$task.desc$size)
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
  names(ms_train) = vcapply(measures, mlr:::measureAggrName)
  
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
    messagef(mlr:::perfsToString(x))
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
  pred = mlr:::makeResamplePrediction(instance = rin, preds.test = pred_test,
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

downsample_clustering_resample = function(learner, task, resampling, measures, 
                                          show_info=T, cluster="negatives", 
                                          cluster_test=F, ...) {
  
  # This is an altered version of the mlr function that does resampling. Here we
  # downsample the negatives by either clustering the positives or negatives.

  library(checkmate)
  
  n = getTaskSize(task)
  # instantiate resampling
  if (inherits(resampling, "ResampleDesc"))
    resampling = makeResampleInstance(resampling, task = task)
  assertClass(resampling, classes = "ResampleInstance")
  
  r = resampling$size
  if (n != r)
    stop(stri_paste("Size of data set:", n, "and resampling instance:", r, 
                    "differ!", sep = " "))
  weights = mlr:::getTaskWeights(task)
  rin = resampling
  
  more_args = list(learner=learner, task=task, rin=rin, weights=weights, 
                   measures=measures, show_info=show_info, cluster=cluster, 
                   cluster_test, ...)
  
  parallelLibrary("mlr", master=F, level="mlr.resample", show.info=F)
  mlr:::exportMlrOptions(level="mlr.resample")
  time1 = Sys.time()
  results = parallelMap(tune_outer_fold, seq_len(rin$desc$iters), 
                        level = "mlr.resample", more.args=more_args)
  time2=Sys.time()
  runtime=as.numeric(difftime(time2, time1, units="secs"))
  
  merged_results <- merge_outer_models(learner, task, results, measures, rin, 
                                        runtime)
  addClasses(merged_results, "ResampleResult")
}