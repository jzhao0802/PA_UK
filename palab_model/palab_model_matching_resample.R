
# ------------------------------------------------------------------------------
# OVERWRITE MLR RESAMPLING WITH PREDEFINED CV FOLDS
# ------------------------------------------------------------------------------

get_matched_cv_folds <- function(ncv, fold="inner_fold"){
  # This function takes in a nested cv matched dataframe, and generates an mlR
  # CV resampling instance, then overwrites the indices with the predefined
  # indices in train_fold_ncv. 
  
  # Make resampling object for inner fold
  fold_n <- max(ncv[fold])
  if (fold == "inner_fold"){
    cv_desc <- makeResampleDesc("CV", iter=fold_n)  
  }else{
    cv_desc <- makeResampleDesc("CV", iter=fold_n, predict="both")
  }
  cv_inst <- makeResampleInstance(cv_desc, size=nrow(ncv))
  
  # mlR uses indices and not rownames to define the CV folds. Therefore When 
  # we index subset the full ncv dataframe to get the samples for a particular 
  # outer train fold, the ids don't correspond to the subsetted data anymore 
  # (the first line is not necessarily id=1, but id=4 for example). We correct 
  # this by overwriting the id col with a 1:nrow(train_foldg_ncv). This is 
  # fine because the matching is already done and preserved across inner folds
  ncv$id = 1:nrow(ncv)
  
  # Overwrite the predefined mlR resampling indices, with ncv indices
  for (i in 1:fold_n){
    test_ix = which(ncv[fold] == i)
    train_ix = which(ncv[fold] != i) 
    cv_inst$test.inds[[i]] = ncv$id[test_ix]
    cv_inst$train.inds[[i]] = ncv$id[train_ix]
  }
  cv_inst
}

# ------------------------------------------------------------------------------
# HIDDEN HELPER FUNCTIONS FROM MLR COPIED IN HERE
# ------------------------------------------------------------------------------

getTaskFactorLevels <-  function(task) {
  cols = vlapply(task$env$data, is.factor)
  lapply(task$env$data[cols], levels)
}

perfsToString = function(y, sep = "=") {
  stri_paste(stri_paste(names(y), "=", formatC(y, digits = 3L), sep = ""),
             collapse = ",", sep = " ")
}

measureAggrName = function(measure) {
  stri_paste(measure$id, measure$aggr$id, sep = ".")
}

makeResamplePrediction = function(instance, preds.test, preds.train) {
  library(data.table)
  tenull = sapply(preds.test, is.null)
  trnull = sapply(preds.train, is.null)
  if (any(tenull)) pr.te = preds.test[!tenull] else pr.te = preds.test
  if (any(trnull)) pr.tr = preds.train[!trnull] else pr.tr = preds.train
  
  data = setDF(rbind(
    rbindlist(lapply(seq_along(pr.te), function(X) 
      cbind(pr.te[[X]]$data, iter = X, set = "test"))),
    rbindlist(lapply(seq_along(pr.tr), function(X) 
      cbind(pr.tr[[X]]$data, iter = X, set = "train")))
  ))
  
  if (!any(tenull) && instance$desc$predict %in% c("test", "both")) {
    p1 = preds.test[[1L]]
    pall = preds.test
  } else if (!any(trnull) && instance$desc$predict == "train") {
    p1 = preds.train[[1L]]
    pall = preds.train
  }
  
  
  makeS3Obj(c("ResamplePrediction", class(p1)),
            instance = instance,
            predict.type = p1$predict.type,
            data = data,
            threshold = p1$threshold,
            task.desc = p1$task.desc,
            time = extractSubList(pall, "time")
  )
}

exportMlrOptions = function(level) {
  .mlr.slave.options = getMlrOptions()
  parallelExport(".mlr.slave.options", level=level, master=F, show.info=F)
}

# ------------------------------------------------------------------------------
# TUNE OUTER FOLD WITH MATCHING AND PREDEFINED CV
# ------------------------------------------------------------------------------

tune_outer_fold <- function(ncv, learner, task, i, ps, ctrl, measures, 
                            show_info=F){
  # This function is a modified replicate of mlR's doResampleIteration function
  # in the resample.R. We do this basically so we end up with the results having
  # the same classes/structure/etc so we can use everything else from the other
  # non-matchinh scripts and the rest of mlR. For implementation details check:
  # https://github.com/mlr-org/mlr/blob/master/R/resample.R
  
  library(stringi)
  
  # Define test and train datasets in the outer fold and print them
  test_fold_ncv <- ncv[ncv$outer_fold == i,]
  test_fold_ids <- test_fold_ncv$id
  test_data <- subsetTask(task, subset=test_fold_ids)
  train_fold_ncv <- ncv[ncv$outer_fold != i,]
  train_fold_ids <- train_fold_ncv$id
  train_data <- subsetTask(task, subset=train_fold_ids)
  
  # Get mlR resampling instance overwritten with predefined indices
  inner_cv <- get_matched_cv_folds(train_fold_ncv)
  
  # Tune parameters with nested CV, preserving matching
  time1 <- Sys.time()
  lrn_inner <- tuneParams(learner, train_data, resampling=inner_cv, 
                          par.set=ps, control=ctrl, show.info=FALSE, 
                          measures=measures)
  time2 <- Sys.time()
  runtime <- as.numeric(difftime(time2, time1, units="mins"))
  # Make learner with best params and predict test data
  lrn_outer <- setHyperPars(learner, par.vals=lrn_inner$x)
  
  # Train on the whole outer train set
  lrn_outer <- train(lrn_outer, train_data)
  
  # Define variables holding results
  err_msgs = c(NA_character_, NA_character_)
  if (isFailureModel(lrn_outer))
    err_msgs[1L] = getFailureModelMsg(lrn_outer)
  
  ms_train = rep(NA, length(measures))
  ms_test = rep(NA, length(measures))
  pred_train = NULL
  pred_test = NULL
  
  # Predict train
  pred_train = predict(lrn_outer, task=task, subset=train_fold_ids)
  if (!is.na(pred_train$error)){
    err_msgs[2L] = pred_train$error
  }
  ms_train = performance(task=task, model=lrn_outer, pred=pred_train, 
                         measures=measures)
  names(ms_train) = vcapply(measures, measureAggrName)
  
  # Predict test
  pred_test = predict(lrn_outer, task=task, subset=test_fold_ids)
  if (!is.na(pred_test$error)){
    err_msgs[2L] = paste(err_msgs[2L], pred_test$error)
  }
  ms_test = performance(task=task, model=lrn_outer, pred=pred_test, 
                        measures=measures)
  names(ms_test) = vcapply(measures, measureAggrName)
  
  # Show.info
  if (show_info) {
    idx_train = which(vlapply(measures, 
                              function(x) "req.train" %in% x$aggr$properties))
    idx_test = which(vlapply(measures, 
                             function(x) "req.test" %in% x$aggr$properties))
    x = c(ms_train[idx_train], ms_test[idx_test])
    messagef(perfsToString(x))
  }
  
  # Compile results
  list(
    measures_test = ms_test,
    measures_train = ms_train,
    model = lrn_outer,
    pred_test = pred_test,
    pred_train = pred_train,
    err_msgs = err_msgs,
    extract = lrn_inner
  )
}

# ------------------------------------------------------------------------------
# MERGE RESULTS FROM TUNE OUTER MODELS
# ------------------------------------------------------------------------------

merge_outer_models <- function(learner, task, results, measures, outer_cv, 
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
  pred = makeResamplePrediction(instance = outer_cv, preds.test = pred_test,
                                preds.train = pred_train)
  
  # Aggregate measures
  aggr = vnapply(seq_along(measures), function(i) {
    m = measures[[i]]
    m$aggr$fun(task, ms_test[, i], ms_train[, i], m, outer_cv$group, pred)
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
# REPLICATE MLR'S NESTED RESAMPLING FUNCTION WITH SAMPLE MATCHING
# ------------------------------------------------------------------------------

palab_resample <- function(learner, task, ncv, ps, ctrl, measures, show_info=F){
  # This function is mimicing the nested cv of mlR. This is usually done with
  # mlR's resample method. Here we do the same but with matching as defined 
  # by the ncv dataframe - see palab_model_matching for details.
  
  # Define extra params for parallelized execution
  args = list(ncv=ncv, learner=learner, task=task, measures=measures, ps=ps, 
              ctrl=ctrl, show_info=show_info)
  
  # Generate outer CV object with predefined indices
  outer_fold_n <- max(ncv$outer_fold)
  outer_cv <- get_matched_cv_folds(ncv, "outer_fold")
  
  # TODO: palab_resample with mlr.resample parallel option breaks the code
  # parallelLibrary("mlr", master=F, level="mlr.resample", show.info=F)
  # exportMlrOptions(level = "mlr.resample")
  
  # Start measuring time, and do outer loop of nested CV in parallel
  time1 = Sys.time()
  results = parallelMap(tune_outer_fold, seq_len(outer_fold_n), 
                        level="mlr.resample", more.args=args)
  time2 = Sys.time()
  runtime = as.numeric(difftime(time2, time1, units="secs"))
  
  # Merge results into mlR's ResampleResult data structure
  merged_results <- merge_outer_models(learner, task, results, measures, 
                                       outer_cv, runtime)
  addClasses(merged_results, "ResampleResult")
}