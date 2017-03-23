# ------------------------------------------------------------------------------
#
#             Utils functions of the modelling part of PAlab
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(ROCR)

# ------------------------------------------------------------------------------
# Function to create custom precision at x% recall in mlR
# ------------------------------------------------------------------------------

make_custom_pr_measure <- function(recall_perc=5, name_str="pr5"){
  
  find_prec_at_recall <- function(pred, recall_perc=5){
    # This function takes in a prediction output from a trained mlR learner. It
    # extracts the predicitons, and finds the highest precision at a given
    # percentage of recall. 
    
    # see docs here: https://cran.r-project.org/web/packages/PRROC/PRROC.pdf
    # the positive class has to be 1, and the negative has to be 0.
    prob = pred$prob_1
    truth = pred$label
    recall_target <- recall_perc/100

    # this part is adopted Hui's code 
    aucobj <- ROCR::prediction(prob, truth)
    
    # generate the recall and ppv and threshold
    prec_rec <- ROCR::performance(aucobj, 'prec', 'rec')
    rec <- prec_rec@x.values[[1]]
    prec <- prec_rec@y.values[[1]]
    thresh <- prec_rec@alpha.values[[1]]
    
    # ignore nans
    rec <- rec[!is.nan(prec)]
    prec <- prec[!is.nan(prec)]
    thresh <- thresh[!is.nan(prec)]
    
    # find closest recall value to target and return corresponding prec value
    recall_diff <- abs(rec - recall_target)
    # Return prec that corresponds to the threshold closest to recall target
    prec[which.min(recall_diff)]
  }
  
  name <- paste("Precision at ", as.character(recall_perc),"%"," recall", sep='')
  
  custom_measure = makeMeasure(
    id = name_str, 
    name = name,
    properties = c("classif", "req.prob", "req.truth"),
    minimize = FALSE, best = 1, worst = 0,
    extra.args = list("threshold" = recall_perc),
    fun = function(task, model, pred, feats, extra.args) {
      find_prec_at_recall(pred, extra.args$threshold)
    }
  )
  custom_measure
}

# ------------------------------------------------------------------------------
# Function to create custom precision at x% recall in mlR
# ------------------------------------------------------------------------------

make_custom_pr_measure_old <- function(recall_perc=5, name_str="pr5"){
  
  find_prec_at_recall2 <- function(pred, recall_perc=5){
    # This function takes in a prediction output from a trained mlR learner. It
    # extracts the predicitons, and finds the highest precision at a given
    # percentage of recall. 
    
    # see docs here: https://cran.r-project.org/web/packages/PRROC/PRROC.pdf
    # the positive class has to be 1, and the negative has to be 0.
    
    prob = pred$prob_1
    truth = pred$label
    
    # Create desc sorted table of probs and truth
    df <- data.frame(truth=truth, prob=prob)
    df_pos <- df[which(df$truth == 1),]
    df_pos <- BBmisc::sortByCol(df_pos, "prob", asc=F)
    pos_N <- nrow(df_pos)
    
    # Find the right threshold for x% recall, by walking through the probs in
    # the df_pos table and using each as a thrsh to calculate recall
    recall_tmp <- 0
    thrsh_tmp <- 0
    ix <- 1
    recall_target <- recall_perc/100
    while (recall_tmp < recall_target){
      # To make sure we pick the thrsh_tmp that leads us the closest to the 
      # desired recall level
      recall_tmp2 <- recall_tmp
      thrsh_tmp2 <- thrsh_tmp
      # Threshold we'll try
      thrsh_tmp <- df_pos$prob[ix]
      # Predictions that this threshold translates to
      # !!! - For some reason profiler says this is the slowest step - !!!
      pred_tmp <- as.numeric(df$prob >= thrsh_tmp)
      # Calculate true positive rate = recall
      recall_tmp <- sum(pred_tmp)/pos_N
      ix <- ix + 1
    }
    
    # Two closest recall levels
    recall_tmps <- c(recall_tmp, recall_tmp2)
    # Two corresponding thresholds
    thrsh_tmps <- c(thrsh_tmp, thrsh_tmp2)
    recall_diff <- abs(recall_tmps - recall_target)
    # Threshold to use in precision calculation
    thrsh <- thrsh_tmps[which(recall_diff == min(recall_diff))]
    # Find precision at this threshold
    tp <- sum(df$truth[df$prob >= thrsh])
    pred_n <- sum(df$prob >= thrsh)
    tp/pred_n
  }
  
  name <- paste("Precision at ", as.character(recall_perc),"%"," recall", sep='')
  
  custom_measure = makeMeasure(
    id = name_str, 
    name = name,
    properties = c("classif", "req.prob", "req.truth"),
    minimize = FALSE, best = 1, worst = 0,
    extra.args = list("threshold" = recall_perc),
    fun = function(task, model, pred, feats, extra.args) {
      find_prec_at_recall(pred, extra.args$threshold)
    }
  )
  custom_measure
}
