library(PRROC)
library(mlR)
make_custom_pr_measure <- function(recall_perc=10, name_str="pr10"){
  
  find_prec_at_recall <- function(test_pred, recall_perc=10){
    library(PRROC)
    # This function takes in a prediction output from a trained mlR learner. It
    # extracts the predicitons, and finds the highest precision at a given
    # percentage of recall. 
    
    # see docs here: https://cran.r-project.org/web/packages/PRROC/PRROC.pdf
    # the positive class has to be 1, and the negative has to be 0.
    d = test_pred$data
    scores = d[,3]
    labels = as.numeric(factor(d[,2]))-1
    pr <- pr.curve(scores.class0=scores, weights.class0=labels, curve = T)
    
    # extract recall and precision from the curve of PRROC package's result
    recall = pr$curve[,1]
    prec = recall = pr$curve[,2]
    
    # find closes recall value(s)
    target_recall = recall_perc/100
    recall_diff <- abs(recall - target_recall)
    # find indice for this (these)
    recall_min_ix <- which(recall_diff == min(recall_diff))
    # find corresponding highest precision
    prec[recall_min_ix]  
  }
  
  custom_measure = makeMeasure(
    id = name_str, name = "Precision at a given percentage of recal",
    properties = c("classif", "req.prob", "req.truth"),
    minimize = FALSE, best = 1, worst = 0,
    extra.args = list("threshold" = recall_perc),
    fun = function(task, model, pred, feats, extra.args) {
      find_prec_at_recall(pred, extra.args$threshold)
    }
  )
  custom_measure
}