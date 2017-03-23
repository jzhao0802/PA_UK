# ------------------------------------------------------------------------------
#
#      General functions for calculating and plotting performance curves
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(ggplot2)
library(ROCR)

# ------------------------------------------------------------------------------
# Helper functions to calculate, bin, and plot PR and ROC curves
# ------------------------------------------------------------------------------

get_truth_pred <- function(pred){
  # This helper function could not have been used in make_custom_pr_measure()
  # because it was not in scope. 
  positive_class <- pred$task.desc$positive
  prob = getPredictionProbabilities(pred, cl=positive_class)
  
  # get truth and turn it into ones (positive) and zeros (negative)
  truth <- getPredictionTruth(pred)
  if (is.factor(truth)) {
    pos_ix <- as.integer(truth) == which(levels(truth) == positive_class)
  } else {
    pos_ix <- truth == positive_class
  }
  truth <- as.integer(pos_ix)
  
  # Retain only the predictions on the test set if pred is from nested cv
  if ("set" %in% colnames(as.data.frame(pred))){
    df <- as.data.frame(pred)
    truth <- truth[df$set=="test"]
    prob <- prob[df$set=="test"]
  }
  
  results <- list("truth"=truth, "prob"=prob)
  return (results)
}

get_curve <- function(prob, truth, x_metric, y_metric){
  # Calculates roc or pr curve using the ROCR package. Based on Hui's code.
  # Then returns a curve_df with x, y, thresh columns.
  aucobj <- ROCR::prediction(prob, truth)
  perf <- ROCR::performance(aucobj, y_metric, x_metric)
  x <- perf@x.values[[1]]
  y <- perf@y.values[[1]]
  thresh <- perf@alpha.values[[1]]
  
  # ignore nans and inf
  non_nan <- (!is.nan(x) & !is.nan(y) & !is.nan(thresh) & !is.infinite(x) & 
                !is.infinite(y) & !is.infinite(thresh))
  x <- x[non_nan]
  y <- y[non_nan]
  thresh <- thresh[non_nan]
  
  # make and return df
  data.frame(x=x, y=y, thresh=thresh)
}

bin_curve <- function(curve_df, bin_num){
  # Takes in a curve data frame and bins it into bin_num of points. 
  curve_df <- curve_df %>%
    group_by(x_binned=cut(x, breaks = seq(0, 1, by=1/bin_num))) %>%
    filter(x == max(x)) %>%
    filter(y == max(y)) %>%
    arrange(as.numeric(x_binned)) %>%
    select(x, y, thresh, x_binned)
}

auc_curve <- function(curve_df){
  # Linearly interpolates the x, y points of curve_df and returns the integral
  f <- approxfun(curve_df$x, curve_df$y)
  min_x <- min(curve_df$x)
  max_x <-max(curve_df$x)
  integrate(f, min_x, max_x, subdivisions=2000)$value
}

ggplot_perf_curve <- function(data, xlab, ylab, title){
  # Plots the ROC and PR curves using ggplot.
  colnames(data)[3] = "Threshold"
  p <- ggplot(data, aes(x=x,y=y))
  p + geom_line(aes(colour=Threshold), size=1) + 
    scale_colour_gradientn(colors = rainbow(7)) +
    xlab(xlab) +
    ylab(ylab) + 
    ggtitle(title)
}

# ------------------------------------------------------------------------------
# Main functions to plot PR and ROC curve and any arbitrary perf curve
# ------------------------------------------------------------------------------

plot_pr_curve <- function(pred, bin_num=1000){
  theme_set(theme_minimal(base_size=10))
  # get probabilities and truth
  tp <- get_truth_pred(pred)
  # get binned pr curve df - bin_num x, y points
  curve_df <- get_curve(tp$prob, tp$truth, x_metric="rec", y_metric="prec")
  # bin if necessary
  if(length(curve_df$x) > bin_num){
    curve_df <- bin_curve(curve_df, bin_num)
  }
  # pr auc - with linear interpolation - this isn't precise but it's fast
  pr_auc <- auc_curve(curve_df)
  # plot curve
  ggplot_perf_curve(data=curve_df, xlab="TPR/Sens/Recall", ylab="Precision/PPV",
                    title=paste("Area under PR: ", decimal_rounder(pr_auc, 4)))
}

plot_roc_curve <- function(pred, bin_num=1000){
  theme_set(theme_minimal(base_size=10))
  # get probabilities and truth
  tp <- get_truth_pred(pred)
  # get binned roc curve df - bin_num x, y points
  curve_df <- get_curve(tp$prob, tp$truth, x_metric="fpr", y_metric="tpr")
  # bin if necessary
  if(length(curve_df$x) > bin_num){
    curve_df <- bin_curve(curve_df, bin_num)
  }
  # roc auc - with linear interpolation
  roc_auc <- auc_curve(curve_df)
  # plot curve
  ggplot_perf_curve(data=curve_df, xlab="FPR", ylab="TPR", 
                    title=paste("Area under ROC: ", decimal_rounder(roc_auc, 4)))
}

plot_perf_curve <- function(pred, bin_num=1000, x_metric="rec", y_metric="prec"){
  # get probabilities and truth
  tp <- get_truth_pred(pred)
  # compute and bin curve
  curve_df <- get_curve(tp$prob, tp$truth, x_metric, y_metric)
  if(length(curve_df$x) > bin_num){
    curve_df <- bin_curve(curve_df, bin_num)
  }
  # get auc
  auc <- auc_curve(curve_df)
  # prepare df that we return
  ggplot_perf_curve(data=curve_df, xlab=x_metric, ylab=y_metric, 
                    title=paste("AUC: ", decimal_rounder(auc, 4)))
}

# ------------------------------------------------------------------------------
# Wrapper function to get performance curves binned
# ------------------------------------------------------------------------------

binned_perf_curve <- function(pred, bin_num=20, x_metric="rec", y_metric="prec"){
  # get probabilities and truth
  tp <- get_truth_pred(pred)
  # compute and bin curve
  curve_df <- get_curve(tp$prob, tp$truth, x_metric, y_metric)
  curve_df <- bin_curve(curve_df, bin_num)
  # get auc
  auc <- auc_curve(curve_df)
  # prepare df that we return
  curve_df <- as.data.frame(curve_df[,c("x_binned", "y", "thresh")])
  colnames(curve_df) <- c(paste(x_metric, "_binned", sep=""), y_metric, "thresh")
  return(list(curve=curve_df, auc=auc))
}

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
    positive_class <- pred$task.desc$positive
    prob = getPredictionProbabilities(pred, cl=positive_class)
    
    # get truth and turn it into ones (positive) and zeros (negative)
    truth <- getPredictionTruth(pred)
    if (is.factor(truth)) {
      pos_ix <- as.integer(truth) == which(levels(truth) == positive_class)
    } else {
      pos_ix <- truth == positive_class
    }
    truth <- as.integer(pos_ix)
    recall_target <- recall_perc/100
    
    # this part is adopted Hui's code 
    aucobj <- ROCR::prediction(prob, truth)
    
    # generate the recall and ppv and threshold
    prec_rec <- ROCR::performance(aucobj, 'prec', 'rec')
    rec <- prec_rec@x.values[[1]]
    prec <- prec_rec@y.values[[1]]
    
    # ignore nans
    non_nan <- !is.nan(prec) & !is.nan(rec)
    rec <- rec[non_nan]
    prec <- prec[non_nan]
    
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