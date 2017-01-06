# ------------------------------------------------------------------------------
#
#                      Functions for tree based models
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(ggplot2)

# ------------------------------------------------------------------------------
# Plot variable importances for all the outer folds
# ------------------------------------------------------------------------------

plot_all_rf_vi <- function(results){
  # Plots the variable importances of RF models
  # Setup the plot
  outer_fold_n <- length(results$models)
  num_rows <- ceiling((outer_fold_n)/2)
  par(mfrow=c(num_rows, 2))
  par(mar=c(5.1,10.1,4.1,2.1))
  
  for (i in 1:outer_fold_n){
    model <- getLearnerModel(results$models[[i]], more.unwrap = T)
    title <- paste("Outer fold", as.character(i))
    plot_rf_vi(model, title)
  }
  par(mfrow=c(1,1))
}

plot_rf_vi <- function(model, title){
  vi <- model$variable.importance
  vi <- vi/max(vi)
  barplot(sort(vi), horiz = T, las=2, main=title)
}

# ------------------------------------------------------------------------------
# Plot decision tree, simple and fancy as well
# ------------------------------------------------------------------------------

plot_dt <- function(model, pretty=F){
  if (pretty){
    library(rattle)
    plt <- fancyRpartPlot(model)
  }else{
    par(mar=c(2.1,4.1,2.1,4.1))
    plt <- plot(model)
    text(model)
  }
  plt
}

# ------------------------------------------------------------------------------
# Extract rules and precision from decision trees in a data.frame format
# ------------------------------------------------------------------------------

get_dt_rules <- function(model){
  # This is a modified version of rattle::asRules() that saves results as a 
  # data.frame instead of just printing them out
  library(rpart)
  library(BBmisc)
  
  if (!inherits(model, "rpart")){
    stop("Not a legitimate rpart tree")
  }
  
  # Basic vars
  ylevels <- attr(model, "ylevels")
  # Regression tree?
  rtree <- length(ylevels) == 0
  target <- as.character(attr(model$terms, "variables")[2])
  frm <- model$frame
  names <- row.names(frm)
  ds.size <-  frm[1,]$n
  
  # Get each leaf node as a rule, and sort them
  if (rtree){
    # Sort rules by coverage
    ordered <- rev(sort(frm$n, index=T)$ix)
  }else{ 
    # Sort rules by prob of positive class i.e. 1 (the 5th col in binary class)
    ordered <- rev(sort(frm$yval2[,5], index=T)$ix)
  }
  
  # Define lists that will hold the cols of the resulting data.frame
  names_list <- c()
  covers <- c()
  pcovers <- c()
  pths <- c()
  yvals <- c()
  probs <- c()
  
  # Iterate through rules one by one
  for (i in ordered){
    if (frm[i,1] == "<leaf>"){
      # Get stats of leaf 
      rule <- frm[i,]
      if (rtree){
        yval <- rule$yval
      }else{
        yval <- ylevels[rule$yval]
      }
      cover <- rule$n
      pcover <- round(100*cover/ds.size)
      if (!rtree){
        prob <- rule$yval2[,5]
      }
      
      # Get the path from root to tip
      pth <- rpart::path.rpart(model, nodes=as.numeric(names[i]), print.it=F)
      pth <- unlist(pth)[-1]
      if (!length(pth)){
        pth <- "True"
      }
      
      # Write row to data.frame
      names_list <- c(names_list, names[i])
      covers <- c(covers, cover)
      pcovers <- c(pcovers, pcover)
      yvals <- c(yvals, yval)
      pths <- c(pths, paste(pth, collapse=" & "))
      if (!rtree){
        probs <- c(probs, prob)
      }
    }
  }
  
  # Compile result dataframe
  result <- data.frame(names_list, covers, pcovers, yvals, pths)
  cols <- c("Rule", "SampleNum", "SamplePercent", "Class")
  if (!rtree){
    result["probs"] <- probs
    # Sort dataframe by the precision then by sample num
    BBmisc::sortByCol(result, c("probs", "covers"), asc=T)
    cols <- c(cols, "Precision")
    reordered_cols <- c("names_list", "covers", "pcovers", "yvals", 
                        "probs", "pths")
    result <- result[, reordered_cols]
  }
  cols <- c(cols, "Splits")
  colnames(result) <- cols
  result
}