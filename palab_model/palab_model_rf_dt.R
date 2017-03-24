# ------------------------------------------------------------------------------
#
#                      Functions for tree based models
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(ggplot2)

# ------------------------------------------------------------------------------
# Get and plot variable importances for all the outer folds
# ------------------------------------------------------------------------------

get_vi_table <- function(model, dataset, decimal=3){
  if (is(model, "TuneModel")){
    model <- getLearnerModel(model, more.unwrap = T)
  }
  if (inherits(model, "xgb.Booster")){
    feat_names <- getTaskFeatureNames(dataset)
    tmp <- xgboost::xgb.importance(feature_names=feat_names, model=model)
    vi <- tmp$Gain
  }else{
    vi <- model$variable.importance  
  }
  vi <- sort(vi/sum(vi)*100, decreasing=T)
  vi <- data.frame(VI=vi)
  if (inherits(model, "xgb.Booster"))
    rownames(vi) <- tmp$Feature
  
  # Calculate correlation between predictors and outcome
  target <- dataset$task.desc$target
  data <- getTaskData(dataset)
  y <- as.numeric(unlist(data[target]))
  X <- data_without_target(data, target)
  r <- unlist(lapply(X, function(x) cor(as.numeric(x), y)))
  r <- data.frame(r)
  
  # Merge data with correlation values after sorting by rownames
  vi$Corr <- r[rownames(vi),]
  decimal_rounder(vi, decimal)
}

plot_all_rf_vi <- function(results, decimal=2, aggregate=F, dataset=NULL){
  library(scales)
  outer_fold_n <- length(results$models)
  vis <- c()
  iters <- c()
  vars <- c()
  models = get_models(results)
  
  for (i in 1:outer_fold_n){
    model <- models[[i]]
    
    if (inherits(model, "xgb.Booster")){
      # extracting VI from XGBoost
      feat_names <- getTaskFeatureNames(dataset)
      tmp <- xgboost::xgb.importance(feature_names=feat_names, model=model)
      vi <- tmp$Gain
      vars <- c(vars, tmp$Feature)
    }else{
        # extracting VI from RF and DT
        vi <- as.numeric(model$variable.importance)
        vars <- c(vars, names(model$variable.importance))
    }
    vi <- vi/sum(vi)*100
    vis <- c(vis, vi)
    iters <- c(iters, rep(i, length(vi)))
  }
  
  # Collate variable importances into one data.frame and format long floats
  data <- data.frame(VI=vis, Outer=factor(iters), Vars=vars)
  data <- decimal_rounder(data, decimal)
  
  # Sort data by average VI across folds
  average_vi_order <- data %>% 
    group_by(Vars) %>% 
    summarise(mean_VI=mean(VI), sd_VI=sd(VI)) %>% 
    arrange(desc(mean_VI))
  data$Vars <- factor(data$Vars, levels=average_vi_order$Vars)
  
  if (aggregate){
    colnames(average_vi_order) <- c("Vars", "VI", "SD")
    average_vi_order$Vars <- factor(average_vi_order$Vars, 
                                    levels=average_vi_order$Vars)
    average_vi_order <- decimal_rounder(average_vi_order, decimal)
    # Plot it as barplot with error bars
    ggplot(average_vi_order, aes(x=Vars)) +
      geom_bar(stat="identity", aes(y=VI), position="dodge") +
      geom_text(aes(x=Vars, y=VI-SD*1.1, label=paste0(VI, "%"), 
                    hjust=ifelse(sign(VI)>0, 1, 0)), 
                position=position_dodge(width=1)) +
      geom_errorbar(aes(ymax=VI+SD, ymin=VI-SD), width=0.25) +
      coord_flip()
  }else{
    # Plot it as stacked barplots with ggplot
    ggplot(data, aes(x=Vars, fill=Outer)) +
      geom_bar(stat="identity", aes(y=VI), position="dodge") +
      geom_text(aes(x=Vars, y=VI, label=paste0(VI, "%"), 
                    hjust=ifelse(sign(VI)>0, 1, 0)), 
                position=position_dodge(width=1)) +
      coord_flip()
  }
}

plot_all_rf_vi_simple <- function(results){
  # Setup the plot
  outer_fold_n <- length(results$models)
  num_rows <- ceiling((outer_fold_n)/2)
  par(mfrow=c(num_rows, 2))
  par(mar=c(5.1,10.1,4.1,2.1))
  models = get_models(results)
  
  for (i in 1:outer_fold_n){
    model <- models[[i]]
    title <- paste("Outer fold", as.character(i))
    plot_rf_vi(model, title, subplot=TRUE)
  }
  par(mfrow=c(1,1))
}

plot_rf_vi <- function(model, title='', subplot=FALSE){
  if (is(model, "TuneModel")){
    model <- getLearnerModel(model, more.unwrap = T)
  }
  par(mar=c(5.1,10.1,4.1,2.1))
  if (inherits(model, "xgb.Booster")){
    feat_names <- getTaskFeatureNames(dataset)
    tmp <- xgboost::xgb.importance(feature_names=feat_names, model=model)
    vi <- tmp$Gain
    names(vi) <- tmp$Feature
  }else{
    vi <- model$variable.importance  
  }
  
  vi <- vi/max(vi)
  barplot(sort(vi), horiz = T, las=2, main=title)
  
  # If this is just one plot and not part of a multiplot, reset faceting
  if (!subplot){
    par(mfrow=c(1,1))
  }
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