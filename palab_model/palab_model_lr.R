# ------------------------------------------------------------------------------
#
#                      Functions for elastic net and LR
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Plot regularisation path for glmnet
# ------------------------------------------------------------------------------

plot_reg_path_glmnet <- function(results, n_feat="all"){
  # Plots the regularisation paths for each model in the outer folds.
  library(plotmo)
  
  # Setup the plot
  outer_fold_n <- length(results$models)
  num_rows <- ceiling((outer_fold_n)/2)
  par(mfrow=c(num_rows, 2))
  models <- get_models(results)
  
  for (i in 1:outer_fold_n){
    # Get best lambda and model
    best_lambda <- results$models[[i]]$learner.model$opt.result$x$s
    model <- models[[i]]
    title <- paste("Outer fold", as.character(i))
    
    # Plot regularisation path with the best lambda=s chosen by CV
    if (n_feat == "all"){
      plotmo::plot_glmnet(model, label=T, s=best_lambda, main=title)
    }else{
      plotmo::plot_glmnet(model, label=n_feat, s=best_lambda, main=title)
      # grid.col="lightgrey" adds grid to the plits
    }
  }
  par(mfrow=c(1,1))
}

# ------------------------------------------------------------------------------
# Calculate odds ratios from the logistic regression coefficients
# ------------------------------------------------------------------------------

get_odds_ratios <- function(model){
  # For details see here: http://www.ats.ucla.edu/stat/r/dae/logit.htm
  exp(cbind(OR = coef(model), confint(model)))
}
