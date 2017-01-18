# ------------------------------------------------------------------------------
#
#             General plotting functions for all kinds of models
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(ggplot2)


# ------------------------------------------------------------------------------
# Plot precision recall and ROC curve
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
  results <- list("truth"=truth, "prob"=prob)
  results
}

plot_pr_curve <- function(results, roc=TRUE){
  library(PRROC)
  # Get probabilities and truth
  tb <- get_truth_pred(results$pred)
  
  # Retain only the predictions on the test set
  df <- as.data.frame(results$pred)
  truth <- tb$truth[df$set=="test"]
  prob <- tb$prob[df$set=="test"]
  
  # Plot ROC curve first so it's the 2nd plot once this function is run
  if (roc){
    roc <- roc.curve(scores.class0=prob, weights.class0=truth, curve = T)
    plot(roc)
  }
  
  # Plot PR curve
  pr <- pr.curve(scores.class0=prob, weights.class0=truth, curve = T)
  plot(pr)
}

# ------------------------------------------------------------------------------
# Plot each hyper-parameter pair and the interpolated performance metric
# ------------------------------------------------------------------------------

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Now that I've started to use the development version, I need to update this 
# function so it uses partial.dep if we have more than 3 params. Also the 
# experiments are not showing which is a bummer..

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


plot_hyperpar_pairs <- function(results, perf_metric, output_folder="", trafo=F){
  # Plots the performance metric surface for all hyper parameter combinations 
  # across all outer folds
  library(gridExtra)
  
  # Setup the plot
  outer_fold_n <- length(results$models)
  subplot_n <- outer_fold_n + 1
  
  # Generate data for plots
  num_of_params <- length(results$models[[1]]$learner.model$opt.result$x)
  if (num_of_params > 2){
    resdata <- generateHyperParsEffectData(results, trafo=trafo, partial.dep=T)
  }else{
    resdata <- generateHyperParsEffectData(results, trafo=trafo)
  }
  
  # Generate hyper param pairs to plot
  all_axes <- t(combn(resdata$hyperparams, 2))
  
  # Check output folder, create it if needed
  create_output_folder(output_folder)
  
  # Set default font size
  theme_set(theme_minimal(base_size = 10))
  
  # Go through all variable pairs  
  for (p in 1:nrow(all_axes)){
    # Go through each outer fold + a combined plot
    subplots <- list()
    # Get axes of the plot
    axes <- all_axes[p, ]
    for (i in 1:subplot_n){
      # df stores the relevant data of each outer fold
      df <- resdata
      # Discard hyperparams that we are not plotting, otherwise we get a nasty
      # bug that took me 3 hours to debug
      df$hyperparams <- axes[1:2]
      if (i==1){
        title = paste("All outer folds combined")
      }else{
        title = paste("Outer fold", as.character(i-1))
        # Faceting with nested_cv_run is not implemented yet, we do it manually
        df$data <- df$data[df$data$nested_cv_run==i-1,]
        
      }
      min_plt <- min(df$data[perf_metric], na.rm=T)
      max_plt <- max(df$data[perf_metric], na.rm=T)
      med_plt <- mean(c(min_plt, max_plt))
      if (num_of_params > 2){
        plt <- plotHyperParsEffect(df, x=axes[1], y =axes[2], z=perf_metric,
                                   plot.type="heatmap", show.experiments=T,
                                   partial.dep.learn="regr.earth")
      }else{
        plt <- plotHyperParsEffect(df, x=axes[1], y =axes[2], z=perf_metric,
                                   plot.type="heatmap", interpolate="regr.earth", 
                                   show.experiments=T)
      }
      plt +
        scale_fill_gradient2(breaks=seq(min_plt, max_plt, length.out=5),
                             low="grey", high="blue", midpoint=med_plt) +
        ggtitle(title) +
        theme(panel.grid.major = element_blank(), 
              panel.grid.minor = element_blank(),
              axis.line = element_line(colour = "grey"))
      
      # Add the sublots to build multiplot
      subplots[[length(subplots)+1]] <- plt
    }
    
    # Save each multiplot of hyper param pair into the output folder
    main_title <- paste(axes[1],"vs", axes[2])
    file_name <- paste(paste(axes[1], axes[2], sep='_'), ".pdf", sep='')
    output_path <- file.path(output_folder, file_name)
    multiplot <- marrangeGrob(subplots, nrow=2, ncol=2, top=main_title)
    ggsave(output_path, multiplot)
  }
}

# ------------------------------------------------------------------------------
# Plot partial dependence plots into a multi page pdf
# ------------------------------------------------------------------------------

plot_partial_deps <- function(model, dataset, cols, individual=F, 
                              output_folder=""){
  library(gridExtra)
  
  # Check output folder, create it if needed
  create_output_folder(output_folder)
  
  subplots <- list()
  for (i in 1:length(cols)){
    if (individual){
      par_dep_data <- generatePartialDependenceData(model, dataset, cols[i], 
                                                    individual=T)
    }else{
      par_dep_data <- generatePartialDependenceData(model, dataset, cols[i], 
                                                    fun=median)
    }
    plt <- plotPartialDependence(par_dep_data)
    subplots[[length(subplots)+1]] <- plt
  }
  
  # Save each multiplot of hyper param pair into the output folder
  main_title <- "Partial dependence plots"
  file_name <- paste(dataset$task.desc$id, ".pdf", sep='')
  output_path <- file.path(output_folder, file_name)
  multiplot <- marrangeGrob(subplots, nrow=3, ncol=3, top=main_title)
  ggsave(output_path, multiplot)
}

# ------------------------------------------------------------------------------
# Get linear fit of partial dependence plots and plot these as barplots
# ------------------------------------------------------------------------------

get_par_dep_plot_slopes <- function(par_dep_data, decimal=5){
  data <- par_dep_data$data
  data_without_probs <- data[, 3:ncol(data)]
  
  # Fit linear model to each column with the predicted probability as outcome
  get_beta <- function(col){
    non_nan <- !is.na(col)
    model <- lm(data$Probability[non_nan] ~ col[non_nan])
    beta <- coef(summary(model))[2,]
  }
  betas <- lapply(data_without_probs, get_beta)
  
  # Make data frame out of models and format it nicely
  betas <- as.data.frame(t(as.data.frame(betas)))
  betas$names <- rownames(betas)
  rownames(betas) <- NULL
  colnames(betas) <- c("Beta", "Std", "Tval", "Pval", "Vars")
  betas <- betas[, c("Vars", "Beta", "Std", "Tval", "Pval")]
  is.num <- sapply(betas, is.numeric)
  betas <- decimal_rounder(betas, decimal)
  betas <- sortByCol(betas, "Beta", asc=F)
  #This is so ggplot preservs the order of the bars
  betas$Vars <- factor(betas$Vars, levels=betas$Vars)
  betas
}

plot_par_dep_plot_slopes <- function(par_dep_data, decimal=5){
  betas <- get_par_dep_plot_slopes(par_dep_data, decimal=decimal)
  ggplot(betas, aes(x = Vars)) +
    geom_bar(stat="identity", aes(y=Beta), position="dodge") +
    geom_text(aes(x=Vars, y=Beta-Std*1.1, label=Pval, 
                  hjust=ifelse(sign(Beta)>0, 1, 0)), 
                  position = position_dodge(width=1)) +
    geom_errorbar(aes(ymax=Beta+Std, ymin=Beta-Std), width=0.25) +
    coord_flip()
  
}