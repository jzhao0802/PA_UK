# ------------------------------------------------------------------------------
#
#             General plotting functions for all kinds of models
#
# ------------------------------------------------------------------------------

library(dplyr)
library(mlr)
library(ggplot2)

# ------------------------------------------------------------------------------
# Plot each hyper-parameter pair and the interpolated performance metric
# ------------------------------------------------------------------------------

plot_hyperpar_pairs <- function(results, perf_metric, output_folder="", trafo=F){
  # Plots the performance metric surface for all hyper parameter combinations 
  # across all outer folds
  library(gridExtra)
  
  # Setup the plot
  outer_fold_n <- length(results$models)
  subplot_n <- outer_fold_n + 1
  
  # Generate data for plots
  resdata <- generateHyperParsEffectData(results, trafo=trafo)
  
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
      
      plt <- plotHyperParsEffect(df, x=axes[1], y =axes[2], z=perf_metric,
                                 plot.type="heatmap", interpolate="regr.earth", 
                                 show.experiments=T) +
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