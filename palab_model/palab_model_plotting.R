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

plot_hyperpar_pairs <- function(results, ps, perf_metric, per_fold=T, 
                                output_folder="", indiv_pars=F){
  # Plots the performance metric surface for all hyper parameter combinations 
  # across all outer folds
  library(gridExtra)
  
  # Setup the plot
  outer_fold_n <- length(results$models)
  subplot_n <- outer_fold_n + 1
  
  # Find out if there is any param with trafo
  trafo <- any(unlist(lapply(extractSubList(ps$pars, "trafo"), is.null)))
  
  # Generate data for plots
  num_of_params <- length(ps$pars)
  if (num_of_params > 2){
    resdata <- generateHyperParsEffectData(results, trafo=trafo, partial.dep=T)
  }else{
    resdata <- generateHyperParsEffectData(results, trafo=trafo)
  }
  
  # Check output folder, create it if needed
  create_output_folder(output_folder)
  
  # Set default font size
  theme_set(theme_minimal(base_size=12))
  
  # Short metric name
  legend_title <- unlist(strsplit(perf_metric, split="\\."))[1]
  
  # Go through all variable pairs or individual params, define axes for plots
  if(indiv_pars){
    all_axes <- resdata$hyperparams
    plot_n <- length(all_axes)
  }else{
    # Generate hyper param pairs to plot
    all_axes <- t(combn(resdata$hyperparams, 2))
    plot_n <- nrow(all_axes)
  }
  
  for (p in 1:plot_n){
    # Go through each outer fold + a combined plot
    subplots <- list()
    
    # Get axes of the multiplot
    if(!indiv_pars) axes <- all_axes[p, ] 
    
    for (i in 1:subplot_n){
      # df stores the relevant data of each outer fold
      df <- resdata
      if (i==1){
        subtitle = paste("All outer folds combined")
      }else{
        subtitle = paste("Outer fold", as.character(i-1))
        # Faceting with nested_cv_run is not implemented yet, do it manually
        df$data <- df$data[df$data$nested_cv_run==i-1,]
      }
      
      # Calculate min and max colors for the gradient
      min_plt <- min(df$data[perf_metric], na.rm=T)
      max_plt <- max(df$data[perf_metric], na.rm=T)
      med_plt <- mean(c(min_plt, max_plt))
      
      # Round long deciamls in legend text
      format_legend <- function(x) sprintf("%.2f", x)
      
      # Create plots with partial dependence
      if (num_of_params > 2){
        if(indiv_pars){
          # Lot more than 2 params, plot them individually - fast
          plt <- plotHyperParsEffect(df, x=all_axes[p], y=perf_metric,
                                     plot.type="line", show.experiments=T,
                                     partial.dep.learn="regr.randomForest") +
                 ggtitle(subtitle)
        }else{
          # More than 2 params, plot them in pairs as partial dependece heatmaps
          plt <- plotHyperParsEffect(df, x=axes[1], y =axes[2], z=perf_metric,
                                     plot.type="heatmap", show.experiments=T,
                                     partial.dep.learn="regr.randomForest")  
        }
      # WE only have 2 params, no par.dep, plot them as interpolated heatmap
      }else{
        plt <- plotHyperParsEffect(df, x=axes[1], y =axes[2], z=perf_metric,
                                   plot.type="heatmap", interpolate="regr.earth", 
                                   show.experiments=T)
      }
      
      # Make them pretty
      if(!indiv_pars){
        plt <- plt +
              scale_fill_gradient2(breaks=seq(min_plt, max_plt, length.out=5),
                                   low="grey", high="blue", midpoint=med_plt,
                                   guide=guide_legend(title=legend_title),
                                   labels=format_legend) +
              ggtitle(subtitle) +
              theme(panel.grid.major = element_blank(), 
                    panel.grid.minor = element_blank(),
                    axis.line = element_line(colour = "grey"))
      }
      
      # Add the sublots to build multiplot
      subplots[[length(subplots)+1]] <- plt
      
      # If we don't want a plot per fold
      if (i==1 & per_fold==F){
        break
      }
    }
    # Save each (multi) plot of hyper param (pairs) into the output folder
    if(indiv_pars){
      main_title <- paste(all_axes[p])
      file_name <- paste(main_title, ".pdf", sep='')
    }
    else{
      main_title <- paste(axes[1],"vs", axes[2])
      file_name <- paste(paste(axes[1], axes[2], sep='_'), ".pdf", sep='')
    }
    output_path <- file.path(output_folder, file_name)
    if (per_fold){
      multiplot <- marrangeGrob(subplots, nrow=2, ncol=2, top=main_title)  
    }else{
      multiplot <- marrangeGrob(subplots, nrow=1, ncol=1, top=main_title)
    }
    ggsave(output_path, multiplot,  width = 16, height = 9, dpi = 120)
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