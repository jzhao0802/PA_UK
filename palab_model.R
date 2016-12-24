# ------------------------------------------------------------------------------
#
#             Utils functions of the modelling part of PAlab
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Get numeric variables from input dataframe
# ------------------------------------------------------------------------------

get_numerical_variables <- function(input, var_config, categorical=F) {
  library(dplyr)  
  # Keeping only those variables in var_config that are in input and
  # are numerical
  
  if (categorical){
    accepted_types <- c("numerical", "categorical")
  }else{
    accepted_types <- c("numerical")
  }
  var_config_numerical <- var_config %>%
    dplyr::filter_(~Column %in% colnames(input)) %>%
    dplyr::filter_(~Type %in% accepted_types)
  
  # Keeping only these variables from the input and returning the dataframe
  output <- input %>%
    select_(.dots = var_config_numerical$Column)
  
  output
}

# ------------------------------------------------------------------------------
# Function to create custom precision at x% recall in mlR
# ------------------------------------------------------------------------------

make_custom_pr_measure <- function(recall_perc=10, name_str="pr5"){
  
  find_prec_at_recall <- function(pred, recall_perc=10){
    library(PRROC)
    # This function takes in a prediction output from a trained mlR learner. It
    # extracts the predicitons, and finds the highest precision at a given
    # percentage of recall. 

    # see docs here: https://cran.r-project.org/web/packages/PRROC/PRROC.pdf
    # the positive class has to be 1, and the negative has to be 0.
    scores = getPredictionProbabilities(pred)
    labels = as.numeric(factor(getPredictionTruth(pred)))-1
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
    max(prec[recall_min_ix])
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
# Extract results from tuned nested CV model
# ------------------------------------------------------------------------------

get_results <- function(results, detailed=F, all_measures=F, write_csv=F){
  # This function calculates a bunch of summary statistics from a learner that
  # was tuned using nested CV. The level of detail that's returned could be 
  # set with detailed=T and all_measure=T. Also if write_csv=T, all data.frames
  # will be saved to a csv with the current timestamp as its name.
  
  # Define return list
  info_col_name <- ""
  info_df <- data.frame(info_col_name, results$task.id, results$learner.id)
  colnames(info_df) <- c("Info", "Dataset", "Learner")
  to_write <- list("info"= info_df)
  to_return <- list("info" = data.frame(results$task.id, results$learner.id))
  
  # Extract results from nested CV results
  o_train <- results$measures.test
  o_train$iter <- NULL
  o_test <- results$measures.train
  o_test$iter <- NULL
  i_test <- as.data.frame(t(extractSubList(results$extract, "y")))
  
  # Unify the columns
  colnames(o_train) <- colnames(i_test)
  colnames(o_test) <- colnames(i_test)
  
  # Discard columns with aggregated SD values in the outer loop
  no_sd_ix <- !grepl("sd", colnames(o_train))
  o_train <- o_train[,no_sd_ix]
  o_test <- o_test[,no_sd_ix]
  
  # Add table name
  o_train_w <- cbind("Outer train"=1:nrow(o_train), o_train)
  o_test_w <- cbind("Outer test"=1:nrow(o_train), o_test)
  i_test_w <- cbind("Inner test"=1:nrow(o_train), i_test)
  
  # Save whole table if user wants them
  if (all_measures){
    to_write <- c(to_write, list("outer_train" = o_train_w,
                                   "outer_test" = o_test_w,
                                   "inner_test" = i_test_w))
    to_return <- c(to_return, list("outer_train" = o_train,
                                   "outer_test" = o_test,
                                   "inner_test" = i_test))
  }
  
  # Now that we saved the outer tables, let's discard SD from the inner one too
  i_test <- i_test[,no_sd_ix]
  
  # Difine difference matrices
  o_train_minus_o_test <- o_train - o_test
  i_test_minus_o_test <- i_test - o_test
  
  # Collect summary stats
  o_train_mean <- unlist(lapply(o_train, mean))
  o_train_sd <- unlist(lapply(o_train, sd))
  o_test_mean <- unlist(lapply(o_test, mean))
  o_test_sd <- unlist(lapply(o_test, sd))
  i_test_mean <- unlist(lapply(i_test, mean))
  i_test_sd <- unlist(lapply(i_test, sd))
  o_train_minus_o_test_mean <- unlist(lapply(o_train_minus_o_test, mean))
  o_train_minus_o_test_sd <- unlist(lapply(o_train_minus_o_test, sd))
  i_test_minus_o_test_mean <- unlist(lapply(i_test_minus_o_test, mean))
  i_test_minus_o_test_sd <- unlist(lapply(i_test_minus_o_test, sd))
  
  if (detailed){
    summary = rbind(o_train_mean, o_train_sd, o_test_mean, o_test_sd,
                    i_test_mean, i_test_sd,
                    o_train_minus_o_test_mean, o_train_minus_o_test_sd,
                    i_test_minus_o_test_mean, i_test_minus_o_test_sd)
    # add row names as a new column
    row_names <- c("Outer train mean", "Outer train std",
                   "Outer test mean", "Outer test std",
                   "Inner test mean", "Inner test std",
                   "(outer train - outer test) mean",
                   "(outer train - outer test) std",
                   "(inner test - outer test) mean",
                   "(inner test - outer test) std")
  }else{
    summary <- rbind(o_test_mean, o_test_sd, o_train_minus_o_test_mean, 
                     i_test_minus_o_test_mean)
    # add row names as a new column
    row_names <- c("Outer test mean", "Outer test std",
                   "(outer train - outer test) mean",
                   "(inner test - outer test) mean")
  }
  summary_w <- cbind("Summary"=row_names, summary)
  rownames(summary) <- row_names
  to_write <- c(to_write, list("summary"=summary_w))
  to_return <- c(to_return, list("summary"=summary))
  
  # extract best params for each outer fold
  params <- getNestedTuneResultsX(results)
  to_return <- c(to_return, list("best_params"=params))
  params_w <- cbind("Best params"=1:nrow(o_train), params)
  to_write <- c(to_write, list("best_params"=params_w))
  
  # write results to csv if needed
  if (write_csv){
    # get time stamp and make a (Windows safe) file name out of it
    t <- as.character(Sys.time())
    t <- gsub(" ", "_", t)
    t <- gsub(":", "-", t)
    output_csv <- paste(t, '.csv', sep='')
    write_df <- function(df){
      write.table(as.data.frame(df), output_csv, append=T, sep=',', row.names=F)
    }
    suppressWarnings(lapply(to_write, write_df))
  }
  
  #return results object
  to_return
}

# ------------------------------------------------------------------------------
# Get paths of the optimized hyper parameters, models and best mean params
# ------------------------------------------------------------------------------

get_opt_paths <- function(result){
  # Returns all the hyper-parameters combinations that were tested in the
  # nested  CV.
  
  opt_paths <- getNestedTuneResultsOptPathDf(result, trafo=T)
  # Drop unnecessary columns
  opt_paths$dob <- NULL
  opt_paths$eol <- NULL
  
  # Check if we have any error messages
  error_ix <- !is.na(opt_paths$error.message)
  
  if (any(error_ix)){
    warning("Some of the hyper-param combinations threw errors, check the
            error.message column!")
  }else{
    opt_paths$error.message <- NULL
  }
  opt_paths
}

get_models <- function(results){
  lapply(res$models, function(x) getLearnerModel(x, more.unwrap=T))
}


get_best_mean_param <- function(results, int=F){
  if (int){
    bmp <- lapply(results$best_params, function(x) round(mean(x)))
  }else{
    bmp <- lapply(results$best_params, mean)
  }
  bmp
}

# ------------------------------------------------------------------------------
# Plot regularisation path for glmnet
# ------------------------------------------------------------------------------

plot_reg_path_glmnet <- function(results){
  # Plots the regularisation paths for each model in the outer folds.
  library(plotmo)
  
  # Setup the plot
  outer_fold_n <- length(results$models)
  num_rows <- ceiling((outer_fold_n)/2)
  par(mfrow=c(num_rows, 2))
  
  for (i in 1:outer_fold_n){
    # Get best lambda and model
    best_lambda <- results$best_params$s[i]
    model <- getLearnerModel(results$models[[i]], more.unwrap = T)
    title <- paste("Outer fold", as.character(i))
    # Plot regularisation path with the best lambda=s chosen by CV
    plotmo::plot_glmnet(model, label=T, s=best_lambda, main=title)
  }
  # Plot regularisation path with labels for only 5 variables, and add grid
  # plotmo::plot_glmnet(model, label=5, s=best_lambda, grid.col="lightgrey")
  par(mfrow=c(1,1))
}

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
  if (output_folder == ""){
    output_folder = getwd()
  }else{
    output_folder =file.path(getwd(), output_folder)
    if (!file.exists(output_folder)){
      dir.create(output_folder)
    }
  }
  
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