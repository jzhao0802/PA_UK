
# Libraries -------------------------------------------------------------------
library(tidyverse)
library(magrittr)

make_binary_dataset <- function(n_rows=10000, n_feat=150, frac_pos=.01, 
                                seed_num=NULL) {
  # This is Dan's fantastic function to generate a binary classification dataset
  # with known coefficients and a set fraction of positives
  
  # set seed if user wants one
  if (!is.null(seed_num))
    seed(seed)
    
  # Define numbers for each feature type
  f_num <- round(n_feat/3)
  f_num2 <- n_feat - 2 * f_num
  col_types <- list(normal = f_num, uniform = f_num, binomial = f_num2)
  
  # Switch for distribution creation 
  col_function <- function(distribution, size) {
    if (distribution == "normal") {
      return(rnorm(
        n = size,
        mean = runif(1),
        sd = runif(1)
      ))
    } else {
      if (distribution == "uniform") {
        return(runif(n = size))
      } else if (distribution == "binomial") {
        return(rbinom(
          n = size,
          size = n_rows,
          prob = runif(1)
        ))
      } else {
        cat("Error: unknown distribution", distribution)
        return(-1)
      }
    }
  }
  
  # Create the dataset
  cols <- col_types %>%
    map2(.x = ., .y = names(.), function(x, y)
      rep(y, x)) %>%
    map(as.list) %>%
    map2(.x = ., .y = names(.), function(x, y)
      setNames(x, stringr::str_c(y, seq_along(x), sep = "_"))) %>%
    map(function(x)
      map(x, function(y)
        col_function(
          distribution = y, size = n_rows
        ))) %>%
    map(as_tibble) %>%
    bind_cols
  
  
  # Build the function for t
  coefs <- runif(n_feat, min = -1, max = 1)
  t_func <- cols %>%
    names %>%
    tibble(x = .,
           coefficient = coefs) %>%
    mutate(term = stringr::str_c(coefficient, x, sep = " * ")) %>%
    mutate(term = stringr::str_c("(", term, ")")) %>%
    .$term %>%
    stringr::str_c(collapse = " + ")
  
  # Add t, sigma_t and the flag to the dataset
  df <- cols %>%
    mutate_(.dots = list(t = t_func)) %>%
    mutate(t = t - mean(t) + rnorm(n = n_rows, mean = 0, sd = 0.01)) %>%
    mutate(sigma_t = 1 / (1 + exp(-t))) %>%
    (function (x) {
      sigma_t_threshod <- quantile(x = x$sigma_t, probs = frac_pos)
      mutate(x, label = ifelse(sigma_t >= sigma_t_threshod, 0, 1))
    })(.) %>%
    mutate(label=as.factor(label))
  
  if (sum(df$label==1)==0)
    stop("Run it again, because we didn't get enough positives!")
  
  df$t <- NULL
  df$sigma_t <- NULL
  list(df=df, coefs=coefs)
}
