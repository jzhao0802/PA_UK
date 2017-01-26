
# Libraries ---------------------------------------------------------------
library(tidyverse)
library(magrittr)
library(caret)

# Configuration -----------------------------------------------------------
set.seed(123456)

n_rows <- 500
col_types <- list(normal = 25, uniform = 25, binomial = 25)
frac_positive <- .7
train_frac <- .7


# Switch for distribution creation ----------------------------------------
col_function <- function(distribution, size){
  if (distribution == "normal") {
    return(rnorm(n = size, mean = runif(1), sd = runif(1)))
  } else {
    if (distribution == "uniform") {
      return(runif(n = size))
    } else if (distribution == "binomial") {
      return(rbinom(n = size, size = n_rows, prob=runif(1)))
    } else {
      cat("Error: unknown distribution", distribution)
      return(-1)
    }
  }
}


# Create the dataset ------------------------------------------------------
cols <- col_types %>%
  map2(.x = ., .y=names(.), function(x, y) rep(y, x)) %>%
  map(as.list) %>%
  map2(.x = ., .y = names(.), function(x, y) setNames(x, stringr::str_c(y, seq_along(x), sep = "_"))) %>%
  map(function(x) map(x, function(y) col_function(distribution = y, size = n_rows))) %>%
  map(as_tibble) %>%
  bind_cols



# Build the function for t ------------------------------------------------
t_func <- cols %>%
  names %>%
  tibble(x=., coefficient = runif(length(x), min = -1, max = 1)) %>% 
  mutate(term = stringr::str_c(coefficient, x, sep = " * ")) %>%
  mutate(term = stringr::str_c("(", term, ")")) %>%
  .$term %>%
  stringr::str_c(collapse = " + ")

# Add t, sigma_t and the flag to the dataset ------------------------------
df <- cols %>%
  mutate_(.dots = list(t = t_func)) %>%
  mutate(t = t - mean(t),
         noise = rnorm(n = n_rows, mean = 0, sd = 0.01)) %>%
  mutate(sigma_t = 1/(1 + exp(-t))) %>%
  (function (x) {
    sigma_t_threshod <- quantile(x = x$sigma_t, probs = frac_positive)
    mutate(x, flag = ifelse(sigma_t >= sigma_t_threshod, 0, 1))
  })(.) %>%
  mutate(flag = as.factor(flag),
         patient_id = seq_len(nrow(.)))

# Seperate out training and testing ---------------------------------------
df_train <- df %>%
  sample_frac(train_frac)

df_test <- df %>%
  filter(!(patient_id %in% df_train$patient_id))


# Try a model out ---------------------------------------------------------
train_control <- trainControl(method="cv", number=10)


# Train the model
model <- train(flag ~ ., 
               data = df_train, 
               trControl = train_control, 
               method = "glm", family = "binomial")

# How good is the model on the test data?
predictions <- predict(model, df_test)
correct_prediction_rate <- sum(predictions == df_test$flag) / nrow(df_test)

