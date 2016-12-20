# ------------------------------------------------------------------------------
#
#             Utils functions of the modelling part of palab
#
# ------------------------------------------------------------------------------

get_numerical_variables <- function(input, var_config) {
  library(dplyr)  
  # Keeping only those variables in var_config that are in input and
  # are numerical
  var_config_numerical <- var_config %>%
    dplyr::filter_(~Column %in% colnames(input)) %>%
    dplyr::filter_(~Type == "numerical")
  
  # Keeping only these variables from the input and returning the dataframe
  output <- input %>%
    select_(.dots = var_config_numerical$Column)
  
  output
}