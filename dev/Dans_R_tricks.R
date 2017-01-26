# Hello hello and welcome to the world of R. Here are some tricks that I find useful.


# Load packages -----------------------------------------------------------

library(tidyverse) # Seriously, the only data manipulation package you'll ever need, except maybe lubridate (for dates)
library(testthat)


# Test data ---------------------------------------------------------------

# Tibbles are a better data frame
df <- tibble(norm_col=rnorm(10),
             ints=seq_along(norm),
             lets=letters[ints],
             LETS=LETTERS[ints])

df_all_norm <- tibble(x_1=rnorm(10),
                      x_2=rnorm(10),
                      x_3=rnorm(10),
                      x_4=rnorm(10))

# List of 1 to 10 with the nammes as letters a to j
list_ints <- seq_len(10) %>% 
  as.list %>%
  setNames(letters[seq_len(10)])


# seq_along and seq_len ---------------------------------------------------

# If you NEED to loop over something, or need to create objects of a given length, iseuse seq_along or seq_len
expect_equal(1:3, seq_len(3))
expect_equal(1:length(norm_10), seq_along(norm_10))

# It's faster and more robust than 1:(size of object). Also the following breaks
expect_equal(1:0, c(1, 0)) # 1:0 gives c(1, 0), this is because R supports reverse sequence construction, e.g. try 1:-5


# map and walk----------------------------------------------------------------

# Map is awesome. you no longer need to loop. What if I want to take a sequence of numbers, add one to each of the, 
result <- list_ints %>%
  # Add two to each of them
  map(function(x) x + 2) %>%
  # Multiply them by 4
  map(function(x) x * 4) %>%
  # Then calculate their sum from left element to right element
  accumulate(function(x, y) x + y) %>%
  # Then calculate their sum
  reduce(function(x, y) x + y)

# Wow that was easy. Also look how readable that is. Isn't map reduce awesome?
# Also check out walk() if you want to do something that you dont want to print the results for,
# e.g. writing a list of data frames to csv
# list_of_dataframes %>% walk2(.x=., .y=list_of_paths, function(x, y) write_csv(x, y))
# oooooo

# Also, data frames are just lists. I could apply a function to each row using mutate_all, but I could also use map!
expect_equal(df_all_norm %>% 
               mutate_all(function (x) 2 * x), 
             df_all_norm %>% 
               map(function (x) 2 * x) %>% 
               as_tibble)


# Standard and non-standard evaluation ------------------------------------
# dplyr uses non - standard evaulation (NSE). That means I can do things like
df %>%
  select(norm_col)

# You might think that's normal (hah, see what I did there?) but really it isn't. Try running asking the global environment what norm is
norm_col # Error: object 'norm_col' not found

# So it doesn't exist. Actually, dplyr is doing something clever with scope and lazy evaulation. This is useful unless you want to
# be more lazy and generate the names of columns for selection
df %>%
  select("norm_col") # Error: All select() inputs must resolve to integer column positions. The following do not: *  "norm_col"

# In this case, you need the standard evaulation (SE) version of the function (indicated by the underscore)
df %>%
  select_(.dots = c("norm_col"))

# or for multiple columns
cols_to_select <- c("norm_col", "ints")

expect_equal(df %>%
               select_(.dots = cols_to_select),
             df %>%
               select(norm_col, ints))

# And if you really want that int column multiplied by a specific number
expect_equal(df %>%
               mutate_(.dots = list("ints" = "2 * ints")), 
             df %>%
               mutate(ints = 2 * ints))

# More generally, we can construct this list to contian many generated mutate statements
my_mutate_statements <- list(
  x_1 = stringr::str_c(names(df_all_norm), collapse = " * "),
  x_2 = names(df_all_norm) %>% 
    map2(.x=.,  .y = seq_len(4), 
         .f= function(x, y) stringr::str_c(x, " ** ", y)) %>% 
    stringr::str_c(collapse = " + "),
  x_3 = "x_1 * x_2",
  x_4 = 1
)

expect_equal(df_all_norm %>%
               mutate_(.dots = my_mutate_statements),
             df_all_norm %>%
               mutate(x_1 = x_1 * x_2 * x_3 * x_4,
                      x_2 = x_1 + x_2 ** 2 + x_3 ** 3 + x_4 ** 4,
                      x_3 = x_1 * x_2,
                      x_4 = 1))
