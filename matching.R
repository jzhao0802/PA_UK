# ------------------------------------------------------------------------------
#
#   Functions for getting nested CV index table with matching samples
#
# ------------------------------------------------------------------------------

nested_cv_matched_ix <- function(match_df, outer_fold_n=5, inner_fold_n=5, 
                               shuffle=TRUE, downsample=1){
  # This function takes in a dataframe with two columns. The first column is 
  # "id" and it is the unique ID of each sample. The 2nd col is "match" which 
  # links the negative cases to the positive ones. Positives patients are 
  # automatically found, as they match themselves. It then returns the same 
  # dataframe with two additional columns: outer_fold and inner_fold, which 
  # hold the place of each sample in nested matching CV.
  
  # find positive patients, i.e. they match themselves
  ids <- match_df$id
  matches <- match_df$match
  pos_ix <- which(ids == matches)
  pos <- match_df[pos_ix,]
  
  # define dimensions
  n <- dim(match_df)[1]
  pos_n <- length(pos_ix)
  
  # do we need to downsample?
  if (downsample != 1){
    if (downsample > 1 | downsample <= 0){
      stop("Downsample must be between 0 and 1.")
    }
    down_pos_n <- floor(downsample * pos_n)
    down_ix = sample(1:pos_n, down_pos_n)
    pos_ix <- pos_ix[down_ix]
    pos <- match_df[pos_ix, ]
    pos_n <- length(pos_ix)

    # downsample the negatives accordingly
    match_df <- match_df[matches %in% pos$id,]
    ids <- match_df$id
    matches <- match_df$match
  }
  pos_id <- pos$id
  
  # do we have enough positive samples?
  if (outer_fold_n * inner_fold_n > pos_n){
    stop("Number of positive samples have to be greater than outer_fold * 
         inner_fold.")
  }
  
  # OUTER SPLIT
  # we first do the two-fold cv for the positives only and then every negative 
  # is following their matched positive. First we create a list of integers, 
  # then split in outer_fold_n number of equal bins
  outer_fold <- split(1:pos_n, cut(1:pos_n, outer_fold_n))
  # take those bins and rename the integers going from 1 to N so they go from 1 
  # to outer_fold_n
  start <- 1
  inner_fold_tmp <- 1:pos_n
  for (i in 1:outer_fold_n) {
    if (i > 1){
      start <- start + outer_len
    }
    outer_len <- length(outer_fold[[i]])
    outer_fold[[i]] <- rep(i, outer_len)
    
    # INNER SPLIT
    # do the same as above but now within each outerfold
    inner_fold <- split(1:outer_len, cut(1:outer_len, inner_fold_n))
    for (j in 1:inner_fold_n) {
      inner_len <- length(inner_fold[[j]])
      inner_fold[[j]] <- rep(j, inner_len)
    }
    end <- start + outer_len - 1
    inner_fold_tmp[start:end] <- unlist(inner_fold, use.names=F)
  }
  outer_fold <- unlist(outer_fold, use.names=F)
  inner_fold <- inner_fold_tmp

  # shuffle pos_ids make sure they're random
  if (shuffle){
    pos_id <- sample(pos_id)
  }
  
  # build a lookup table for the CV-ed positive samples
  cv_pos <- data.frame(outer_fold, inner_fold)
  rownames(cv_pos) <- pos_id
  
  # add outer and inner cv positions to match_df
  match_df <- data.frame(match_df, cv_pos[as.character(matches),])
  rownames(match_df) <- 1:dim(match_df)[1]
  match_df
}

matching_to_indices <- function(match_df, key){
  # match_df is a dataframe with two columns: "id", and "match". Each row in 
  # match_df links samples with unique string identifiers to each other. 
  # Samples from the positive class are linked with themselves. The key input 
  # variable is a column from the dataframe we want to use for modelling and it
  # holds unique string IDs for each row/sample. The function returns a data-
  # frame with "id" and "match" columns, where the matching information is
  # represented by row indicies in the dataframe, as defined by the key, i.e.
  # it describes which row is linked with which in the dataframe.
  
  library(hash)
  # check if IDs are unique
  n_unique <- length(unique(match_df$id))
  n_id <- length(match_df$id)
  if( n_unique != n_id){
    stop("Not all IDs are unique in the match_df dataframe.")
  }
  n_unique_key <- length(unique(key))
  if( n_unique_key != n_id){
    stop("Not all IDs are unique in the key.")
  }
  
  # check if there are any positives
  ids <- as.character(match_df$id)
  matches <- as.character(match_df$match)
  if(sum(ids == matches) == 0){
    stop("There aren't any positive samples in this matching dataframe.")
  }
  
  # TODO!! check if all IDs in the key are in the match
  
  # reorder match_df by the key column
  rownames(match_df)  <- match_df$id
  match_df <- match_df[key,]
  
  # turn IDs into integers
  id <- 1:nrow(match_df)
  
  # create hash from match_df for quick lookup
  h <-hash(keys=as.character(match_df$id), values=id)
  
  # lookup the index of matching samples
  match <- unlist(lapply(match_df$match, function(x) h[[as.character(x)]]))
  
  #list("orig"=match_df, "new"=match_df_ix)
  data.frame(id, match)
}

# ------------------------------------------------------------------------------
#
# Functions for simulating matching data
#
# ------------------------------------------------------------------------------

sim_matched_samples_int <- function(freq=.1, N=100){
  # Given the frequency of the positive class and the size of the population
  # it returns a dataframe with two columns ("id", "match") where each positive
  # patient is matched with the right amount of negatives. This function simply
  # uses integers to denote samples.
  
  if (freq <= 0 | N <= 2){
    stop("N has to be at least 2, and freq needs to be larger than zero.")
  }
  inv_freq <- 1/freq
  pos_N <- floor(freq * N)
  if (pos_N < 1){
    stop("We need to have at least one positive sample. Increase freq or N.")
  }
  neg_N <- N - pos_N
  
  # The first pos_N will match themselves as they are positives but from 
  # pos_N + 1 to N we have negatives, and these will matches the positives
  match <- rep(seq(1, pos_N), (inv_freq + 1))[1:N]
  id <- seq(1, N)
  data.frame(id, match)
}

sim_matched_samples_str <- function(freq=.1, N=100){
  # Given the frequency of the positive class and the size of the population
  # it returns a data.frame with two columns ("id", "match") where each positive
  # patient is matched with the right amount of negatives. This function uses 
  # random strings to denote samples. 
  
  if (freq <= 0 | N <= 2){
    stop("N has to be at least 2, and freq needs to be larger than zero.")
  }
  inv_freq <- 1/freq
  pos_N <- floor(freq * N)
  if (pos_N < 1){
    stop("We need to have at least one positive sample. Increase freq or N.")
  }
  neg_N <- N - pos_N
  
  # The first pos_N will match themselves as they are positives but from 
  # pos_N + 1 to N we have negatives, and these will matches the positives
  id <- replicate(N, paste(sample(letters, 5), collapse=''))
  match <- rep(id[1:pos_N], (inv_freq + 1))[1:N]
  data.frame(id, match)
}

# ------------------------------------------------------------------------------
#
# Test the functions with integer sample names which are easy to understand
#
# ------------------------------------------------------------------------------

test <- function(sampleType="int", freq=0.5, N=16, o=2, i=2, downsample=1){
  # Tests the above functions with simple examples. Try it with default params
  # and once you understand the output start playing around with it.
  
  if(sampleType=="int"){
    print_type <- "INTEGER"
    match_df <- sim_matched_samples_int(freq=freq, N=N)
    # create random shuffle of the IDs as it would happen in a dataset
    key <- sample(match_df$id)
  }else{
    print_type <- "STRING"
    match_df <- sim_matched_samples_str(freq=freq, N=N)
    # create random shuffle of the IDs as it would happen in a dataset
    key <- as.character(sample(match_df$id))
  }
  cat("---------------------------------------------------------------------\n")
  cat("            TEST WITH", print_type, "SAMPLE NAMES\n")
  cat("---------------------------------------------------------------------\n")
  
  
  # get linkage with indices
  match_df_ix <- matching_to_indices(match_df, key)
  # let's do a 2-fold outer and 2-fold inner CV
  ncv <- nested_cv_matched_ix(match_df_ix, outer_fold_n=o, inner_fold_n=i, 
                           shuffle=F, downsample=downsample)
  # same example but randomly shuffled folds
  # ncv <- nested_cv(match_df_ix, outer_fold_n=3, inner_fold_n=2, shuffle=T)
  
  print_header <- function(str){
    cat("\n-----------------------------------------\n")
    cat(str, "\n")
    cat("-----------------------------------------\n")  
  }
  
  print_header("original matching dataframe")
  print(match_df)
  
  print_header("key, i.e. order of samples in real data")
  print(key)
  
  match_df_sh <- match_df
  rownames(match_df_sh)  <- match_df_sh$id
  match_df_sh <- match_df_sh[key,]
  rownames(match_df_sh)  <- NULL
  
  print_header("matching dataframe reordered by key")
  print(match_df_sh)
  
  print_header("linkage with indices")
  print(match_df_ix)
  
  print_header("CV info")
  print(ncv)
}

# run tests
# test()
# test(N=32, downsample=.5)
# test("str", N=48, downsample=.5, freq=.25, i=3)