
make_matched_samples = function(freq=.1, N=100){
  # Given the frequency of the positive class and the size of the population
  # it returns a data.frame with two columns (id, match) where each positive
  # patient is matched with the right amount of negatives.
  
  pos_N = floor(freq * N)
  neg_N = N - pos_N
  
  # The first pos_N will match themselves as they are positives but from 
  # pos_N + 1 to N we have negatives, and then this number matches the positives
  match = as.character(rep(seq(1, pos_N), inv_freq))
  id = as.character(seq(1, N))
  
  patient_table = data.frame(id, match)
  return(patient_table)
}

nested_cv = function(patient_table, outer_fold_n=5, inner_fold_n=5, shuffle=TRUE){
  # This function takes in a data.frame with two columns. The first column is "id" 
  # and it is the unique ID of each sample. The 2nd col is "match" which what links 
  # the negative cases to the positive ones. The function assumes that the top N
  # rows of this data.frame are the positive ones. Positives are automatically 
  # found, as they match themselves. It then returns the same data.frame with 
  # two additional columns: outer and inner, which hold the membership of each
  # sample in the nested CV process.
  
  # find positive patients, i.e. they match themselves
  ids = as.character(patient_table$id)
  matches = as.character(patient_table$match)
  pos_ix = which(ids == matches)
  pos = patient_table[pos_ix,]
  pos_id = pos$id
  
  # define dimensions
  N = dim(patient_table)[1]
  pos_len = length(pos_id)
  
  # do we have enough positive samples?
  if (outer_fold_n * inner_fold_n >= length(pos_id)){
    stop("Number of positive samples have to be greater than outer_fold * inner_fold.")
  }
  
  # OUTER SPLIT
  # we first do the two-fold cv for the positives only and then every negative is following their 
  # matched positive. First we create a list of integers, then split in outer_fold_n number 
  # of equal bins
  outer_fold = split(1:pos_len, cut(1:pos_len, outer_fold_n))
  # take those bins and rename the integers going from 1 to N so they go from 1 to outer_fold_n
  start = 1
  end = 0
  inner_fold_tmp = 1:pos_len
  for (i in 1:outer_fold_n) {
    outer_len = length(outer_fold[[i]])
    if (i > 1){
      start = start + outer_len
    }
    end = end + outer_len
    outer_fold[[i]] = rep(i, outer_len)
    
    # INNER SPLIT
    # do the same as above but now within each outerfold
    inner_fold = split(1:outer_len, cut(1:outer_len, inner_fold_n))
    for (j in 1:inner_fold_n) {
      inner_len = length(inner_fold[[j]])
      inner_fold[[j]] = rep(j, inner_len)
    }
    inner_fold_tmp[start:end] = unlist(inner_fold, use.names = F)
  }
  outer_fold = unlist(outer_fold, use.names = F)
  inner_fold = inner_fold_tmp
  
  # bring in the matching negatives for each positive sample
  outer = 1:N
  inner = 1:N
  
  # shuffle pos_ids make sure they're random
  if (shuffle){
    pos_id = sample(pos_id)
    patient_table[pos_ix,] = pos_id
  }
  
  # build a lookup table for the CV-ed positive samples
  cv_pos = data.frame(outer_fold, inner_fold)
  rownames(cv_pos) = pos_id
  
  # get matching negative samples for each positive
  outer = unlist(lapply(patient_table$match, function(x) cv[x,]$outer_fold), use.names=F)
  inner = unlist(lapply(patient_table$match, function(x) cv[x,]$inner_fold), use.names=F)
  
  # add new columns to patient_table
  patient_table["outer_fold"] = outer
  patient_table["inner_fold"] = inner
  
  return (patient_table)
}
  