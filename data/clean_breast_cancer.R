# ------------------------------------------------------------------------------
#
#     Clean breast cancer  dataset and make unmatched and matched versions
#
# ------------------------------------------------------------------------------

library(readr)
source("palab_model/palab_model.R")

# ------------------------------------------------------------------------------
# MAKE UNMATCHED VERSION
# ------------------------------------------------------------------------------

# Load Breast Cancer dataset, impute missing
data(BreastCancer, package="mlbench")
df <- BreastCancer
target <- "Class"
df$Id <- NULL

# Make sure that the negative class is 0 and the positive is 1, otherwise the
# custom prec@recal perf metric will not work
df[[target]] <- as.factor(as.numeric(factor(df[[target]]))-1)

df <- impute_data(df, target)

# Add randomly generated ID column and check that all of them are unique
df['ID'] = replicate(nrow(df), paste(sample(letters, 5), collapse=''))
if (length(unique(df$ID)) != nrow(df)){
  stop("Try this ID generation again, looks like we have some duplicates.")
}

readr::write_csv(df, "~/PAlab/palab_model/data/breast_cancer.csv")

# ------------------------------------------------------------------------------
# MAKE VERSION WITH ARBITRARY MATCHING 
# ------------------------------------------------------------------------------

# Make random linkage between malignant and benign samples. The positives will
# match themselves, that's how we know they're positive. 
id <- 1:nrow(df)
match <- id
target_col <- df[[target]]
pos_ix <- which(target_col == 1)
neg_ix <- which(target_col == 0)
pos_N <- length(pos_ix)
neg_N <- length(neg_ix)
freq <- pos_N / neg_N
pos_multiplier <- round(1/freq*pos_N)
match[neg_ix] = rep(match[pos_ix], pos_multiplier)[1:length(neg_ix)]
match_df = data.frame(id, match)

# Define nested CV with maching: outer 3-fold, inner 3-fold
ncv <- nested_cv_matched_ix(match_df, outer_fold_n=3, inner_fold_n=3, shuffle=F)

# Add it to df and save it
df["outer_fold"] <- ncv$outer_fold
df["inner_fold"] <- ncv$inner_fold
readr::write_csv(df, "~/PAlab/palab_model/data/breast_cancer_matched.csv")