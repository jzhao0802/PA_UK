# ------------------------------------------------------------------------------
#
#                Clean breast cancer subgroup dataset
#
# ------------------------------------------------------------------------------

library("dplyr")
library("tidyr")
library("readr")

input <- "data/subgroup_data.csv"
target <- "Diff_12monVA_IdxVA"
data <- readr::read_csv(input, guess_max=8000)

# delete cols that make up the unique ID
data[["Eye_Laterality"]] <- NULL
data[["Patient_ID"]] <- NULL

# the target is the difference of two measurements, remove the 2nd
data[["VA_12mon"]] <- NULL
data[["Perc_12monVA_IdxVA"]] <- NULL

# convert sex to numbers
data$Gender <- as.numeric(factor(data$Gender))

# convert dates to UNIX timestamps
data$Idx_Dt <- as.numeric(as.POSIXct(data$Idx_Dt, format="%d/%m/%Y"))

# let's have a look at the data
dplyr::glimpse(data)

# check the target
hist(data[[target]])

# check if all input vars are numeric
if(sum(sapply(data, is.numeric)) < dim(data)[2]){
  warning("There are non-numeric columns. These will be removed now.")
  to_remove <- -which(!(sapply(data, is.numeric)))
  data <- data[,to_remove]
}

#check if there are missing values in the target
if(sum(is.na(data[[target]])) > 0){
  warning("There are missing values in the target column. 
          These will be removed now.")
  to_remove <- -which(is.na(data[[target]]))
  data <- data[to_remove,]
}

# replace NA's with median in each column
impute_median=function(x){
  x <- as.numeric(as.character(x))
  x[is.na(x)] <- median(x, na.rm=TRUE)
  x 
}
data <- as.data.frame(sapply(data, impute_median))

# save resulting data_frame
readr::write_csv(data, "data/subgroup_data_cleared.csv", row.names=T)

# ------------------------------------------------------------------
# RUN SIMPLE LIN REG MODEL AND SAVE RESULTS
# ------------------------------------------------------------------

input <- "data/subgroup_data_cleared.csv"
target <- "Diff_12monVA_IdxVA"
data <- readr::read_csv(input, guess_max=8000)

# simple linear model
lin_reg <- lm(Diff_12monVA_IdxVA ~ ., data)
sink("summary.txt")
summary(lin_reg)
sink()
lin_reg_sum <- summary(lin_reg)
write.csv(lin_reg_sum$coefficients, file="summary_coef.csv") 