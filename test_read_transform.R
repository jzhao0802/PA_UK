library(palab)
setwd("~/PAlab/palab_test")

# ---------------------------------------------------------
# TEST WITH SUBGROUP
# ---------------------------------------------------------

input = "input_data/subgroup_data.csv"
var_conf = var_config_generator(input_csv = input, output = "subgroup_var_config",
                     output_dir = ".", sample_rows = 5000)

# edit the subgroup_var_config.csv and make the ID column key and binary variables categorical
r = read_transform(input_csv = input, var_config = "subgroup_var_config_edited.csv",
                   missing_values = "NA", output = "subgroup_transformed",
                   output_dir = "results", output_csv = T,
                   outcome_var = "Diff_VA12mon_VAIdx", report_csv = "subgroup_report")

# we need to read the var_config in temporarily (till Tim fixes the function)
var_conf = readr::read_csv("subgroup_var_config_edited.csv")
univariate_stats(r$output, var_config = var_conf, output="subgroup_univariate", output_dir = "results")


# ---------------------------------------------------------
# TEST WITH BI
# ---------------------------------------------------------

# generate and curate the var_config on the smaller positive rows
input = "\\\\woksfps01\\RWES Central team P&BD\\Predictive_Analytics\\Projects\\BI_IPF\\01_data\\all_features_pos.csv"
var_conf = var_config_generator(input_csv = input, output = "BI_var_config",
                                output_dir = ".", sample_rows = 5000)

# edit the BI_var_config_edited.csv and make the ID column key and binary variables categorical
r = read_transform(input_csv = input, var_config = "BI_var_config_edited.csv",
                   missing_values = "NA", output = "BI_pos_transformed",
                   output_dir = "results", output_csv = T, report_csv = "BI_pos_report")

var_conf = readr::read_csv("BI_var_config_edited.csv")
univariate_stats(r$output, var_config = var_conf, output="BI_pos_univariate", output_dir = "results")

# run it on the much larger negative cohort as well
input = "\\\\woksfps01\\RWES Central team P&BD\\Predictive_Analytics\\Projects\\BI_IPF\\01_data\\all_features_neg.csv"
r = read_transform(input_csv = input, var_config = "BI_var_config_edited.csv",
                   missing_values = "NA", output = "BI_neg_transformed",
                   output_dir = "results", output_csv = F, report_csv = "BI_neg_report")

var_conf = readr::read_csv("BI_var_config_edited.csv")
univariate_stats(r$output, var_config = var_conf, output="BI_neg_univariate", output_dir = "results")

