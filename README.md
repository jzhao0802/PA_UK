# PAlab modelling module

This repo holds the modelling functionality of PAlab. Several models are supported both with and without matching. Parallelized nested CV, random and regular grid search, numerous performance metrics and many useful diagnostic plots are included as well.

So far we support classification with:
* Logistic regression
* Logistic regression with elastic net and lasso
* Decision trees
* Random forest
* SVM with radial kernel

All of the code is heavily relying ong the mlR package. Whenever in doubt (if the comments in the code aren't helpful enough), please check the documentation of mlR or the notebooks. 

## Module structure

* __data__: A few example scripts how to clean, impute, preprocess datasets. The resulting datasets of these scripts are used by the template scripts.
* __dev__: Uncategorised, development scripts, probably not very important unless you want to add functionality to palab_model.
* __matched__: Template scripts with matching.
* __notebooks__: Please read these before asking for help. It's highly likely that you'll find your answer here, or at least learn a great deal about mlR and palab_model.
* __output__: Folder for some output plots and text files, not very interesting.
* __palab_model:__ Holds the helper functions that make the actual analysis templates in the matched and unmatched folders neater and cleaner. mlR uses camelCase names for functions, I used pythonic_names, so it's easy to see if a function of interest is part of PAlab or mlR.
* __unmatched__: Template scripts without matching.
