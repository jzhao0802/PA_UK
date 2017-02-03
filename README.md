# PAlab modelling module

This repo holds the modelling functionality of PAlab. Several models are supported both with and without matching. Parallelized nested CV, random and regular grid search, numerous performance metrics and many useful diagnostic plots are included as well.

All of the code is heavily relying ong the mlR package. Whenever in doubt (if the comments in the code aren't helpful enough), please check the documentation of mlR or the notebooks. 


## Supported models
All models work with matched and unmatched data, see corresponding folders for the template scripts.

* Binary classification
    * Logistic regression
    * Logistic regression with elastic net and lasso
    * Decision trees
    * Random forest
    * SVM with radial and polynomial kernel
    * XGboost


## Module structure

* __data__: A few example scripts how to clean, impute, preprocess datasets. The resulting datasets of these scripts are used by the template scripts.
* __dev__: Uncategorised, development scripts, probably not very important unless you want to add functionality to palab_model.
* __templates__: Template scripts with for all available models.
* __notebooks__: Please read these before asking for help. It's highly likely that you'll find your answer here, or at least learn a great deal about mlR and palab_model.
* __palab_model:__ Holds the helper functions that make the actual analysis templates in the matched and unmatched folders neater and cleaner. mlR uses camelCase names for functions, I used pythonic_names, so it's easy to see if a function of interest is part of PAlab or mlR.


## How to use this?
* Make the working directory the this top folder. Each script will assume you are at the top and  not in matched or unmatched. 
* A number of packages are required to run these scripts. Please pay attention to the error messages you get, you might be missing some packages that are not listed below. Here's a non-exhaustive list:
    * mlr - development branch
        * Some functionality in these scripts require the development version of mlr. This can easily install from github (usually, if you're not sitting behind IMS's firewall). 
        * To do this follow the github tutorial I wrote at: \\woksfps01\RWES Central team P&BD\Predictive_Analytics\Infrastructure\Tutorials\Git\HowToGetGitToWorkAtIMS.txt
        * Then install `devtools`, `httr` and do:
        ```
        library(httr)
        library(devtools)
        # on your laptop after you've setup Cntlm
        set_config(use_proxy(url="http://localhost", port=3128, username="username",password="password"))
        # on server100 or server101 
        set_config(use_proxy(url="frboucpxy01.internal.imsglobal.com", port=8080, username="username",password="password"))
        devtools::install_github("mlr-org/mlr")
        ```
    * ParamHelpers - dev version
        * Just like above:
        ```
        library(httr)
        library(devtools)
        # on your laptop after you've setup Cntlm
        set_config(use_proxy(url="http://localhost", port=3128, username="username",password="password"))
        # on server100 or server101 
        set_config(use_proxy(url="frboucpxy01.internal.imsglobal.com", port=8080, username="username",password="password"))
        devtools::install_github("berndbischl/ParamHelpers")
        ```
    * ranger
    * randomForest
    * kernlab
    * xgboost
    * glmnet
    * rpart
    * rpart.plot
    * rpart.utils
    * prroc
    * earth
    * rattle
    * gridExtra
    * plotmo
    * readr, dtplyr, tidyr, stringi 
    * parallel
    * parallelMap
    * ggplot2
    * yakmoR
    * fastcluster