# ------------------------------------------------------------------------------
#
#             Various ways of using clustering to downsample data
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Define wrapper functions for hierarchical and k-means clustering
# ------------------------------------------------------------------------------

do_hclust <- function(data, k=3, dist_m="euclidean", agg_m="complete"){
  # This function performs fast hierarchical clustering on a piece of data and 
  # cuts the resulting tree into k clusters. It returns a list where each sample
  # of data has a cluster membership number.
  library(fastcluster)
  
  # sanity checks
  dists <- c("euclidean", "maximum", "manhattan", "canberra", "binary", 
             "minkowski")
  if(!dist_m %in% dists)
    stop("dist_m must be one of: ", paste(dists, collapse=", "))
  
  aggs <- c("single", "complete", "average", "mcquitty", "ward.D", "ward.D2", 
            "centroid","median")
  if(!agg_m %in% aggs)
    stop("agg_m must be one of: ", paste(aggs, collapse=", "))
  
  # calculate pairwise distance within negative class
  d <- dist(data, method=dist_m)
  
  # do fast hierarchical clustering
  hc <- fastcluster::hclust(d, method=agg_m)
  
  # cut the resulting tree so we get pos_N clusters
  clusters <- cutree(hc, k=k)
}

do_kmeans <- function(data, k=3, rounds=5){
  # This functions performs fast kmeans clustering rounds times with kmeans++
  # initialisation. It then choosest the best run and returns it's centroids and 
  # clusters.
  library(yakmoR)
  
  kmod <- yakmoR::orthoKMeansTrain(as.matrix(data), k=k, rounds=rounds)
  
  # choose best run and return its cluster_memberships and centroids
  best_ix <- order(kmod$obj, decreasing=T)[1]
  centroids <- as.data.frame(kmod$centers[best_ix])
  colnames(centroids) <- colnames(data)
  cluster_membership <- kmod$cluster[best_ix]
  return(list(centroids=centroids, cluster_membership=cluster_membership))
}

# ------------------------------------------------------------------------------
# Functions for calculating centroids, and nearest points to centroids
# ------------------------------------------------------------------------------

get_centroids <- function(data, cluster_membership, method=mean){
  # Given a dataset and a clusters vector, holding the cluster membership for 
  # each sample in the data, it will return the centroids (mean or median) for
  # each cluster.
  
  # this will extract the mean of each feature for each centroid
  get_centroid_mean_median <- function(i, method){
    unlist(lapply(data[cluster_membership==i,], mean))
  }
  
  # get centroid for all clusters
  clusters <- unique(cluster_membership)
  centroids <- lapply(clusters, get_centroid_mean_median, method=method)
  centroids <- as.data.frame(t(as.data.frame(centroids)))
  rownames(centroids) <- as.character(unique(cluster_membership))
  centroids
}

get_closest_witihin_cluster_points <- function(data, cluster_membership, 
                                              centroids, dist_m="euclidean", n){
  # Given a dataset, a vector of cluster membership of each sample and a list of
  # centroids in the space spanned by the data, we return the n closest samples
  # within each cluster. It returns a index of the samples in data, that should 
  # be kept. n should be a list with length = number of centroids, each element 
  # defining the number of negatives that should be returned for the given 
  # centroid. If it's just an integer, it will be repeated #centroid times.
  # Centroids should be a data frame with rownames corresponding to the actual
  # items in cluster_membership, i.e. it needn't to be an increasing list of int
  
  library(fields)

  # sanity checks
  clusters <- rownames(centroids)
  n_clusters <- length(unique(cluster_membership))
  #if (n_clusters != dim(centroids)[1])
  #  stop("The number of clusters must match the number of centroids.")
  if (dim(data)[2] != dim(centroids)[2])
    stop("The dimension of the data and the centroids must be the same.")
  dists <- c("euclidean", "maximum", "manhattan", "canberra", "binary", 
             "minkowski")
  if(!dist_m %in% dists)
    stop("dist_m must be one of: ", paste(dists, collapse=", "))
  if (inherits(n, c("numeric", "integer")))
    n <- setNames(as.list(rep(n, n_clusters)), clusters)
  if (length(n) != dim(centroids)[1])
    stop("Num of centroids does not match num of required closest points.")
  
  get_pair_dist <- function(i){
    # This simply returns the distance between the centroid and a sample from
    # the cluster.
    dist(rbind(cluster_data[i, ], centroid), method=dist_m)
  }
  
  # variables for keeping track of results
  data_ix <- 1:nrow(data)
  samples_closest_ix <- c()
  
  # Iterate through the cluster centroids and get the closest points to them
  for (c in clusters){
    centroid <- centroids[as.character(c), ]
    cluster_data_ix <- data_ix[cluster_membership==c]
    cluster_data <- data[cluster_data_ix, ]
    # get distance of each point in the cluster to the centroid
    centroid_cluster_dists <- fields::rdist(centroid, cluster_data)
    # get closest n
    closest_n_ix <- order(centroid_cluster_dists)[1:n[[c]]]
    # if the user asked for more n than we have, ignore it
    closest_n_ix <- closest_n_ix[!is.na(closest_n_ix)]
    # save selected rows
    samples_closest_ix <- c(samples_closest_ix, cluster_data_ix[closest_n_ix])
  }
  samples_closest_ix
}

# ------------------------------------------------------------------------------
# Clustering negatives (majority class)
# ------------------------------------------------------------------------------

cluster_negatives <- function(dataset, ratio=1, method="hclust", matching=F, 
                              ncv=NULL, dist_m="euclidean", agg_m="complete"){
  # This function takes in an binary mlr dataset, clusters the negative samples
  # hierarchically based on the distance measure and agglomeration method. Then
  # it cuts the tree to form as many clusters as many positive samples we have.
  # This is the default, but changing the ratio>1, this could be different, e.g.
  # 10 negative to 1 positive, with ratio=10. Alternatively it can cluster the 
  # negatives using kmeans. Finally it returns a new mlr dataset, where the 
  # negatives have been replaced with their cluster ceontroids. 
  
  # If matching=T, user must provide an ncv data frame that has an id and a 
  # match column and which defines the matching between the positive samples and
  # negatives (positives match their own id). For ratio=1, the function then 
  # finds the centroid for each negative cluster, i.e. centroid of the negative 
  # samples that are matched to a given positive. If the ratio > 1, it gets the 
  # ratio number of negatives which are closest to the centroids.
  
  # sanity checks
  if (!inherits(dataset, "ClassifTask"))
    stop("dataset must be an mlr classification dataset ")
  if (length(dataset$task.desc$class.levels) > 2)
    stop("This function only works for binary outcome variables.")
  
  # cut and split data 
  data <- getTaskData(dataset)
  target <- dataset$task.desc$target
  positive_flag <- dataset$task.desc$positive
  pos_ix <- data[[target]] == positive_flag
  neg_ix <- data[[target]] == dataset$task.desc$negative
  pos_N <- sum(pos_ix)
  if (sum(neg_ix) < pos_N)
    stop("This function only makes sense if we have more neg than pos samples.")
  neg_data <- data[neg_ix,]
  neg_data <- data_without_target(neg_data, target)
  # normalize so euclidean distance based clustering is sensibe across features
  # neg_data <- normalizeFeatures(neg_data, method="standardize")
  pos_data <- data[pos_ix,]
  
  if (matching){
    # we have matching, no clustering is needed, we just find centroids of 
    # negatives for each positive and return the closest n to the centroid
    if (is.null(ncv))
      stop("You must provide an ncv data frame!")
    
    # store the positives part of the ncv
    pos_ncv <- ncv[pos_ix,]
    # cluster membership is simply defined by the matching
    cluster_membership <- ncv[neg_ix, "match"]
    # find the centroids of each negative match cluster
    centroids <- get_centroids(neg_data, cluster_membership, method=mean)
    # keep only the first match of every cluster in the negative part of ncv
    neg_ncv <- ncv[which(neg_ix),]
    neg_ncv <-  ncv[!duplicated(ncv$match),]
    
    # if we want more than 1 sample per positive we have to find closest samples 
    # to the centroid and we cannot simply use the centroid
    if (ratio > 1){
      closest <- get_closest_witihin_cluster_points(neg_data, cluster_membership,
                                                    centroids, n=ratio)
      # downsample negative data to only keep the closest
      centroids <- neg_data[closest,]
      # downsample the negative part of ncv to only hold the closest samples
      neg_ncv <- ncv[which(neg_ix)[closest],]
    }
    # merge pos and neg ncv
    ncv <- rbind(pos_ncv, neg_ncv)
  }else{
    # cluster data with either hclust or kmeans and get the clusters' centroids
    c_num <- as.integer(pos_N * ratio)
    if (c_num >= sum(neg_ix))
      stop("The number of clusters have to be less than the number of negs.")
    if (method == "hclust"){
      cluster_membership <- do_hclust(neg_data, c_num, dist_m, agg_m)
      centroids <- get_centroids(neg_data, cluster_membership, method=mean)
    }else if (method == "kmeans"){
      km <- do_kmeans(neg_data, c_num)
      centroids <- km$centroids
    }
  }
  
  # add back the factor target - bunch of zeros
  centroids[target] <- factor(rep(dataset$task.desc$negative, dim(centroids)[1]), 
                              levels=levels(data[[target]]))
  
  # merge positives with the downsampled negatives
  new_data <- rbind(pos_data, centroids)
  new_dataset <- makeClassifTask(id=dataset$task.desc$id, data=new_data, 
                                 target=target, positive=positive_flag)
  
  # return results
  if (matching){
    return(list(dataset=new_dataset, ncv=ncv))
  }else{
    return(new_dataset)
  }
}

# ------------------------------------------------------------------------------
# Clustering positives (minority class)
# ------------------------------------------------------------------------------

cluster_positives <- function(dataset, ratio=1, k, method="hclust", matching=F, 
                              ncv=NULL, dist_m="euclidean", agg_m="complete"){
  # This function clusters the positive (minority) class into k clusters using
  # kmeans or hierarchical clustering. Then it finds the closest negative 
  # samples to the centroids of these clusters. It can preserve matching if
  # needed.
  
  library(hash)
  library(fields)
  
  # sanity checks
  if (!inherits(dataset, "ClassifTask"))
    stop("dataset must be an mlr classification dataset ")
  if (length(dataset$task.desc$class.levels) > 2)
    stop("This function only works for binary outcome variables.")
  
  # cut and split data 
  data <- getTaskData(dataset)
  target <- dataset$task.desc$target
  positive_flag <- dataset$task.desc$positive
  negative_flag <- dataset$task.desc$negative
  pos_ix <- data[[target]] == positive_flag
  neg_ix <- data[[target]] == negative_flag
  pos_N <- sum(pos_ix)
  neg_N <- sum(neg_ix)
  if (neg_N <= pos_N)
    stop("This function only makes sense if we have more neg than pos samples.")
  neg_data <- data[neg_ix,]
  neg_data <- data_without_target(neg_data, target)
  # normalize so euclidean distance based clustering is sensibe across features
  # neg_data <- normalizeFeatures(neg_data, method="standardize")
  pos_data <- data[pos_ix,]
  pos_data <- data_without_target(pos_data, target)
  # pos_data <- normalizeFeatures(pos_data, method="standardize")
  
  # cluster pos data with hclust or kmeans into k clusters, then get centroids
  if (method == "hclust"){
    pos_cluster_membership <- do_hclust(pos_data, k, dist_m, agg_m)
    pos_centroids <- get_centroids(pos_data, pos_cluster_membership, method=mean)
  }else if (method == "kmeans"){
    km <- do_kmeans(pos_data, k, rounds=10)
    pos_cluster_membership <- unlist(km$cluster_membership)+1
    pos_centroids <- km$centroids
  }
  
  # get number of samples in each positive cluster and turn it into a named list
  ns <- table(pos_cluster_membership)[unique(pos_cluster_membership)]
  ns <- as.list(ns*ratio)
  
  # depending on matching we need to proceed differently
  if (matching){
    if (is.null(ncv))
      stop("You must provide an ncv data frame!")
    pos_ncv <- ncv[pos_ix,]
    neg_ncv <- ncv[neg_ix,]
  
    # make a hash: pos_id: cluster_mem
    h <-hash(keys=as.character(pos_ncv$id), values=pos_cluster_membership)
    # map match column of negs to pos centroid centers
    cluster_membership <- unlist(lapply(neg_ncv$match, 
                                        function(x) h[[as.character(x)]]))
    # get closest matching samples to the positive centroids
    closest <- get_closest_witihin_cluster_points(neg_data, cluster_membership,
                                                  pos_centroids, n=ns)
    # downsample negative data to only keep the closest
    centroids <- neg_data[closest,]
    # downsample the negative part of ncv to only hold the closest samples
    neg_ncv <- ncv[which(neg_ix)[closest],]
    # merge pos and neg ncv
    ncv <- rbind(pos_ncv, neg_ncv)
  }else{
    # find which pos_centroid is closest to each negative data point. kmeans
    # does not work here because many times certain pos_centroids end up with 0
    # assigned negative samples and this causes an error from base::kmeans, try:
    # km <- kmeans(neg_data, pos_centroids, iter.max=1)
    dists <- fields::rdist(neg_data, pos_centroids)
    # if zero negatives are assigned to a positive cluster/centroid we will not
    # select any negatives for that cluster later
    cluster_membership <- apply(dists, 1, function(x) which(x==min(x)))
    
    # then find the negs that are closest to their pos cluster
    closest <- get_closest_witihin_cluster_points(neg_data, cluster_membership,
                                                  pos_centroids, n=ns)
    # downsample negative data to only keep the closest
    centroids <- neg_data[closest,]
  }
  
  # add back the factor target to both pos and neg data
  pos_data[target] <- factor(rep(positive_flag, dim(pos_data)[1]), 
                              levels=levels(data[[target]]))
  centroids[target] <- factor(rep(negative_flag, dim(centroids)[1]), 
                              levels=levels(data[[target]]))
  
  # merge positives with the downsampled negatives
  new_data <- rbind(pos_data, centroids)
  new_dataset <- makeClassifTask(id=dataset$task.desc$id, data=new_data, 
                                 target=target, positive=positive_flag)
  
  # return results
  if (matching){
    return(list(dataset=new_dataset, ncv=ncv))
  }else{
    return(new_dataset)
  }
}
  