# ------------------------------------------------------------------------------
#
#             Various ways of using clustering to downsample data
#
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
  
  kmod <- yakmoR::orthoKMeansTrain(as.matrix(neg_data), k=k, rounds=rounds)
  
  # choose best run and return its cluster_memberships and centroids
  best_ix <- order(kmod$obj, decreasing=T)[1]
  centroids <- as.data.frame(kmod$centers[best_ix])
  colnames(centroids) <- colnames(data)
  cluster_membership <- kmod$cluster[best_ix]
  return(list(centroids=centroids, cluster_membership=cluster_membership))
}

get_centroids <- function(data, cluster_membership, method=mean){
  # Given a dataset and a clusters vector, holding the cluster membership for 
  # each sample in the data, it will return the centroids (mean or median) for
  # each cluster.
  
  # this will extract the mean of each feature for each centroid
  get_centroid_mean_median <- function(i, method){
    unlist(lapply(data[clusters==i,], mean))
  }
  
  # get centroid for all clusters
  clusters <- unique(cluster_membership)
  centroids <- lapply(clusters, get_centroid_mean_median, method=method)
  centroids <- as.data.frame(t(as.data.frame(centroids)))
  rownames(centroids) <- NULL
  centroids
}

get_closest_witihin_cluster_points <- function(data, cluster_membership, 
                                              centroids, dist_m="euclidean", n){
  # Given a dataset, a vector of cluster membership of each sample and a list of
  # centroids in the space spanned by the data, we return the n closest samples
  # within each cluster. It returns a dataframe where the 
  
  # sanity checks
  clusters <- unique(cluster_membership)
  n_clusters <- length(clusters)
  if (n_clusters != dim(centroids)[1])
    stop("The number of clusters must match the number of centroids.")
  if (dim(data)[2] != dim(centroids)[2])
    stop("The dimension of the data and the centroids must be the same.")
  dists <- c("euclidean", "maximum", "manhattan", "canberra", "binary", 
             "minkowski")
  if(!dist_m %in% dists)
    stop("dist_m must be one of: ", paste(dists, collapse=", "))
  
  get_pair_dist <- function(i){
    # This simply returns the distance between the centroid and a sample from
    # the cluster.
    dist(rbind(cluster_data[i, ],centroid), method=dist_m)
  }
  
  data_ix <- 1:nrow(data)
  n <- as.integer(n)
  # variables holding the results
  samples_closest_ix <- c()
  cluster_num <- c()
  
  # Iterate through the cluster centroids and get the closest points to them
  for (c in clusters){
    centroid <- centroids[c, ]
    cluster_data_ix <- data_ix[clusters==c]
    cluster_data <- data[clusters==c, ]
    cluster_data_N <- length(cluster_data_ix)
    # get distance of each point in the cluster to the centroid
    centroid_cluster_dists <- unlist(lapply(1:cluster_data_N, get_pair_dist))
    # get closest n
    closest_n_ix <- order(centroid_cluster_dists)[1:n]
    # save selected rows
    samples_closest_ix <- c(samples_closest_ix, cluster_data_ix[closest_n_ix])
    cluster_num <- c(cluster_num, rep(c, n))
  }
  data.frame(closest=samples_closest_ix, cluster=cluster_num)
}

cluster_negatives <- function(dataset, ratio=1, method="hclust", matched=F, 
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
  pos_data <- data[pos_ix,]
  
  if (matched){
    # we have matching, no clustering is needed, we just find centroids of 
    # negatives for each positive and return the closest n to the centroid
    if (is.null(ncv))
      stop("You must provide an ncv data frame!")
    cluster_membership <- ncv[neg_ix,"match"]
    centroids <- get_centroids(neg_data, cluster_membership, method=mean)
    # if we want more than 1 sample per positive we have to find closest ones
    if (ratio > 1){
      closest <- get_closest_witihin_cluster_points(neg_data, cluster_membership,
                                                    centroids, n=ratio)
      centroids <- neg_data[closest$closest,]
    }
  }else{
    # cluster data with either hclust or kmeans and get the clusters' centroids
    c_num <- as.integer(pos_N * ratio)
    if (method == "hclust"){
      cluster_membership <- do_hclust(neg_data, c_num, dist_m, agg_m)
      centroids <- get_centroids(neg_data, cluster_membership, method=mean)
    }else if (method == "kmeans"){
      km <- do_kmeans(neg_data, c_num)
      centroids <- km$centroids
    }
  }
  
  # add back the factor target - bunch of zeros
  centroids[target] <- factor(rep(dataset$task.desc$negative, pos_N), 
                              levels=levels(data[[target]]))
  
  # merge positives with the downsampled negatives
  new_data <- rbind(pos_data, centroids)
  new_dataset <- makeClassifTask(id=dataset$task.desc$id, data=new_data, 
                                 target=target, positive=positive_flag)
  
  # return results
  if (matching){
    # downsample the ncv as well
    ncv <- ncv[c(which(pos_ix), c$closest),] 
    return(list(dataset=new_dataset, ncv=ncv))
  }else{
    return(new_dataset)
  }
}