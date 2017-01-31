# ------------------------------------------------------------------------------
#
#             Various ways of using clustering to downsample data
#
# ------------------------------------------------------------------------------

hclust_downsampling <- function(dataset, dist_m="euclidean", agg_m="complete"){
  # This function takes in an binary mlr dataset, clusters the negative samples
  # hierarchically based on the distance measure and agglomeration method. Then
  # it cuts the tree to form as many clusters as many positive samples we have.
  # Finally it returns a new mlr dataset, where the negatives have been replaced
  # with the cluster ceontroids, so that now #pos=#neg.
  
  library(fastcluster)
  
  # sanity checks
  if (!inherits(dataset, "ClassifTask"))
    stop("dataset must be an mlr classification dataset ")
  if (length(dataset$task.desc$class.levels) > 2)
    stop("This function only works for binary outcome variables.")
  
  dists <- c("euclidean", "maximum", "manhattan", "canberra", "binary", 
             "minkowski")
  if(!dist_m %in% dists)
    stop("dist_m must be one of: ", paste(dists, collapse=", "))
  
  aggs <- c("single", "complete", "average", "mcquitty", "ward.D", "ward.D2", 
            "centroid","median")
  if(!agg_m %in% aggs)
    stop("agg_m must be one of: ", paste(aggs, collapse=", "))
  
  # cut and split data 
  data <- getTaskData(dataset)
  target <- dataset$task.desc$target
  pos_ix <- data[[target]] == dataset$task.desc$positive
  neg_ix <- data[[target]] == dataset$task.desc$negative
  pos_N <- sum(pos_ix)
  if (sum(neg_ix) < pos_N)
    stop("This function only makes sense if we have more neg than pos samples.")
  neg_data <- data[neg_ix,]
  
  # calculate pairwise distance within negative class
  d <- dist(neg_data, method=dist_m)
  
  # do fast hierarchical clustering
  clusters <- fastcluster::hclust(d, method=agg_m)
  
  # cut the resulting tree so we get pos_N clusters
  cluster_membership <- cutree(clusters, k=pos_N)
  
  # this will extract the mean of each feature for each centroid
  get_centroid_means <- function(i){
    unlist(lapply(neg_x[which(cluster_membership==i),], mean))
  }
  
  # get mean for all cluster centroids
  centroids <- lapply(c(1:pos_N), get_centroid_means)
  centroids <- as.data.frame(t(as.data.frame(centroids)))
  rownames(centroids) <- NULL
  
  # add back the factor target - bunch of zeros
  centroids[target] <- factor(rep(dataset$task.desc$negative, pos_N), 
                              levels=levels(data[[target]]))
  
  new_data <- rbind(pos_data, centroids)
  dataset$task.desc$id
  makeClassifTask(id=dataset$task.desc$id, data=new_data, target=target, 
                  positive=dataset$task.desc$positive)
}
