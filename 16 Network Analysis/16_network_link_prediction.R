required_libs <- c('igraph', 'reshape2', 'LICORS', 'linkprediction',
                   'caret', 'xgboost', 'corrplot', 'e1071', 
                   'MLmetrics', 'ROCR', 'DMwR', 'plyr')
lapply(required_libs, require, character.only=T)

g_orig <- read_graph("dsilt-ml-code/16 Network Analysis/karate/karate.gml", format='gml')
plot(g_orig)

# Verify the graph is fully connected
if (length(unique(clusters(g_orig)$membership))>1) {
  print(clusters(g_orig))
} else {
  print('All OK')
}


# Set up utility functions


# Function to randomly delete x links from the data, making sure not to disconnect the graph
delete_random_links <- function(g, sample_size=10) {
  set.seed(14)
  nbr_links <- nrow(as_edgelist(g))
  disconnected <- TRUE
  while(disconnected) {
    newg <- g
    links_to_delete <- sample(1:nbr_links, sample_size, replace=FALSE)
    # Zero out the link and its reverse
    fromlist <- c(as_edgelist(newg)[links_to_delete, 1], as_edgelist(newg)[links_to_delete, 2])
    tolist <- c(as_edgelist(newg)[links_to_delete, 2], as_edgelist(newg)[links_to_delete, 1])
    newg[from=fromlist, to=tolist] <- 0
    if(is.connected(newg)) {
      disconnected <- FALSE
    }
  }
  return(list(newg, links_to_delete))
}

# Function to convert the feature matrix to a feature list
feature_matrix_to_feature_list <- function(fm, degree_vector) {
  # Make sure the row and column names match the node IDs
  row.names(fm) <- names(degree_vector)
  colnames(fm) <- names(degree_vector)
  fl <- melt(fm)
}

# Function to calculate features for a graph
create_topological_features <- function(g) {
  
  # Adjacency matrix
  ajm <- as_adj(g, type="both", sparse=F)/2  # Divide by 2 to reduce the max to 1
  
  # degree vector
  degv <- degree(g, mode='all')/2
  
  # common neighbors matrix (number of shared connections)
  cnm <- cocitation(g)/2
  
  # Jaccard similarity matrix
  jsm <- similarity(g, method="jaccard")
  
  # Dice similarity matrix
  dsm <- similarity(g, method="dice")
  
  # Adamic-Adar Index
  aaim <- similarity.invlogweighted(g)
  
  # Resource Allocation
  n <- vcount(g)
  ram <- matrix(integer(n^2), nrow=n)   
  neighbors <- neighborhood(g, order=1)
  neighbors <- lapply(neighbors, function(x) x[-1])
  for (k in seq(n)){
    tmp <- neighbors[[k]]
    l <- degv[[k]]
    if (l > 1){
      for (i in 1:(l-1)){
        n1 <- tmp[i]
        for (j in (i+1):l){
          n2 <- tmp[j]
          ram[n1, n2] <- ram[n1, n2] + 1/l
          ram[n2, n1] <- ram[n2, n1] + 1/l
        }
      }
    }
  }
  
  # Preferential Attachment
  pam <- proxfun(g, method='pa', value='matrix')
  
  # Geodesic Distance
  gdm <- shortest.paths(g)
  gdm[is.infinite(gdm)] <- vcount(g) + 1
  gdm <- 1/gdm
  diag(gdm) <- 0
  
  # Katz Index
  kim <- proxfun(g, method='katz', value='matrix')
  
  # Local Paths Index
  epsilon <- 0.01
  lpim <- ajm %*% ajm
  lpim <- lpim + lpim %*% ajm * epsilon
  
  # Matrix Forest Index
  mfim <- proxfun(g, method='mf', value='matrix')
  
  # Random Walk with Restart
  rwrm <- proxfun(g, method='rwr', value='matrix', alpha=0.3)
  
  # Average Commute Time
  actm <- proxfun(g, method='act', value='matrix')
  
  # Rooted PageRank
  alpha <- 0.01
  identity_m <- diag(nrow(ajm))
  # Normalize rows sums of the adjacency matrix
  najm <- LICORS::normalize(ajm, byrow=T)
  rprm <- (1-alpha)*(1/(identity_m-alpha*najm))
  #rprm <- (1-alpha)*(1/(identity_m-alpha*(degv/ajm)))
  rprm[is.infinite(rprm)] <- 0
  
  #Combine all features
  ajdf <- feature_matrix_to_feature_list(ajm, degv)
  colnames(ajdf) <- c('node1', 'node2', 'link')
  aaidf <- feature_matrix_to_feature_list(aaim, degv)
  colnames(aaidf) <- c('node1', 'node2', 'adamic_adar_index')
  cndf <- feature_matrix_to_feature_list(cnm, degv)
  colnames(cndf) <- c('node1', 'node2', 'common_neighbors')
  dsdf <- feature_matrix_to_feature_list(dsm, degv)
  colnames(dsdf) <- c('node1', 'node2', 'dice_similarity')
  gddf <- feature_matrix_to_feature_list(gdm, degv)
  colnames(gddf) <- c('node1', 'node2', 'geodesic_dist')
  jsdf <- feature_matrix_to_feature_list(jsm, degv)
  colnames(jsdf) <- c('node1', 'node2', 'jaccard_similarity')
  kidf <- feature_matrix_to_feature_list(kim, degv)
  colnames(kidf) <- c('node1', 'node2', 'katz_index')
  lpidf <- feature_matrix_to_feature_list(lpim, degv)
  colnames(lpidf) <- c('node1', 'node2', 'local_paths_index')
  mfidf <- feature_matrix_to_feature_list(mfim, degv)
  colnames(mfidf) <- c('node1', 'node2', 'matrix_forest_index')
  padf <- feature_matrix_to_feature_list(pam, degv)
  colnames(padf) <- c('node1', 'node2', 'preferential_attachment')
  rprdf <- feature_matrix_to_feature_list(rprm, degv)
  colnames(rprdf) <- c('node1', 'node2', 'rooted_pagerank')
  radf <- feature_matrix_to_feature_list(ram, degv)
  colnames(radf) <- c('node1', 'node2', 'resource_allocation')
  rwrdf <- feature_matrix_to_feature_list(rwrm, degv)
  colnames(rwrdf) <- c('node1', 'node2', 'random_walk_w_reset')
  actdf <- feature_matrix_to_feature_list(actm, degv)
  colnames(actdf) <- c('node1', 'node2', 'average_commute_time')
  df <- cbind(ajdf, adamic_adar_index=aaidf[,3], common_neighbors=cndf[,3], 
              dice_similarity=dsdf[,3], geodesic_dist=gddf[,3], 
              jaccard_similarity=jsdf[,3], katz_index=kidf[,3], 
              local_paths_index=lpidf[,3], matrix_forest_index=mfidf[,3],
              preferential_attachment=padf[,3], 
              rooted_pagerank=rprdf[,3], resource_allocation=radf[,3],
              random_walk_w_reset=rwrdf[,3], average_commute_time=actdf[,3])
  degdf <- data.frame(keyName=as.integer(V(g)), value=degv, row.names=NULL)
  df <- merge(x=df, y=rename(degdf, c('keyName'='keyName', 'value'='node2_deg')), by.x='node2', by.y='keyName', all.x=TRUE)
  df <- merge(x=df, y=rename(degdf, c('keyName'='keyName', 'value'='node1_deg')), by.x='node1', by.y='keyName', all.x=TRUE)
  df <- df[df$node1!=df$node2,]  # Remove node relationships with themselves
  df$edge_key <- paste(df$node1, df$node2, sep="_")
  df$link <- ifelse(df$link>0, 1, 0) # Normalize links
  row.names(df) <- df$edge_key
  df$edge_key <- NULL
  
  return(df)
  
}


# Create Train/Test Sets, Remove Links, Create Features


# Make sure to delete links before calculating graph features!!
print(paste("Total Edges:", nrow(as_edgelist(g_orig))))
tmp <- delete_random_links(g_orig, sample_size=10)
g_test <- tmp[[1]]
deleted_links_test <- tmp[[2]]
print(paste("Test Edges:", nrow(as_edgelist(g_test))))
tmp <- delete_random_links(g_test, sample_size=10)
g_train <- tmp[[1]]
deleted_links_train <- tmp[[2]]
print(paste("Train Edges:", nrow(as_edgelist(g_train))))

ground_truth <- create_topological_features(g_orig)
test <- create_topological_features(g_test)
test$link <- ground_truth[row.names(ground_truth) %in% row.names(test), 'link']
train <- create_topological_features(g_train)
train$link <- ground_truth[row.names(ground_truth) %in% row.names(train), 'link']
# Remove extraneous columns
test[,c(1,2)] <- NULL
train[,c(1,2)] <- NULL

# Resample to balance classes using SMOTE
train_rs <- train
train_rs$link <- as.factor(train_rs$link)
train_rs <- SMOTE(link ~ ., data=train_rs, perc.over=200, k=4)
train_rs$link <- as.integer(train_rs$link)-1
table(train$link)
table(train_rs$link)


# Modeling


# XGBoost
train_x <- as.matrix(train_rs[,!colnames(train_rs) %in% c('link')])
train_y <- train_rs$link
test_x <- as.matrix(test[,!colnames(test) %in% c('link')])
test_y <- test$link
xgb_model <- xgboost(data=train_x, label=train_y, 
                     max.depth=15, eta=0.1, nrounds=25, 
                     subsample=0.5, colsample_bytree=0.5,
                     seed=14, nthread=2, 
                     objective="binary:logistic")
xgb_preds_p <- predict(xgb_model, test_x)
xgb_preds <- as.numeric(xgb_preds_p > 0.5)  # Convert the predicted score to a binary value
# Check importances
features <- colnames(train_rs[,!colnames(train_rs) %in% c('link')])
impm <- xgb.importance(features, xgb_model)
xgb.plot.importance(impm, main="Feature Importances by Information Gain")

# Now view the true accuracy of the model, based on the hidden links in the test set
names(xgb_preds) <- row.names(test)
table(predicted=xgb_preds, true_value=test$link)
table(predicted=xgb_preds[deleted_links_test], true_value=test[deleted_links_test, 'link'])

# check for highly correlated variables
corm <- cor(train_rs, use="pairwise.complete.obs")
corrplot(corm, method="circle")

# Logistic Regression
logreg_model <- glm(as.factor(link) ~ ., data=train_rs, family=binomial(link='logit'))
summary(logreg_model)
logreg_preds_p <- predict(logreg_model, test[,!(colnames(test) %in% c('link'))], type='response')
logreg_preds <- as.numeric(logreg_preds_p > 0.5)  #Convert the predicted score to a binary value

# Naive Bayes
nb_model <- naiveBayes(as.factor(link) ~ ., data=train_rs)
nb_model
nb_preds_p <- predict(nb_model, test[,!(colnames(test) %in% c('link'))], type='raw')[,2]
nb_preds <- as.integer(predict(nb_model, test[,!(colnames(test) %in% c('link'))]))-1


# Model evaluation


# Plot ROC and Precision-Recall curves
xgb_perf <- prediction(xgb_preds_p, test$link)
logreg_perf <- prediction(logreg_preds_p, test$link)
nb_perf <- prediction(nb_preds_p, test$link)
roc_curve_xgb <- performance(xgb_perf, 'tpr', 'fpr')
roc_curve_logreg <- performance(logreg_perf, 'tpr', 'fpr')
roc_curve_nb <- performance(nb_perf, 'tpr', 'fpr')
prec_rec_curve_xgb <- performance(xgb_perf, 'prec', 'rec')
prec_rec_curve_logreg <- performance(logreg_perf, 'prec', 'rec')
prec_rec_curve_nb <- performance(nb_perf, 'prec', 'rec')
plot(roc_curve_xgb, col='red', lwd=3, main='Sensitivity on Specificity (ROC)')
plot(roc_curve_logreg, col='blue', lwd=3, add=T)
plot(roc_curve_nb, col='green', lwd=3, add=T)
abline(c(0,0), c(1,1), lty=8, lwd=3)
legend('bottomright', legend=c('XG Boost', 'Logistic Reg', 'Naive Bayes', 'No Skill'),
       col=c('red', 'blue', 'green', 'black'), lty=c(1,1,1,8), lwd=3)
plot(prec_rec_curve_xgb, col='red', lwd=3, main='Positive Predictive Power on True Positive Rate')
plot(prec_rec_curve_logreg, col='blue', lwd=3, add=T)
plot(prec_rec_curve_nb, col='green', lwd=3, add=T)
abline(h=nrow(test[test$link==1,])/nrow(test), lty=8, lwd=3)
legend('bottomleft', legend=c('XG Boost', 'Logistic Reg', 'Naive Bayes', 'No Skill'),
       col=c('red', 'blue', 'green', 'black'), lty=c(1,1,1,8), lwd=3)

# Find the optimal threshold to use for classification (maximizes TP while minimizing FP)
# Thanks to: https://stackoverflow.com/questions/16347507/obtaining-threshold-values-from-a-roc-curve
best_prob_threshold <- function(predicted, response) {
  perf <- ROCR::performance(ROCR::prediction(predicted, response), "tpr", "fpr")
  cutoffs <- data.frame(cut=perf@alpha.values[[1]], 
                        fpr=perf@x.values[[1]], 
                        tpr=perf@y.values[[1]])
  cutoffs[which.max(cutoffs$tpr-cutoffs$fpr), 'cut']
}

xgb_thresh <- best_prob_threshold(xgb_preds_p, test$link)
logreg_thresh <- best_prob_threshold(logreg_preds_p, test$link)
nb_thresh <- best_prob_threshold(nb_preds_p, test$link)

xgb_preds <- as.numeric(xgb_preds_p > xgb_thresh)
logreg_preds <- as.numeric(logreg_preds_p > logreg_thresh)
nb_preds <- as.numeric(nb_preds_p > nb_thresh)

# Final model performance after applying threshold

# XGBoost
LogLoss(y_pred=xgb_preds, y_true=test_y)
AUC(y_pred=xgb_preds, y_true=test_y)
Accuracy(y_pred=xgb_preds, y_true=test_y)
prop.table(table(y_pred=xgb_preds, y_true=test_y))
# Logistic Reg
LogLoss(y_pred=logreg_preds, y_true=test$link)
AUC(y_pred=logreg_preds, y_true=test$link)
Accuracy(y_pred=logreg_preds, y_true=test$link)
prop.table(table(y_pred=logreg_preds, y_true=test$link))
# Naive Bayes
LogLoss(y_pred=nb_preds, y_true=test$link)
AUC(y_pred=nb_preds, y_true=test$link)
Accuracy(y_pred=nb_preds, y_true=test$link)
prop.table(table(y_pred=nb_preds, y_true=test$link))
