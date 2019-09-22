
setwd("dsilt-ml-code/05-07 Multivariate Outliers SVMs Ensembles")

library(caret)
library(randomForest)
library(MLmetrics)
library(pROC)
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
library(xgboost)

d <- read.csv("WBCdata_clean.csv", header=TRUE)
head(d)
d <- d[,-which(colnames(d) %in% c('group', 'id'))] # Will not need these

seed <- 14
num_trees <- 99
kfold <- 5

#createDataPartition stratifies y automatically
train_ind <- createDataPartition(y=d$diagnosis, p=0.66, list=F)
train_set <- d[train_ind,]
test_set <- d[-train_ind,]

train_norm <- predict(preProcess(train_set, method="range"), train_set)
test_norm <- predict(preProcess(test_set, method="range"), test_set)

#-------------------------------------------------------------------------------------------------#
#----------------------------------------Random Forest--------------------------------------------#
#-------------------------------------------------------------------------------------------------#

cv <- trainControl(method="oob", search="grid")
param_grid <- expand.grid(mtry=c(3,5,7))
rf_model_tuned <- train(as.factor(diagnosis) ~ ., data=train_norm, 
                        method='rf', metric='Kappa', seed=seed,
                        tuneGrid=param_grid, trControl=cv)
rf_model_tuned

rf_model <- randomForest(y=as.factor(train_norm$diagnosis),
                         x=train_norm[,-which(colnames(train_norm) %in% c('diagnosis'))], 
                         mtry=3,
                         ntree=59, 
                         replace=TRUE,
                         importance=TRUE,
                         proximity=TRUE,
                         strata=as.factor(train_norm$diagnosis),
                         seed=seed)

# Evaluate results
plot(rf_model, main='OOB Error Rate by Nbr Trees')
varImpPlot(rf_model)
test_preds <- predict(rf_model, test_norm[,2:10])
ConfusionMatrix(test_preds, test_norm$diagnosis)
Accuracy(test_preds, test_norm$diagnosis)
AUC(test_preds, test_norm$diagnosis)
roc_data <- roc(test_norm$diagnosis, predict(rf_model, test_norm[,2:10], type='prob')[,2])
plot.roc(roc_data, main="Random Forest ROC Curve")

#-------------------------------------------------------------------------------------------------#
#--------------------------------------------XGBoost----------------------------------------------#
#-------------------------------------------------------------------------------------------------#

x_train_norm <- as.matrix(train_norm[,2:ncol(train_norm)])
x_test_norm <- as.matrix(test_norm[,2:ncol(train_norm)])
y_train_norm <- train_norm$diagnosis
y_test_norm <- test_norm$diagnosis

xgb_model <- xgboost(data=x_train_norm, label=y_train_norm,
                     eta=0.1, gamma=1, max_depth=10,
                     subsample=0.5, colsample_bytree=0.5,
                     lambda=0, lambda_bias=0, alpha=0,
                     nrounds=59, seed=seed, nthread=3,
                     objective="binary:logistic",
                     eval_metric="logloss")

# Evaluate results
plot(xgb_model$evaluation_log, type='l', main='Train Log Loss by Iteration')
xgb_var_imp <- xgb.importance(colnames(train_norm[,2:ncol(train_norm)]), 
                              model=xgb_model)
xgb.plot.importance(xgb_var_imp, 
                    main='XGBoost Variable Importance by Information Gain')
test_preds <- as.integer(predict(xgb_model, x_test_norm)>0.5)
ConfusionMatrix(test_preds, y_test_norm)
Accuracy(test_preds, y_test_norm)
AUC(test_preds, y_test_norm)
roc_data <- roc(y_test_norm, predict(xgb_model, x_test_norm))
plot.roc(roc_data, main="XGBoost ROC Curve")
