
setwd("dsilt-ml-code/05-07 Multivariate Outliers SVMs Ensembles")

library(caret)
library(e1071)
library(MLmetrics)

d <- read.csv("WBCdata_clean.csv", header=TRUE)
head(d)

seed <- 14

train_ind <- createDataPartition(y=d$diagnosis, p=0.66, list=F)
train <- d[train_ind, 3:12]
test <- d[-train_ind, 3:12]

#Setting up model parameters and training models
boundary_smoothness <- 1/ncol(train)
svc_lin_model <- svm(diagnosis ~ ., data=train, 
                     scale=TRUE, type='C-classification',
                     kernel='linear', seed=seed)
svc_pol_model <- svm(diagnosis ~ ., data=train, 
                     scale=TRUE, type='C-classification',
                     kernel='polynomial', degree=3,
                     gamma=boundary_smoothness,
                     seed=seed)
svc_rbf_model <- svm(diagnosis ~ ., data=train, 
                     scale=TRUE, type='C-classification',
                     kernel='radial', 
                     gamma=boundary_smoothness,
                     seed=seed)
svc_sig_model <- svm(diagnosis ~ ., data=train, 
                     scale=TRUE, type='C-classification',
                     kernel='sigmoid', 
                     gamma=boundary_smoothness,
                     seed=seed)

#Predictions
svc_lin_preds <- as.integer(predict(svc_lin_model, test[,2:10]))
svc_pol_preds <- as.integer(predict(svc_pol_model, test[,2:10]))
svc_rbf_preds <- as.integer(predict(svc_rbf_model, test[,2:10]))
svc_sig_preds <- as.integer(predict(svc_sig_model, test[,2:10]))

#Evaluation
print(paste("Linear SVC AUC", AUC(svc_lin_preds, test$diagnosis)))
print(paste("Polynomial SVC AUC", AUC(svc_pol_preds, test$diagnosis)))
print(paste("Radial Basis SVC AUC", AUC(svc_rbf_preds, test$diagnosis)))
print(paste("Sigmoid SVC AUC", AUC(svc_sig_preds, test$diagnosis)))
print(paste(" "))
print(paste("Linear SVC Log Loss", LogLoss(svc_lin_preds, test$diagnosis)))
print(paste("Polynomial SVC Log Loss", LogLoss(svc_pol_preds, test$diagnosis)))
print(paste("Radial Basis SVC Log Loss", LogLoss(svc_rbf_preds, test$diagnosis)))
print(paste("Sigmoid SVC Log Loss", LogLoss(svc_sig_preds, test$diagnosis)))
print(paste(" "))
print("Linear SVC Confusion Matrix")
ConfusionMatrix(svc_lin_preds, test$diagnosis)
print("Polynomial SVC Confusion Matrix")
ConfusionMatrix(svc_pol_preds, test$diagnosis)
print("Radial Basis SVC Confusion Matrix")
ConfusionMatrix(svc_rbf_preds, test$diagnosis)
print("Sigmoid SVC Log Confusion Matrix")
ConfusionMatrix(svc_sig_preds, test$diagnosis)
