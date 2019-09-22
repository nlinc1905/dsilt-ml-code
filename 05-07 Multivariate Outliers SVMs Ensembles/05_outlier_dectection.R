
setwd("dsilt-ml-code/05-07 Multivariate Outliers SVMs Ensembles")

library(plotly)
library(devtools)
#install_github("vqv/ggbiplot")
#install_github("Zelazny7/isofor")
library(ggbiplot)
library(GGally)
library(dbscan)
library(e1071)
library(isofor)
library(MLmetrics)

d <- read.delim("WBCdata.data", header=TRUE, sep="\t")

#-------------------------------------------------------------------------------------------------#
#---------------------------------------Data Cleaning---------------------------------------------#
#-------------------------------------------------------------------------------------------------#

colnames(d) <- c('group', 'id', 'diagnosis', 'clump_thickness',
                 'cell_size_uniformity', 'cell_shape_uniformity',
                 'marginal_adhesion', 'epithelial_cell_size',
                 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli',
                 'mitoses')

head(d)
str(d)

# Label encode the target variable (0 = benign, 1 = malignant)
target_var_cat_map <- levels(d$diagnosis)
d$diagnosis <- as.integer(as.integer(d$diagnosis)-1)

naCol <- function(x){
  y <- sapply(x, function(y) sum(length(which(is.na(y) 
                                              | is.nan(y) 
                                              | is.infinite(y) 
                                              | y=='NA' 
                                              | y=='NaN' 
                                              | y=='Inf' 
                                              | y==''))))
  y <- data.frame('feature'=names(y), 'count.nas'=y)
  row.names(y) <- c()
  y
}

naCol(d)

d <- d[complete.cases(d),]
str(d)

#write.csv(d, 'wbcdata_clean.csv', row.names=FALSE)

#-------------------------------------------------------------------------------------------------#
#-----------------------------Visual Inspection for Outliers--------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Plotly version of parallel coordinates
options(viewer=NULL)  # Avoids viewing bug by opening plot in browser
plot_ly(type = 'parcoords',
        line = list(color = ~d$diagnosis,
                    colorscale = list(c(0, '#0158FE'), c(1, '#FE0101')),
                    showscale = TRUE,
                    cmin=0,
                    cmax=1),
        dimensions = list(
          list(label = colnames(d)[4], values = ~d[,4]),
          list(label = colnames(d)[5], values = ~d[,5]),
          list(label = colnames(d)[6], values = ~d[,6]),
          list(label = colnames(d)[7], values = ~d[,7]),
          list(label = colnames(d)[8], values = ~d[,8]),
          list(label = colnames(d)[9], values = ~d[,9]),
          list(label = colnames(d)[10], values = ~d[,10]),
          list(label = colnames(d)[11], values = ~d[,11]),
          list(label = colnames(d)[12], values = ~d[,12])
        )
)

possible_outliers <- row.names(d[((d[,'epithelial_cell_size']==1) & (d[,'diagnosis']==1)),])

#Plotting first 2 latent features
pca_model <- prcomp(d[,4:12], center=TRUE, scale.=TRUE)
summary(pca_model)
pca_dim <- pca_model$x

ggbiplot(pca_model, groups=d$diagnosis) +
  ggtitle("PCA 2-Dimension Plot with Observation Class")

possible_outliers <- c(possible_outliers, 
                       row.names(pca_dim[(pca_dim[,1]<(-2) 
                                          | pca_dim[,2]>1 
                                          | pca_dim[,2]<(-2)),])
                       )
print(unique(possible_outliers))

#Pairs plot/SPLOM
ggpairs(d[,4:12]) +
  ggtitle('Pairs Plot of Wisconsin Breast Cancer Data')

#Mosaic plot
mosaicplot(~ clump_thickness + cell_size_uniformity, 
           data=d, color=TRUE, 
           main='Mosaic Plot of 2 Features from Wisconsin Breast Cancer Data')

#Plot parallel coordinates of potential outliers
options(viewer=NULL)  # Avoids viewing bug by opening plot in browser
plot_ly(type = 'parcoords',
        line = list(color = ~d[row.names(d) %in% possible_outliers, 'diagnosis'],
                    colorscale = list(c(0, '#0158FE'), c(1, '#FE0101')),
                    showscale = TRUE,
                    cmin=0,
                    cmax=1),
        dimensions = list(
          list(label = colnames(d)[4], values = ~d[row.names(d) %in% possible_outliers,4]),
          list(label = colnames(d)[5], values = ~d[row.names(d) %in% possible_outliers,5]),
          list(label = colnames(d)[6], values = ~d[row.names(d) %in% possible_outliers,6]),
          list(label = colnames(d)[7], values = ~d[row.names(d) %in% possible_outliers,7]),
          list(label = colnames(d)[8], values = ~d[row.names(d) %in% possible_outliers,8]),
          list(label = colnames(d)[9], values = ~d[row.names(d) %in% possible_outliers,9]),
          list(label = colnames(d)[10], values = ~d[row.names(d) %in% possible_outliers,10]),
          list(label = colnames(d)[11], values = ~d[row.names(d) %in% possible_outliers,11]),
          list(label = colnames(d)[12], values = ~d[row.names(d) %in% possible_outliers,12])
        )
)

#-------------------------------------------------------------------------------------------------#
#-------------------------------------Local Outlier Factor----------------------------------------#
#-------------------------------------------------------------------------------------------------#

lof_fit <- lof(d[,4:12], k=5)
lof_outliers <- lof_fit[lof_fit>1]
d$lof_outlier <- ifelse(lof_fit>1, 1, 0)

#-------------------------------------------------------------------------------------------------#
#------------------------------------------One-Class SVM------------------------------------------#
#-------------------------------------------------------------------------------------------------#

ocsvm <- svm(d[,4:12], d$diagnosis, type='one-classification')
ocsvm_outliers <- names(ocsvm$fitted[ocsvm$fitted==FALSE])
d$ocsvm_outlier <- ifelse(ocsvm$fitted==FALSE, 1, 0)

#-------------------------------------------------------------------------------------------------#
#---------------------------------------Isolation Forest------------------------------------------#
#-------------------------------------------------------------------------------------------------#

samples_per_tree <- as.integer(round(0.2*nrow(d),0))
isoforest <- iForest(d[,4:12], nt=99, phi=samples_per_tree, seed=14)
isoforest_fit <- predict(isoforest, d[,4:12])
isoforest_outliers <- isoforest_fit[isoforest_fit>quantile(isoforest_fit, 0.95)]
d$isoforest_outlier <- ifelse(isoforest_fit>quantile(isoforest_fit, 0.95), 1, 0)

#-------------------------------------------------------------------------------------------------#
#-------------------------------------Evaluation as Classifiers-----------------------------------#
#-------------------------------------------------------------------------------------------------#

#Evaluate outlier detection methods as classifier for minority class
print(paste0("Local Outlier Factor AUC ", AUC(d$lof_outlier, d$diagnosis)))
print(paste0("One-Class SVM AUC ", AUC(d$ocsvm_outlier, d$diagnosis)))
print(paste0("Isolation Forest AUC ", AUC(d$isoforest_outlier, d$diagnosis)))

#Create a weighted ensemble score
d$ensemble_outlier_score = round((d$lof_outlier*AUC(d$lof_outlier, d$diagnosis)
                                  + d$ocsvm_outlier*AUC(d$ocsvm_outlier, d$diagnosis)
                                  + d$isoforest_outlier*AUC(d$isoforest_outlier, d$diagnosis))
                                 /(AUC(d$lof_outlier, d$diagnosis)
                                   + AUC(d$ocsvm_outlier, d$diagnosis)
                                   + AUC(d$isoforest_outlier, d$diagnosis)), 0)
print(paste0("Ensemble AUC ", AUC(d$ensemble_outlier_score, d$diagnosis)))

#Export and use conditional formatting in Excel to compare
d.to_csv('wbcdata_clean_outlier_tagging.csv', row.names=F)
