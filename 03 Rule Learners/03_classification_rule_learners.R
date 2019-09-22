
setwd("dsilt-ml-code/03 Rule Learners")

library(ggparallel)
library(gmodels)
library(RWeka)
library(MLmetrics)

set.seed(14)

#-------------------------------------------------------------------------------------------------#
#----------------------------------------EDA and Cleaning-----------------------------------------#
#-------------------------------------------------------------------------------------------------#


shrooms <- read.delim("agaricus-lepiota.data", header=F, 
                      stringsAsFactors=T, sep=",")
colnames(shrooms) <- c("poisonous_or_edible", "cap_shape", "cap_surface", 
                       "cap_color", "bruises", "odor", "gill_attachment", 
                       "gill_spacing", "gill_size", "gill_color", 
                       "stalk_shape", "stalk_root", 
                       "stalk_surface_above_ring", 
                       "stalk_surface_below_ring", 
                       "stalk_color_above_ring", "stalk_color_below_ring", 
                       "veil_type", "veil_color", "ring_number", 
                       "ring_type", "spore_print_color", "population", 
                       "habitat")
head(shrooms)

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

naCol(shrooms)
sapply(shrooms, class)
str(shrooms)

# Drop veil_type because it only has 1 level
# Features with 1 value across all observations are useless predictors
shrooms$veil_type <- NULL

# This loop only works for this data bc all features are factors
for (feat in colnames(shrooms)) {
  writeLines(paste("\nCategory Spread of", feat))
  print(table(shrooms[, feat]))
}

# See if categories with few samples should be binned
ggparallel(list('poisonous_or_edible', 'cap_shape', 
                'cap_surface', 'cap_color'), 
           shrooms[shrooms$cap_shape=='c',], order=0)
ggparallel(list('poisonous_or_edible', 'cap_shape', 
                'cap_surface', 'cap_color'), 
           shrooms[shrooms$cap_surface=='g',], order=0)
CrossTable(shrooms$poisonous_or_edible, 
           shrooms$cap_shape, 
           prop.chisq=F)
CrossTable(shrooms$poisonous_or_edible, 
           shrooms$cap_surface, 
           prop.chisq=F)

train_ind <- sample(seq_len(nrow(shrooms)), 
                    size=0.75*nrow(shrooms), replace=F)
train <- shrooms[train_ind,]
test <- shrooms[-train_ind,]

#-------------------------------------------------------------------------------------------------#
#------------------------------------------Rule Learners------------------------------------------#
#-------------------------------------------------------------------------------------------------#

# Single rule classifier
shroom1r <- OneR(poisonous_or_edible ~ ., data=train)
shroom1r
summary(shroom1r)

# RIPPER
shroom_ripper <- JRip(poisonous_or_edible ~., data=train)
shroom_ripper
summary(shroom_ripper)

# Evaluate RIPPER on test set
test_preds <- predict(shroom_ripper, test[,-1])
Accuracy(y_true=test[,1], y_pred=test_preds)
ConfusionMatrix(y_true=test[,1], y_pred=test_preds)
Sensitivity(y_true=test[,1], y_pred=test_preds)
Specificity(y_true=test[,1], y_pred=test_preds)
F1_Score(y_true=test[,1], y_pred=test_preds)
