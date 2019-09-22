"""
Chapter 7: Ensemble Models
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

d = pd.read_csv("WBCdata_clean.csv")
print(d.head())
d.drop(['group', 'id'], axis=1, inplace=True) # Will not need these

seed = 14
num_trees = 99
kfold = 5

x = d.iloc[:, 1:].values
y = d['diagnosis'].values

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.33, random_state=seed,
                                                    shuffle=True, stratify=y)

# Ignore the int64 to float64 conversion error caused by next 3 lines
normalizer = MinMaxScaler().fit(x_train)
x_train_norm = normalizer.transform(x_train)
x_test_norm = normalizer.transform(x_test)

#-------------------------------------------------------------------------------------------------#
#----------------------------------------Random Forest--------------------------------------------#
#-------------------------------------------------------------------------------------------------#

rf_model = RandomForestClassifier(n_estimators=num_trees, criterion='entropy',
                                  max_features='sqrt', min_samples_split=5,
                                  random_state=seed)

# Note that stratified k-fold cross validation is used if cv argument is an integer
rf_cv_scores = cross_validate(rf_model, x_train_norm, y_train, cv=kfold,
                              scoring=('neg_log_loss', 'f1', 'roc_auc', 'accuracy'),
                              return_train_score=False)
print('Random Forest Validation Set Avg Neg Log Loss:',
      rf_cv_scores['test_neg_log_loss'].mean())
print('Random Forest Validation Set Avg F1:',
      rf_cv_scores['test_f1'].mean())
print('Random Forest Validation Set Avg AUC:',
      rf_cv_scores['test_roc_auc'].mean())
print('Random Forest Validation Set Avg Accuracy:',
      rf_cv_scores['test_accuracy'].mean())

# Grid search optimal random forest hyperparameters
params = {'min_samples_split': range(2, 10, 2),
          'max_features': ['sqrt', 6, None]}
scoring_metrics = {'Neg Log Loss': 'neg_log_loss',
                   'F1': 'f1',
                   'AUC': 'roc_auc',
                   'Accuracy': 'accuracy'}
best_model_metric = list(scoring_metrics.keys())[0]
gs = GridSearchCV(rf_model, param_grid=params, scoring=scoring_metrics, iid=True,
                  cv=kfold, refit=best_model_metric, return_train_score=False)
gs.fit(x_train_norm, y_train)
# There will be 1 avg score for each model (each combo of hyperparameters)
print('Avg validation set neg log loss', gs.cv_results_['mean_test_Neg Log Loss'])
print('Avg validation set F1', gs.cv_results_['mean_test_F1'])
print('Avg validation set AUC', gs.cv_results_['mean_test_AUC'])
print('Avg validation set accuracy', gs.cv_results_['mean_test_Accuracy'])
print('Best model chosen using', best_model_metric)
print('Hyperparameters of best model:', gs.best_params_)
print('Score of best model:', gs.best_score_)
rf_best_model = gs.best_estimator_ # Only possible if using refit

#-------------------------------------------------------------------------------------------------#
#-----------------------------------------Sklearn GBM---------------------------------------------#
#-------------------------------------------------------------------------------------------------#

gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                                      n_estimators=num_trees, criterion='friedman_mse',
                                      max_features='sqrt', min_samples_split=4,
                                      random_state=seed)

# Note that stratified k-fold cross validation is used if cv argument is an integer
gb_cv_scores = cross_validate(gb_model, x_train_norm, y_train, cv=kfold,
                              scoring=('neg_log_loss', 'f1', 'roc_auc', 'accuracy'),
                              return_train_score=False)
print('GBM Validation Set Avg Neg Log Loss:',
      gb_cv_scores['test_neg_log_loss'].mean())
print('GBM Validation Set Avg F1:',
      gb_cv_scores['test_f1'].mean())
print('GBM Validation Set Avg AUC:',
      gb_cv_scores['test_roc_auc'].mean())
print('GBM Validation Set Avg Accuracy:',
      gb_cv_scores['test_accuracy'].mean())

# Grid search optimal GBM hyperparameters
params = {'min_samples_split': range(2, 6, 2),
          'max_features': ['sqrt', 6, None],
          'learning_rate': [0.01, 0.1]}
scoring_metrics = {'Neg Log Loss': 'neg_log_loss',
                   'F1': 'f1',
                   'AUC': 'roc_auc',
                   'Accuracy': 'accuracy'}
best_model_metric = list(scoring_metrics.keys())[0]
gs = GridSearchCV(gb_model, param_grid=params, scoring=scoring_metrics, iid=True,
                  cv=kfold, refit=best_model_metric, return_train_score=False)
gs.fit(x_train_norm, y_train)
# There will be 1 avg score for each model (each combo of hyperparameters)
print('Avg validation set neg log loss', gs.cv_results_['mean_test_Neg Log Loss'])
print('Avg validation set F1', gs.cv_results_['mean_test_F1'])
print('Avg validation set AUC', gs.cv_results_['mean_test_AUC'])
print('Avg validation set accuracy', gs.cv_results_['mean_test_Accuracy'])
print('Best model chosen using', best_model_metric)
print('Hyperparameters of best model:', gs.best_params_)
print('Score of best model:', gs.best_score_)
gb_best_model = gs.best_estimator_ # Only possible if using refit

#-------------------------------------------------------------------------------------------------#
#-------------------------------------------XGBoost-----------------------------------------------#
#-------------------------------------------------------------------------------------------------#

xgb_model = XGBClassifier(n_estimators=num_trees,
                          learning_rate=0.1, gamma=1, max_depth=10,
                          subsample=0.5, colsample_bytree=0.5,
                          reg_lambda=0, reg_alpha=0,
                          nround=59, seed=seed, nthread=3,
                          objective='binary:logistic')
#xgb_model.fit(x_train_norm, y_train, eval_metric="logloss")

# Note that stratified k-fold cross validation is used if cv argument is an integer
xgb_cv_scores = cross_validate(xgb_model, x_train_norm, y_train, cv=kfold,
                               scoring=('neg_log_loss', 'f1', 'roc_auc', 'accuracy'),
                               return_train_score=False)
print('XGB Validation Set Avg Neg Log Loss:',
      xgb_cv_scores['test_neg_log_loss'].mean())
print('XGB Validation Set Avg F1:',
      xgb_cv_scores['test_f1'].mean())
print('XGB Validation Set Avg AUC:',
      xgb_cv_scores['test_roc_auc'].mean())
print('XGB Validation Set Avg Accuracy:',
      xgb_cv_scores['test_accuracy'].mean())

# Grid search optimal XGBoost hyperparameters
params = {'learning_rate': [0.01, 0.1],
          'max_depth': range(2, 12, 2),
          'colsample_bytree': [0.33, 0.5, 1],
          'reg_lambda': [0, 1],
          'min_samples_split': range(2, 6, 2)}
scoring_metrics = {'Neg Log Loss': 'neg_log_loss',
                   'F1': 'f1',
                   'AUC': 'roc_auc',
                   'Accuracy': 'accuracy'}
best_model_metric = list(scoring_metrics.keys())[0]
gs = GridSearchCV(xgb_model, param_grid=params, scoring=scoring_metrics,
                  cv=kfold, refit=best_model_metric, return_train_score=False)
gs.fit(x_train_norm, y_train)
# There will be 1 avg score for each model (each combo of hyperparameters)
print('Avg validation set neg log loss', gs.cv_results_['mean_test_Neg Log Loss'])
print('Avg validation set F1', gs.cv_results_['mean_test_F1'])
print('Avg validation set AUC', gs.cv_results_['mean_test_AUC'])
print('Avg validation set accuracy', gs.cv_results_['mean_test_Accuracy'])
print('Best model chosen using', best_model_metric)
print('Hyperparameters of best model:', gs.best_params_)
print('Score of best model:', gs.best_score_)
xgb_best_model = gs.best_estimator_ # Only possible if using refit

#-------------------------------------------------------------------------------------------------#
#-------------------------------------------Catboost----------------------------------------------#
#-------------------------------------------------------------------------------------------------#

catboost_model = CatBoostClassifier(iterations=num_trees,
                                    learning_rate=0.1,
                                    depth=10,
                                    l2_leaf_reg=0,
                                    loss_function='Logloss',
                                    border_count=32,
                                    simple_ctr=None,
                                    thread_count=3,
                                    random_seed=seed,
                                    logging_level='Silent')
catboost_model.fit(x_train_norm, y_train)

#-------------------------------------------------------------------------------------------------#
#---------------------------------------Compare All Models----------------------------------------#
#-------------------------------------------------------------------------------------------------#

# Evaluate models on test set
rf_test_preds = rf_best_model.predict(x_test_norm)
gb_test_preds = gb_best_model.predict(x_test_norm)
xgb_test_preds = xgb_best_model.predict(x_test_norm)
catboost_test_preds = catboost_model.predict(x_test_norm)

print("Random Forest AUC", roc_auc_score(y_test, rf_test_preds))
print("GBM AUC", roc_auc_score(y_test, gb_test_preds))
print("XGBoost AUC", roc_auc_score(y_test, xgb_test_preds))
print("CatBoost AUC", roc_auc_score(y_test, catboost_test_preds))
print("\n")
print("Random Forest Log Loss", log_loss(y_test, rf_test_preds))
print("GBM Log Loss", log_loss(y_test, gb_test_preds))
print("XGBoost Log Loss", log_loss(y_test, xgb_test_preds))
print("CatBoost Log Loss", log_loss(y_test, catboost_test_preds))
print("\n")
print("Confusion Matrix Format: \n[TN, FP]\n[FN, TP]")
print("Random Forest Confusion Matrix \n", confusion_matrix(y_test, rf_test_preds))
print("GBM Confusion Matrix \n", confusion_matrix(y_test, gb_test_preds))
print("XGBoost Confusion Matrix \n", confusion_matrix(y_test, xgb_test_preds))
print("CatBoost Confusion Matrix \n", confusion_matrix(y_test, catboost_test_preds))

# View validation set scores by model
results = [rf_cv_scores['test_roc_auc'],
           gb_cv_scores['test_roc_auc'],
           xgb_cv_scores['test_roc_auc']]
model_names = ['RF', 'GBM', 'XGBoost']
fig = plt.figure()
fig.suptitle('Model AUC Comparison On Validation Set')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(model_names)
plt.show()
