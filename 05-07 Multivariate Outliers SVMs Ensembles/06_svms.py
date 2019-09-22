"""
Chapter 6: Support Vector Machines
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix

d = pd.read_csv("WBCdata_clean.csv")

seed = 14

x_train, x_test, y_train, y_test = train_test_split(d.iloc[:, 3:], d.diagnosis,
                                                    test_size=0.33, random_state=seed,
                                                    shuffle=True, stratify=d.diagnosis)


#Setting up model parameters
boundary_smoothness = 1/len(d.columns[3:])
svc_lin_model = SVC(kernel='linear', random_state=seed)
svc_pol_model = SVC(kernel='poly', degree=3, gamma=boundary_smoothness, random_state=seed)
svc_rbf_model = SVC(kernel='rbf', gamma=boundary_smoothness, random_state=seed)
svc_sig_model = SVC(kernel='sigmoid', gamma=boundary_smoothness, random_state=seed)

#Training models
svc_lin_model.fit(x_train, y_train)
svc_pol_model.fit(x_train, y_train)
svc_rbf_model.fit(x_train, y_train)
svc_sig_model.fit(x_train, y_train)

#Predictions
svc_lin_preds = svc_lin_model.predict(x_test)
svc_pol_preds = svc_pol_model.predict(x_test)
svc_rbf_preds = svc_rbf_model.predict(x_test)
svc_sig_preds = svc_sig_model.predict(x_test)

#Evaluation
print("Linear SVC AUC", roc_auc_score(y_test, svc_lin_preds))
print("Polynomial SVC AUC", roc_auc_score(y_test, svc_pol_preds))
print("Radial Basis SVC AUC", roc_auc_score(y_test, svc_rbf_preds))
print("Sigmoid SVC AUC", roc_auc_score(y_test, svc_sig_preds))
print("\n")
print("Linear SVC Log Loss", log_loss(y_test, svc_lin_preds))
print("Polynomial SVC Log Loss", log_loss(y_test, svc_pol_preds))
print("Radial Basis SVC Log Loss", log_loss(y_test, svc_rbf_preds))
print("Sigmoid SVC Log Loss", log_loss(y_test, svc_sig_preds))
print("\n")
print("Confusion Matrix Format: \n[TN, FP]\n[FN, TP]")
print("Linear SVC Confusion Matrix \n", confusion_matrix(y_test, svc_lin_preds))
print("Polynomial SVC Confusion Matrix \n", confusion_matrix(y_test, svc_pol_preds))
print("Radial Basis SVC Confusion Matrix \n", confusion_matrix(y_test, svc_rbf_preds))
print("Sigmoid SVC Log Confusion Matrix \n", confusion_matrix(y_test, svc_sig_preds))

