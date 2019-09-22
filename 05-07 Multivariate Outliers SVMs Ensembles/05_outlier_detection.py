"""
Chapter 5: Multivariate Outlier Detection
"""

import pandas as pd
import json
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import plotly.plotly as ply
import plotly.graph_objs as go
from plotly.offline import plot as plyplot
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

d = pd.read_csv("WBCdata.data", sep="\t")

#-------------------------------------------------------------------------------------------------#
#--------------------------------------Data Cleaning----------------------------------------------#
#-------------------------------------------------------------------------------------------------#

d.columns = ['group', 'id', 'diagnosis', 'clump_thickness',
             'cell_size_uniformity', 'cell_shape_uniformity',
             'marginal_adhesion', 'epithelial_cell_size',
             'bare_nuclei', 'bland_chromatin', 'normal_nucleoli',
             'mitoses']
print(d.head())
print(d.info())

# Label encode the target variable (0 = benign, 1 = malignant)
target_var_cat_map = {n: cat for n, cat in enumerate(d['diagnosis'].astype('category').cat.categories)}
d['diagnosis'] = d['diagnosis'].astype('category').cat.codes.astype('int')

def naCol(df, print_result=True):
    """
    Checks for null or missing values in all columns of a Pandas dataframe

    Arguments:
    df: A Pandas dataframe
    print_result: indicates whether or not the output should be printed to the console

    Returns:
    dict: (key:value) = (column_name:number_missing_values)    
    """
    y = dict.fromkeys(df.columns)
    for idx, key in enumerate(y.keys()):
        if df.dtypes[list(y.keys())[idx]] == 'object':
            y[key] = pd.isnull(df[list(y.keys())[idx]]).sum() + (df[list(y.keys())[idx]]=='').sum()
        else:
            y[key] = pd.isnull(df[list(y.keys())[idx]]).sum()
    if print_result:
        print("Number of nulls by column:")
        for k, v in y.items():
            print(k, v)
    return y

def naRow(df, threshold=0.5, print_result=True):
    """
    Checks for null or missing values in all rows of a Pandas dataframe

    Arguments:
    df: A Pandas dataframe
    threshold: The percentage of missing values out of all dataframe
            columns that are allowed to be null
    print_result: indicates whether or not the output should be printed to the console

    Returns:
    dict: (key:value) = (row_index:number_missing_values)    
    """
    y = dict.fromkeys(df.index)
    for idx, key in enumerate(y.keys()):
        y[key] = sum(df.iloc[[idx]].isnull().sum())
    if print_result:
        print("Rows with more than 50% null columns:")
        print([r for r in y if y[r]/df.shape[1] > threshold])
    return y

naCol(d)
naRow(d)

d.dropna(axis=0, how='any', inplace=True)
d['bare_nuclei'] = d['bare_nuclei'].astype('int')
print(d.info())

d.to_csv('wbcdata_clean.csv', index=False)
with open('wbcdata_target_cat_map.json', 'w') as file:
    json.dump(target_var_cat_map, file, sort_keys=True, indent=4)

#-------------------------------------------------------------------------------------------------#
#-----------------------------Visual Inspection for Outliers--------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Pandas version of parallel coordinates
fig = plt.figure()
fig.suptitle('Parallel Coordinates Plot of Wisconsin Breast Cancer Data')
parallel_coordinates(d, class_column='diagnosis',
                     cols=d.columns[3:],
                     color=('#0158FE', '#FE0101'))
plt.show()

#Plotly version of parallel coordinates
data = [
    go.Parcoords(
        line = dict(color = d['diagnosis'],
                    colorscale = [[0, '#0158FE'], [1, '#FE0101']],
                    showscale = True,
                    cmin=0,
                    cmax=1),
        dimensions = list([
            dict(label=d.columns[3], values = d.iloc[:, 3]),
            dict(label=d.columns[4], values = d.iloc[:, 4]),
            dict(label=d.columns[5], values = d.iloc[:, 5]),
            dict(label=d.columns[6], values = d.iloc[:, 6]),
            dict(label=d.columns[7], values = d.iloc[:, 7]),
            dict(label=d.columns[8], values = d.iloc[:, 8]),
            dict(label=d.columns[9], values = d.iloc[:, 9]),
            dict(label=d.columns[10], values = d.iloc[:, 10]),
            dict(label=d.columns[11], values = d.iloc[:, 11])
            ])
        )
    ]

fig = go.Figure(data=data)
plyplot(fig)

possible_outliers = d[(d['epithelial_cell_size']==1) & (d['diagnosis']==1)].index.tolist()

#Plotting first 2 latent features
pca_model = PCA(n_components=None, whiten=False, random_state=14)
pca_dim = pca_model.fit_transform(d.iloc[:, 3:])

plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1 (explains most variance)')
plt.ylabel('Latent Variable 2 (explains 2nd most variance)')
plt.title('PCA 2-Dimension Plot with Observation Class')
plt.scatter(pca_dim[:, 0], pca_dim[:, 1], c=d['diagnosis'].values.ravel())
plt.colorbar()
plt.show()

possible_outliers = np.where((pca_dim[:, 0]>12) & (pca_dim[:, 1]>4))[0].tolist()
possible_outliers += np.where((pca_dim[:, 0]<5) & (pca_dim[:, 1]>6))[0].tolist()
possible_outliers += np.where((pca_dim[:, 0]>0) & (d['diagnosis'].values.ravel()==0))[0].tolist()
possible_outliers += np.where((pca_dim[:, 0]<0) & (d['diagnosis'].values.ravel()==1))[0].tolist()
possible_outliers += np.where((pca_dim[:, 0]>4) & (d['diagnosis'].values.ravel()==0))[0].tolist()
print(list(set(possible_outliers)))

#Pairs plot/SPLOM
splom = sns.pairplot(d.iloc[:, 3:])#, diag_kind="kde")
fig = splom.fig
fig.suptitle('Pairs Plot of Wisconsin Breast Cancer Data')
plt.show()

#Mosaic plot
mosaic(d, ['clump_thickness', 'cell_size_uniformity'],
       title='Mosaic Plot of 2 Features from Wisconsin Breast Cancer Data')
plt.show()

#Plot parallel coordinates of potential outliers
fig = plt.figure()
fig.suptitle('Parallel Coordinates Plot of Potential Outliers in Wisconsin Breast Cancer Data')
parallel_coordinates(d.iloc[possible_outliers, :], class_column='diagnosis',
                     cols=d.columns[3:],
                     color=('#0158FE', '#FE0101'))
plt.show()

#-------------------------------------------------------------------------------------------------#
#----------------------------------------Robust Covariance----------------------------------------#
#-------------------------------------------------------------------------------------------------#

robust_cov = MinCovDet(assume_centered=False, random_state=14)
robust_cov.fit(d.iloc[:, 3:12])

#View covariance matrix before and after reweighting
sns.heatmap(robust_cov.raw_covariance_, annot=True)
plt.title('Raw Covariance Matrix')
plt.show()
sns.heatmap(robust_cov.covariance_, annot=True)
plt.title('Robust Covariance Matrix')
plt.show()

#View the Mahalanobis distances on the PCA plot
pca_model = PCA(n_components=None, whiten=False, random_state=14)
pca_dim = pca_model.fit_transform(d.iloc[:, 3:12])

plt.figure(figsize=(10, 5))
plt.xlabel('Latent Variable 1 (explains most variance)')
plt.ylabel('Latent Variable 2 (explains 2nd most variance)')
plt.title('PCA 2-Dimension Plot with Mahalanobis Outlier Score')
plt.scatter(pca_dim[:, 0], pca_dim[:, 1], c=robust_cov.dist_.ravel())
plt.colorbar()
plt.show()

#Use chi-square distribution to find cutoff distance for outliers
critical_value = chi2.ppf(q=0.975, df=len(d.columns[3:12]))
robust_cov_outliers = np.where(robust_cov.dist_ > critical_value)[0].tolist()
print("Indices of outliers found by robust covariance: \n", robust_cov_outliers)
d['robust_cov_outlier'] = [1 if i > critical_value else 0 for i in robust_cov.dist_]

#-------------------------------------------------------------------------------------------------#
#-------------------------------------Local Outlier Factor----------------------------------------#
#-------------------------------------------------------------------------------------------------#

expected_perc_outliers = round(sum(d.diagnosis)/len(d), 1)
lof = LocalOutlierFactor(n_neighbors=5, metric="minkowski", p=2,
                         contamination=expected_perc_outliers)
lof.fit(d.iloc[:, 3:12])

lof_outliers = np.where(lof.negative_outlier_factor_*-1 > 1)[0].tolist()
print("Indices of outliers found by LOF: \n", lof_outliers)
print("Indices flagged as outliers by both robust covariance and LOF: \n",
      np.intersect1d(robust_cov_outliers, lof_outliers))
d['lof_outlier'] = [1 if i > 1 else 0 for i in lof.negative_outlier_factor_*-1]

#-------------------------------------------------------------------------------------------------#
#------------------------------------------One-Class SVM------------------------------------------#
#-------------------------------------------------------------------------------------------------#

expected_perc_outliers = round(sum(d.diagnosis)/len(d), 1)
boundary_smoothness = 1/len(d.columns[3:12])
ocsvm = OneClassSVM(kernel='rbf', nu=expected_perc_outliers,
                    gamma=boundary_smoothness, random_state=14)
ocsvm.fit(d.iloc[:, 3:12])

ocsvm_outliers = np.where(ocsvm.predict(d.iloc[:, 3:12])==-1)[0].tolist()
print("Indices of outliers found by One-Class SVM: \n", ocsvm_outliers)
d['ocsvm_outlier'] = [1 if i == -1 else 0 for i in ocsvm.predict(d.iloc[:, 3:12])]

#-------------------------------------------------------------------------------------------------#
#---------------------------------------Isolation Forest------------------------------------------#
#-------------------------------------------------------------------------------------------------#

expected_perc_outliers = round(sum(d.diagnosis)/len(d), 1)
isoforest = IsolationForest(n_estimators=99,
                            contamination=expected_perc_outliers,
                            max_features=1.0, random_state=14)
isoforest.fit(d.iloc[:, 3:12])

isoforest_outliers = np.where(isoforest.predict(d.iloc[:, 3:12])==-1)[0].tolist()
print("Indices of outliers found by Isolation Forest: \n", isoforest_outliers)
d['isoforest_outlier'] = [1 if i == -1 else 0 for i in isoforest.predict(d.iloc[:, 3:12])]

#-------------------------------------------------------------------------------------------------#
#-------------------------------------Evaluation as Classifiers-----------------------------------#
#-------------------------------------------------------------------------------------------------#

#Evaluate outlier detection methods as classifier for minority class
print("Robust Covariance AUC ", roc_auc_score(d.diagnosis, d.robust_cov_outlier))
print("Local Outlier Factor AUC ", roc_auc_score(d.diagnosis, d.lof_outlier))
print("One-Class SVM AUC ", roc_auc_score(d.diagnosis, d.ocsvm_outlier))
print("Isolation Forest AUC ", roc_auc_score(d.diagnosis, d.isoforest_outlier))

#Create a weighted ensemble score
d['ensemble_outlier_score'] = round((d.robust_cov_outlier*roc_auc_score(d.diagnosis, d.robust_cov_outlier) \
                                    + d.lof_outlier*roc_auc_score(d.diagnosis, d.lof_outlier) \
                                    + d.ocsvm_outlier*roc_auc_score(d.diagnosis, d.ocsvm_outlier) \
                                    + d.isoforest_outlier*roc_auc_score(d.diagnosis, d.isoforest_outlier)\
                                    )/(roc_auc_score(d.diagnosis, d.robust_cov_outlier) \
                                       + roc_auc_score(d.diagnosis, d.lof_outlier) \
                                       + roc_auc_score(d.diagnosis, d.ocsvm_outlier) \
                                       + roc_auc_score(d.diagnosis, d.isoforest_outlier)), 0)
print("Ensemble AUC ", roc_auc_score(d.diagnosis, d.ensemble_outlier_score))

#Export and use conditional formatting in Excel to compare
d.to_csv('wbcdata_clean_outlier_tagging.csv', index=False)
