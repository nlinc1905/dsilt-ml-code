
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist, pdist
from numpy.matlib import repmat
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
#from hdbscan import HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

seed = 14

#-------------------------------------------------------------------------------------------------#
#---------------------------------------EDA and Cleaning------------------------------------------#
#-------------------------------------------------------------------------------------------------#

raw_d = pd.read_excel("dsilt-ml-code/03 Rule Learners/Online Retail.xlsx", sheet_name="Online Retail")
print(raw_d.head())

# Add basic features
raw_d['Date'] = raw_d['InvoiceDate'].dt.date
raw_d['Time'] = raw_d['InvoiceDate'].dt.time
raw_d['Hour'] = raw_d['InvoiceDate'].dt.hour
print(min(raw_d.Date), max(raw_d.Date)) #Should match documentation
raw_d['Amount'] = raw_d.Quantity*raw_d.UnitPrice
raw_d['InvoiceNbrItems'] = raw_d[['StockCode', 'InvoiceNo']].groupby(['InvoiceNo']).transform('count')
raw_d['InvoiceQuantity'] = raw_d[['Quantity', 'InvoiceNo']].groupby(['InvoiceNo']).transform('sum')
raw_d['InvoiceTotal'] = raw_d[['Amount', 'InvoiceNo']].groupby(['InvoiceNo']).transform('sum')
raw_d['Day'] = raw_d['InvoiceDate'].dt.day
raw_d['Month'] = raw_d['InvoiceDate'].dt.month
raw_d['Year'] = raw_d['InvoiceDate'].dt.year

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

naCol(raw_d)
print(raw_d.info())

# Remove some of the same records as before (see chapter 3)
clean_d = raw_d[(~raw_d['InvoiceNo'].str.slice(0,1).isin(['C', 'A'])) \
                & (raw_d['Quantity']>0) \
                & (raw_d['UnitPrice']>0) \
                & (~raw_d['StockCode'].isin(['AMAZONFEE', 'M', 'POST', 'DOT', 'B'])) \
                & (~raw_d['Description'].isna())].copy()
# Before removing missing customer IDs, see if they share a pattern
clean_d['MissingCust'] = list(np.where(clean_d['CustomerID'].isna(), 1, 0))
sns.boxplot(x=clean_d[clean_d['Quantity']<50]['MissingCust'],
            y=clean_d[clean_d['Quantity']<50]['Quantity'])
plt.show()
sns.boxplot(x=clean_d[clean_d['InvoiceTotal']<1000]['MissingCust'],
            y=clean_d[clean_d['InvoiceTotal']<1000]['InvoiceTotal'])
plt.show()
# Missing custs buy fewer items of each type but have larger average bills

clean_d['InvoiceNo'] = clean_d['InvoiceNo'].astype('int')
clean_d['Description'] = clean_d['Description'].str.strip()

# Explore countries
print(clean_d['Country'].value_counts().sort_values(ascending=False))
sns.distplot(clean_d.Amount)
plt.show()
print(clean_d[clean_d['Amount']>10000].head())
sns.distplot(clean_d[clean_d['Amount']<1000]['InvoiceTotal'])
plt.show()
print(clean_d[clean_d['InvoiceTotal']>50000].head())
print(clean_d['Quantity'].value_counts())
print(clean_d[clean_d['Quantity']>5000].head())
# No outliers appear to be data errors

#-------------------------------------------------------------------------------------------------#
#--------------------------------------Feature Engineering----------------------------------------#
#-------------------------------------------------------------------------------------------------#

# Separate unknown customers
unknown_custs = clean_d[clean_d['MissingCust']==1].copy()
unknown_custs.drop('MissingCust', axis=1, inplace=True)
clean_d = clean_d[~clean_d['MissingCust'].isna()]
clean_d.drop('MissingCust', axis=1, inplace=True)

print(clean_d.info())

# Create customer level features
clean_d['CustNumInvoices'] = clean_d[['CustomerID', 'InvoiceNo']].groupby(['CustomerID']).transform('count')

clean_d['CustTotalItems'] = clean_d[['CustomerID', 'StockCode']].groupby(['CustomerID']).transform('count')
clean_d['CustAvgItems'] = clean_d[['CustomerID', 'InvoiceNbrItems']].groupby(['CustomerID']).transform(np.mean)
clean_d['CustStdItems'] = clean_d[['CustomerID', 'InvoiceNbrItems']].groupby(['CustomerID']).transform(np.std)
clean_d['CustMinItems'] = clean_d[['CustomerID', 'InvoiceNbrItems']].groupby(['CustomerID']).transform('min')
clean_d['CustMaxItems'] = clean_d[['CustomerID', 'InvoiceNbrItems']].groupby(['CustomerID']).transform('max')
clean_d['CustItemsRange'] = clean_d.CustMaxItems-clean_d.CustMinItems

clean_d['CustTotalQuant'] = clean_d[['CustomerID', 'Quantity']].groupby(['CustomerID']).transform('sum')
clean_d['CustAvgQuant'] = clean_d[['CustomerID', 'InvoiceQuantity']].groupby(['CustomerID']).transform(np.mean)
clean_d['CustStdQuant'] = clean_d[['CustomerID', 'InvoiceQuantity']].groupby(['CustomerID']).transform(np.std)
clean_d['CustMinQuant'] = clean_d[['CustomerID', 'InvoiceQuantity']].groupby(['CustomerID']).transform('min')
clean_d['CustMaxQuant'] = clean_d[['CustomerID', 'InvoiceQuantity']].groupby(['CustomerID']).transform('max')
clean_d['CustQuantRange'] = clean_d.CustMaxQuant-clean_d.CustMinQuant

clean_d['CustTotalAmt'] = clean_d[['CustomerID', 'UnitPrice']].groupby(['CustomerID']).transform('sum')
clean_d['CustAvgAmt'] = clean_d[['CustomerID', 'InvoiceTotal']].groupby(['CustomerID']).transform(np.mean)
clean_d['CustStdAmt'] = clean_d[['CustomerID', 'InvoiceTotal']].groupby(['CustomerID']).transform(np.std)
clean_d['CustMinAmt'] = clean_d[['CustomerID', 'InvoiceTotal']].groupby(['CustomerID']).transform('min')
clean_d['CustMaxAmt'] = clean_d[['CustomerID', 'InvoiceTotal']].groupby(['CustomerID']).transform('max')
clean_d['CustAmtRange'] = clean_d.CustMaxAmt-clean_d.CustMinAmt

clean_d['CustAvgHr'] = clean_d[['CustomerID', 'Hour']].groupby(['CustomerID']).transform(np.mean)
clean_d['CustStdHr'] = clean_d[['CustomerID', 'Hour']].groupby(['CustomerID']).transform(np.std)
clean_d['CustMinHr'] = clean_d[['CustomerID', 'Hour']].groupby(['CustomerID']).transform('min')
clean_d['CustMaxHr'] = clean_d[['CustomerID', 'Hour']].groupby(['CustomerID']).transform('max')
clean_d['CustHrRange'] = clean_d.CustMaxHr-clean_d.CustMinHr

temp = clean_d[['CustomerID', 'InvoiceNo', 'Date']]\
       .copy()\
       .sort_values(['CustomerID', 'Date'])\
       .groupby('CustomerID')
temp = (temp['Date'].diff()/(60*60*24)).reset_index(drop=True).dt.days
clean_d['RunDaysSinceLastPurch'] = temp

clean_d['AvgDaysBtwnPurch'] = clean_d[['RunDaysSinceLastPurch', 'CustomerID']].groupby(['CustomerID']).transform(np.mean)
clean_d['StdDaysBtwnPurch'] = clean_d[['RunDaysSinceLastPurch', 'CustomerID']].groupby(['CustomerID']).transform(np.std)
clean_d['MinDaysBtwnPurch'] = clean_d[['RunDaysSinceLastPurch', 'CustomerID']].groupby(['CustomerID']).transform(np.min)
clean_d['MaxDaysBtwnPurch'] = clean_d[['RunDaysSinceLastPurch', 'CustomerID']].groupby(['CustomerID']).transform(np.max)
clean_d['DaysBtwnPurchRange'] = clean_d.MaxDaysBtwnPurch-clean_d.MinDaysBtwnPurch

# Market to new customers
clean_d['CustFirstPurchaseDt'] = clean_d[['Date', 'CustomerID']].groupby(['CustomerID']).transform('min')
clean_d['NewCust'] = np.where((max(clean_d['Date'])-clean_d['CustFirstPurchaseDt']).dt.days<90, 1, 0)
# Entice lost customers to return
clean_d['CustLastPurchaseDt'] = clean_d[['Date', 'CustomerID']].groupby(['CustomerID']).transform('max')
clean_d['LostCust'] = np.where((clean_d['CustLastPurchaseDt']-min(clean_d['Date'])).dt.days<90, 1, 0)
# Market to recent customers (not necessarily new)
clean_d['DaysSinceLastPurch'] = (dt.date.today()-clean_d['CustLastPurchaseDt']).dt.days

clean_d.dropna(axis=0, how='any', inplace=True)
clean_d.reset_index(drop=True, inplace=True)
print(clean_d.head())
clean_d.to_csv('online_retail_clean.csv', index=False)

#-------------------------------------------------------------------------------------------------#
#------------------------------------------Clustering Prep----------------------------------------#
#-------------------------------------------------------------------------------------------------#

clean_d = pd.read_csv('online_retail_clean.csv')

cols_not_to_cluster = ['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                       'InvoiceDate', 'UnitPrice', 'Country',
                       'Date', 'Time', 'Hour', 'Amount', 'InvoiceNbrItems',
                       'InvoiceQuantity', 'InvoiceTotal', 'Day', 'Month',
                       'Year', 'RunDaysSinceLastPurch', 'NewCust', 'LostCust']
d_to_clust = clean_d.drop(cols_not_to_cluster, axis=1).copy()
d_to_clust['CustFirstPurchaseDt'] = pd.factorize(d_to_clust['CustFirstPurchaseDt'])[0]
d_to_clust['CustLastPurchaseDt'] = pd.factorize(d_to_clust['CustLastPurchaseDt'])[0]
d_to_clust.drop_duplicates(inplace=True)
d_to_clust.reset_index(drop=True, inplace=True)

def logTransform(x):
    return np.log(x+0.001)  # Add small number to prevent Inf
standardizer = StandardScaler()

cols_to_log_transform = ['CustNumInvoices', 'CustTotalItems', 'CustAvgItems',
                         'CustStdItems', 'CustTotalQuant', 'CustAvgQuant',
                         'CustStdQuant', 'CustTotalAmt', 'CustAvgAmt',
                         'CustStdAmt', 'AvgDaysBtwnPurch', 'StdDaysBtwnPurch']
# Ignore the warning from the line below - it's covered by .replace()
d_to_clust[cols_to_log_transform] = d_to_clust[cols_to_log_transform].apply(lambda x: logTransform(x)).replace(-np.inf, 0)
d_to_clust.iloc[:,1:] = standardizer.fit_transform(d_to_clust.iloc[:,1:])
d_to_clust.fillna(0)
print(d_to_clust.head())

x_std = d_to_clust.iloc[:,1:] # Makes things easier

def silhouettePlot(d, cluster_labels):
    fig, ax = plt.subplots(1, 1)
    y_lower_bound = 10
    sil_avg = silhouette_score(d, labels=cluster_labels,
                               metric='euclidean',
                               random_state=seed)
    silhouette_values = silhouette_samples(d, cluster_labels)
    nbr_clusters = len(set(cluster_labels))
    for i in range(nbr_clusters):
        ith_cluster_silhouette_values = silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        ith_cluster_size = ith_cluster_silhouette_values.shape[0]
        y_upper_bound = y_lower_bound + ith_cluster_size
        color = cm.nipy_spectral(float(i)/nbr_clusters)
        ax.fill_betweenx(np.arange(y_lower_bound, y_upper_bound),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color)
        ax.text(-0.05, y_lower_bound+0.5*ith_cluster_size, str(i))
        y_lower_bound = y_upper_bound+10
    ax.set_title("Silhouette Plot for {} Clusters".format(nbr_clusters))
    ax.set_xlabel("Silhouette Coefficients")
    ax.set_ylabel("Cluster Label by Sample")
    ax.axvline(x=sil_avg, color="red", linestyle="--")
    ax.set_yticks([])
    plt.show()
    return

#-------------------------------------------------------------------------------------------------#
#---------------------------------------------k-Means---------------------------------------------#
#-------------------------------------------------------------------------------------------------#

km_model = KMeans(n_clusters=5, n_init=20, random_state=seed)
km_results = km_model.fit(x_std)
print("Total Within Cluster Sum of Squares:",
      km_results.inertia_)
cluster_colormap = np.array(['red', 'orange', 'lime', 'blue', 'violet'])
plt.scatter(d_to_clust['CustNumInvoices'], d_to_clust['CustAvgAmt'],
            c=cluster_colormap[km_model.labels_], s=10)
plt.title('5-Means Clusters on Two Features')
plt.show()

# Try a few different k's
K = range(2, 20)
kms = [KMeans(n_clusters=k, n_init=20, random_state=seed)\
       .fit(x_std) for k in K]
centroids = [k.cluster_centers_ for k in kms]
dists_to_k = [cdist(x_std, center, 'euclidean') for center in centroids]
min_dists = [np.min(d, axis=1) for d in dists_to_k]
avg_within_ss = [sum(d)/x_std.shape[0] for d in min_dists]
inerts = [k.inertia_ for k in kms]
plt.plot([k for k in K], inerts, 'b*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Total Within Cluster Sum of Squares (Error)')
plt.title('Elbow for K-Means Clustering')
plt.show()
print("Smallest error when k =", np.where(inerts==min(inerts))[0][0])

# Find optimal k with elbow method
nPoints = len(inerts)
allCoord = np.vstack((range(1, nPoints+1), inerts)).T
print(np.array([range(1, nPoints+1), inerts]))
firstPoint = allCoord[0]
lineVec = allCoord[-1] - allCoord[0]
lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
vecFromFirst = allCoord - firstPoint
scalarProduct = np.sum(vecFromFirst*repmat(lineVecNorm, nPoints, 1), axis=1)
vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
vecToLine = vecFromFirst - vecFromFirstParallel
distToLine = np.sqrt(np.sum(vecToLine**2, axis=1))
bestK = [k for k in K][np.argmax(distToLine)]
print("Best k According to Elbow Method =", bestK)

# Find optimal k with silhouette plot
silhouette_scores = []
for km_idx, km in enumerate(kms):
    sil_avg = silhouette_score(x_std, labels=km.labels_,
                               metric='euclidean',
                               random_state=seed)
    silhouette_scores.append(sil_avg)
    print("For k = {}".format(K[km_idx]),
          "the average silhouette score is:", sil_avg)
bestK = [k for k in K][np.argmax(silhouette_scores)]
print("Best k According to Silhouette Method =", bestK)

for km in kms[:3]:
    silhouettePlot(x_std, km.labels_)

#-------------------------------------------------------------------------------------------------#
#----------------------------------Hierarchical Clustering----------------------------------------#
#-------------------------------------------------------------------------------------------------#

# Agglomerative clustering
# Linkage methods: single (default), complete, average, weighted, centroid, median, ward
# Distance measures: euclidean (default), minkowski, correlation, hamming, and others, see: help(pdist)
hc_single_link = linkage(x_std, method='single', metric='euclidean')
hc_ward_link = linkage(x_std, method='ward', metric='euclidean')

#Check the Cophenetic correlation coefficient
#The closer to 1 this is, the better the original point distances are preserved
coph_coeffs = [cophenet(link_method)[0] for link_method in [hc_single_link, hc_ward_link]]
print("Cophenetic correlation coefficients:", coph_coeffs)

plt.figure(figsize=(12, 5))
plt.title("Hierarchical Agglomerative Clustering Dendrogram Truncated (Single Linkage)")
plt.xlabel("Sample Index or Cluster Size")
plt.ylabel("Euclidean Distance")
dendrogram(hc_single_link,
           truncate_mode='lastp', #Show only the last p merged clusters
           p=30,
           leaf_rotation=90.,     #Rotate the x-axis labels for ease of reading
           leaf_font_size=10.,
           show_contracted=True   #Shows distribution of leaves in branch
           )
plt.show()

# Find optimal clusters using elbow method
dists = hc_single_link[-20:, 2]
# ^ Column index 2 is the distance, truncate to last 40 distances (these are actually the first 40 splits from top down)
dists_rev = dists[::-1]
# ^ Reverse view of the array to go from top of hierarchy to bottom (view distance from top down)
idxs = np.arange(1, len(dists)+1)
plt.plot(idxs, dists_rev)
plt.title('Elbow Plot for Hierarchical Clusters with 2nd Derivative')
plt.xlabel('Split or Partition')
plt.ylabel('Cluster Distance')
plt.grid(True)
plt.xticks(np.arange(min(idxs), max(idxs)+1, 1.0))
#Take the second derivative of the elbow graph to mathematically find optimal number of clusters
dists_2derv = np.diff(dists, 2)
dists_2derv_rev = dists_2derv[::-1]
plt.plot(idxs[:-2], dists_2derv_rev)
plt.show()
bestKSplits = dists_2derv_rev.argmax() + 2  #The number of clusters is always 2 more than the argmax index
print ('Optimal number of clusters according to scree plot:', bestKSplits)
num_clusts = bestKSplits
clusters = fcluster(hc_single_link, num_clusts, criterion='maxclust')
plt.figure(figsize=(10, 5))
plt.scatter(d_to_clust['CustNumInvoices'], d_to_clust['CustAvgAmt'],
            c=clusters, cmap='prism')
plt.title('Hiearchical Clusters on 2 Features, Optimized by Elbow Method')
plt.show()

# Alernative way to find best number of clusters is to specify a distance threshold
max_dist = 4
h_clusters = fcluster(hc_single_link, max_dist, criterion='distance')
print(h_clusters)
plt.figure(figsize=(10, 5))
plt.scatter(d_to_clust['CustNumInvoices'], d_to_clust['CustAvgAmt'],
            c=h_clusters, cmap='prism')
plt.title('Hiearchical Clusters on 2 Features, Optimized by Distance Threshold')
plt.show()

#-------------------------------------------------------------------------------------------------#
#----------------------------------Density Based Clustering---------------------------------------#
#-------------------------------------------------------------------------------------------------#

# DBSCAN
minpts = x_std.shape[1]+1
db_model = DBSCAN(eps=3, min_samples=minpts, metric='euclidean', leaf_size=30)
db_results = db_model.fit(x_std)
plt.scatter(d_to_clust['CustNumInvoices'], d_to_clust['CustAvgAmt'],
            c=db_model.labels_, cmap="prism", s=10)
plt.title('DBSCAN Clusters on Two Features')
plt.show()
'''
# Hierarchical DBSCAN
hdbs_model = HDBSCAN(algorithm='best', allow_single_cluster=False, 
                     approx_min_span_tree=True, gen_min_span_tree=True, 
                     metric='euclidean', min_cluster_size=10)
hdbs_results = hdbs_model.fit_predict(x_std)

#Plot the minimum spanning tree (only possible if gen_min_span_tree=True)
hdbs_model.minimum_spanning_tree_.plot(edge_cmap='viridis', 
                                       edge_alpha=0.6, node_size=80, 
                                       edge_linewidth=2)
plt.show()
#Plot condensed cluster hierarchy with circles around clusters (set to False to turn off)
hdbs_model.condensed_tree_.plot(select_clusters=True)
plt.show()
#Plot clusters and first 2 features of dataset
plt.scatter(d_to_clust['CustNumInvoices'], d_to_clust['CustAvgAmt'],
            c=hdbs_results, cmap="prism")
plt.title("HDBScan Clusters on 2 Features")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
'''

#-------------------------------------------------------------------------------------------------#
#--------------------------------------Spectral Clustering----------------------------------------#
#-------------------------------------------------------------------------------------------------#

spec_clust_model = SpectralClustering(n_clusters=5,
                                      random_state=seed, n_init=20,
                                      gamma=1.0, affinity='rbf')
spec_clust_results = spec_clust_model.fit(x_std)
plt.scatter(d_to_clust['CustNumInvoices'], d_to_clust['CustAvgAmt'],
            c=spec_clust_results.labels_, cmap="prism", s=10)
plt.title('Spectral Clusters on Two Features')
plt.show()

#-------------------------------------------------------------------------------------------------#
#-----------------------------------Gaussian Mixture Models---------------------------------------#
#-------------------------------------------------------------------------------------------------#

gm_model = GaussianMixture(n_components=5, covariance_type='full',
                           max_iter=100, n_init=10,
                           init_params='kmeans',
                           random_state=14)

gm_results = gm_model.fit_predict(x_std)
plt.scatter(d_to_clust['CustNumInvoices'], d_to_clust['CustAvgAmt'],
            c=gm_results, cmap="prism", s=10)
plt.title('Gaussian Mixture Labels on Two Features')
plt.show()

# Find optimal number of mixtures with silhouette plot
K = range(2, 10)
gms = [GaussianMixture(n_components=k, covariance_type='full',
                       max_iter=100, n_init=10,
                       init_params='kmeans',
                       random_state=14).fit_predict(x_std)\
       for k in K]
silhouette_scores = []
for gm_idx, gm in enumerate(gms):
    sil_avg = silhouette_score(x_std, labels=gm,
                               metric='euclidean',
                               random_state=seed)
    silhouette_scores.append(sil_avg)
    print("For number of mixtures = {}".format(K[gm_idx]),
          "the average silhouette score is:", sil_avg)
bestK = [k for k in K][np.argmax(silhouette_scores)]
print("Best Number of Mixtures According to Silhouette Method =", bestK)
silhouettePlot(x_std, gms[np.argmax(silhouette_scores)])

#-------------------------------------------------------------------------------------------------#
#-----------------------------------Embedding with t-SNE------------------------------------------#
#-------------------------------------------------------------------------------------------------#

tsne_model = TSNE(n_components=2,
                  perplexity=30,
                  learning_rate=200,
                  random_state=seed)
tsne_model.fit_transform(x_std)
tsne_embed = tsne_model.embedding_

plt.scatter(tsne_embed[:,0], tsne_embed[:,1], c=km_model.labels_)
plt.title('t-SNE Plot with k-Means Cluster Assignment')
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.colorbar()
plt.show()

plt.scatter(tsne_embed[:,0], tsne_embed[:,1], c=h_clusters)
plt.title('t-SNE Plot with Agglomerative Single Linkage Cluster Assignment')
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.colorbar()
plt.show()

plt.scatter(tsne_embed[:,0], tsne_embed[:,1], c=db_model.labels_)
plt.title('t-SNE Plot with DBSCAN Cluster Assignment')
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.colorbar()
plt.show()

plt.scatter(tsne_embed[:,0], tsne_embed[:,1], c=spec_clust_results.labels_)
plt.title('t-SNE Plot with Spectral Cluster Assignment')
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.colorbar()
plt.show()

plt.scatter(tsne_embed[:,0], tsne_embed[:,1], c=gm_results)
plt.title('t-SNE Plot with Gaussian Mixture Cluster Assignment')
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.colorbar()
plt.show()

#-------------------------------------------------------------------------------------------------#
#------------------------------------Improved Clustering------------------------------------------#
#-------------------------------------------------------------------------------------------------#

new_d_to_clust = clean_d[['CustAvgItems', 'CustAvgAmt', 'AvgDaysBtwnPurch', 'DaysSinceLastPurch']].copy()
new_d_to_clust.drop_duplicates(inplace=True)
new_d_to_clust.reset_index(drop=True, inplace=True)
new_d_to_clust = new_d_to_clust.apply(lambda x: logTransform(x)).replace(-np.inf, 0)
new_x_std = x_std[['CustAvgItems', 'CustAvgAmt', 'AvgDaysBtwnPurch', 'DaysSinceLastPurch']].copy()

# Try a GMM with 4 clusters (note that optimal k is 2 if evaluated with silhouette)
gm_model = GaussianMixture(n_components=4, covariance_type='full',
                           max_iter=100, n_init=10,
                           init_params='kmeans',
                           random_state=14)

gm_results = gm_model.fit_predict(new_x_std)
plt.scatter(new_d_to_clust['CustAvgAmt'], new_d_to_clust['AvgDaysBtwnPurch'],
            c=gm_results, cmap=plt.cm.jet, s=10)
plt.title('Gaussian Mixture Labels on Two Features')
plt.xlabel('Log Customer Average Amount')
plt.ylabel('Log Average Days Between Purchase')
plt.show()
plt.scatter(new_d_to_clust['CustAvgItems'], new_d_to_clust['CustAvgAmt'],
            c=gm_results, cmap=plt.cm.jet, s=10)
plt.title('Gaussian Mixture Labels on Two Features')
plt.xlabel('Log Customer Average Items')
plt.ylabel('Log Customer Average Amount')
plt.show()
plt.scatter(new_d_to_clust['CustAvgAmt'], new_d_to_clust['DaysSinceLastPurch'],
            c=gm_results, cmap=plt.cm.jet, s=10)
plt.title('Gaussian Mixture Labels on Two Features')
plt.xlabel('Log Customer Average Amount')
plt.ylabel('Log Days Since Last Purchase')
plt.show()

tsne_model = TSNE(n_components=2,
                  perplexity=30,
                  learning_rate=200,
                  random_state=seed)
tsne_model.fit_transform(new_x_std)
tsne_embed = tsne_model.embedding_

plt.scatter(tsne_embed[:,0], tsne_embed[:,1], c=gm_results)
plt.title('t-SNE Plot with Gaussian Mixture Cluster Assignment')
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.colorbar()
plt.show()
