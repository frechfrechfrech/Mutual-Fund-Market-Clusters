## Import data
## Import methods
import numpy as np
import pandas as pd
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
pd.set_option("display.max_columns",300)
from pandas.tools.plotting import scatter_matrix
from scipy import stats
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_samples, silhouette_score
import pickle
from sklearn.preprocessing import MinMaxScaler


# # # Prep the data
# #
# filepath = 'data/ALL_CLIENTS_Q417only_cat_bd_zip_redemptions_20180417.txt'
# df_inc_red = pd.read_csv(filepath, sep='\t')
# #set zip as str
# df_inc_red['ZIP_CODE']=df_inc_red['ZIP_CODE'].astype(str)
# # fill NA with 0
# df_inc_red.fillna(0, inplace=True)
# #drop redemptions for now
# df = df_inc_red.drop('GLOBAL_REDEMPTIONS', axis=1)
# #pivot
# row_headers = ['BROKER_NAME', 'FDS_BROKER_ID', 'ZIP_CODE','STREET_ADDRESS','STATE']
# column_headers = ['BROAD_FUND_CATEGORY']
# agg_func = 'sum'
# df_pivot = pd.pivot_table(df, index=row_headers,columns = column_headers, aggfunc = agg_func, fill_value=0)
# # reset index
# df_pivot = pd.DataFrame(df_pivot.to_records())
# # drop offices with less than $1M in sales
# df_pivot = df_pivot[df_pivot.sum(axis=1)>1000000]
# # change negative values to zero
# num = df_pivot._get_numeric_data()
# num[num < 0] = 0
# # add totals column
# df_pivot['Total'] = num.sum(axis=1)
# # df_pivot %
# df_pivot_pct_all = df_pivot.copy()
# for i in range(5,12):
#     df_pivot_pct_all.iloc[:,i]= df_pivot_pct_all.iloc[:,i]/df_pivot_pct_all['Total']
#
# # Test that division worked, this should be True
# df_pivot_pct_all.iloc[:, 5:12].sum(axis=1).mean() == 1
#
# df_pivot_pct_all.to_pickle('data/df_pivot_pct_all.pkl')

'''read in data and get the bd you care about'''
df_pivot_pct_all = pd.read_pickle('data/df_pivot_pct_all.pkl')
# list of all BDs in the file
df_bd_id = df_pivot_pct_all.loc[:,['FDS_BROKER_ID','BROKER_NAME']].drop_duplicates()
# search for a particular BD
df_bd_id[df_bd_id['BROKER_NAME'].str.contains('EDWARD', na=False)]

#Ameriprise only
df_amp = df_pivot_pct_all[df_pivot_pct_all['BROKER_NAME']=='AMERIPRISE FINANCIAL SERVICES, INC.']
df_amp = df_amp.reset_index()
df_amp.drop(['index'],inplace=True, axis=1)
# X=df_amp._get_numeric_data().iloc[:,:-1].values


'''old way of doing it'''
# # visualize in fewer dims
# tsne = TSNE(n_components=2, perplexity = 30)
# data_tsne = tsne.fit_transform(X)
# df_tsne = pd.DataFrame(data_tsne, columns = ['tsne_one','tsne_two'])
# sns.lmplot(x='tsne_one', y ='tsne_two', data = df_tsne , fit_reg=False, legend=False)
# plt.show()
#
#
# pca = PCA(n_components=2)
# data_pca = pca.fit_transform(X)
# df_pca = pd.DataFrame(data_pca, columns = ['pca_one','pca_two'])
# sns.lmplot(x='pca_one', y ='pca_two', data = df_pca , fit_reg=False, legend=False)
# plt.show()
#
# # Make clusters: try clusters 2-10
#
# results = {}
# for num_clusters in range(2, 21):
#     kmeans= KMeans(n_clusters = num_clusters, random_state=0).fit(X)
#     results[num_clusters] = silhouette_score(X, kmeans.labels_)
#
# optimal_number_of_clusters = max([(value, key) for key, value in results.items()])[1]
#
#
# # 5 clusters on KMeans
# kmeans5 = KMeans(n_clusters = 3, random_state=0).fit(X)
# df_amp['kmeans5'] = kmeans5.labels_
# df_tsne['kmeans5'] = kmeans5.labels_
# df_pca['kmeans5'] = kmeans5.labels_
#
# sns.lmplot(x='tsne_one', y ='tsne_two', hue= 'kmeans5', data = df_tsne , fit_reg=False, legend=False)
# sns.lmplot(x='pca_one', y ='pca_two', hue= 'kmeans5', data = df_pca , fit_reg=False, legend=False)
#




'''try the base proportions with the clustering class'''
# from clustering import Clustering
run clustering.py
c = Clustering(df_amp.iloc[:,:-1]) # initialize clustering class
# calculate and store dimension reductions
c._PCA()
c._TSNE()
# # plot dimension reductions
# c.plotPCA()
# c.plotTSNE()
# KMeans
c._kmeans()
c.plot_kmeans()

'''What's in these clusters?'''

row_headers = ['kmeans']
column_headers = ['STATE']
agg_func = 'count'
df_cluster_states = pd.pivot_table(c.df, index=row_headers,columns = column_headers, aggfunc = agg_func, fill_value=0)


df_cluster_props = pd.DataFrame(data = c.kclusters, columns = c.df._get_numeric_data().iloc[:,:-1].columns)

# reset index



'''try the base proportions + normalized size with the clustering class'''
df_amp_total_normed = df_amp.copy()
scaler = MinMaxScaler()
df_amp_total_normed['Total']=scaler.fit_transform(df_amp_total_normed['Total'].values.reshape(-1,1))
c2 = Clustering(df_amp_total_normed) # initialize clustering class
# calculate and store dimension reductions
c2._PCA()
c2._TSNE()
# # plot dimension reductions
# c.plotPCA()
# c.plotTSNE()
# KMeans
c2._kmeans()
c2.plot_kmeans()




''' try it with Edward Jones'''
df_ej = df_pivot_pct_all[df_pivot_pct_all['FDS_BROKER_ID']=='LMS13460']
cej = Clustering(df_ej.iloc[:,:-1]) # initialize clustering class
# calculate and store dimension reductions
cej._PCA()
cej._TSNE()
# # plot dimension reductions
# c.plotPCA()
# c.plotTSNE()
# KMeans
cej._kmeans()
cej.plot_kmeans()


'''okay now what do there clusters mean?'''
df['argmax']=np.argmax(df._get_numeric_data().values, axis=1) +3
df['max']= [df.columns[i] for i in df['argmax']


df_results_eps = pd.DataFrame([], columns = ['sil_score','labels','counts', 'pct_counts'], index=np.linspace(0.01,0.2,10))
for eps in np.linspace(0.01,0.2,10):
    dbscan = DBSCAN(eps = eps).fit(X)
    labels, counts = np.unique(dbscan.labels_, return_counts=True)
    df_results_eps.loc[eps,'labels'] = labels
    df_results_eps.loc[eps,'counts'] = counts
    df_results_eps.loc[eps,'pct_counts'] = np.round(counts/counts.sum(),2)
    if len(labels)==1:
        sil_score = -999
    else:
        sil_score = silhouette_score(X, dbscan.labels_)

    df_results_eps.loc[eps,'sil_score'] = sil_score


dbscan = DBSCAN(eps = 0.01).fit(X)
sil_score = silhouette_score(X, dbscan.labels_)
labels, counts = np.unique(dbscan.labels_, return_counts=True)
# pct_counts = counts/counts.sum()
results_eps[eps] = [sil_score, labels, counts]
#
#
# dbscan1000000 = DBSCAN(eps = 1000000).fit(df_amp._get_numeric_data().values)
# print('1M eps: {} clusters'.format(len(np.unique(dbscan1000000.labels_))))
# dbscan1000 =  DBSCAN(eps = 1000).fit(df_amp._get_numeric_data().values)
# print('1 thousand eps: {} clusters'.format(len(np.unique(dbscan1000.labels_))))
# dbscan10 = DBSCAN(eps = 10).fit(df_amp._get_numeric_data().values)
# print('10 eps: {} clusters'.format(len(np.unique(dbscan10.labels_))))

# what if we add in normalized size of office?
