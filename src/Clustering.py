import numpy as np
import pandas as pd
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
pd.set_option("display.max_columns",300)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_samples, silhouette_score

class Clustering(object):

    def __init__(self, df):
        self.df = df
        self.df_original = df.copy()
        self.num_values = self.df._get_numeric_data().values # features
        self.df_tsne = pd.DataFrame([]) # contains values of tsne decomp for each observation
        self.df_pca = pd.DataFrame([]) # contains values of PCA decomp for each observation
        self.num_clusters = {} # optimal number of clusters
        self.k_cluster_centers = {} #cluster centers
        self.k_silhouette = {} #silhouette score of the optimal number of clusters. will be set in kmeans

    def _PCA(self):
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.num_values)
        self.df_pca = pd.DataFrame(data_pca, columns = ['pca_one','pca_two'])

    def plotPCA(self, hue = None):
        sns.lmplot(x='pca_one', y ='pca_two', data = self.df_pca , fit_reg=False, legend=False)
        plt.show()

    def _TSNE(self, perplexity=30):
        tsne = TSNE(n_components=2, perplexity = perplexity)
        data_tsne = tsne.fit_transform(self.num_values)
        self.df_tsne = pd.DataFrame(data_tsne, columns = ['tsne_one','tsne_two'])

    def plotTSNE(self, hue = None):
        sns.lmplot(x='tsne_one', y ='tsne_two', data = self.df_tsne, hue=hue, fit_reg=False, legend=False)
        plt.show()

    def _kmeans(self, range_to_test = range(2,11), dim_red = 'raw'):
        '''
        calculates kmeans clustering for raw features, pca features, tsne features

        Inputs
        dim_red: (optional) None = raw features, tsne = kmeans run on tsne, pca = kmeans run on pca
        '''
        # Choose features: raw or dimension reduced
        if dim_red == 'raw':
            features = self.num_values
        elif dim_red == 'tsne':
            features = self.df_tsne.iloc[:,:2]
        elif dim_red == 'pca':
            features = self.df_pca.iloc[:,:2]
        # Calculate optimal number of kmeans clusters by maximizing silhouette scor
        results = {}
        for num_clusters in range_to_test:
            kmeans = KMeans(n_clusters = num_clusters, random_state=0).fit(features)
            results[num_clusters] = silhouette_score(features, kmeans.labels_)
        sil_score, optimal_number_of_clusters = max([(value, key) for key, value in results.items()])
        self.k_silhouette[dim_red] = sil_score
        self.num_clusters[dim_red] = optimal_number_of_clusters

        # get the optimal # of kmeans
        kmeans = KMeans(n_clusters = optimal_number_of_clusters, random_state=0).fit(features)

        # add the cluster label to the dataframes
        cluster_label_name = 'kmeans_{}'.format(dim_red)
        self.df_tsne[cluster_label_name] = kmeans.labels_
        self.df_pca[cluster_label_name] = kmeans.labels_
        self.df[cluster_label_name] = kmeans.labels_
        self.k_cluster_centers[dim_red] = kmeans.cluster_centers_


    def plot_kmeans(self, dim_red = 'raw'):
        '''Plot the kmeans clusters as hues overlayed on the tsne and pca plots
        Requirements: _PCA,_TSNE,optimal have already been run
        Input: dim_red: (optional) None = raw features, tsne = kmeans run on tsne, pca = kmeans run on pca
        '''
        cluster_label_name = 'kmeans_{}'.format(dim_red)
        sns.lmplot(x='tsne_one', y ='tsne_two', hue= cluster_label_name, data = self.df_tsne , fit_reg=False, legend=False)
        sns.lmplot(x='pca_one', y ='pca_two', hue= cluster_label_name, data = self.df_pca , fit_reg=False, legend=False)
        plt.show()

    def describe_clusters(self, dim_red = 'raw'):
        if dim_red == 'raw':
            feature_names = self.df_original._get_numeric_data().columns
        elif dim_red == 'tsne':
            features = self.df_tsne.iloc[:,:2].columns
        elif dim_red == 'pca':
            features = self.df_pca.iloc[:,:2].columns

        df_clusters_w_labels = pd.DataFrame(data = self.k_cluster_centers[dim_red], columns = feature_names)
        sorted_indices = np.argsort(-1*df_clusters_w_labels.values, axis=1)
        sorted_cols = np.array([np.array(df_clusters_w_labels.columns[i]) for i in sorted_indices])
        df_top3 = pd.DataFrame(data=sorted_cols[:,:3],columns=['1st','2nd','3rd'])
        return df_clusters_w_labels, df_top3

# if __name__ == '__main__':
#     df_pivot_pct_all = pd.read_pickle('data/df_pivot_pct_all.pkl')
#     df_amp = df_pivot_pct_all[df_pivot_pct_all['BROKER_NAME']=='AMERIPRISE FINANCIAL SERVICES, INC.']
#     df_amp = df_amp.reset_index()
#     df_amp.drop(['index'],inplace=True, axis=1)
#     c_amp = Clustering(df_amp.iloc[:,:-1]) # initialize clustering class
#     # calculate and store dimension reductions
#     c_amp._PCA()
#     c_amp._TSNE()
#     # KMeans
#     c_amp._kmeans(dim_red='tsne')
#     c_amp._kmeans(dim_red='pca')
#     c_amp._kmeans()
#     c_amp.plot_kmeans(dim_red='tsne')
