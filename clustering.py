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
        self.num_values = self.df._get_numeric_data().values # features
        self.df_tsne = pd.DataFrame([]) # contains values of tsne decomp for each observation
        self.df_pca = pd.DataFrame([]) # contains values of PCA decomp for each observation
        self.k_clusters = 0 #optimal number of clusters. will be set in kmeans
        self.k_silhouette = 0 #silhouette score of the optimal number of clusters. will be set in kmeans
        self.kclusters = 0

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


    def optimal_k_means(self, range_to_test = range(2,11)):
        '''
        Calculate optimal number of kmeans clusters by maximizing silhouette score
        '''
        results = {}
        for num_clusters in range_to_test:
            kmeans = KMeans(n_clusters = num_clusters, random_state=0).fit(self.num_values)
            results[num_clusters] = silhouette_score(self.num_values, kmeans.labels_)
        sil_score, optimal_number_of_clusters = max([(value, key) for key, value in results.items()])
        self.k_clusters = optimal_number_of_clusters
        self.k_silhouette = sil_score

    def _kmeans(self, range_to_test = range(2,11)):
        # get the optimal # of kmeans
        self.optimal_k_means(range_to_test)
        kmeans = KMeans(n_clusters = self.k_clusters, random_state=0).fit(self.num_values)
        self.df_tsne['kmeans'] = kmeans.labels_
        self.df_pca['kmeans'] = kmeans.labels_
        self.df['kmeans'] = kmeans.labels_
        self.kclusters = kmeans.cluster_centers_

    def plot_kmeans(self):
        '''Plot the kmeans clusters as hues overlayed on the tsne and pca plots
        Requirements: _PCA,_TSNE,optimal have already been run
        '''
        sns.lmplot(x='tsne_one', y ='tsne_two', hue= 'kmeans', data = self.df_tsne , fit_reg=False, legend=False)
        sns.lmplot(x='pca_one', y ='pca_two', hue= 'kmeans', data = self.df_pca , fit_reg=False, legend=False)
        plt.show()
