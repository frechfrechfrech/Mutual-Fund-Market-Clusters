import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
pd.set_option("display.max_columns",300)
from pandas.tools.plotting import scatter_matrix
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Process_Data(object):
    '''Process data for clustering

    load the data into a datafram
    replace Nans with zeros
    repalce
    pca/lstme down, put results in a different dataframe



    '''

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.df = pd.DataFrame([])
        self.df_pca = pd.DataFrame([])
        self.df_tsne = pd.DataFrame([])

    def _load_data(self):
        self.df = pd.read_csv(self.filepath, sep='\t')

    def _format_data(self):
        #replace Nans with zeros
        self.df.fillna(0, inplace=True)
        #make a column with rolling4Q sum
        self.df['R4Q Total'] = self.df.iloc[:,-4:].sum(axis=1)
        #drop funds where rolling4Q sum <1
        self.df = self.df[self.df['R4Q Total']>=1]

    def _pca(self):
        '''
        Find pca for numerical columns
        Output: assign for self.df_pca a copy of input dataframe with
                pca columns appended
        '''
        pca = PCA(n_components = 2)
        self.df_pca = self.df.copy()
        #use only the numeric columns
        numerical_df = self.df_pca.select_dtypes(include=[np.number])


        pca_result = pca.fit_transform(numerical_df.values)
        self.df_pca['pca_one'] = pca_result[:,0]
        self.df_pca['pca_two'] = pca_result[:,1]

    def _tsne(self):
        '''
        Find tsne for numerical columns
        Output: assign for self.df_tsne a copy of input dataframe with
                tsne columns appended
        '''
        tsne = TSNE(n_components = 2)
        self.df_tsne = self.df.copy()
        #use only the numeric columns
        numerical_df = self.df_tsne.select_dtypes(include=[np.number])


        tsne_result = tsne.fit_transform(numerical_df.values)
        self.df_tsne['tsne_one'] = tsne_result[:,0]
        self.df_tsne['tsne_two'] = tsne_result[:,1]


    def _plot_lower_dims(self, dim_red = 'pca'):
        '''
        plot the features in lower-dimensional space
        dim_red: the dimensionality reduction type to plot. Can be 'pca' or 'tsne'
        '''
        if dim_red == 'pca':
            pca_one = self.df_pca['pca_one']
            pca_two = self.df_pca['pca_two']
            plt.scatter(pca_one.values, pca_two.values)

        elif dim_red == 'tsne':
            tsne_one = self.df_tsne['tsne_one']
            tsne_two = self.df_tsne['tsne_two']
            plt.scatter(tsne_one.values, tsne_two.values)
        plt.show()


if __name__ == '__main__':
    # process = Process_Data('data_dump-20180220.txt')
    # process._load_data()
    # process._format_data()
    # process._pca()
    # process._plot_lower_dims(dim_red='pca')
    p2 = Process_Data('data_dump-20180220.txt')
    p2._load_data()
    # p2._format_data()
    # p2._tsne()
    # p2._plot_lower_dims(dim_red='tsne')
