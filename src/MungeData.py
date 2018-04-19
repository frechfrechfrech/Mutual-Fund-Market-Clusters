# Prep the data
import pandas as pd

class MungeData(object):

    def __init__(self):
        self

    def fit(self):
        self

    def transform(self, df, row_headers, column_headers, agg_func='sum',filepath_to_save=None):
        '''
        Pivot, clean up missing values and negatives, turn features into
        proportions of the column, add totals column

        Inputs:

        df: dataframe to transform
        row_headers: row headers for the pivot
        column_headers: features to use in the pivot
        agg_func: (optional) use any of accepted pivot_table agg functions
        filepath_to_save: (optional) filepath to save a pkl of the transformed dataframe

        Outputs:

        df_pivot_pct_all: pivoted version of the dataframe where
            - numeric features are represented as '%' of that feature in the row
            - Total column added to show total $$ of the office (sans negatives)
        '''
        #set zip as str
        df['ZIP_CODE']=df['ZIP_CODE'].astype(str)
        # fill NA with 0
        df = df.fillna(0)
        #drop redemptions for now
        df = df.drop('GLOBAL_REDEMPTIONS', axis=1)

        ## pivot to get features into columns
        df_pivot = pd.pivot_table(df, index=row_headers,columns = column_headers, aggfunc = agg_func, fill_value=0)
        df_pivot = pd.DataFrame(df_pivot.to_records()) # reset index
        df_pivot = df_pivot[df_pivot.sum(axis=1)>1000000] # drop offices with less than $1M in sales
        num = df_pivot._get_numeric_data() # change negative values to zero
        num_indices = num.columns
        num[num < 0] = 0
        # add totals column
        df_pivot['Total'] = num.sum(axis=1)

        # df_pivot %
        df_pivot_pct_all = df_pivot.copy()
        for i in num_indices:
            df_pivot_pct_all.loc[:,i]= df_pivot_pct_all.loc[:,i]/df_pivot_pct_all['Total']
        # print(df_pivot_pct_all.iloc[:, num_indices].sum(axis=1).mean())
        # print(type(df_pivot_pct_all.iloc[:, num_indices].sum(axis=1).mean()))
        # Test that division worked, this should be True
        if df_pivot_pct_all.iloc[:,:-1].sum(axis=1).mean() == 1:
            print('Division worked real real good')
        else:
            print('IT FAILED!')

        if filepath_to_save != None:
            df_pivot_pct_all.to_pickle(filepath_to_save)

        return df_pivot_pct_all

# 
# if __name__ == '__main__':
#     filepath = './data/ALL_CLIENTS_Q417only_cat_bd_zip_redemptions_20180417.txt'
#     #import data
#     df_inc_red = pd.read_csv(filepath, sep='\t')
#     munge_bd_zip_broad = MungeData()
#     row_headers = ['BROKER_NAME', 'FDS_BROKER_ID', 'ZIP_CODE','STREET_ADDRESS', 'STATE']
#     column_headers = ['BROAD_FUND_CATEGORY']
#     df_bd_zip_broad = munge_bd_zip_broad.transform(df_inc_red, row_headers, column_headers)
#
#     #
#     munge_bd_broad = MungeData()
#     row_headers = ['BROKER_NAME', 'FDS_BROKER_ID']
#     column_headers = ['BROAD_FUND_CATEGORY']
#     df_bd_broad = munge_bd_broad.transform(df_inc_red, row_headers, column_headers)
