# Mutual-Fund-Market-Clusters

1. Research Questions

    Our clients often ask the following two questions:
    - How can I segment my clients into groups so that my wholesalers can tailor their sales strategy for each client?
        Cluster on Broker Dealer
    - Can you help me figure out which mutual funds I can compete with outside of the Morningstar Category that my fund falls into?
![](http://i.imgur.com/OUkLi.gif)



2. Data
    - sales data for broker dealer, zip, and cusip combinations going back monthly 2 years.
    
    
![](http://media.giphy.com/media/l1J9R1Q7LJGSZOxFe/giphy.gif)


3. Approach
    - I intend to address both questions with clustering. My features will be quarterly sales for the rolling 2 years, by bd+zip for the first question and by fund for the second. The features will be correlated because this is time series data. I will try using PCA/tsne to project onto lower-dimensions. If that doesn’t prove successful, I will attempt go down the road
of time series analysis.
4. MVP
    - Client segmentation will be to segment Ameriprise bd+zip combos into groups with similar buying habits.
        - We know that there are at least two distinct groups within Ameriprise offices (private wealth offices which manage investments for high net worth individuals and regular offices for regular people), my goal would be to either further segment the offices or show that there is no futher meaningful clustering.
    - Fund grouping: create alternate groups of funds within the Equity broad morningstar category based on the rolling 2 years sales data. I will compare these to current Morningstar categorization, which are based on investment strategy, to find the funds that wouldn’t have been captured by the traditional M* categorization.


Steps:
1. Process data + feature engineer to isolate signal that we care about:
    - feature engineering
        - mitigate time series element
        - experiment with ways to deal with massive differences in scale between fund sales
    - PCA/t-SNE down into lower dimensional space for plotting and review
2. Use M* categories as clusters
3. Research clustering methods to determine most appropriate for this problem


Results (so far):

Fund Clustering by Rolling 24M sales:

![alt text](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/fundid_pca_scree.png "PCA Scree")

![alt text](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/dendrogram_allocation.png)

![alt text](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/dendrogram_taxpreferred.png)



Broker Dealer Locaiton Clustering broad category proportions of most recent quarter sales:

![alt text](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/amp_pca_kmeans.png)

![alt text](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/amp_tsne_kmeans.png)




# Resources
- dtw module https://github.com/pierre-rouanet/dtw
    ```python -m pip install dtw```
