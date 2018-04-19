# Put the FUN in Mutual Funds

# Research Questions
  
  ### 1. Can we segment broker dealer offices into groups with similar buying behavior?
  Motivation: I, asset manager x, can better arm my salesforce to pursue opportunities if we have targeted strategies for each segment of the market.
    
  ### 2. Can we group open-end mutual funds based on sales patterns in the last 2 years?
  Motivation: If I, asset manager x, can find funds similar to my own which are not currently captured by Morningstra Category relationships, I can target those funds for competition.


2. Data: Open End Mutual Fund Sales aka What was sold, when, and where?
    - When: Monthly aggregates January 2016 - January 2017
    - Where: Broker Dealer Locations
    - What: CUSIP, Fund ID, Morningstar Category

----

# Fund Mapping
- Features are time series so they are correlated


![alt_text](https://media.giphy.com/media/zcVOyJBHYZvX2/giphy.gif "ugh")

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
