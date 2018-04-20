# Put the FUN in Mutual Funds

# Research Questions
  
  ### 1. Can we segment broker dealer offices into groups with similar buying behavior?
  **Motivation**: I, asset manager x, can better arm my salesforce to pursue opportunities if we have targeted strategies for each segment of the market.
    
  ### 2. Can we group open-end mutual funds based on sales patterns in the last 2 years?
 **Motivation**: If I, asset manager x, can find funds similar to my own which are not currently captured by Morningstra Category relationships, I can target those funds for competition.


# Data: Open End Mutual Fund Sales aka What was sold, when, and where?
  - **When**: Monthly aggregates January 2016 - January 2017
  - **Where**: Broker Dealer Locations
  - **What**: CUSIP, Fund ID, Morningstar Category

----

# Fund Clustering

### Time Series
- Features are time series so they are correlated
(insert heatmap of correlation)
- What we're really interested in is: how closely do the sales movements of these funds align? So this is a great problem for the distance metric dynamic time warp.

### Dynamic Time Warp

![](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/images/time_warp.jpg)

Dynamic time warp is an algorithm to measure similarity between two temporal sequences, which may vary in speed.

I used an implementation written by Pierre Rouanet: https://github.com/pierre-rouanet/dtw

Examples:

    Dynamic Time Warp for sin vs sin: 0.0
    Dynamic Time Warp for sin vs cos: 0.04
    Dynamic Time Warp for sin vs sin*2: 0.28
    Dynamic Time Warp for sin vs sin+2: 1.0



### Hierarchical Clustering

Well-suited to situations where want to understand the relationships within clusters. For example, within this cluster, which funds are the most similar? Are there any funds that are total snowflakes?

![alt_text](link_to_dendrogram)


![alt_text](https://media.giphy.com/media/zcVOyJBHYZvX2/giphy.gif "ugh")

---

# Broker Dealer Clustering

### Broker Dealer Office Clustering

## Ameriprise
![](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/images/amp_pca_kmeans.png)

**Cluster Centroids**

Allocation | Alternative	| Commodities	| Convertibles |	Equity  | Fixed Income	| Tax Preferred 
--- | --- | --- | --- | --- | --- | --- 
7%	|3%	|0%	|0%	|28%	|**56%**	|6%
8%	|3%	|0%	|0%	|**56%**	|27%|	6%
29%	|6%	|0%	|0%	|27%	|25%	|12%



Steps:
1. Process data + feature engineer to isolate signal that we care about:
    - feature engineering
        - mitigate time series element
        - experiment with ways to deal with massive differences in scale between fund sales
    - PCA/t-SNE down into lower dimensional space for plotting and review
2. Use M* categories as clusters
3. Research clustering methods to determine most appropriate for this problem


Results (so far):





Broker Dealer Locaiton Clustering broad category proportions of most recent quarter sales:




# Resources
- dtw module by pierre-rouanet https://github.com/pierre-rouanet/dtw
    ```python -m pip install dtw```
