# Put the FUN in Mutual Funds

## Research Questions

  ### 1. Can we group open-end mutual funds based on sales patterns in the last 2 years?
 **Motivation**: If I, asset manager x, can find funds similar to my own which are not currently captured by Morningstar Category relationships, I can target those funds for competition.
  
  ### 2. Can we segment broker dealer (BD) offices into groups with similar buying behavior?
  **Motivation**: I, asset manager x, can better arm my salesforce to pursue opportunities if we have targeted strategies for each segment of the market.
    


## Data: Open End Mutual Fund Sales aka What was sold, when, and where?
  - **When**: Monthly aggregates January 2016 - December 2017
  - **Where**: Broker Dealer Locations
  - **What**: CUSIP, Fund ID, Morningstar Category

----

# Fund Clustering

### Time Series
- **Hypothesis: funds that are similar have similar sales patterns.**
- **Features**: *monthly* sales into each FundID in the rolling 24 months
- **Distance Metric**: We want to know how closely the sales movements of these funds align. Dynamic Time warp can help us measure this.

### Dynamic Time Warp

![](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/images/time_warp.jpg)

Dynamic time warp is an algorithm to measure similarity between two temporal sequences, which may vary in speed.

I used an implementation written by Pierre Rouanet: https://github.com/pierre-rouanet/dtw


    Dynamic Time Warp for sin vs sin: 0.0
    Dynamic Time Warp for sin vs cos: 0.04
    Dynamic Time Warp for sin vs sin*2: 0.28
    Dynamic Time Warp for sin vs sin+2: 1.0

![alt_text](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/images/dtw_illustrated.jpeg)


### Hierarchical Clustering

Well-suited to situations where want to understand the relationships within clusters. For example, within this cluster, which funds are the most similar? Are there any funds that are total snowflakes?

![alt_text](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/images/dendrogram_allocation_scaled.png)


**Interpretation:**
  - Most funds are pretty similar by this metric
  - There are some funds that are definite outliers - really far away from the rest of the pack. 


---

# Broker Dealer Clustering

- **Hypothesis: Broker Dealers/Broker Dealer Offices with similar buying behavior can constitute meaningful consumer segments. **
- **Features**: Proportion of sales into each broad category within the broker dealer or broker dealer location
  - Included scaled broker dealer size for the broker-dealer level analysis because size is a meaningful differentiator for sales opportunity.

### Broker Dealer Office Clustering by Broad Category

#### Ameriprise
![](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/images/amp_pca_kmeans.png)

**Ameriprise Cluster Centroids**

Allocation | Alternative	| Commodities	| Convertibles |	Equity  | Fixed Income	| Tax Preferred 
--- | --- | --- | --- | --- | --- | --- 
7%	|3%	|0%	|0%	|28%	|**56%**	|6%
8%	|3%	|0%	|0%	|**56%**	|27%|	6%
29%	|6%	|0%	|0%	|27%	|25%	|12%



### Broker Dealer Clustering by Broad Category

BD Clusters overlayed on PCA        |  BD Clusters overlayed on t-SNE
:-------------------------:|:-------------------------:
![](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/images/bd_broad_inc_size_pca_kmeans.png)  |  ![](https://github.com/frechfrechfrech/Mutual-Fund-Market-Clusters/blob/master/images/bd_broad_inc_size_tsne_kmeans.png)




**Broker Dealer Cluster Centroids**

| Allocation | Alternative | Commodities | Convertibles | Equity | Fixed Income | Tax Preferred | BD Size vs Largest | 
| --- | --- | --- | --- | --- | --- | --- |--- |
| **45%**        | 0%          | 0%          | 0%           | **36%**    | 13%          | 5%            | 0%    | 
| 8%         | 6%          | 1%          | 0%           | 19%    | 17%          | **48%**        | 0%                     | 
| 12%        | 0%          | 0%          | 0%           | **72%**    | 11%          | 4%            | 0%             | 
| 9%         | 1%          | 0%          | 0%           | 14%    | 72%          | 5%            | 0%                     | 
| 17%        | 2%          | 0%          | 0%        | **42%**   | **32%**      | 7%            | **1%**              | 
| 11%        | 2%          | 0%          | 0%           |**36%**   | **40%**        | 10%           | **68%**              | 



# Continued Exploration:

### FundID
- Scale the fundid data to make all time series the same magnitude.
- Add in redemptions and returns features.
- Feature engineer a "target" to represent which funds actually do sell together at the same location or unseat one another.

### Broker Dealer + Broker Dealer Offices
- Try category instead of broad category.
- Build deep neural net to predict sales into categories within each broker dealer office for the next month/quarter.


# Resources
- dtw module by pierre-rouanet https://github.com/pierre-rouanet/dtw
    ```python -m pip install dtw```
