# Mutual-Fund-Market-Clusters

1. Research Questions
    Our clients often ask the following two questions:
    - How can I segment my clients into groups so that my wholesalers can tailor their sales strategy for each client?
    - Can you help me figure out which mutual funds I can compete with outside of the Morningstar Category that my fund falls into?
2. Data
a. Specifically, I will have sales data and redemptions data for broker dealer, zip, and cusip
combinations going back quarterly to 2014.
3. Approach
a. I intend to address both questions with clustering. My features will be quarterly sales for
the rolling 2 years, by bd+zip for the first question and by fund for the second. The
features will be correlated because this is time series data. I will try using PCA to project
onto lower-dimensions. If that doesn’t prove successful, I will attempt go down the road
of time series analysis.
4. MVP
a. My mvp for the client segmentation will be to segment Ameriprise bd+zip combos into
groups with similar buying habits. We know that there are at least two distinct groups
within Ameriprise offices (private wealth offices which manage investments for high net
worth individuals and regular offices for regular people), my goal would be to either
further segment the offices or show that there is no futher meaningful clustering.
i.
Bonus: currently, many salesforces are broken up into groups, each covering
one channel of broker dealers (wire/independent/regional/bank), can I zoom
out from Ameripise to create clusters within all BD+zip combos.
b. My mvp for the fund grouping question will be to create alternate groups of funds
within the Equity broad morningstar category based on the rolling 2 years sales data. I
will compare these to current Morningstar categorization, which are based on
investment strategy, to find the funds that wouldn’t have been captured by the
traditional M* categorization.
i.
Bonus: generalize the model to all broad morningstar categories.
ii.
Bonus: Make a tool that our team could easily use to build these alternate
comparison groups.
iii.
Bonus: get and incorporate returns data.
iv.
Bonus: feature engineer a target of “eligible for cross-selling” based on products
that have historically been bought together.
5. Extra: I would really like to incorporate neural nets into my project but I have not yet been
struck with any inspiration as to how to do this. If you have any ideas, please let me know.
