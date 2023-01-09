# ExposomeChallenge

In this part of the repository,
I saved all the I code wrote to evaluate the deep clustering models
on the [ExposomeChallenge](https://www.isglobal.org/-/exposome-data-analysis-challenge) data set.

Here I explain briefly the process I followed.
At the end of the document I show a summary of the results.


## Data exploration

First, I made a brief exploration of the data sets,
provided as a RData file,
and saved the data as CSV files for later use with Python.
On the [ExposomeChallengeData](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/ExposomeChallenge/ExposomeChallengeData) folder
are stored the original RData and my R code as both Rmd and HTML files.

All the data sets were saved on the [ExposomeChallengeData/datasets](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/ExposomeChallenge/ExposomeChallengeData/datasets) folder.

Then, I explored the phenotype and covariates data sets
and selected a few variables
that I found would be interesting to compare against
the cluster assignments.
Some of the variables where continuous,
so I separated them on equally sized bins using quantiles.
All the categorical variables where codified with numbers starting from 0.
[[code in data_exploration.ipynb]](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/ExposomeChallenge/data_exploration.ipynb)



## Clustering on the metabolomics data

I joined the serum and urine metabolomics data
into one data set
and then applied a min-max normalization.
Then I compared three classic clustering methods:
K-means, GMM, and agglomerative, 
first on the raw data
and next on the PCA representation
(selecting the number of PCs that explain 80% of the variance).
Each method was evaluated on four metrics:
accuracy (Acc), adjusted Rand Index (ARI), Adjusted Mutual Information (AMI), silhouette (Sil).
This should serve as a base metric for the deep clustering methods.
None of the methods achieved clustering quality.
[[classic.ipynb]](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/ExposomeChallenge/cluster_metabolome_uncorrected/classic.ipynb)

Next,
I evaluated two deep clustering methods
(DEC, VaDE)
on the same data
[[deepclust_mpl.ipynb]](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/ExposomeChallenge/cluster_metabolome_uncorrected/deepclust_mpl.ipynb).
The results did not improve over the classic methods,
so I tried a couple of things:
- To compensate for the relatively small size of the data set,
  apply a technique of data augmentation,
  then reevaluate both deep clustering methods.
  [same notebook]
- For the same reason,
  try lowering the number of trainable parameters of the models
  by replacing the original MPL intermediate layers
  with convolutional layers (1D and 2D).
  [[deepclust_conv.ipynb]](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/ExposomeChallenge/cluster_metabolome_uncorrected/deepclust_conv.ipynb)
- Try raising the signal to noise ratio of the data
  by selecting a subset of the most variable features,
  then reevaluating both the classical methods
  [[classic_feature_sel.ipynb]](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/ExposomeChallenge/cluster_metabolome_uncorrected/classic_feature_sel.ipynb)
  and the deep clustering methods
  [[deepclust_mpl_feature_sel.ipynb]](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/ExposomeChallenge/cluster_metabolome_uncorrected/deepclust_mpl_feature_sel.ipynb).
  
None of the methods achieved an improvement over the base metrics.

All the code can be found on the corresponding notebooks 
on the [cluster_metabolome_uncorrected](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/ExposomeChallenge/cluster_metabolome_uncorrected) folder.


## Clustering on the exposome data

The same clustering methods and deep clustering models
(only the MPL based ones)
where applied also to the exposome data set.
Here,
when the number of clusters was set to 6
all the methods overlapped almost perfectly
with the categorical variable *cohort*
(and so achieving a high clustering quality).

The code can be found on the corresponding notebooks 
on the [cluster_exposome_uncorrected](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/ExposomeChallenge/cluster_exposome_uncorrected) folder.


## Clustering on the batch effect corrected data

The results on the exposome data
suggested a strong batch effect on the data,
so I tried correcting for it by
standardizing each variable within each cohort class,
before min-max normalizing.

I applied this correction on both data sets
and reevaluated the same clustering methods.
Again, none of them achieved good clustering quality.

The code can be found on the corresponding notebooks 
on the [cluster_metabolome_corrected](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/ExposomeChallenge/cluster_metabolome_corrected)
and [cluster_exposome_corrected](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/ExposomeChallenge/cluster_exposome_corrected)
folders.


# Discussion

Here I show a summary of the results (mean for each combination of methods).
The full results for each method can be found at the end of its corresponding notebook.
Next, I show the top results for each metric.

It is evident that for the external validation metrics
(Accuracy, Adjusted Rand Index, Adjusted Mutual Information),
the best clustering quality is achieved with number of clusters of 6,
where all the methods almost perfectly matched the cohort classes.
This is the reason I suspected the presence of the batch effect.

Also, the asthma and native categorical variables
show a high degree of accuracy,
but this is only because on both variables
there is one big group that dominates over all the others.
The ARI and AMI correct for this,
and they stay low also on this variables.

For the internal validation metric (Silhouette),
the DEC model consistently achieves the best results,
specially with a small number of clusters.
But the found clusters do not seem to have a biological interpretation.


## Results summary

| Dataset                | Method                             | FL method   | CL method   |   Acc. |   ARI. |   AMI. |   Sil. |
|:-----------------------|:-----------------------------------|:------------|:------------|-------:|-------:|-------:|-------:|
| Exposome               | Classic                            | PCA         | Agglo.      |   0.6  |   0.17 |   0.18 |   0.13 |
| Exposome               | Classic                            | PCA         | GMM         |   0.59 |   0.16 |   0.18 |   0.12 |
| Exposome               | Classic                            | PCA         | K-Means     |   0.59 |   0.16 |   0.18 |   0.13 |
| Exposome               | Classic                            | Raw data    | Agglo.      |   0.6  |   0.17 |   0.18 |   0.1  |
| Exposome               | Classic                            | Raw data    | GMM         |   0.59 |   0.17 |   0.19 |   0.09 |
| Exposome               | Classic                            | Raw data    | K-Means     |   0.59 |   0.16 |   0.18 |   0.1  |
| Exposome               | Deep clustering                    | DEC         | K-Means     |   0.6  |   0.15 |   0.16 |   0.81 |
| Exposome               | Deep clustering                    | VaDE        | GMM         |   0.6  |   0.14 |   0.15 |   0.52 |
| Exposome (corrected)   | Classic                            | PCA         | Agglo.      |   0.46 |   0    |   0    |   0.02 |
| Exposome (corrected)   | Classic                            | PCA         | GMM         |   0.46 |   0    |   0    |   0.02 |
| Exposome (corrected)   | Classic                            | PCA         | K-Means     |   0.47 |   0    |   0    |   0.03 |
| Exposome (corrected)   | Classic                            | Raw data    | Agglo.      |   0.46 |   0    |   0    |   0.02 |
| Exposome (corrected)   | Classic                            | Raw data    | GMM         |   0.49 |   0.03 |   0.04 |  -0    |
| Exposome (corrected)   | Classic                            | Raw data    | K-Means     |   0.47 |   0    |   0    |   0.02 |
| Exposome (corrected)   | Deep clustering                    | DEC         | K-Means     |   0.49 |   0    |   0    |   0.63 |
| Metabolome             | Classic                            | PCA         | Agglo.      |   0.46 |   0    |   0    |   0.06 |
| Metabolome             | Classic                            | PCA         | GMM         |   0.46 |   0    |   0    |   0.01 |
| Metabolome             | Classic                            | PCA         | K-Means     |   0.46 |   0    |  -0    |   0.09 |
| Metabolome             | Classic                            | Raw data    | Agglo.      |   0.46 |   0    |   0    |   0.05 |
| Metabolome             | Classic                            | Raw data    | GMM         |   0.46 |   0    |   0    |   0.06 |
| Metabolome             | Classic                            | Raw data    | K-Means     |   0.46 |   0    |   0    |   0.07 |
| Metabolome             | Classic (fet. sel.)                | PCA         | Agglo.      |   0.46 |   0    |   0    |   0.07 |
| Metabolome             | Classic (fet. sel.)                | PCA         | GMM         |   0.46 |   0    |   0    |   0.04 |
| Metabolome             | Classic (fet. sel.)                | PCA         | K-Means     |   0.46 |   0    |   0    |   0.08 |
| Metabolome             | Classic (fet. sel.)                | Raw data    | Agglo.      |   0.46 |  -0    |  -0    |   0.06 |
| Metabolome             | Classic (fet. sel.)                | Raw data    | GMM         |   0.46 |   0    |   0    |   0.05 |
| Metabolome             | Classic (fet. sel.)                | Raw data    | K-Means     |   0.46 |   0    |   0    |   0.07 |
| Metabolome             | Deep clustering                    | DEC         | K-Means     |   0.48 |   0    |   0    |   0.53 |
| Metabolome             | Deep clustering                    | DEC + DA    | K-Means     |   0.48 |  -0    |  -0    |   0.82 |
| Metabolome             | Deep clustering                    | VaDE        | GMM         |   0.48 |   0    |   0    |   0.19 |
| Metabolome             | Deep clustering                    | VaDE + DA   | GMM         |   0.48 |   0    |   0    |   0.19 |
| Metabolome             | Deep clustering (convolutional AE) | DEC-conv1D  | K-Means     |   0.48 |  -0    |  -0    |   0.42 |
| Metabolome             | Deep clustering (convolutional AE) | DEC-conv2D  | K-Means     |   0.48 |  -0    |  -0    |   0.43 |
| Metabolome             | Deep clustering (convolutional AE) | VaDE-conv1D | GMM         |   0.48 |   0    |   0    |   0.19 |
| Metabolome             | Deep clustering (fet. sel.)        | DEC         | K-Means     |   0.48 |  -0    |  -0    |   0.51 |
| Metabolome             | Deep clustering (fet. sel.)        | DEC + DA    | K-Means     |   0.48 |   0    |   0    |   0.75 |
| Metabolome             | Deep clustering (fet. sel.)        | VaDE        | GMM         |   0.48 |   0    |   0    |   0.24 |
| Metabolome             | Deep clustering (fet. sel.)        | VaDE + DA   | GMM         |   0.48 |   0    |   0    |   0.22 |
| Metabolome (corrected) | Classic                            | PCA         | Agglo.      |   0.46 |   0    |   0    |   0.07 |
| Metabolome (corrected) | Classic                            | PCA         | GMM         |   0.46 |  -0    |  -0    |   0.02 |
| Metabolome (corrected) | Classic                            | PCA         | K-Means     |   0.46 |  -0    |  -0    |   0.09 |
| Metabolome (corrected) | Classic                            | Raw data    | Agglo.      |   0.46 |   0    |   0    |   0.06 |
| Metabolome (corrected) | Classic                            | Raw data    | GMM         |   0.46 |   0    |  -0    |   0.07 |
| Metabolome (corrected) | Classic                            | Raw data    | K-Means     |   0.46 |   0    |   0    |   0.07 |
| Metabolome (corrected) | Deep clustering                    | DEC         | K-Means     |   0.48 |  -0    |  -0    |   0.55 |
| Metabolome (corrected) | Deep clustering                    | VaDE        | GMM         |   0.48 |   0    |   0    |   0.22 |


## Top accuracy results

| Dataset                | Method              | FL method   | CL method   |   num clusters | variable   |   Acc. |   ARI. |   AMI. |   Sil. |
|:-----------------------|:--------------------|:------------|:------------|---------------:|:-----------|-------:|-------:|-------:|-------:|
| Exposome               | Classic             | Raw data    | Agglo.      |              6 | cohort     |   0.99 |   0.98 |   0.97 |   0.12 |
| Exposome               | Classic             | Raw data    | K-Means     |              6 | cohort     |   0.99 |   0.98 |   0.97 |   0.12 |
| Exposome               | Classic             | PCA         | K-Means     |              6 | cohort     |   0.99 |   0.98 |   0.97 |   0.16 |
| Exposome               | Classic             | PCA         | GMM         |              6 | cohort     |   0.99 |   0.97 |   0.96 |   0.16 |
| Exposome               | Classic             | PCA         | Agglo.      |              6 | cohort     |   0.99 |   0.97 |   0.96 |   0.16 |
| Exposome               | Deep clustering     | DEC         | K-Means     |              6 | cohort     |   0.99 |   0.97 |   0.96 |   0.73 |
| Exposome               | Deep clustering     | VaDE        | GMM         |              6 | cohort     |   0.98 |   0.95 |   0.94 |   0.6  |
| Exposome               | Classic             | Raw data    | GMM         |              6 | cohort     |   0.96 |   0.93 |   0.94 |   0.12 |
| Metabolome (corrected) | Classic             | PCA         | GMM         |              2 | asthma     |   0.89 |   0    |   0    |   0.03 |
| Metabolome             | Classic (fet. sel.) | PCA         | K-Means     |              2 | asthma     |   0.89 |   0    |   0    |   0.09 |




## Top ARI results

| Dataset   | Method          | FL method   | CL method   |   num clusters | variable   |   Acc. |   ARI. |   AMI. |   Sil. |
|:----------|:----------------|:------------|:------------|---------------:|:-----------|-------:|-------:|-------:|-------:|
| Exposome  | Classic         | Raw data    | Agglo.      |              6 | cohort     |   0.99 |   0.98 |   0.97 |   0.12 |
| Exposome  | Classic         | PCA         | K-Means     |              6 | cohort     |   0.99 |   0.98 |   0.97 |   0.16 |
| Exposome  | Classic         | Raw data    | K-Means     |              6 | cohort     |   0.99 |   0.98 |   0.97 |   0.12 |
| Exposome  | Classic         | PCA         | GMM         |              6 | cohort     |   0.99 |   0.97 |   0.96 |   0.16 |
| Exposome  | Classic         | PCA         | Agglo.      |              6 | cohort     |   0.99 |   0.97 |   0.96 |   0.16 |
| Exposome  | Deep clustering | DEC         | K-Means     |              6 | cohort     |   0.99 |   0.97 |   0.96 |   0.73 |
| Exposome  | Deep clustering | VaDE        | GMM         |              6 | cohort     |   0.98 |   0.95 |   0.94 |   0.6  |
| Exposome  | Classic         | Raw data    | GMM         |              6 | cohort     |   0.96 |   0.93 |   0.94 |   0.12 |
| Exposome  | Classic         | Raw data    | GMM         |              7 | age        |   0.59 |   0.48 |   0.59 |   0.1  |
| Exposome  | Classic         | Raw data    | Agglo.      |              7 | age        |   0.59 |   0.48 |   0.58 |   0.1  |



## Top AMI results

| Dataset   | Method          | FL method   | CL method   |   num clusters | variable   |   Acc. |   ARI. |   AMI. |   Sil. |
|:----------|:----------------|:------------|:------------|---------------:|:-----------|-------:|-------:|-------:|-------:|
| Exposome  | Classic         | Raw data    | Agglo.      |              6 | cohort     |   0.99 |   0.98 |   0.97 |   0.12 |
| Exposome  | Classic         | PCA         | K-Means     |              6 | cohort     |   0.99 |   0.98 |   0.97 |   0.16 |
| Exposome  | Classic         | Raw data    | K-Means     |              6 | cohort     |   0.99 |   0.98 |   0.97 |   0.12 |
| Exposome  | Classic         | PCA         | GMM         |              6 | cohort     |   0.99 |   0.97 |   0.96 |   0.16 |
| Exposome  | Deep clustering | DEC         | K-Means     |              6 | cohort     |   0.99 |   0.97 |   0.96 |   0.73 |
| Exposome  | Classic         | PCA         | Agglo.      |              6 | cohort     |   0.99 |   0.97 |   0.96 |   0.16 |
| Exposome  | Deep clustering | VaDE        | GMM         |              6 | cohort     |   0.98 |   0.95 |   0.94 |   0.6  |
| Exposome  | Classic         | Raw data    | GMM         |              6 | cohort     |   0.96 |   0.93 |   0.94 |   0.12 |
| Exposome  | Classic         | Raw data    | GMM         |              7 | age        |   0.59 |   0.48 |   0.59 |   0.1  |
| Exposome  | Classic         | Raw data    | Agglo.      |              7 | age        |   0.59 |   0.48 |   0.58 |   0.1  |


## Top silhouette results

| Dataset    | Method                      | FL method   | CL method   |   num clusters | variable   |   Acc. |   ARI. |   AMI. |   Sil. |
|:-----------|:----------------------------|:------------|:------------|---------------:|:-----------|-------:|-------:|-------:|-------:|
| Metabolome | Deep clustering             | DEC + DA    | K-Means     |              2 | asthma     |   0.89 |   0    |   0    |   0.95 |
| Metabolome | Deep clustering             | DEC + DA    | K-Means     |              2 | sex        |   0.53 |   0    |   0    |   0.95 |
| Metabolome | Deep clustering (fet. sel.) | DEC + DA    | K-Means     |              2 | sex        |   0.53 |   0    |   0    |   0.93 |
| Metabolome | Deep clustering (fet. sel.) | DEC + DA    | K-Means     |              2 | asthma     |   0.89 |   0    |   0    |   0.93 |
| Exposome   | Deep clustering             | DEC         | K-Means     |              3 | education  |   0.52 |   0.05 |   0.02 |   0.91 |
| Exposome   | Deep clustering             | DEC         | K-Means     |              3 | native     |   0.84 |   0    |   0    |   0.91 |
| Exposome   | Deep clustering             | DEC         | K-Means     |              3 | parity     |   0.45 |   0    |   0    |   0.91 |
| Metabolome | Deep clustering             | DEC + DA    | K-Means     |              3 | parity     |   0.45 |   0    |   0    |   0.86 |
| Metabolome | Deep clustering             | DEC + DA    | K-Means     |              3 | native     |   0.84 |   0    |   0    |   0.86 |
| Metabolome | Deep clustering             | DEC + DA    | K-Means     |              3 | education  |   0.51 |   0    |   0    |   0.86 |
