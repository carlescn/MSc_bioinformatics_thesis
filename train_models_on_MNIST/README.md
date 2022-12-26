# Summary of the metrics on clustering quality

| Repr. method      | Clustering method | Acc.      | ARI       | AMI       | Sil.      |
|:-----------------:|:-----------------:|----------:|----------:|----------:|----------:|
| None (raw data)   | K-means           |  0.59     |  0.41     |  0.53     |  0.06     |
| None (raw data)   | GMM               |  0.44     |  0.22     |  0.34     |  0.02     |
| PCA               | K-Means           |  0.59     |  0.41     |  0.53     |  0.09     |
| PCA               | GMM               |  0.47     |  0.23     |  0.43     |  0.02     |
| **AE embeddings** | K-means           |  **0.83** |  0.69     |  0.73     |  0.19     |
| AE embeddings     | GMM               |  0.77     |  0.57     |  0.68     |  0.14     |
| **DEC**           | K-means*          |  **0.83** |  0.69     |  0.74     |  **0.93** |
| DEC               | GMM*              |  0.76     |  0.56     |  0.69     |  **0.93** |
| VAE embeddings    | K-means           |  0.59     |  0.41     |  0.54     |  0.16     |
| VAE embeddings    | GMM               |  0.48     |  0.25     |  0.39     | -0.01     |
| Clustering VAE    | K-means*          |  0.57     |  0.37     |  0.49     |  0.20     |
| Clustering VAE    | GMM*              |  0.59     |  0.36     |  0.49     |  0.12     |
| VAE-DEC           | K-Means*          |  0.61     |  0.42     |  0.54     |  **0.90** |
| VAE-DEC           | GMM*              |  0.57     |  0.31     |  0.47     |  **0.87** |
| **VaDE**          | GMM*              |  **0.94** |  **0.88** |  **0.88** |  0.23     |

\* Clustering method used to initialize the model weights when training.
