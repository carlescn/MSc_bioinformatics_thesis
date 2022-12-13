# import numpy as np
# import matplotlib.pyplot as plt

import seaborn as sns
import sklearn.metrics


def evaluate_clustering_performance(clustering_method, data, true_labels, silhouette=False, encode_method=None):
    """
    Evaluates the clustering performance of the model on multiple metrics and
    shows the confusion matrix of the clustering assignments vs the true labels as a heatmap.
    
    Parameters:
    -----------
    clustering_method : method
        Model method (or function) which returns the cluster assignments.
    data : array_like
        Data to evaluate the model.
    true_labels : list
        True labels of the data.
    silhouette : bool, optional(default=False)
        If True, compute the Silhouette metric (slow!).
    encode_method : method, optional(default=None)
        Model method (or function) which returns the embeddings on the latent space.
        Necessary for computing the Silhouette.
    """
    
    cluster_assignments = clustering_method(data)
    confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, cluster_assignments)
    cluster_matched_labels = confusion_matrix.argmax(0)[cluster_assignments]
    
    heatmap = sns.heatmap(confusion_matrix,  cmap='magma')
    heatmap.set(xlabel='clusters', ylabel='true labels')

    print(f"Acc: {sklearn.metrics.accuracy_score(true_labels, cluster_matched_labels):.4f}")
    print(f"ARI: {sklearn.metrics.adjusted_rand_score(true_labels, cluster_assignments):.4f}")
    print(f"AMI: {sklearn.metrics.adjusted_mutual_info_score(true_labels, cluster_assignments):.4f}")

    if silhouette and encode_method is not None:
        z = encode_method(data)
        print(f"Sil: {sklearn.metrics.silhouette_score(z, cluster_assignments):.4f}")