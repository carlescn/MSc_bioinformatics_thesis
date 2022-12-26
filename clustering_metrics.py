import seaborn as sns
import sklearn.metrics


def _draw_heatmap(confusion_matrix):
    """
    Draws a confusion matrix as a heatmap.
    
    Parameters:
    -----------
    confusion_matrix : confusion_matrix object
        The output of sklearn.metrics.confusion_matrix().
    """
    heatmap = sns.heatmap(confusion_matrix,  cmap='magma')
    heatmap.set(xlabel='clusters', ylabel='true labels')
    

def evaluate_clustering(data, true_labels, cluster_assignments, heatmap=True):
    """
    Evaluates the clustering performance given the cluster assignments.
    Also, calls _draw_heatmap() on the confusion matrix of the cluster assignments vs the true labels.
    
    Parameters:
    -----------
    data : array_like
        Data to evaluate the model.
    true_labels : list
        True labels of the data.
    cluster_assignments : array_like
        Cluster assignments of the data.
        
    Returns:
    --------
    return : dictionary
        A dictionary containing some performance metrics.
    """
    confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, cluster_assignments)
    matched_labels = confusion_matrix.argmax(0)[cluster_assignments]
    if heatmap:
        _draw_heatmap(confusion_matrix)
    return {"Acc" : sklearn.metrics.accuracy_score(true_labels, matched_labels),
            "ARI" : sklearn.metrics.adjusted_rand_score(true_labels, matched_labels),
            "AMI" : sklearn.metrics.adjusted_mutual_info_score(true_labels, matched_labels),
            "Sil" : sklearn.metrics.silhouette_score(data, cluster_assignments),
           }


def evaluate_model(data, true_labels, clustering_method, encode_method, heatmap=True):
    """
    Evaluates the model clustering performance:
    Computes the cluster assignment and embeddings and calls evaluate_clustering().
    
    Parameters:
    -----------
    data : array_like
        Data to evaluate the model.
    true_labels : list
        True labels of the data.
    clustering_method : method
        Model method (or function) which returns the cluster assignments.
    encode_method : method
        Model method (or function) which returns the embeddings on the latent space.
        
    Returns:
    --------
    return : dictionary
        A dictionary containing some performance metrics.
    """
    embeddings = encode_method(data)
    cluster_assignments = clustering_method(data)
    return evaluate_clustering(embeddings, true_labels, cluster_assignments, heatmap=heatmap)