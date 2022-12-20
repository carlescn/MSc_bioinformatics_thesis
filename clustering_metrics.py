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
    

def evaluate_clustering(cluster_assignments, data, true_labels):
    """
    Evaluates the clustering performance given the cluster assignments.
    Also, calls _draw_heatmap() on the confusion matrix of the cluster assignments vs the true labels.
    
    Parameters:
    -----------
    cluster_assignments : array_like
        Cluster assignments of the data.
    data : array_like
        Data to evaluate the model.
    true_labels : list
        True labels of the data.
        
    Returns:
    --------
    return : dictionary
        A dictionary containing some performance metrics.
    """
    confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, cluster_assignments)
    _draw_heatmap(confusion_matrix)
    matched_labels = confusion_matrix.argmax(0)[cluster_assignments]
    return {"Acc" : sklearn.metrics.accuracy_score(true_labels, matched_labels),
            "ARI" : sklearn.metrics.adjusted_rand_score(true_labels, matched_labels),
            "AMI" : sklearn.metrics.adjusted_mutual_info_score(true_labels, matched_labels),
            "Sil" : sklearn.metrics.silhouette_score(data, cluster_assignments),
           }


def evaluate_model(clustering_method, encode_method, data, true_labels):
    """
    Evaluates the model clustering performance:
    Computes the cluster assignment and embeddings and calls evaluate_clustering().
    
    Parameters:
    -----------
    clustering_method : method
        Model method (or function) which returns the cluster assignments.
    encode_method : method
        Model method (or function) which returns the embeddings on the latent space.
    data : array_like
        Data to evaluate the model.
    true_labels : list
        True labels of the data.
        
    Returns:
    --------
    return : dictionary
        A dictionary containing some performance metrics.
    """
    embeddings = encode_method(data)
    cluster_assignments = clustering_method(data)
    return evaluate_clustering(cluster_assignments, embeddings, true_labels)