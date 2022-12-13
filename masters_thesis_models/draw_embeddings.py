"""
TODO LIST:
- Write documentation on _draw_plot
- Write documentation on evaluate_model
- Update documentation on draw_embeddings
- Update documentation on draw_clusters_assignments
"""


import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture



def _draw_plot(ax, points, labels, centroids=None, legend_labels=None, legend_title="Labels", cmap="tab10", alpha=0.2):    
    """
    Missing documentation
    """
    
    scatter = ax.scatter(points[:,0], points[:,1], label=labels, c=labels, cmap=cmap, alpha=alpha, linewidths=0)

    if centroids is not None:
        plt.scatter(centroids[:,0], centroids[:,1], c="black", marker="x")
    
    if legend_labels is None:
        leg = ax.legend(*scatter.legend_elements(), title=legend_title)
    else:
        leg = ax.legend((scatter.legend_elements()[0]), legend_labels, title=legend_title)
    # Make points in legend opaque
    for lh in leg.legendHandles: 
        # lh._legmarker.set_alpha(1) # old version of matplotlib
        lh.set_alpha(1)
    
    

def draw_embeddings(z, labels, centroids=None, legend_title="Labels", figsize=(10,10), alpha=0.2):
    
    """
    Draw a scatterplot of the first 2 dimensions of the embeddings on the latent space.
    Optionally, superimpose the centroids of the found clusters on the plot.
    
    Args:
        z (float array of dim ((D >= 2), N)): embeddings (points on the latent space) to plot.
        labels (float list of len N): indeces to color and label the plotted points.
        centroids (float array of dim ((D >= 2), K)): centroid (points on the latent space) to plot.
        figsize (float tuple of len 2): sets the size of the draen figure.
        legend_title (str): text displayed as the title of the legend.
        alpha (float): sets the transparency of the plotted points.
        
        (where D: number of dimmentions of the latent space.
               N: number of embeddings)
    """
    
    plt.figure(figsize=figsize)
    _draw_plot(plt, z, labels, centroids=centroids, legend_title=legend_title, alpha=alpha)
    plt.show()
    
    
def draw_clusters_assignments(z, labels, clusters, centroids=None, figsize=(16,8), alpha=0.2):
    """
    Compare the original labels and the cluster assignments by drawing 
    two side by side scatterplots of the first 2 dimensions of the embeddings on the latent space.
    Color the points by the original labels (left plot) and the cluster assignments (right plot)
    Optionally, superimpose the centroids of the found clusters on the plot.
    
    Args:
        z (float array of dim ((D >= 2), N)): embeddings (points on the latent space) to plot.
        labels (float list of len N): indeces to color and label the plotted points.
        clusters (float list of len N): indeces to color and label the plotted points.
        centroids (float array of dim ((D >= 2), K)): centroid (points on the latent space) to plot.
        figsize (float tuple of len 2): sets the size of the draen figure.
        legend_title (str): text displayed as the title of the legend.
        alpha (float): sets the transparency of the plotted points.
        
        (where D: number of dimmentions of the latent space.
               N: number of embeddings.
               K: number of clusters.)
    """
    
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, clusters)
    clusters = confusion_matrix.argmax(0)[clusters]
    correct_labels = [x for x in (clusters == labels)]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _draw_plot(axes[0], z, clusters, centroids=centroids, legend_title="Clusters", cmap="tab10", alpha=alpha)
    _draw_plot(axes[1], z, correct_labels, centroids=centroids, legend_labels=["Incorrect", "Correct"], legend_title="Assignment", cmap="Set1", alpha=alpha)
    
    plt.tight_layout()
    
    
def compare_reconstructed_images_MNIST(dataset, encoder, decoder, labels, old_figure=None, n=5):
    """
    Take the first n images of the passed MNIST dataset, pass them through the AE encoder and decoder,
    and show the original and reconstructed images side by side.
    
    If an old_figure is passed, it adds a column to it with the new reconstruction. 
    
    Args:
        dataset (float array of dim (784, (N >= 5))): data set with the original images.
        encoder (model): model to generate the embeddings.
        decoder (model): model to reconstruct the original image from the embeddings.
        labels (str list of len C): text to display on the X axis.
        old_figure (figure): the output of an earlier call to this function. It will add a new column to the figure.
        n (int): number of images to be displayed.
        
        (where N: number of images in the dataset.
               C: number of columns (one column per compared model).)
    
    Returns:
        figure: copy of displayed image.
    """
    
    index = np.arange(0,n)
    res = 28
       
    if old_figure is not None:
        assert old_figure.shape[0] == n*res
        figure = np.zeros((old_figure.shape[0], old_figure.shape[1]+res))
        figure[0:old_figure.shape[0], 0:old_figure.shape[1]] = old_figure
    else:
        figure = np.zeros((n*res, 2*res))
        for i in index:
            figure[i*res:(i+1)*res, 0:res] = dataset[i].reshape(res, res)
    
    for i in index:
        z, _ = encoder.predict(dataset[[i]], verbose=0)
        reconstructed = decoder.predict(z, verbose=0).reshape(res, res)
        figure[i*res:(i+1)*res, figure.shape[1]-res:figure.shape[1]] = reconstructed

    plt.figure()
    # plt.axis("off")
    
    x_ticks = np.arange(res/2, figure.shape[1], res)
    assert len(labels) == len(x_ticks)
    plt.xticks(x_ticks, labels, rotation=90)
    plt.yticks([])
    
    plt.imshow(figure, cmap="Greys_r")
    
    return figure


def evaluate_model(model, x, true_labels, silhouette=False):
    """
    Missing documentation
    """
    cluster_assignments = model.classify(x)
    confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, cluster_assignments)
    cluster_matched_labels = confusion_matrix.argmax(0)[cluster_assignments]
    
    heatmap = sns.heatmap(confusion_matrix,  cmap='magma')
    heatmap.set(xlabel='clusters', ylabel='true labels')

    print(f"Acc: {sklearn.metrics.accuracy_score(true_labels, cluster_matched_labels):.4f}")
    print(f"ARI: {sklearn.metrics.adjusted_rand_score(true_labels, cluster_assignments):.4f}")
    print(f"AMI: {sklearn.metrics.adjusted_mutual_info_score(true_labels, cluster_assignments):.4f}")

    if silhouette:
        z, _ = model.encode(x)
        print(f"Sil: {sklearn.metrics.silhouette_score(z, cluster_assignments):.4f}")