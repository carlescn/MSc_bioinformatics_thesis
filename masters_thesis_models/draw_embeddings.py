import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics


def _draw_plot(ax, points, labels, centroids=None, legend_labels=None, legend_title="Labels", cmap="tab10", alpha=0.2):    
    """
    Should by called by draw_embeddings() or draw_clusters_assignments().
    Draw a 2D scatterplot of the first two dimensions of the argument points.
    
    Parameters:
    -----------
    ax : Axes object
        Axes on which to draw the plot
    points : array_like(float, shape=(N, D))*
        Embeddings (points on the latent space) to plot.
    labels : list(float, len=N)*
        Indeces to color and label the plotted points.
    centroids : array_like(float, shape=(K, D))*, optional(default=None)
        Centroids (points on the latent space) to plot.
    legend_labels : list(str, len=C)*, optional(default=None)
        Text to display as the legend labels.
    legend_title : str, optional(default="Labels")
        Text to display as the legend title.
    cmap : str, optional(default="tab10")
        Colormap to use
    alpha : float (optional, default=0.2)
        Sets the transparency of the plotted points.

    *(where D >= 2: number of dimmentions of the latent space.
            N: number of points
            K: number of clusters
            C = len(unique(labels)): number of categories in legend)
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
    
    

def draw_embeddings(z, labels, centroids=None, legend_title="Labels", alpha=0.2, figsize=(10,10)):
    """
    Draw a scatterplot of the first 2 dimensions of the embeddings on the latent space.
    Optionally, superimpose the cluster centroids on the plot.
    
    Parameters:
    -----------
    z : array_like(float, shape=(N, D))*
        Embeddings (points on the latent space) to plot.
    labels : list(float, len=N)*
        Indeces to color and label the plotted points.
    centroids : array_like(float, shape=(K, D))*, optional(default=None)
        Centroids (points on the latent space) to plot.
    legend_title : str, optional(default="Labels")
        Text to display as the legend title.
    alpha : float, optional(default=0.2)
        Sets the transparency of the plotted points.
    figsize : tuple(float, len=2), optional(default=(10,10))
        Size of the figure to display.

    *(where D >= 2: number of dimmentions of the points space.
            N: number of points
            K: number of clusters)
    """
    
    plt.figure(figsize=figsize)
    _draw_plot(plt, z, labels, centroids=centroids, legend_title=legend_title, alpha=alpha)
    plt.show()
    
    
def draw_clusters_assignments(z, true_labels, clusters, centroids=None, alpha=0.2, figsize=(16,8)):
    """
    Draw two side by side scatterplots of the first 2 dimensions of the embeddings on the latent space.
    Left plot: color the points depending on its cluster assignment.
    Right plot: color the points depending on if its cluster assignment matches de true label.
    Optionally, superimpose the centroids of the found clusters on the plot.
        
    Parameters:
    -----------
    z : array_like(float, shape=(N, D))*
        Embeddings (points on the latent space) to plot.
    true_labels : list(float, len=N)*
        True labels for the embeddings.
    clusters : list(float, len=N)*
        Cluster assignments for the embeddings.
    centroids : array_like(float, shape=(K, D))*, optional(default=None)
        Centroids (points on the latent space) to plot.
    alpha : float, optional(default=0.2)
        Sets the transparency of the plotted points.
    figsize : tuple(float, len=2), optional(default=(16, 8))
        Size of the figure to display.

    *(where D >= 2: number of dimmentions of the latent space.
            N: number of points
            K: number of clusters)
    """
    
    confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, clusters)
    clusters = confusion_matrix.argmax(0)[clusters]
    correct_labels = [x for x in (clusters == true_labels)]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _draw_plot(axes[0], z, clusters, centroids=centroids, legend_title="Clusters", cmap="tab10", alpha=alpha)
    _draw_plot(axes[1], z, correct_labels, centroids=centroids, legend_labels=["Incorrect", "Correct"], legend_title="Assignment", cmap="Set1", alpha=alpha)
    
    plt.tight_layout()
    
    
def compare_reconstructed_images_MNIST(dataset, encoder, decoder, labels, old_figure=None, n=5):
    """
    Take the first n images of the passed dataset, pass them through the AE encoder and decoder,
    and show the original and reconstructed images side by side.
    If an old_figure is passed, it adds a new column to it with the new reconstruction. 
    
    Parameters:
    -----------
    dataset : array_like(float, shape=(N, 784))*
        Dataset with the original (flattened) MNIST images.
    encoder : model
        Encoder model of the AE, to generate the embeddings.
    decoder : model
        Decoder model of the AE, to reconstruct the original image from the embeddings.
    labels : list(str, len=C)*
        Text to display on the X axis labels.
    old_figure : figure, optional(default=None)
        The output of an earlier call to this function. If not None, it will add a new column to the figure.
    n : int, optional(default=5)
        Number of images from the dataset to be displayed.
    
    Returns:
    --------
    figure : array_like(float, shape=(28*N, 28*C))*
        Copy of displayed image, to be passed as old_figure argument.
        
    
    *(where: N >= n: number of images in the dataset.
             C >= 2: number of columns, one per compared model.)
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