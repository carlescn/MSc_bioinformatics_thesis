import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

def _draw_plot(ax, points, labels, centroids=None, 
               title=None, legend_title=None, legend_labels=None, 
               cmap="tab10", alpha=0.7):    
    """
    Should by called by draw_embeddings() or draw_clusters_assignments().
    Draw a 2D scatterplot of the first two dimensions of the argument points.
    
    Parameters:
    -----------
    ax : Axes object
        Axes on which to draw the plot
    points : array_like(float, shape=(N, D))*
        Embeddings (points on the latent space) to plot.
    labels : array_like(int, shape=(N, 1))*
        Indeces to color and label the plotted points.
    centroids : array_like(float, shape=(K, D))*, optional(default=None)
        Centroids (points on the latent space) to plot.
    title : str, optional(default=None)
        Text to display as the plot title.
    legend_title : str, optional(default=None)
        Text to display as the legend title.
    legend_labels : list(str, len=C)*, optional(default=None)
        Text to display as the legend labels.
    cmap : str, optional(default="tab10")
        Colormap to use
    alpha : float (optional, default=0.7)
        Sets the transparency of the plotted points.

    *(where D >= 2: number of dimmentions of the latent space.
            N: number of points
            K: number of clusters
            C = len(unique(labels)): number of categories in legend)
    """
    
    scatter = ax.scatter(points[:,0], points[:,1], label=labels, c=labels, cmap=cmap, alpha=alpha, linewidths=0)
    ax.title.set_text(title)
    if centroids is not None:
        ax.scatter(centroids[:,0], centroids[:,1], c="black", marker="x")
    if legend_labels is None:
        leg = ax.legend(*scatter.legend_elements(), title=legend_title)
    else:
        leg = ax.legend((scatter.legend_elements()[0]), legend_labels, title=legend_title)
    # Make points in legend opaque
    for lh in leg.legendHandles: 
        # lh._legmarker.set_alpha(1) # old version of matplotlib
        lh.set_alpha(1)
    
    

def draw_embeddings(z, labels, centroids=None, legend_title=None, alpha=0.7, figsize=(5,5)):
    """
    Draw a scatterplot of the first 2 dimensions of the embeddings on the latent space.
    Optionally, superimpose the cluster centroids on the plot.
    
    Parameters:
    -----------
    z : array_like(float, shape=(N, D))*
        Embeddings (points on the latent space) to plot.
    labels : array_like(int, shape=(N, 1))*
        Indeces to color and label the plotted points.
    centroids : array_like(float, shape=(K, D))*, optional(default=None)
        Centroids (points on the latent space) to plot.
    legend_title : str, optional(default=None)
        Text to display as the legend title.
    alpha : float, optional(default=0.7)
        Sets the transparency of the plotted points.
    figsize : tuple(float, len=2), optional(default=(5,5))
        Size of the figure to display.

    *(where D >= 2: number of dimmentions of the points space.
            N: number of points
            K: number of clusters)
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    _draw_plot(ax, z, labels, centroids=centroids, legend_title=legend_title, alpha=alpha)
    plt.show()
    
    
def draw_matched_labels(z, true_labels, cluster_labels, centroids=None, alpha=0.7, figsize=(12,4)):
    """
    Draw three side by side scatterplots of the first 2 dimensions of the embeddings on the latent space.
    Left plot: cluster assignment.
    Middle plot: true labels.
    Right plot: matches between both labels.
    Optionally, superimpose the centroids of the found clusters on the plot.
        
    Parameters:
    -----------
    z : array_like(float, shape=(N, D))*
        Embeddings (points on the latent space) to plot.
    true_labels : array_like(int, shape=(N, 1))*
        True labels for the embeddings.
    cluster_labels : array_like(int, shape=(N, 1))*
        Cluster assignments for the embeddings.
    centroids : array_like(float, shape=(K, D))*, optional(default=None)
        Centroids (points on the latent space) to plot.
    alpha : float, optional(default=0.7)
        Sets the transparency of the plotted points.
    figsize : tuple(float, len=2), optional(default=(12, 4))
        Size of the figure to display.

    *(where D >= 2: number of dimmentions of the latent space.
            N: number of points
            K: number of clusters)
    """
    
    confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, cluster_labels)
    clusters = confusion_matrix.argmax(0)[cluster_labels]
    matched_labels = [x for x in (clusters == true_labels)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _draw_plot(axes[0], z, cluster_labels, centroids=centroids, alpha=alpha, 
               title="Cluster labels")
    _draw_plot(axes[1], z, true_labels, centroids=centroids, alpha=alpha, 
               title="True labels")
    _draw_plot(axes[2], z, matched_labels, centroids=centroids, alpha=alpha, cmap="Set1",
               title="Matched labels", legend_labels=("Mismatch", "Match"))
    plt.tight_layout()

    
def draw_multiple_labels(z, labels_dict, centroids=None, alpha=0.7, subplot_size=(4, 4), max_cols=5):
    """
    Draws the same scatterplot of the first 2 dimensions of the embeddings on the latent space,
    each one with a different labels array.
    The number of rows and columns of the subplots is determined by the number of plots to be drawn
    and the max_cols argument.
    Optionally, superimpose the centroids of the found clusters on the plot.
        
    Parameters:
    -----------
    z : array_like(float, shape=(N, D))*
        Embeddings (points on the latent space) to plot.
    labels_dict : dictionary (len=L)*
        keys: str
            The title for each plot.
        values: array_like(int, shape=(N, 1))*
            The labels for each plot.
    centroids : array_like(float, shape=(K, D))*, optional(default=None)
        Centroids (points on the latent space) to plot.
    alpha : float, optional(default=0.7)
        Sets the transparency of the plotted points.
    subplot_size: tuple(float, len=2), optional(default=(4,4)
        Size of each subplot.
    max_cols : int, optional(default=5)
        Maximum number of columns for the subplots.

    *(where D >= 2: number of dimmentions of the latent space.
            N: number of points
            K: number of clusters
            L: number of arrays of labels)
    """
    num_plots = len(labels_dict)
    num_cols = min(num_plots, max_cols)
    num_rows = num_plots // max_cols + 1
    figsize = (subplot_size[0] * num_cols, subplot_size[1] * num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i, (title, labels) in enumerate(labels_dict.items()):
        _draw_plot(axes[i], z, labels, centroids=centroids, alpha=alpha, title=title)
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
    assert len(encoder.outputs) <= 2
    
    dataset = dataset[0:n]
    res = 28
       
    if old_figure is not None:
        assert old_figure.shape[0] == n*res
        figure = np.zeros((old_figure.shape[0], old_figure.shape[1]+res))
        figure[0:old_figure.shape[0], 0:old_figure.shape[1]] = old_figure
    else:
        figure = np.zeros((n*res, 2*res))
        figure[0:n*res, 0:res] = dataset.reshape(n*res, res)
    
    if len(encoder.outputs) == 1:
        z = encoder.predict(dataset, verbose=0)
    else:
        z, _ = encoder.predict(dataset, verbose=0)
        
    reconstructed = decoder.predict(z, verbose=0).reshape(n*res, res)
    figure[0:n*res, figure.shape[1]-res:figure.shape[1]] = reconstructed[0:n*res, 0:res]
    
    plt.figure()
    x_ticks = np.arange(res/2, figure.shape[1], res)
    assert len(labels) == len(x_ticks)
    plt.xticks(x_ticks, labels, rotation=90)
    plt.yticks([])
    plt.imshow(figure, cmap="Greys_r")
    
    return figure