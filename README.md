# Deep clustering models for metabolomics data

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3.0-blue.svg?logo=)](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/LICENSE)
[![made-with-latex](https://img.shields.io/badge/Made%20with-Latex-1f425f.svg?logo=latex)](https://www.latex-project.org/)
[![made-with-python 3.9](https://img.shields.io/badge/Made%20with-Python%203.9-1f425f.svg?logo=python)](https://www.python.org/)
[![tensorflow 2.9.1](https://img.shields.io/badge/Tensorflow-2.9.1-green.svg?logo=tensorflow)](https://github.com/tensorflow/tensorflow)
[![keras 2.9.0](https://img.shields.io/badge/Keras-2.9.0-green.svg?logo=keras)](https://github.com/keras-team/keras)

This repository contains
my implementation of some deep clustering models
written for my MSc thesis.
It also contains
the code written to train the models on multiple datasets
and a summary of the clustering performances achieved.

The objective of the thesis was
to implement a VAE based deep clustering model
and apply it on metabolomics data,
and then compare the resulting clustering with
more classical techniques.
I expected that the found clusters
would lend themselves to some biological interpretation.

The deep learning models implemented here are:

- AE (Autoencoder)
- DEC (Deep Embedded Clustering) [https://arxiv.org/abs/1312.6114v10]()
- VAE (Variational Autoencoder) [https://arxiv.org/abs/1511.06335v2]()
- VaDE (Variational Deep Embedding) [https://arxiv.org/abs/1611.05148v3]()

All the models where implemented
using [Keras](https://github.com/keras-team/keras)
over [Tensorflow](https://github.com/tensorflow/tensorflow),
using Python.
For the training process, I leveraged the virtual machines provided by
[Paperspace Gradient](https://www.paperspace.com/gradient) (over a paid subscription).


The models where first trained on the
[MNIST data set](https://doi.org/10.1109/MSP.2012.2211477)
and the clustering performance was measured
using multiple metrics
(Accuracy, Rand Index, Mutual Information, Silhouette).
The results where then compared to some classic clustering techniques
(K-Means and GMM over the raw data and over the PCA transformation).

Then, the best performing models (DEC, VaDE)
where evaluated on the [ExposomeChallenge2021 data set](https://arxiv.org/abs/2202.01680),
which contains metabolomics data.
The results where also compared with come classic clustering techniques.
*(work in progress)*

## File structure

- `models.py` Python module that contains
  the implementation of all the models.
- `draw_embeddings.py` Python module that contains
  some functions used to draw graphical representations
  of the embeddings and cluster assignments
  obtained by the models.
- `clustering_metrics.py` Python module that contains
  some functions used to evaluate
  the performance of the models,
  measuring some metrics of clustering quality.
- `MNIST` Folder that contains
  some Jupyter Notebooks
  where I train the different models on the MNIST dataset,
  then I evaluate their clustering performance
  and compare it to some classic clustering techniques
  (see `results.ipynb` for a summary).
- `ExposomeChallenge` Folder that contains
  some Jupyter Notebooks
  where I train the different models on the ExposomeChallenge2021 data set,
  then I evaluate their clustering performance
  and compare it to some classic clustering techniques.
- `_learning_keras` Folder that contains
  some Jupyter Notebooks
  where I trained myself on the use of Keras over Tensorflow
  for the implementation of artificial neural network models.
