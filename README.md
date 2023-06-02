# Implementation of deep clustering models for metabolomics data.

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3.0-blue.svg?logo=)](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/LICENSE)
[![made-with-latex](https://img.shields.io/badge/Made%20with-Latex-1f425f.svg?logo=latex)](https://www.latex-project.org/)
[![made-with-python 3.9](https://img.shields.io/badge/Made%20with-Python%203.9-1f425f.svg?logo=python)](https://www.python.org/)
[![tensorflow 2.9.1](https://img.shields.io/badge/Tensorflow-2.9.1-darkgreen.svg?logo=tensorflow)](https://github.com/tensorflow/tensorflow)
[![keras 2.9.0](https://img.shields.io/badge/Keras-2.9.0-darkgreen.svg?logo=keras)](https://github.com/keras-team/keras)

This repository contains
my implementation of some deep clustering models 
I wrote for my MSc thesis.
It also contains the code written 
to train and evaluate the models on multiple datasets.
The original thesis report can be read 
[here](https://github.com/carlescn/MSc_bioinformatics_thesis/raw/main/thesis_report/CriadoNina_Carles_TFM.pdf),
but the document is in Catalan
(I plan on translating it into English,
but I have not set a deadline).

The original objective of the thesis was
to implement a VAE based deep clustering model
and apply it on metabolomics data,
then compare the results with
more established techniques.
I expected that the found clusters
would lend themselves to some biological interpretation.

The VAE based model did not perform well,
which prompted me to try other models,
also based on the autoencoder architecture.
The deep learning models I implemented are:

- AE (Autoencoder) [https://arxiv.org/abs/2201.03898](https://arxiv.org/abs/2201.03898)
- DEC (Deep Embedded Clustering) [https://arxiv.org/abs/1312.6114v10](https://arxiv.org/abs/1312.6114v10)
- VAE (Variational Autoencoder) [https://arxiv.org/abs/1511.06335v2](https://arxiv.org/abs/1511.06335v2)
- VaDE (Variational Deep Embedding) [https://arxiv.org/abs/1611.05148v3](https://arxiv.org/abs/1611.05148v3)

All the models where implemented
using [Keras](https://github.com/keras-team/keras)
with [Tensorflow](https://github.com/tensorflow/tensorflow),
using [Python](https://www.python.org/).
For the training process,
I leveraged the virtual machines provided by
[Paperspace Gradient](https://www.paperspace.com/gradient) (paid subscription).

## File structure

- [models.py](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/models.py):
  Python module that contains 
  my implementation of all the deep clustering models.
- [draw_embeddings.py](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/draw_embeddings.py)
  Python module that contains some functions
  to draw graphical representations of the embeddings 
  and cluster assignments.
- [clustering_metrics.py](https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/clustering_metrics.py):
  Python module that contains some functions
  to evaluate the performance of the models.
- [thesis_report](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/thesis_report)
  Folder that contains my full thesis report,
  both the PDF file and the latex source.
- [MNIST](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/MNIST)
  Folder that contains the Jupyter Notebooks I wrote 
  to train and evaluate the models on the MNIST data set.
  Also contaings the metrics and cluster assignments
  on CSV files.
- [ExposomeChallenge](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/ExposomeChallenge)
  Same as above, for the Exposome Data Challenge Event data set.
- [PrivateDataset](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/PrivateDataset)
  Same as above, for the DCH-NG data set.
- [_learning_keras](https://github.com/carlescn/MSc_bioinformatics_thesis/tree/main/_learning_keras)
  Folder that contains some Jupyter Notebooks I wrote 
  while training myself on the use of Keras and Tensorflow.
  
## Required software

The DNNs provided here are implemented using
Python and Keras over Tensorflow.

The implementation of the models
defined in the module models.py
requires the following python packages:

- python >= 3.9
- tensorflow >= 2.9.1
- keras >= 2.9.0
- numpy

To reproduce the provided notebooks,
you will also need:

- matplotlib
- numpy
- pandas
- scikit-learn
- scipy
- seaborn

## Abstract

I implemented several deep clustering models 
based on the Autoencoder architecture 
with the aim of evaluating their performance in metabolomics datasets. 
Using the MNIST dataset and two metabolomic datasets, 
I evaluated the performance of several variations of the VAE, DEC and VaDE architectures 
using internal and external validation metrics 
to measure clustering quality. 
I compared the results with more established methods 
such as K-means, GMM and agglomerative clustering.
I found found that the VAE architecture is not conducive to good clustering quality. 
The clusters obtained with the DEC, Vade and consolidated techniques 
show a high level of overlap with each other, 
but yield low performances according to the validation metrics. 
The DEC model excels over the rest in the internal validation metric, 
but is very sensitive to the initialization parameters. 
The VaDE model achieves similar results to the rest of the techniques, 
and has the added value of having generative capacity,
which could be used in artificial data augmentation techniques.
The multivariate distribution of the covariates 
(as well as that of the most variable metabolites) 
shows a differential distribution by the clusters obtained, 
although the results are not clear. 
This suggests a possible biological interpretation of the clusters, 
but it will be necessary to study it in more depth to draw conclusions.
