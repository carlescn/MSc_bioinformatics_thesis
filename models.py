import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

import math
import numpy as np


##### Autoencoder #####

def get_encoder(input_dim, latent_dim, intermediate_dims=[]):
    encoder_inputs = keras.Input(shape=(input_dim), name="encoder_inputs")
    x = encoder_inputs
    for dim in intermediate_dims:
        x = layers.Dense(dim, activation="relu") (x)
    encoder_outputs = layers.Dense(latent_dim, activation="relu", name="z") (x)
    return models.Model(encoder_inputs, encoder_outputs, name="Encoder")
    
def get_decoder(output_dim, latent_dim, intermediate_dims=[]):
    decoder_inputs = keras.Input(shape=(latent_dim), name="decoder_inputs")
    x = decoder_inputs
    for dim in intermediate_dims[::-1]:
        x = layers.Dense(dim, activation="relu") (x)
    decoder_outputs = layers.Dense(output_dim, activation="sigmoid", name="recon_x") (x)
    return models.Model(decoder_inputs, decoder_outputs, name="Decoder")

def get_autoencoder_model(input_dim, latent_dim, intermediate_dims=[]):
    encoder = get_encoder(input_dim, latent_dim, intermediate_dims)
    decoder = get_decoder(input_dim, latent_dim, intermediate_dims)    
    return AutoEncoder(encoder, decoder)

class AutoEncoder(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x

    
##### DEC #####

def _compute_soft_assignment(z, centroids):
    """ student t-distribution, as same as used in t-SNE algorithm.
     Measure the similarity between embedded point z_i and centroid µ_j.
             q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
             q_ij can be interpreted as the probability of assigning sample i to cluster j.
             (i.e., a soft assignment)
    Arguments:
        z: embedded points, shape=(n_samples, n_features)
        centroids: centroids, shape=(n_clusters, n_features)
    Return:
        q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
    Source: https://github.com/rezacsedu/Deep-Learning-for-Clustering-in-Bioinformatics/blob/master/Notebooks/DEC_Gene_Clustering.ipynb
    """
    alpha = 1.0
    q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(z, axis=1) - centroids), axis=2) / alpha))
    q **= (alpha + 1.0) / 2.0
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
    return q

def clustering_loss_function(true_dist, pred_dist):
    return K.mean(tf.keras.losses.kld(true_dist, pred_dist))

def get_dec_model(encoder, n_clusters):
    return DEC(encoder, n_clusters)

def compute_p(q):
    """
    Compute the auxiliary distirbution P.
    Source: https://github.com/rezacsedu/Deep-Learning-for-Clustering-in-Bioinformatics/blob/master/Notebooks/DEC_Gene_Clustering.ipynb
        def target_distribution(q):
            weight = q ** 2 / q.sum(0)
            return (weight.T / weight.sum(1)).T
    """
    weight = q ** 2 / tf.reduce_sum(q, axis=0)
    p = tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))
    return p.numpy()


def compute_delta(c_new, c_last):
    """
    Compute the proportion of points that changed clusters.
    """
    delta = np.sum(c_new != c_last).astype(np.float32) / c_new.shape[0]
    return delta


class DEC(models.Model):
    def __init__(self, encoder, n_clusters, **kwargs):
        super(DEC, self).__init__(**kwargs)
        self.encoder = encoder
        
        latent_dim = encoder.outputs[0].shape[1]
        self.centroids = self.add_weight(name='centroids',
                                         shape=(n_clusters, latent_dim),
                                         trainable=True,
                                         initializer=tf.keras.initializers.random_normal)
        
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker,]
    
    def encode(self, x):
        return self.encoder(x)
    
    def soft_assignment(self, x):
        z = self.encode(x)
        return _compute_soft_assignment(z, self.centroids)

    def classify(self, x):
        q = self.soft_assignment(x)
        return q.numpy().argmax(1)
    
    def call(self, x):
        return self.classify(x)

    def train_step(self, data):
        x, p = data
        with tf.GradientTape() as tape:
            q = self.soft_assignment(x)
            loss = clustering_loss_function(q, p)   # Reverse KL (zero forcing) achieves tighter clusters
            # loss = clustering_loss_function(p, q) # Forward KL (zero avoiding)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}



##### VAE #####

def _reparameterize(mu, logvar):
    """Reparameterization trick."""
    std = K.exp(0.5 * logvar)
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + eps * std

def get_vae_encoder(input_dim, latent_dim, intermediate_dims):
    assert len(intermediate_dims) >= 1
    encoder = get_encoder(input_dim, intermediate_dims[-1], intermediate_dims[:-1])
    h = encoder(encoder.inputs)
    z_mu = layers.Dense(latent_dim, name="z_mu") (h)
    z_logvar = layers.Dense(latent_dim, name="z_logvar") (h)
    return models.Model(encoder.inputs, [z_mu, z_logvar], name="Encoder")

def get_vae_model(input_dim, latent_dim, intermediate_dims):
    vae_encoder = get_vae_encoder(input_dim, latent_dim, intermediate_dims)
    vae_decoder = get_decoder(input_dim, latent_dim, intermediate_dims)
    return VAE(vae_encoder, vae_decoder)

def get_clustering_vae_model(vae_model, n_clusters, clustering_loss_weight=1):
    return ClusteringVAE(vae_model.encoder, vae_model.decoder, n_clusters, clustering_loss_weight)


class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.regularization_loss_tracker = keras.metrics.Mean(name="reg_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="rec_loss")
        
    @property
    def metrics(self):
        return [self.total_loss_tracker,
               self.regularization_loss_tracker,
               self.reconstruction_loss_tracker]

    def encode_only_mu(self, x):
        mu, _ = self.encoder(x)
        return mu
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mu, z_logvar = self.encode(x)
        z = _reparameterize(z_mu, z_logvar)
        recon_x = self.decode(z)
        return recon_x, z_mu, z_logvar
    
    def regularization_loss(self, z_mu, z_logvar):
        reg_loss = -0.5 * K.sum(1 + z_logvar - K.square(z_mu) - K.exp(z_logvar), axis=-1)
        return K.mean(reg_loss)
    
    def reconstruction_loss(self, x, recon_x):
        return K.sum(keras.losses.binary_crossentropy(x, recon_x))

    def train_step(self, x):
        with tf.GradientTape() as tape:
            recon_x, z_mu, z_logvar = self(x)
            regularizatoin_loss = self.regularization_loss(z_mu, z_logvar)
            reconstruction_loss = self.reconstruction_loss(x, recon_x)
            total_loss = regularizatoin_loss + reconstruction_loss
            
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.regularization_loss_tracker.update_state(regularizatoin_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {"total_loss": self.total_loss_tracker.result(),
                "reg_loss": self.regularization_loss_tracker.result(),
                "rec_loss": self.reconstruction_loss_tracker.result()}


class ClusteringVAE(VAE):
    def __init__(self, encoder, decoder, n_clusters, clustering_loss_weight=1, **kwargs):
        super(ClusteringVAE, self).__init__(encoder, decoder, **kwargs)
        latent_dim = encoder.outputs[0].shape[1]
        self.centroids = self.add_weight(name='centroids',
                                         shape=(n_clusters, latent_dim),
                                         trainable=True,
                                         initializer=tf.keras.initializers.random_normal)
        self.clustering_loss_weight=clustering_loss_weight
        self.clustering_loss_tracker = keras.metrics.Mean(name="clust_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
               self.regularization_loss_tracker,
               self.reconstruction_loss_tracker,
               self.clustering_loss_tracker]
    
    def soft_assignment(self, x):
        z_mu, z_logvar = self.encode(x)
        z = _reparameterize(z_mu, z_logvar)
        return _compute_soft_assignment(z, self.centroids)

    def classify(self, x):
        q = self.soft_assignment(x)
        return q.numpy().argmax(1)

    def train_step(self, data):
        x, p = data
        
        with tf.GradientTape() as tape:
            z_mu, z_logvar = self.encode(x)
            z = _reparameterize(z_mu, z_logvar)
            recon_x = self.decode(z)
            q = _compute_soft_assignment(z, self.centroids)
            
            regularization_loss = self.regularization_loss(z_mu, z_logvar)
            reconstruction_loss = self.reconstruction_loss(x, recon_x)
            clustering_loss = clustering_loss_function(q, p)   # Reverse KL (zero forcing) achieves tighter clusters
            # clustering_loss = clustering_loss_function(p, q) # Forward KL (zero avoiding)
            total_loss = regularization_loss + reconstruction_loss + self.clustering_loss_weight*clustering_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.regularization_loss_tracker.update_state(regularizatoin_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.clustering_loss_tracker.update_state(clustering_loss)
        
        return {"total_loss": self.total_loss_tracker.result(),
                "reg_loss": self.regularization_loss_tracker.result(),
                "rec_loss": self.reconstruction_loss_tracker.result(),
                "clust_loss": self.clustering_loss_tracker.result()}



##### VaDE #####

def get_vade_models(n_classes, input_dim, latent_dim, intermediate_dims):
    vae_encoder = get_vae_encoder(input_dim, latent_dim, intermediate_dims)
    vae_decoder = get_decoder(input_dim, latent_dim, intermediate_dims)
    autoencoder_model = AutoEncoderForPretrain(vae_encoder, vae_decoder)
    vade_model = VaDE(n_classes, vae_encoder, vae_decoder)
    return autoencoder_model, vade_model


class VaDE(models.Model):
    """
    Implementation of Variational Deep Embedding(VaDE)
    Original paper: https://arxiv.org/pdf/1611.05148.pdf
    Code adapted from https://github.com/mori97/VaDE/ (implementation on PyTorch)
    """
    
    def __init__(self, n_classes, vae_encoder, vae_decoder, **kwargs):
        super(VaDE, self).__init__(**kwargs)
        latent_dim = vae_encoder.outputs[0].shape[1]
        self._pi = self.add_weight(name='pi',
                                   shape=(n_classes),
                                   trainable=True,
                                   initializer=tf.keras.initializers.zeros)
        self.mu = self.add_weight(name='mu',
                                  shape=(n_classes, latent_dim),
                                  trainable=True,
                                  initializer=tf.keras.initializers.random_normal)
        self.logvar =  self.add_weight(name='logvar',
                                       shape=(n_classes, latent_dim),
                                       trainable=True,
                                       initializer=tf.keras.initializers.random_normal)
        self.encoder = vae_encoder
        self.decoder = vae_decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
            
    @property
    def metrics(self):
        return [self.loss_tracker,]
    
    @property
    def weights(self):
        return K.softmax(self._pi, axis=0)

    def encode_only_mu(self, x):
        mu, _ = self.encoder(x)
        return mu
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mu, z_logvar = self.encode(x)
        z = _reparameterize(z_mu, z_logvar)
        recon_x = self.decode(z)
        return recon_x, z_mu, z_logvar

    def classify(self, x, n_samples=8):
        mu, logvar = self.encode(x)
        z = K.stack([_reparameterize(mu, logvar) for _ in range(n_samples)], axis=1)
        z = tf.expand_dims(z, 2)
        
        h = z - self.mu
        h = K.exp(-0.5 * K.sum(h * h / K.exp(self.logvar), axis=3))
        h = h / K.exp(K.sum(0.5 * self.logvar, axis=1))
        
        p_z_given_c = h / (2 * math.pi)
        p_z_c = p_z_given_c * self.weights
        y = p_z_c / K.sum(p_z_c, axis=2, keepdims=True)
        y = K.sum(y, axis=1)
        return K.argmax(y, axis=1)    
    
    def loss_function(self, x, recon_x, mu, logvar):
        batch_size = tf.cast(tf.shape(x)[0], dtype=tf.float32)
        data_dim = tf.cast(tf.shape(x)[1], dtype=tf.float32)
        
        # Compute gamma ( q(c|x) )
        z = tf.expand_dims(_reparameterize(mu, logvar), 1)
        h = z - self.mu
        h = K.exp(-0.5 * K.sum((h * h / K.exp(self.logvar)), axis=2))
        h = h / K.exp(K.sum(0.5 * self.logvar, axis=1))
        p_z_given_c = h / (2 * math.pi)
        p_z_c = p_z_given_c * self.weights
        gamma = p_z_c / K.sum(p_z_c, axis=1, keepdims=True)

        h = tf.expand_dims(K.exp(logvar), 1) + K.pow((tf.expand_dims(mu, 1) - self.mu), 2)
        h = K.sum(self.logvar + h / K.exp(self.logvar), axis=2)
        loss = data_dim * K.sum(keras.losses.binary_crossentropy(x, recon_x)) \
            + 0.5 * K.sum(gamma * h) \
            - K.sum(gamma * K.log(self.weights + 1e-9)) \
            + K.sum(gamma * K.log(gamma + 1e-9)) \
            - 0.5 * K.sum(1 + logvar)
        loss = loss / batch_size
        return loss
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            recon_x, mu, logvar = self(x)
            loss = self.loss_function(x, recon_x, mu, logvar)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    
class AutoEncoderForPretrain(models.Model):
    """Auto-Encoder for pretraining VaDE."""
    def __init__(self, vae_encoder, vae_decoder, **kwargs):
        super(AutoEncoderForPretrain, self).__init__(**kwargs)
        self.encoder = vae_encoder
        self.decoder = vae_decoder

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x
    
    # there is no gradient for variable z_logvar
    # so tensorflow prints a warning every time
    # workaround: set unconnected_gradients=tf.UnconnectedGradients.ZERO
    def train_step(self, x):
        with tf.GradientTape() as tape:
            recon_x = self(x)
            loss = self.compiled_loss(x, recon_x, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_weights,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(x, recon_x,x)
        return {m.name: m.result() for m in self.metrics}