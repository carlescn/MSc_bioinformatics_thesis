
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

import math
import numpy as np


def get_autoencoder_models(input_dim, latent_dim):
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dim), name="encoder_inputs")
    x = layers.Dense(512, activation="relu") (encoder_inputs)
    x = layers.Dense(512, activation="relu") (x)
    x = layers.Dense(2048, activation="relu") (x)
    encoder_mu = layers.Dense(latent_dim) (x)
    encoder_logvar = layers.Dense(latent_dim) (x)
    encoder = models.Model(encoder_inputs, [encoder_mu, encoder_logvar], name="encoder")
    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim), name="decoder_inputs")
    x = layers.Dense(2048, activation="relu") (decoder_inputs)
    x = layers.Dense(512, activation="relu") (x)
    x = layers.Dense(512, activation="relu") (x)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid") (x)
    decoder = models.Model(decoder_inputs, decoder_outputs, name="decoder")
    return (encoder, decoder)


def _reparameterize(mu, logvar):
    """Reparameterization trick.
    """
    std = K.exp(0.5 * logvar)
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + eps * std


class VaDE(models.Model):
    """
    Implementation of Variational Deep Embedding(VaDE)
    Original paper: https://arxiv.org/pdf/1611.05148.pdf
    Code adapted from https://github.com/mori97/VaDE/

    Args:
        n_classes (int): Number of clusters.
        data_dim (int): Dimension of observed data.
        latent_dim (int): Dimension of latent space.
    """
    
    def __init__(self, n_classes, data_dim, latent_dim, encoder, decoder, **kwargs):
        super(VaDE, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
        self._pi = self.add_weight(name='pi', shape=(n_classes), trainable=True, initializer=tf.keras.initializers.zeros)
        self.mu = self.add_weight(name='mu', shape=(n_classes, latent_dim), trainable=True, initializer=tf.keras.initializers.random_normal)
        self.logvar =  self.add_weight(name='logvar', shape=(n_classes, latent_dim), trainable=True, initializer=tf.keras.initializers.random_normal)
            
    @property
    def metrics(self):
        return [self.loss_tracker,]
    
    @property
    def weights(self):
        return K.softmax(self._pi, axis=0)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mu, logvar = self.encode(x)
        z = _reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

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
        loss = self.data_dim * K.sum(keras.losses.binary_crossentropy(x, recon_x)) \
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
    """Auto-Encoder for pretraining VaDE.

    Args:
        data_dim (int): Dimension of observed data.
        latent_dim (int): Dimension of latent space.
    """
    def __init__(self, data_dim, latent_dim, encoder, decoder, **kwargs):
        super(AutoEncoderForPretrain, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x
    

