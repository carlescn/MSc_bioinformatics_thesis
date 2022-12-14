import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K


def _reparameterize(mu, logvar):
    """Reparameterization trick."""
    std = K.exp(0.5 * logvar)
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + eps * std


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
        reg_loss = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), axis=-1)
        return tf.reduce_mean(reg_loss)
    
    def reconstruction_loss(self, x, recon_x):
        return tf.reduce_sum(keras.losses.binary_crossentropy(x, recon_x))

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
    def __init__(self, n_clusters, latent_dim, encoder, decoder, clustering_loss_weight=1, **kwargs):
        super(ClusteringVAE, self).__init__(encoder, decoder, **kwargs)
        self.clustering_loss_weight=clustering_loss_weight
        self.n_clusters = n_clusters
        self.clustering_loss_tracker = keras.metrics.Mean(name="clust_loss")
        self.centroids = self.add_weight(name='centroids', shape=(n_clusters, latent_dim), trainable=True, initializer=tf.keras.initializers.random_normal)
        
    @property
    def metrics(self):
        return [self.total_loss_tracker,
               self.regularization_loss_tracker,
               self.reconstruction_loss_tracker,
               self.clustering_loss_tracker]

    def _compute_soft_assignment(self, z):
        """
        Compute cluster assignments.
        Source: https://github.com/Tony607/Keras_Deep_Clustering/blob/master/Keras-DEC.ipynb
        """
        q = 1.0 / (1.0 + K.sum(K.square(K.expand_dims(z, axis=1) - self.centroids), axis=2))
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q
    
    def soft_assignment(self, x):
        z_mu, z_logvar = self.encode(x)
        z = _reparameterize(z_mu, z_logvar)
        return self._compute_soft_assignment(z)

    def classify(self, x):
        q = self.soft_assignment(x)
        return q.numpy().argmax(1)
    
    def clustering_loss(self, q, p):
        return tf.reduce_sum(tf.keras.losses.kld(q, p))

    def train_step(self, data):
        x, p = data
        
        with tf.GradientTape() as tape:
            z_mu, z_logvar = self.encode(x)
            z = _reparameterize(z_mu, z_logvar)
            recon_x = self.decode(z)
            q = self._compute_soft_assignment(z)
            
            regularizatoin_loss = self.regularization_loss(z_mu, z_logvar)
            reconstruction_loss = self.reconstruction_loss(x, recon_x)
            clustering_loss = self.clustering_loss(q, p)
            total_loss = regularizatoin_loss + reconstruction_loss + self.clustering_loss_weight*clustering_loss
            
            gradients = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            
            self.total_loss_tracker.update_state(total_loss)
            self.regularization_loss_tracker.update_state(regularizatoin_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.clustering_loss_tracker.update_state(clustering_loss)
        
        return {"total_loss": self.total_loss_tracker.result(),
                "reg_loss": self.regularization_loss_tracker.result(),
                "rec_loss": self.reconstruction_loss_tracker.result(),
                "clust_loss": self.clustering_loss_tracker.result(),
               }
