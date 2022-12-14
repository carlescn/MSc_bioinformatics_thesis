from tensorflow import keras
from tensorflow.keras import layers, models

def get_encoder(input_dim, latent_dim, intermediate_dims=[]):
    encoder_inputs = keras.Input(shape=(input_dim), name="encoder_inputs")
    x = encoder_inputs
    for dim in intermediate_dims:
        x = layers.Dense(dim, activation="relu") (x)
    x = layers.Dense(latent_dim, activation="relu", name="z") (x)
    return models.Model(encoder_inputs, x, name="Encoder")
    
def get_decoder(output_dim, latent_dim, intermediate_dims=[]):
    decoder_inputs = keras.Input(shape=(latent_dim), name="decoder_inputs")
    x = decoder_inputs
    for dim in intermediate_dims[::-1]:
        x = layers.Dense(dim, activation="relu") (x)
    decoder_outputs = layers.Dense(output_dim, activation="sigmoid", name="recon_x") (x)
    return models.Model(decoder_inputs, decoder_outputs, name="Decoder")
    
def get_autoencoder(input_dim, latent_dim, intermediate_dims=[]):
    encoder = get_encoder(input_dim, latent_dim, intermediate_dims)
    decoder = get_decoder(input_dim, latent_dim, intermediate_dims)
    z = encoder(encoder.inputs)
    recon_x = decoder(z)
    return models.Model(encoder.inputs, [z, recon_x], name="Autoencoder")

def get_models_for_vae(input_dim, latent_dim, intermediate_dims=[]):
    encoder_inputs = keras.Input(shape=(input_dim), name="encoder_inputs")
    x = encoder_inputs
    for dim in intermediate_dims:
        x = layers.Dense(dim, activation="relu") (x)
    z_mu = layers.Dense(latent_dim, name="z_mu") (x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar") (x)
    encoder = models.Model(encoder_inputs, [z_mu, z_logvar], name="Encoder")
    decoder = get_decoder(input_dim, latent_dim, intermediate_dims)
    return (encoder, decoder)