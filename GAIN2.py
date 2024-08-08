import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
data_path = '/mnt/data/EI_2024_FOR_ANALYSIS.csv'
data = pd.read_csv(data_path)

# Identify missing values
data = data.replace("?", np.nan)  # Assuming missing values are marked with "?"
data = data.astype(float)  # Convert all data to float

# Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Create masks for missing data
mask = ~np.isnan(data_normalized)

# Parameters
input_dim = data_normalized.shape[1]
h_dim = 128
batch_size = 128
iterations = 5000
alpha = 10

# Generator
def generator(x):
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(h_dim, activation='relu')(inputs)
    h = layers.Dense(h_dim, activation='relu')(h)
    outputs = layers.Dense(input_dim, activation='sigmoid')(h)
    return models.Model(inputs, outputs)

# Discriminator
def discriminator(x):
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(h_dim, activation='relu')(inputs)
    h = layers.Dense(h_dim, activation='relu')(h)
    outputs = layers.Dense(input_dim, activation='sigmoid')(h)
    return models.Model(inputs, outputs)

# Build models
G = generator(data_normalized)
D = discriminator(data_normalized)

# Loss and optimizer
bce = tf.keras.losses.BinaryCrossentropy()
G_optimizer = tf.keras.optimizers.Adam(0.001)
D_optimizer = tf.keras.optimizers.Adam(0.001)

# Training step
@tf.function
def train_step(X, M):
    Z = tf.random.normal(shape=X.shape)

    # Combine observed data with noise
    X_hat = X * M + Z * (1 - M)

    with tf.GradientTape() as tape:
        G_sample = G(X_hat)
        D_prob = D(G_sample)
        G_loss = -tf.reduce_mean(M * tf.math.log(D_prob + 1e-8))

    G_gradients = tape.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))

    with tf.GradientTape() as tape:
        D_prob = D(X * M + G_sample * (1 - M))
        D_loss = -tf.reduce_mean(M * tf.math.log(D_prob + 1e-8) + (1 - M) * tf.math.log(1 - D_prob + 1e-8))

    D_gradients = tape.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_gradients, D.trainable_variables))

    return G_loss, D_loss

# Training loop
for iteration in range(iterations):
    idx = np.random.randint(0, data_normalized.shape[0], batch_size)
    X_batch = data_normalized[idx]
    M_batch = mask[idx]

    G_loss, D_loss = train_step(X_batch, M_batch)

    if iteration % 100 == 0:
        print(f'Iteration: {iteration}, G_loss: {G_loss.numpy()}, D_loss: {D_loss.numpy()}')

# Impute missing values
def impute_missing(data, mask, generator):
    Z = np.random.normal(size=data.shape)
    X_hat = data * mask + Z * (1 - mask)
    imputed_data = generator.predict(X_hat)
    imputed_data = mask * data + (1 - mask) * imputed_data
    return imputed_data

imputed_data = impute_missing(data_normalized, mask, G)

# Reverse normalization
imputed_data = scaler.inverse_transform(imputed_data)

# Save imputed data to CSV
imputed_data_df = pd.DataFrame(imputed_data, columns=data.columns)
imputed_data_df.to_csv('/mnt/data/EI_2024_FOR_ANALYSIS_imputed.csv', index=False)
