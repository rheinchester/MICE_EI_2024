import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Preprocess the data
data = pd.read_csv(file_path)
data_values = data.values

# Impute missing values with mean for normalization purposes
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_values)

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Create masks for missing values
mask = np.isnan(data_values)

# Define the generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=44),
        layers.Dense(256, activation='relu'),
        layers.Dense(44, activation='sigmoid')
    ])
    return model

# Define the discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=44),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build and compile the GAN
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gan_input = layers.Input(shape=(44,))
generated_data = generator(gan_input)
discriminator.trainable = False
gan_output = discriminator(generated_data)

gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training the GAN
def train_gan(generator, discriminator, gan, data, epochs=1000, batch_size=32):
    half_batch = int(batch_size / 2)
    
    for epoch in range(epochs):
        # Generate fake data
        noise = np.random.normal(0, 1, (half_batch, 44))
        gen_data = generator.predict(noise)
        
        # Select a random half batch of real data
        idx = np.random.randint(0, data_scaled.shape[0], half_batch)
        real_data = data_scaled[idx]
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 44))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        
        # Print progress
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

# Train the GAN on the dataset
train_gan(generator, discriminator, gan, data_scaled)

# Impute missing data
def impute_missing_data(generator, data, mask):
    imputed_data = data.copy()
    for i in range(data.shape[0]):
        sample = data[i].reshape(1, -1)
        gen_sample = generator.predict(sample)
        imputed_data[i][mask[i]] = gen_sample[0][mask[i]]
    return imputed_data

# Impute missing values in the original dataset
imputed_data = impute_missing_data(generator, data_scaled, mask)
imputed_data = scaler.inverse_transform(imputed_data)

# Convert imputed data back to DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=data.columns)

# Display the first few rows of the imputed DataFrame
imputed_df.head()
