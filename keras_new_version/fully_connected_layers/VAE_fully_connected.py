from keras.layers import Input, Dense, Lambda, Conv2D
from keras.layers import MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras import backend as K
from keras import metrics
from scipy import misc
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

class Settings:
    img_size_rows = 64
    img_size_cols = 64
    img_size_chnl = 1
    full_img_size = img_size_rows * img_size_cols * img_size_chnl
    batch_size = 1
    latent_dim = 2
    intermediate_dim = full_img_size / 4
    epsilon = 1.0
    num_of_epoch = 100
    log = True

def load_image(image_path):
    grayImage = misc.imread(image_path, mode="L") # shape is (x,x)
    x_train = grayImage.reshape((1,) + grayImage.shape) # shape is (1,x,x)
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # shape is (1, x)
    print(x_train.shape)
    return x_train

def show_images(decoder, n=10, img_size=Settings.img_size_rows):
    figure = np.zeros((img_size * n, img_size * n))
    grid_x = np.linspace(-1.0, 1.0, n)
    grid_y = np.linspace(-1.0, 1.0, n)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * Settings.epsilon
            x_decoded = decoder.predict(z_sample) # we ganerate latent layer with latent_dim
            image = x_decoded[0].reshape(img_size, img_size)
            figure[i * img_size: (i + 1) * img_size,
                    j * img_size: (j + 1) * img_size] = image
    plt.figure(figsize=(6, 6))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

class VAE:
    def __init__(self):
        # CODER
        self.input_layer = Input(batch_shape=(Settings.batch_size, Settings.full_img_size))
        self.intermediate_layer = Dense(Settings.intermediate_dim, activation='relu')(self.input_layer)
        self.z_mean = Dense(Settings.latent_dim)(self.intermediate_layer)
        self.z_random = Dense(Settings.latent_dim)(self.intermediate_layer)
        self.z = Lambda(self.sampling)([self.z_mean, self.z_random])
        # DECODER (connected with VAE)
        self.decoder_h = Dense(Settings.intermediate_dim, activation='relu')
        self.decoder_mean = Dense(Settings.full_img_size, activation='sigmoid')
        self.h_decoded = self.decoder_h(self.z)
        self.x_decoded_mean = self.decoder_mean(self.h_decoded)
        # DECODER
        self.decoder_input = Input(shape=(Settings.latent_dim,))
        self._h_decoded = self.decoder_h(self.decoder_input)
        self._x_decoded_mean = self.decoder_mean(self._h_decoded)

    def sampling(self, hidden_layers):
        z_mean, z_random = hidden_layers
        normal = K.random_normal(shape=(Settings.batch_size, Settings.latent_dim), 
                                 mean=0.,
                                 stddev=Settings.epsilon)
        return z_mean + z_random * normal

    def loss(self, x, x_decoded_mean):
        xent_loss = Settings.full_img_size * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_random - K.square(self.z_mean) - K.exp(self.z_random), axis=-1)
        return xent_loss + kl_loss

    def get_VAE_model(self):
        return Model(self.input_layer, self.x_decoded_mean)

    def get_encoder_model(self):
        return Model(self.input_layer, self.z_mean)

    def get_decoder_model(self):
        return Model(self.decoder_input, self._x_decoded_mean)

if __name__ == "__main__":
    
    images = glob("/Users/Maria/Documents/input_faces/*.jpg")
    load_x_train = [load_image(image) for image in images]
    x_train = np.concatenate(load_x_train)
    
    vae = VAE()
    vae_model = vae.get_VAE_model()
    encoder = vae.get_encoder_model()
    decoder = vae.get_decoder_model()

    vae_model.compile(optimizer='rmsprop', loss=vae.loss)
    vae_model.fit(x_train, x_train,
              shuffle=True,
              epochs=Settings.num_of_epoch,
              batch_size=Settings.batch_size,
              validation_data=(x_train, x_train))

    show_images(decoder, n=10)

