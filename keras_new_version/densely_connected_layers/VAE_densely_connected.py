from keras.layers import Input, Dense, Lambda, Conv2D
from keras.layers import MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras import backend as K
from keras import metrics
from scipy import misc
from glob import glob
from matplotlib.widgets import Slider

import pylab
import matplotlib.pyplot as plt
import numpy as np

class Settings:
    img_size_rows = 64
    img_size_cols = 64
    img_size_chnl = 1
    full_img_size = img_size_rows * img_size_cols * img_size_chnl
    batch_size = 1
    latent_dim = 2
    high_dim = 256
    low_dim = 64
    epsilon = 1.0
    min_input = -3.0
    max_input = 3.0
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
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    z_sample = np.array([[0.0, 0.0]]) * Settings.epsilon
    x_decoded = decoder.predict(z_sample)
    image = x_decoded[0].reshape(img_size, img_size)

    img = ax.imshow(image, cmap='Greys_r')
    cb = plt.colorbar(img)
    axcolor = 'lightgoldenrodyellow'

    ax_cmin = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_cmax  = plt.axes([0.25, 0.15, 0.65, 0.03])

    s_first = Slider(ax_cmin, 'first', Settings.min_input, Settings.max_input, valinit=0)
    s_second = Slider(ax_cmax, 'second', Settings.min_input, Settings.max_input, valinit=0)

    def update(val):
       _first = s_first.val
       _second = s_second.val
       z_sample = np.array([[_first, _second]]) * Settings.epsilon
       x_decoded = decoder.predict(z_sample)
       image = x_decoded[0].reshape(img_size, img_size)
       ax.imshow(image, cmap='Greys_r')
       plt.draw()

    s_first.on_changed(update)
    s_second.on_changed(update)

    plt.show()

class VAE:
    def __init__(self):
        # CODER
        self.input_layer = Input(batch_shape=(Settings.batch_size, Settings.full_img_size))
        self.high_layer = Dense(Settings.high_dim, activation='relu')(self.input_layer)
        self.low_layer = Dense(Settings.low_dim, activation='relu')(self.high_layer)
        # latent distribution parameterized by hidden encoding
        # z ~ N(z_mean, np.exp(z_log_sigma)**2)
        self.z_mean = Dense(Settings.latent_dim)(self.low_layer)
        self.z_log_sigma = Dense(Settings.latent_dim)(self.low_layer)
        self.z = Lambda(self.sampling)([self.z_mean, self.z_log_sigma])
        # DECODER (connected with VAE)
        self.decoder_h_low = Dense(Settings.low_dim, activation='relu')
        self.decoder_h_high = Dense(Settings.high_dim, activation='relu')
        self.decoder_mean = Dense(Settings.full_img_size, activation='sigmoid')
        self.h_decoded_low = self.decoder_h_low(self.z)
        self.h_decoded_high = self.decoder_h_high(self.h_decoded_low)
        self.x_decoded_mean = self.decoder_mean(self.h_decoded_high)
        # DECODER
        self.decoder_input = Input(shape=(Settings.latent_dim,))
        self._h_decoded_low = self.decoder_h_low(self.decoder_input)
        self._h_decoded_high = self.decoder_h_high(self._h_decoded_low)
        self._x_decoded_mean = self.decoder_mean(self._h_decoded_high)

    def sampling(self, hidden_layers):
        z_mean, z_log_sigma = hidden_layers
        normal = K.random_normal(shape=(Settings.batch_size, Settings.latent_dim), 
                                 mean=0.,
                                 stddev=Settings.epsilon)
        return z_mean + K.exp(z_log_sigma) * normal # N(mu, sigma**2)

    def loss(self, x, x_decoded_mean):
        # loss = reconstruction loss + KL loss
        xent_loss = Settings.full_img_size * metrics.binary_crossentropy(x, x_decoded_mean)
        # Kullback-Leibler divergence
        # KL[q(z|x) || p(z)]
        kl_loss = 0.5 * K.sum(K.exp(self.z_log_sigma) + K.square(self.z_mean) - 1. - self.z_log_sigma, axis=1)
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
    vae_model.summary() # log
    vae_model.compile(optimizer='rmsprop', loss=vae.loss)
    vae_model.fit(x_train, x_train,
              shuffle=True,
              epochs=Settings.num_of_epoch,
              batch_size=Settings.batch_size,
              validation_data=(x_train, x_train))

    show_images(decoder, n=30)

