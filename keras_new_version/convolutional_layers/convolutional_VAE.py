from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, AveragePooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras import metrics
from scipy import misc
from glob import glob
from matplotlib.widgets import Slider
from scipy.stats import norm
import pydot
import graphviz
import theano

import numpy as np
import pylab
import matplotlib.pyplot as plt

class Settings:
    img_size_rows = 64
    img_size_cols = 64
    img_size_chnl = 1
    full_img_size = (img_size_rows, img_size_cols, img_size_chnl)
    batch_size = 1
    latent_dim = 4
    intermediate_dim = 128
    epsilon = 1.0
    num_of_epoch = 1
    min_input = -3.0
    max_input = 3.0
    log = True

def load_image(image_path):
    grayImage = misc.imread(image_path, mode="L")
    image = grayImage.reshape((1,) + grayImage.shape + (Settings.img_size_chnl,)) # (1,64,64,1)
    image = image.astype('float32') / 255.
    return image

def show_images(decoder, n=10, img_size=Settings.img_size_rows):
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.45)

    z_sample = np.array([[0.0, 0.0, 0.0, 0.0]]) * Settings.epsilon
    x_decoded = decoder.predict(z_sample)
    image = x_decoded[0].reshape(img_size, img_size)

    img = ax.imshow(image, cmap='Greys_r')
    cb = plt.colorbar(img)
    axcolor = 'lightgoldenrodyellow'

    ax_1 = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_2  = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_3  = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_4  = plt.axes([0.25, 0.25, 0.65, 0.03])

    s_1 = Slider(ax_1, '1', Settings.min_input, Settings.max_input, valinit=0)
    s_2 = Slider(ax_2, '2', Settings.min_input, Settings.max_input, valinit=0)
    s_3 = Slider(ax_3, '3', Settings.min_input, Settings.max_input, valinit=0)
    s_4 = Slider(ax_4, '4', Settings.min_input, Settings.max_input, valinit=0)

    def update(val):
       _f_1 = s_1.val
       _f_2 = s_2.val
       _f_3 = s_3.val
       _f_4 = s_4.val
       z_sample = np.array([[_f_1, _f_2, _f_3, _f_4]]) * Settings.epsilon
       x_decoded = decoder.predict(z_sample)
       image = x_decoded[0].reshape(img_size, img_size)
       ax.imshow(image, cmap='Greys_r')
       plt.draw()

    s_1.on_changed(update)
    s_2.on_changed(update)
    s_3.on_changed(update)
    s_4.on_changed(update)
    plt.show()

def showOutputOfModel(model):
    conv_output = model.predict(x_train)
    im = conv_output
    count_of_images = conv_output.shape[0]
    size = conv_output.shape[1]
    count_filters = conv_output.shape[3]
    
    figure = np.zeros((size * count_of_images, size * count_filters))
    
    for i in range(count_of_images):
        for j in range(count_filters):
            tmp_im = np.zeros((size,size))
            tmp_im[:,:] = im[i,:,:,j]
            tmp_im = tmp_im.reshape(size, size)
            figure[i * size: (i + 1) * size,
                   j * size: (j + 1) * size] = tmp_im

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

class VAE:
    def __init__(self):
        # CODER
        self.input_layer = Input(shape=Settings.full_img_size)
        self.conv_layer_1 = Conv2D(filters=8,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    strides=(1, 1), 
                                    activation='relu')(self.input_layer) # (64,64,8)
        self.conv_layer_2 = Conv2D(filters=16, 
                                    kernel_size=(3, 3), 
                                    padding='same',
                                    strides=(1, 1), 
                                    activation='relu')(self.conv_layer_1) # (64,64,16)
        
        self.conv_layer_3 = Conv2D(filters=24,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    strides=(2, 2),  
                                    activation='relu')(self.conv_layer_2) # (32,32,16)
        self.flat_layer = Flatten()(self.conv_layer_3)
        self.hidden_layer = Dense(Settings.intermediate_dim, activation='relu')(self.flat_layer)
        self.z_mean = Dense(Settings.latent_dim)(self.hidden_layer)
        self.z_log_var = Dense(Settings.latent_dim)(self.hidden_layer)
        self.z = Lambda(self.sampling)([self.z_mean, self.z_log_var])
        # Just layers for decoder
        arr_size_conv_layer_4 = [i.value for i in self.conv_layer_3.shape.dims if i.value is not None]
        size_flat_layer = np.prod(arr_size_conv_layer_4)
        size_conv_layer_4 = tuple(arr_size_conv_layer_4)

        self.decoder_hidden = Dense(Settings.intermediate_dim, activation='relu')
        self.decoder_flat = Dense(size_flat_layer, activation='relu')
        self.decoder_reshape = Reshape(size_conv_layer_4)
        self.decoder_conv_4 = Conv2DTranspose(filters=24,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            strides=(1, 1),
                                            activation='relu')
        self.decoder_conv_3 = Conv2DTranspose(filters=16,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            strides=(2, 2),
                                            activation='relu')
        self.decoder_conv_2 = Conv2D(filters=1,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            strides=(1, 1),
                                            activation='sigmoid') # return pixels [0..1]
        # DECODER (connected with full VAE)
        self.hidden_decoded = self.decoder_hidden(self.z)
        self.flat_decoded = self.decoder_flat(self.hidden_decoded)
        self.reshape_decoded = self.decoder_reshape(self.flat_decoded)
        self.conv_4_decoded = self.decoder_conv_4(self.reshape_decoded)
        self.conv_3_decoded = self.decoder_conv_3(self.conv_4_decoded)
        self.conv_2_decoded = self.decoder_conv_2(self.conv_3_decoded)

    def sampling(self, hidden_layers):
        print('sampling')
        z_mean, z_log_var = hidden_layers
        normal = K.random_normal(shape=(Settings.batch_size, Settings.latent_dim),
                                 mean=0.,
                                 stddev=Settings.epsilon)
        return z_mean + K.exp(self.z_log_var) * normal
    
    def loss(self, x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = Settings.full_img_size * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss
    
    def get_conv_model(self):
        return Model(self.input_layer, self.conv_layer_3) # conv_layer_1, conv_layer_2, conv_layer_3
    
    def get_VAE_model(self):
        return Model(self.input_layer, self.conv_2_decoded)
    
    def get_encoder_model(self):
        return Model(self.input_layer, self.z_mean)
    
    def get_decoder_model(self):
        decoder_input = Input(shape=(Settings.latent_dim,))
        hidden_decoded = self.decoder_hidden(decoder_input)
        flat_decoded = self.decoder_flat(hidden_decoded)
        reshape_decoded = self.decoder_reshape(flat_decoded)
        conv_4_decoded = self.decoder_conv_4(reshape_decoded)
        conv_3_decoded = self.decoder_conv_3(conv_4_decoded)
        conv_2_decoded = self.decoder_conv_2(conv_3_decoded)
        return Model(decoder_input, conv_2_decoded)

if __name__ == "__main__":
    images = glob("/Users/Maria/Documents/input_faces/test/*.jpg")
    load_x_train = [load_image(image) for image in images]
    x_train = np.concatenate(load_x_train)
    
    vae = VAE()
    vae_model = vae.get_VAE_model()
    encoder = vae.get_encoder_model()
    decoder = vae.get_decoder_model()
    
    sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    vae_model.compile(optimizer="rmsprop", loss=vae.loss)
    vae_model.summary() # log
    vae_model.fit(x_train, x_train,
                  shuffle=True,
                  epochs=Settings.num_of_epoch,
                  batch_size=Settings.batch_size,
                  validation_data=(x_train, x_train))
                  
    plot_model(vae_model, to_file='/Users/Maria/Documents/FaceTransfer/keras_new_version/convolutional_layers/vae_model.png', show_shapes=True)
    plot_model(encoder, to_file='/Users/Maria/Documents/FaceTransfer/keras_new_version/convolutional_layers/encoder.png', show_shapes=True)
    plot_model(decoder, to_file='/Users/Maria/Documents/FaceTransfer/keras_new_version/convolutional_layers/decoder.png', show_shapes=True)
    
#    # Show output of conv layer
#    conv_model = vae.get_conv_model()
#    showOutputOfModel(conv_model)
#
#    # Show decoder output
    show_images(decoder, n=20)

