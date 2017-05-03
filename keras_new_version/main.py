import densely_connected_layers.VAE_densely_connected as dense_VAE
import convolutional_layers.convolutional_VAE as conv_VAE
import discriminant_model as dis_model
import sys
import numpy as np

from glob import glob

# CONV

train_images = glob("/Users/Maria/Documents/input_faces/*.jpg")
load_x_train = [conv_VAE.load_image(image) for image in train_images]
x_train = np.concatenate(load_x_train)
print x_train.shape
y_train = [1] * len(load_x_train)

conv_VAE.Settings.batch_size = 1
conv_VAE.Settings.num_of_epoch = 100

vae = conv_VAE.VAE()
vae_model = vae.get_VAE_model()
encoder = vae.get_encoder_model()
decoder = vae.get_decoder_model()
    
vae_model.compile(optimizer='rmsprop', loss=vae.loss)
vae_model.summary() # log
vae_model.fit(x_train, x_train,
                shuffle=True,
                epochs=conv_VAE.Settings.num_of_epoch,
                batch_size=conv_VAE.Settings.batch_size,
                validation_data=(x_train, x_train))

n = 5
img_size = 64
grid_x = np.linspace(-2.0, 2.0, n)
grid_y = np.linspace(-2.0, 2.0, n)

#generated_images
all_labels = y_train + [0] * (n*n)
all_images = x_train

generated_images = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, conv_VAE.Settings.batch_size).reshape(conv_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=conv_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        img = img.reshape((1,) + img.shape) # (1,64,64)
        img = img.astype('float32') / 255.
        img = img.reshape((img.shape[0],) + conv_VAE.Settings.full_img_size) # (1,64,64,1)
        generated_images.append(img)

all_images = np.concatenate([np.concatenate(generated_images), all_images])
print 'all_images.shape is ', all_images.shape

# TRAINING MODEL

dis_model.epochs = 50
#all_labels = np.concatenate(all_labels, y_train)
model = dis_model.load(x_train=all_images, y_train=all_labels, x_test=all_images, y_test=all_labels)
print model.predict(generated_images[0])
print model.predict(load_x_train[0])



