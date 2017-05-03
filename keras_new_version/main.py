import densely_connected_layers.VAE_densely_connected as dense_VAE
import convolutional_layers.convolutional_VAE as conv_VAE
import discriminant_model as dis_model
import sys
import keras
import numpy as np

from glob import glob

# 1 is real
# 0 is fake

# CONV

train_images = glob("/Users/Maria/Documents/input_faces/train/*.jpg")
test_images = glob("/Users/Maria/Documents/input_faces/test/*.jpg")
load_x_train = [conv_VAE.load_image(image) for image in train_images]
load_x_test = [conv_VAE.load_image(image) for image in test_images]

# first is real images

x_train = np.concatenate(load_x_train)
x_test = np.concatenate(load_x_test)

y_train = [1] * len(load_x_train)
y_test = [1] * len(load_x_train)

#-----------

conv_VAE.Settings.batch_size = 1
conv_VAE.Settings.num_of_epoch = 10

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

n = 3 # 9 fake images
img_size = 64
grid_x = np.linspace(-2.0, 2.0, n)
grid_y = np.linspace(-2.0, 2.0, n)

all_labels_train = y_train + [0] * (n*n)
all_images_train = x_train

generated_images_train = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, conv_VAE.Settings.batch_size).reshape(conv_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=conv_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        img = img.reshape((1,) + img.shape) # (1,64,64)
        img = img.astype('float32') / 255.
        img = img.reshape((img.shape[0],) + conv_VAE.Settings.full_img_size) # (1,64,64,1)
        generated_images_train.append(img)

all_images_train = np.concatenate([all_images_train, np.concatenate(generated_images_train)])
print 'all_images.shape is ', all_images_train.shape

#-----------

conv_VAE.Settings.batch_size = 1
conv_VAE.Settings.num_of_epoch = 10

vae = conv_VAE.VAE()
vae_model = vae.get_VAE_model()
encoder = vae.get_encoder_model()
decoder = vae.get_decoder_model()

vae_model.compile(optimizer='rmsprop', loss=vae.loss)
vae_model.summary() # log
vae_model.fit(x_test, x_test,
              shuffle=True,
              epochs=conv_VAE.Settings.num_of_epoch,
              batch_size=conv_VAE.Settings.batch_size,
              validation_data=(x_test, x_test))

n = 3 # 9 fake images
img_size = 64
grid_x = np.linspace(-2.0, 2.0, n)
grid_y = np.linspace(-2.0, 2.0, n)

all_labels_test = y_test + [0] * (n*n)
all_images_test = x_test

generated_images_test = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, conv_VAE.Settings.batch_size).reshape(conv_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=conv_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        img = img.reshape((1,) + img.shape) # (1,64,64)
        img = img.astype('float32') / 255.
        img = img.reshape((img.shape[0],) + conv_VAE.Settings.full_img_size) # (1,64,64,1)
        generated_images_test.append(img)

all_images_test = np.concatenate([all_images_test, np.concatenate(generated_images_test)])
print 'all_images_test.shape is ', all_images_test.shape




# TRAINING MODEL

dis_model.epochs = 5
model = dis_model.load(x_train=all_images_train, y_train=all_labels_train, x_test=all_images_test, y_test=all_labels_test)
print model.predict(generated_images_train[0])
print model.predict(load_x_train[0])

all_labels_test = keras.utils.to_categorical(all_labels_test, 2)
test_score = model.evaluate(all_images_test, all_labels_test, verbose=0) # for test
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
print("Baseline Error: %.2f%%" % (100-test_score[1]*100))


