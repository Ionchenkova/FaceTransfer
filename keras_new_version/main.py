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
load_x_train_conv = [conv_VAE.load_image(image) for image in train_images]
load_x_test_conv = [conv_VAE.load_image(image) for image in test_images]

# first is real images

x_train_conv = np.concatenate(load_x_train_conv)
x_test_conv = np.concatenate(load_x_test_conv)

y_train_conv = [1] * len(load_x_train_conv)
y_test_conv = [1] * len(load_x_test_conv)

#-----------

conv_VAE.Settings.num_of_epoch = 100

vae = conv_VAE.VAE()
vae_model = vae.get_VAE_model()
encoder = vae.get_encoder_model()
decoder = vae.get_decoder_model()
    
vae_model.compile(optimizer='rmsprop', loss=vae.loss)
vae_model.summary() # log
vae_model.fit(x_train_conv, x_train_conv,
                shuffle=True,
                epochs=conv_VAE.Settings.num_of_epoch,
                batch_size=conv_VAE.Settings.batch_size,
                validation_data=(x_train_conv, x_train_conv))

n = 3 # 9 fake images
img_size = 64
grid_x = np.linspace(-2.0, 2.0, n)
grid_y = np.linspace(-2.0, 2.0, n)

all_labels_train_conv = y_train_conv + [0] * (n*n)
all_images_train_conv = x_train_conv

generated_images_train_conv = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, conv_VAE.Settings.batch_size).reshape(conv_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=conv_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        img = img.reshape((1,) + img.shape) # (1,64,64)
        img = img.astype('float32') / 255.
        img = img.reshape((img.shape[0],) + conv_VAE.Settings.full_img_size) # (1,64,64,1)
        generated_images_train_conv.append(img)

all_images_train_conv = np.concatenate([all_images_train_conv, np.concatenate(generated_images_train_conv)])
print 'all_images_train_conv.shape is ', all_images_train_conv.shape

#-----------

conv_VAE.Settings.num_of_epoch = 100

vae = conv_VAE.VAE()
vae_model = vae.get_VAE_model()
encoder = vae.get_encoder_model()
decoder = vae.get_decoder_model()

vae_model.compile(optimizer='rmsprop', loss=vae.loss)
vae_model.summary() # log
vae_model.fit(x_test_conv, x_test_conv,
              shuffle=True,
              epochs=conv_VAE.Settings.num_of_epoch,
              batch_size=conv_VAE.Settings.batch_size,
              validation_data=(x_test_conv, x_test_conv))

n = 3 # 9 fake images
img_size = 64
grid_x = np.linspace(-2.0, 2.0, n)
grid_y = np.linspace(-2.0, 2.0, n)

all_labels_test_conv = y_test_conv + [0] * (n*n)
all_images_test_conv = x_test_conv

generated_images_test_conv = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, conv_VAE.Settings.batch_size).reshape(conv_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=conv_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        img = img.reshape((1,) + img.shape) # (1,64,64)
        img = img.astype('float32') / 255.
        img = img.reshape((img.shape[0],) + conv_VAE.Settings.full_img_size) # (1,64,64,1)
        generated_images_test_conv.append(img)

all_images_test_conv = np.concatenate([all_images_test_conv, np.concatenate(generated_images_test_conv)])
print 'all_images_test_conv.shape is ', all_images_test_conv.shape

# DENSE

load_x_train_dense = [dense_VAE.load_image(image) for image in train_images]
load_x_test_dense = [dense_VAE.load_image(image) for image in test_images]

# first is real images

x_train_dense = np.concatenate(load_x_train_dense)
x_test_dense = np.concatenate(load_x_test_dense)

y_train_dense = [1] * len(x_train_dense)
y_test_dense = [1] * len(x_test_dense)

#----

dense_VAE.Settings.num_of_epoch = 100

vae = dense_VAE.VAE()
vae_model = vae.get_VAE_model()
encoder = vae.get_encoder_model()
decoder = vae.get_decoder_model()

vae_model.compile(optimizer='rmsprop', loss=vae.loss)
vae_model.summary() # log
vae_model.fit(x_train_dense, x_train_dense,
              shuffle=True,
              epochs=conv_VAE.Settings.num_of_epoch,
              batch_size=conv_VAE.Settings.batch_size,
              validation_data=(x_train_dense, x_train_dense))

n = 3 # 9 fake images
img_size = 64
grid_x = np.linspace(-2.0, 2.0, n)
grid_y = np.linspace(-2.0, 2.0, n)

all_labels_train_dense = y_train_dense + [0] * (n*n)
all_images_train_dense = x_train_conv

generated_images_train_dense = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample, batch_size=dense_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        img = img.reshape((1,) + img.shape) # (1,64,64)
        img = img.astype('float32') / 255.
        img = img.reshape(img.shape + (1,))
        print 'img.shape is ', img.shape
        generated_images_train_dense.append(img)

all_images_train_dense = np.concatenate([all_images_train_dense, np.concatenate(generated_images_train_dense)])
print 'all_images_train_dense.shape is ', all_images_train_dense.shape

#----

dense_VAE.Settings.num_of_epoch = 100

vae = dense_VAE.VAE()
vae_model = vae.get_VAE_model()
encoder = vae.get_encoder_model()
decoder = vae.get_decoder_model()

vae_model.compile(optimizer='rmsprop', loss=vae.loss)
vae_model.summary() # log
vae_model.fit(x_test_dense, x_test_dense,
              shuffle=True,
              epochs=conv_VAE.Settings.num_of_epoch,
              batch_size=conv_VAE.Settings.batch_size,
              validation_data=(x_test_dense, x_test_dense))

n = 3 # 9 fake images
img_size = 64
grid_x = np.linspace(-2.0, 2.0, n)
grid_y = np.linspace(-2.0, 2.0, n)

all_labels_test_dense = y_test_dense + [0] * (n*n)
all_images_test_dense = x_test_conv

generated_images_test_dense = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
#        z_sample = np.tile(z_sample, conv_VAE.Settings.batch_size).reshape(conv_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=conv_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        img = img.reshape((1,) + img.shape) # (1,64,64)
        img = img.astype('float32') / 255.
        img = img.reshape(img.shape + (1,))
        print 'img.shape is ', img.shape
        generated_images_test_dense.append(img)

all_images_test_dense = np.concatenate([all_images_test_dense, np.concatenate(generated_images_test_dense)])
print 'all_images_test_dense.shape is ', all_images_test_dense.shape

# Now we have 9 conv fake images, 9 dense fake images and 10 real images

# TRAINING MODEL

all_train_images = np.concatenate([all_images_train_conv, all_images_train_dense])
all_train_labels = all_labels_train_conv + all_labels_train_dense

all_train_images = np.concatenate([all_images_test_conv, all_images_test_dense])
all_train_labels = all_labels_test_conv + all_labels_test_dense

dis_model.epochs = 50
model = dis_model.load(x_train=all_train_images, y_train=all_train_labels, x_test=all_train_images, y_test=all_train_labels)
#print model.predict(generated_images_train[0])
#print model.predict(load_x_train[0])

all_labels_test_conv = keras.utils.to_categorical(all_labels_test_conv, 2)
test_score_conv = model.evaluate(all_images_test_conv, all_labels_test_conv, verbose=0) # for test
print('Test CONV loss:', test_score_conv[0])
print('Test CONV accuracy:', test_score_conv[1])
print("Baseline CONV Error: %.2f%%" % (100-test_score_conv[1]*100))

all_labels_test_dense = keras.utils.to_categorical(all_labels_test_dense, 2)
test_score_dense = model.evaluate(all_images_test_dense, all_labels_test_dense, verbose=0) # for test
print('Test dense loss:', test_score_dense[0])
print('Test dense accuracy:', test_score_dense[1])
print("Baseline dense Error: %.2f%%" % (100-test_score_dense[1]*100))




