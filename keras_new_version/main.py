import densely_connected_layers.VAE_densely_connected as dense_VAE
import convolutional_layers.convolutional_VAE as conv_VAE
import discriminant_model as dis_model
from sklearn.metrics import classification_report,confusion_matrix
import sys
import keras
import numpy as np
from glob import glob
from scipy import misc

# 0 is real
# 1 is fake

EPOCH_VAE = 100
EPOCH_DIS = 10

def load_image(image_path):
    grayImage = misc.imread(image_path, mode="L")
    image = grayImage.reshape((1,) + grayImage.shape + (1,)) # (1,64,64,1)
    image = image.astype('float32') / 255.
    return image

def printMetrix(model, x_test, y_test_labels):
    Y_pred = model.predict(x_test)
    print(Y_pred)
    y_pred = np.floor(Y_pred + np.array(0.5)).ravel() # np.argmax(Y_pred, axis=1)
    print(y_pred)
    y_pred = model.predict_classes(x_test)
    print(y_pred.ravel())
    p = model.predict_proba(x_test) # to predict probability
    target_names = ['class 0(REAL)', 'class 1(FAKE)']
    print(classification_report(y_test_labels, y_pred, target_names=target_names))
    # Tp | Fp
    #---------
    # Fn | Tn
    print(confusion_matrix(y_test_labels, y_pred))

def trainDiscrModel(x_trein_real, x_train_fake):
    y_train_real = [0.] * 25
    y_train_fake = [1.] * 24
    y_train_labels = y_train_real + y_train_fake # 49 real + fake
    x_train = np.concatenate([x_trein_real, x_train_fake])
    dis_model.epochs = EPOCH_DIS
    model = dis_model.load(x_train=x_train, y_train=y_train_labels, x_test=x_train, y_test=y_train_labels)
    return model

def testDiscrModel(model, x_test_real, x_test_fake):
    x_test = np.concatenate([x_test_real, x_test_fake])
    y_test_real = [0.] * 25
    y_test_fake = [1.] * 25
    y_test_labels = y_test_real + y_test_fake
    test_score = model.evaluate(x_test, y_test_labels, verbose=0) # for test
    print('Test loss:', test_score[0])
    print('Test accuracy:', test_score[1])
    print("Baseline CONV Error: %.2f%%" % (100-test_score[1]*100))
    printMetrix(model, x_test, y_test_labels)

train_images = glob("/Users/Maria/Documents/input_faces/train/*.jpg") # 25 images now
test_images = glob("/Users/Maria/Documents/input_faces/test/*.jpg") # 25 images now
load_x_train_real = [load_image(image) for image in train_images] # pixels are [0..1], size of image is (1,64,64,1)
load_x_test_real = [load_image(image) for image in test_images] # pixels are [0..1], size of image is (1,64,64,1)

# first is real images

x_train_real_image = np.concatenate(load_x_train_real) # all train images as np.array()
x_test_real_image = np.concatenate(load_x_test_real) # all test images as np.array()

#--------------------

all_real_images = np.concatenate([x_train_real_image, x_test_real_image]) # all real images (train + test), shape is (50, 64, 64, 1)

print('--------------------------------------------------------------------------------------------------------')
print('-------------------------------------------------- CNN -------------------------------------------------')

conv_VAE.Settings.num_of_epoch = 1

vae = conv_VAE.VAE()
vae_model = vae.get_VAE_model()
encoder = vae.get_encoder_model()
decoder = vae.get_decoder_model()

vae_model.compile(optimizer='rmsprop', loss=vae.loss)
vae_model.summary() # log
vae_model.fit(all_real_images, all_real_images,
              shuffle=True,
              epochs=conv_VAE.Settings.num_of_epoch,
              batch_size=conv_VAE.Settings.batch_size,
              validation_data=(all_real_images, all_real_images))

n = 7 # 49 fake images
img_size = 64
grid_x = np.linspace(-1.0, 1.0, n)
grid_y = np.linspace(-1.0, 1.0, n)

generated_images_conv = [] # array of NORM ([0..1]) fake images

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, conv_VAE.Settings.batch_size).reshape(conv_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=conv_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # get norm image (64,64)
        misc.imsave('/Users/Maria/Documents/input_faces/fake_cnn/%d%d.jpg' % (i,j), img)
        img = img.reshape((1,) + img.shape + (1,)) # size is (1,64,64,1)
        generated_images_conv.append(img)

fake_cnn_train = generated_images_conv[0:24] # 24 images
fake_cnn_test = generated_images_conv[-25:] # 25 images

fake_cnn_train = np.concatenate(fake_cnn_train) # np.array of NORM ([0..1]) fake images for train
fake_cnn_test = np.concatenate(fake_cnn_test) # np.array of NORM ([0..1]) fake images for test

print('--------------------------------------------------------------------------------------------------------')
print('-------------------------------------------------- FN --------------------------------------------------')

load_x_train_dense = [dense_VAE.load_image(image) for image in train_images] # size of image is (1, 64*64*chnl) = (1, 4096)
load_x_test_dense = [dense_VAE.load_image(image) for image in test_images] # size of image is (1, 64*64*chnl) = (1, 4096)

x_train_dense = np.concatenate(load_x_train_dense) # fn train images as np.array() (size is (1,4096))
x_test_dense = np.concatenate(load_x_test_dense) # fn train images as np.array() (size is (1,4096))

all_real_images_dense = np.concatenate([x_train_dense, x_test_dense])

dense_VAE.Settings.num_of_epoch = EPOCH_VAE

vae = dense_VAE.VAE()
vae_model = vae.get_VAE_model()
encoder = vae.get_encoder_model()
decoder = vae.get_decoder_model()

vae_model.compile(optimizer='rmsprop', loss=vae.loss)
vae_model.summary() # log
vae_model.fit(all_real_images_dense, all_real_images_dense,
              shuffle=True,
              epochs=dense_VAE.Settings.num_of_epoch,
              batch_size=dense_VAE.Settings.batch_size,
              validation_data=(all_real_images_dense, all_real_images_dense))

n = 7 # 49 fake images
img_size = 64
grid_x = np.linspace(-3.0, 3.0, n)
grid_y = np.linspace(-3.0, 3.0, n)

generated_images_dense = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, dense_VAE.Settings.batch_size).reshape(dense_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=dense_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        misc.imsave('/Users/Maria/Documents/input_faces/fake_fn/%d%d.jpg' % (i,j), img)
        # SHOW IMAGES
        img = img.reshape((1,) + img.shape + (1,)) # (1,64,64,1)
        generated_images_dense.append(img)

fake_fn_train = generated_images_dense[0:24] # 24 train NORM images
fake_fn_test = generated_images_dense[-25:] # 25 test NORM images

fake_fn_train = np.concatenate(fake_fn_train)
fake_fn_test = np.concatenate(fake_fn_test)

print('--------------------------------------------------------------------------------------------------------')
print('-------------------------------------------------- CNN -------------------------------------------------')

model_cnn = trainDiscrModel(x_trein_real=x_train_real_image,
                            x_train_fake=fake_cnn_train)

testDiscrModel(model=model_cnn,
               x_test_real=x_test_real_image,
               x_test_fake=fake_cnn_test)

print('--------------------------------------------------------------------------------------------------------')
print('-------------------------------------------------- FN --------------------------------------------------')

model_fn = trainDiscrModel(x_trein_real=x_train_real_image,
                           x_train_fake=fake_fn_train)

testDiscrModel(model=model_fn,
               x_test_real=x_test_real_image,
               x_test_fake=fake_fn_test)

