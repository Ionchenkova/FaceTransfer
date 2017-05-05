import densely_connected_layers.VAE_densely_connected as dense_VAE
import convolutional_layers.convolutional_VAE as conv_VAE
import discriminant_model as dis_model
from sklearn.metrics import classification_report,confusion_matrix
import sys
import keras
import numpy as np

from glob import glob

# 0 is real
# 1 is fake

EPOCH_VAE = 1
EPOCH_DIS = 1

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
    # Tp | Fn
    #---------
    # Fp | Tn
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
    print('Test CONV loss:', test_score[0])
    print('Test CONV accuracy:', test_score[1])
    print("Baseline CONV Error: %.2f%%" % (100-test_score[1]*100))
    printMetrix(model, x_test, y_test_labels)

print('--------------------------------------------------------------------------------------------------------')
print('-------------------------------------------------- CNN -------------------------------------------------')

train_images = glob("/Users/Maria/Documents/input_faces/train/*.jpg") # 25 images now
test_images = glob("/Users/Maria/Documents/input_faces/test/*.jpg") # 25 images now
load_x_train_conv = [conv_VAE.load_image(image) for image in train_images]
load_x_test_conv = [conv_VAE.load_image(image) for image in test_images]

# first is real images

x_train_conv = np.concatenate(load_x_train_conv)
print 'x_train_conv shape is ', x_train_conv.shape
x_test_conv = np.concatenate(load_x_test_conv)

y_train_conv = [1] * len(load_x_train_conv)
y_test_conv = [1] * len(load_x_test_conv)

#--------------------

all_real_images_conv = np.concatenate([x_train_conv, x_test_conv]) # all real images, shape is (50, 64, 64, 1)

#--------------------

conv_VAE.Settings.num_of_epoch = EPOCH_VAE

vae = conv_VAE.VAE()
vae_model = vae.get_VAE_model()
encoder = vae.get_encoder_model()
decoder = vae.get_decoder_model()

vae_model.compile(optimizer='rmsprop', loss=vae.loss)
vae_model.summary() # log
vae_model.fit(all_real_images_conv, all_real_images_conv,
              shuffle=True,
              epochs=conv_VAE.Settings.num_of_epoch,
              batch_size=conv_VAE.Settings.batch_size,
              validation_data=(all_real_images_conv, all_real_images_conv))

n = 7 # 49 fake images
img_size = 64
grid_x = np.linspace(-1.0, 1.0, n)
grid_y = np.linspace(-1.0, 1.0, n)

generated_images_conv = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, conv_VAE.Settings.batch_size).reshape(conv_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=conv_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        img = img.reshape((1,) + img.shape) # (1,64,64)
        img = img.astype('float32') / 255.
        img = img.reshape((img.shape[0],) + conv_VAE.Settings.full_img_size) # (1,64,64,1)
        generated_images_conv.append(img)

print 'generated_images_conv len is ', len(generated_images_conv)
fake_cnn_train = generated_images_conv[0:24] # 24 images
print 'fake_cnn_train len is ', len(fake_cnn_train)
fake_cnn_test = generated_images_conv[-25:] # 25 images
print 'fake_cnn_test len is ', len(fake_cnn_test)

fake_cnn_train = np.concatenate(fake_cnn_train)
fake_cnn_test = np.concatenate(fake_cnn_test)

print('--------------------------------------------------------------------------------------------------------')
print('-------------------------------------------------- FN --------------------------------------------------')

load_x_train_dense = [dense_VAE.load_image(image) for image in train_images]
load_x_test_dense = [dense_VAE.load_image(image) for image in test_images]

x_train_dense = np.concatenate(load_x_train_dense)
x_test_dense = np.concatenate(load_x_test_dense)

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
grid_x = np.linspace(-1.0, 1.0, n)
grid_y = np.linspace(-1.0, 1.0, n)

generated_images_dense = []

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, dense_VAE.Settings.batch_size).reshape(dense_VAE.Settings.batch_size, conv_VAE.Settings.latent_dim)
        x_decoded = decoder.predict(z_sample, batch_size=dense_VAE.Settings.batch_size)
        img = x_decoded[0].reshape(img_size, img_size) # (64,64)
        img = img.astype('float32') / 255.
        img = img.reshape((1,) + img.shape + (1,)) # (1,64,64,1)
        generated_images_dense.append(img)

fake_fn_train = generated_images_dense[0:24] # 24 train images
fake_fn_test = generated_images_dense[-25:] # 25 test images

fake_fn_train = np.concatenate(fake_fn_train)
fake_fn_test = np.concatenate(fake_fn_test)

print('--------------------------------------------------------------------------------------------------------')
print('-------------------------------------------------- CNN -------------------------------------------------')

model_cnn = trainDiscrModel(x_trein_real=x_train_conv,
                            x_train_fake=fake_cnn_train)

testDiscrModel(model=model_cnn,
               x_test_real=x_test_conv,
               x_test_fake=fake_cnn_test)

print('--------------------------------------------------------------------------------------------------------')
print('-------------------------------------------------- FN --------------------------------------------------')

model_fn = trainDiscrModel(x_trein_real=x_train_conv,
                           x_train_fake=fake_fn_train)

testDiscrModel(model=model_fn,
               x_test_real=x_test_conv,
               x_test_fake=fake_fn_test)

