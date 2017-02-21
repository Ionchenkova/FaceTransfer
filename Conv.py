import cv2
import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

input_img = Input(shape=(28,28,1))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


img = cv2.imread('/Users/Maria/Documents/FaceTransfer/28.jpg')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # type: numpy.ndarray

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

x_train = img_to_array(grayImg)  # this is a Numpy array with shape (28, 28, 1)
print(x_train.shape)
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((1,) + x_train.shape)  # this is a Numpy array with shape (1, 28, 28, 1)
print(x_train.shape)

from keras.callbacks import TensorBoard

autoencoder.fit(x_train,
                x_train,
                nb_epoch=5000,
                batch_size=128,
                shuffle=True,
                validation_data=(x_train, x_train),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


decoded_imgs = autoencoder.predict(x_train)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
ax = plt.subplot(1, 2, 1)
plt.imshow(x_train[0].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(1, 2, 2)
plt.imshow(decoded_imgs[0].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()


