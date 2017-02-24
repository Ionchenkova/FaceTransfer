import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import time

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
    
    def __exit__(self, type, value, traceback):
        print "Elapsed time: {:.3f} sec".format(time.time() - self._startTime)

IMAGE_SIZE = 128

input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(input_img)
print(x.shape)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
print(x.shape)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
print(x.shape)

encoded = MaxPooling2D((2, 2), border_mode='same')(x)

x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 9, 9, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Convolution2D(1, 9, 9, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

img = cv2.imread('/Users/Maria/Documents/FaceTransfer/input_images/128.jpg')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # type: numpy.ndarray

x_train = img_to_array(grayImg)  # this is a Numpy array with shape (28, 28, 1)
print(x_train.shape)
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((1,) + x_train.shape)  # this is a Numpy array with shape (1, 28, 28, 1)
print(x_train.shape)

with Profiler() as p:
    autoencoder.fit(x_train,
                x_train,
                nb_epoch=100,
                batch_size=1,
                shuffle=True,
                validation_data=(x_train, x_train),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]) # log: tensorboard --logdir=/tmp/autoencoder

decoded_imgs = autoencoder.predict(x_train)

res_x_train = x_train[0].reshape(IMAGE_SIZE, IMAGE_SIZE).ravel() # 1D array
res_decoded_image = decoded_imgs[0].reshape(IMAGE_SIZE, IMAGE_SIZE).ravel() # 1D array

# found correlation

correlation = np.linalg.norm(res_x_train - res_decoded_image) # euclidean distance
print("%.5f" % correlation)

# show images

plt.figure(figsize=(8, 6))
ax = plt.subplot(1, 2, 1)
plt.imshow(x_train[0].reshape(IMAGE_SIZE, IMAGE_SIZE)) # x_train[0].reshape(32, 32)
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(1, 2, 2)
plt.imshow(decoded_imgs[0].reshape(IMAGE_SIZE, IMAGE_SIZE)) # decoded_imgs[0].reshape(32, 32)
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()


