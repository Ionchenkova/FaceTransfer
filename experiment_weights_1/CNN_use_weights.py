import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.optimizers import SGD
import time

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
    
    def __exit__(self, type, value, traceback):
        print "Elapsed time: {:.3f} sec".format(time.time() - self._startTime)

IMAGE_SIZE = 128

input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 9, 9, activation='relu', border_mode='same')(x)

encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 9, 9, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
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
                nb_epoch=1000,
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

# use weights for second image

print 'autoencoder layers: ', len(autoencoder.layers)
#for layer in autoencoder.layers:
#    print layer.get_weights()

img = cv2.imread('/Users/Maria/Documents/FaceTransfer/input_images/mila_128.jpg')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # type: numpy.ndarray

res_train = img_to_array(grayImg)  # this is a Numpy array with shape (28, 28, 1)
res_train = res_train.astype('float32') / 255.
res_train = res_train.reshape((1,) + res_train.shape)  # this is a Numpy array with shape (1, 28, 28, 1)

result_model = autoencoder

for i in range(len(autoencoder.layers)):
    result_model.layers[i].set_weights(autoencoder.layers[i].get_weights())

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
result_model.compile(optimizer=sgd, loss='categorical_crossentropy')

result_img = result_model.predict(res_train)

# show images

plt.figure(figsize=(8, 6))
ax = plt.subplot(1, 3, 1)
plt.imshow(x_train[0].reshape(IMAGE_SIZE, IMAGE_SIZE))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(1, 3, 2)
plt.imshow(decoded_imgs[0].reshape(IMAGE_SIZE, IMAGE_SIZE))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(1, 3, 3)
plt.imshow(result_img[0].reshape(IMAGE_SIZE, IMAGE_SIZE))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()


