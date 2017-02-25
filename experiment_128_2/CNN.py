from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

IMAGE_SIZE = 128
NUMBER_OF_EPOCH = 100

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
    
    def __exit__(self, type, value, traceback):
        print "Elapsed time: {:.3f} sec".format(time.time() - self._startTime)

def showImages(first, second):
    plt.figure(figsize=(8, 6))
    # first
    ax = plt.subplot(1, 2, 1)
    plt.imshow(first.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # second
    ax = plt.subplot(1, 2, 2)
    plt.imshow(second.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # show
    plt.show()

def foundCorrelation(firstArray2D, secondArray2D):
    res_x_train = firstArray2D.reshape(IMAGE_SIZE, IMAGE_SIZE).ravel() # 1D array
    res_decoded_image = secondArray2D.reshape(IMAGE_SIZE, IMAGE_SIZE).ravel() # 1D array
    correlation = np.linalg.norm(res_x_train - res_decoded_image) # euclidean distance
    print("%.5f" % correlation)

def createTrainDataFromImage(imagePath):
    img = cv2.imread(imagePath)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # type: numpy.ndarray
    x_train = img_to_array(grayImg) # this is a Numpy array
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((1,) + x_train.shape)
    return x_train

def createModel():
    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
    return Model(input_img, decoded)

def train(model, train_data, nb_epoch):
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    with Profiler() as p:
        autoencoder.fit(train_data,
                        train_data,
                        nb_epoch=nb_epoch,
                        batch_size=1,
                        shuffle=True,
                        validation_data=(train_data, train_data),
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]) # log: tensorboard --logdir=/tmp/autoencoder

if __name__ == '__main__':
    # create and train model
    autoencoder = createModel()
    own_train_data = createTrainDataFromImage('/Users/Maria/Documents/FaceTransfer/input_images/128.jpg')
    train(model=autoencoder, train_data=own_train_data, nb_epoch=NUMBER_OF_EPOCH)
    decoded_imgs = autoencoder.predict(own_train_data)
    # found correlation
    foundCorrelation(own_train_data[0], decoded_imgs[0])
    # show images
    showImages(own_train_data[0], decoded_imgs[0])


