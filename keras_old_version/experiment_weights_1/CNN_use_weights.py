from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
#from keras.layers.core import Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.optimizers import SGD
#from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2

IMAGE_SIZE = 128
NUMBER_OF_EPOCH = 10

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
    
    def __exit__(self, type, value, traceback):
        print "Elapsed time: {:.3f} sec".format(time.time() - self._startTime)

def showImages(first, second, third):
    # first
    plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 3, 1)
    plt.imshow(first.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # second
    ax = plt.subplot(1, 3, 2)
    plt.imshow(second.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # third
    ax = plt.subplot(1, 3, 3)
    plt.imshow(third.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # show
    plt.show()

def createTrainDataFromImage(imagePath):
    img = cv2.imread(imagePath)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # type: numpy.ndarray
    x_train = img_to_array(grayImg) # this is a Numpy array
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((1,) + x_train.shape)
    return x_train

def foundCorrelation(firstArray2D, secondArray2D):
    res_x_train = firstArray2D.reshape(IMAGE_SIZE, IMAGE_SIZE).ravel() # 1D array
    res_decoded_image = secondArray2D.reshape(IMAGE_SIZE, IMAGE_SIZE).ravel() # 1D array
    correlation = np.linalg.norm(res_x_train - res_decoded_image) # euclidean distance
    print("%.5f" % correlation)

def createModel():
    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    # print input_img.get_weights()
    x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Dropout(0.3)(x)
    x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Dropout(0.3)(x)
    x = Convolution2D(8, 9, 9, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    encoded = Dropout(0.3)(encoded)
    x = Convolution2D(8, 9, 9, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 9, 9, activation='sigmoid', border_mode='same')(x)
    return Model(input_img, decoded)

def createModelFromModel(model):
    result_model = createModel()
    print 'autoencoder layers: ', len(model.layers)
    for i in range(len(model.layers)):
        #print 'Layer : ', i, ' is ', model.layers[i].get_weights()
        result_model.layers[i].set_weights(model.layers[i].get_weights())
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    result_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return result_model

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
    # use weights for second image
    res_train = createTrainDataFromImage('/Users/Maria/Documents/FaceTransfer/input_images/mila_128.jpg')
    result_model = createModelFromModel(autoencoder)
    result_img = result_model.predict(res_train)
    # show images
    showImages(first=own_train_data[0], second=decoded_imgs[0], third=result_img[0])
    # plot(autoencoder, to_file='/Users/Maria/Documents/FaceTransfer/model.png', show_layer_names=False, show_shapes=True)
