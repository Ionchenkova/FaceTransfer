from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.optimizers import SGD, Adagrad
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2

IMAGE_SIZE = 128
NUMBER_OF_EPOCH = 500

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
    
    def __exit__(self, type, value, traceback):
        print "Elapsed time: {:.3f} sec".format(time.time() - self._startTime)

def showImages(first, second, third, fourth):
    # first
    plt.figure(figsize=(8, 3))
    ax = plt.subplot(1, 4, 1)
    plt.imshow(first.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # second
    ax = plt.subplot(1, 4, 2)
    plt.imshow(second.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # third
    ax = plt.subplot(1, 4, 3)
    plt.imshow(third.reshape(IMAGE_SIZE, IMAGE_SIZE))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # fourth
    ax = plt.subplot(1, 4, 4)
    plt.imshow(fourth.reshape(IMAGE_SIZE, IMAGE_SIZE))
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

def createFirstModel():
    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = Convolution2D(8, 11, 11, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
    return Model(input_img, decoded)

def createSecondModel():
    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = Convolution2D(8, 11, 11, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Dropout(0.3)(x)
    x = Convolution2D(16, 9, 9, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Dropout(0.3)(x)
    x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
    return Model(input_img, decoded)

def createModelFromModel(model):
    result_model = createSecondModel()
    print 'autoencoder layers: ', len(model.layers)
    print 'new layers: ', len(result_model.layers)
    new_weights = [layer.get_weights() for layer in model.layers]
    new_weights.insert(3, [])
    new_weights.insert(6, [])
    for i, j in zip(new_weights, range(len(new_weights))):
        result_model.layers[j].set_weights(i)
    adagrad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
    result_model.compile(optimizer=adagrad, loss='categorical_crossentropy')
    return result_model

def train(model, train_data_x, train_data_y, nb_epoch, test_data):
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    with Profiler() as p:
        autoencoder.fit(train_data_x,
                train_data_y,
                nb_epoch=nb_epoch,
                batch_size=1,
                shuffle=True,
                validation_data=(test_data, test_data),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]) # log: tensorboard --logdir=/tmp/autoencoder

if __name__ == '__main__':
    # create and train model
    autoencoder = createFirstModel()
    own_train_data_x = createTrainDataFromImage('/Users/Maria/Documents/FaceTransfer/input_images/boy_128.jpg')
    own_train_data_y = createTrainDataFromImage('/Users/Maria/Documents/FaceTransfer/input_images/boy_128.jpg')
    own_test_data = createTrainDataFromImage('/Users/Maria/Documents/FaceTransfer/input_images/girl_128.jpg')
    #two_own_train_data_x = own_train_data_x + own_test_data

    train(model=autoencoder, train_data_x=own_train_data_x, train_data_y=own_train_data_x, test_data=own_test_data, nb_epoch=300)
    train(model=autoencoder, train_data_x=own_test_data, train_data_y=own_train_data_x, test_data=own_test_data, nb_epoch=5)
    train(model=autoencoder, train_data_x=own_train_data_x, train_data_y=own_test_data, test_data=own_train_data_x, nb_epoch=2)
    train(model=autoencoder, train_data_x=own_test_data, train_data_y=own_train_data_x, test_data=own_test_data, nb_epoch=5)

    decoded_img = autoencoder.predict(own_train_data_x)
    result_img = autoencoder.predict(own_test_data)
    showImages(first=own_train_data_x[0], second=decoded_img[0], third=own_test_data[0], fourth=result_img[0])
