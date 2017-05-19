from __future__ import print_function
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
import keras

batch_size = 5
num_classes = 1 # real or fake image
epochs = 100

# input image dimensions
img_rows, img_cols = 64, 64
full_img_size = (img_rows, img_cols, 1)

def createModel():
    model = Sequential()
    model.add(Conv2D(filters=1,
                        kernel_size=(7, 7),
                        padding='same',
                        input_shape=full_img_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
#    model.add(Dense(256))
#    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(filters=16,
#                     kernel_size=(5, 5),
#                     padding='same'))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(filters=16,
#                     kernel_size=(5, 5),
#                     padding='same'))
#    model.add(Activation('relu'))
#    model.add(Flatten())
#    model.add(Dropout(0.5))
#    model.add(Dense(num_classes))
#    model.add(Activation('sigmoid'))
    return model

def load(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
#    x_train = x_train.astype('float32')
#    x_test = x_test.astype('float32')
#    x_train /= 255
#    x_test /= 255
    model = createModel()
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=1.9)
    model.compile(loss=keras.losses.binary_crossentropy,
                    optimizer="rmsprop",
                    metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Baseline Error: %.2f%%" % (100-score[1]*100))
    return model
