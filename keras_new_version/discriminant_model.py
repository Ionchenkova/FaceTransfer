from __future__ import print_function
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 10
num_classes = 2 # real or fake image
epochs = 100

# input image dimensions
img_rows, img_cols = 64, 64
full_img_size = (img_rows, img_cols, 1)

def createModel():
    input_layer = Input(shape=full_img_size)
    x = Conv2D(filters=8, 
                kernel_size=(5, 5), 
                padding='same', 
                activation='relu')(x) # 64x64
    x = MaxPooling2D(pool_size=(2, 2))(x) # 32x32
    x = Conv2D(filters=16,
                kernel_size=(5, 5), 
                padding='same', 
                activation='relu')(x) # 32x32
    x = MaxPooling2D(pool_size=(2, 2))(x) # 16x16
    x = Conv2D(filters=16, 
                kernel_size=(5, 5), 
                padding='same', 
                activation='relu')(x) # 16x16
    x = Flatten()(x) # 256
    x = Dropout(0.5)(x)
    decoded = Dense(num_classes, activation='softmax')(x)
    return Model(input_layer, decoded)

def load(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = createModel()
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
