'''
Denoising Autoencoder on MNIST using Keras
keras 2.0.6
'''

import pandas as pd
import numpy as np
np.random.seed(1337) # for reproducibility

from keras import backend as K
from keras.models import Model
from keras.layers import Input, UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import Callback, RemoteMonitor
from keras.utils import np_utils
from keras.utils.layer_utils import print_summary

# enable multi-CPU
import theano
theano.config.openmp = True

monitor = RemoteMonitor(root='http://localhost:9000')

# input image dimensions
img_rows, img_cols = 28, 28

batch_size = 128 # Number of images used in each optimization step
nb_classes = 10 # One class per digit
nb_epoch = 70 # Number of times the whole data is used to learn

# Read the train and test datasets
train = pd.read_csv("mnist/train.csv").values
test = train # output = input

print('train shape:', train.shape)
print('test shape:', test.shape)

# Reshape the data to be used by a Theano CNN. Shape is
# (nb_of_samples, nb_of_color_channels, img_width, img_heigh)
X_train = train[:, 1:].reshape(train.shape[0], 1, img_rows, img_cols)
X_test = test[:, 1:].reshape(test.shape[0], 1, img_rows, img_cols)
in_shape = (1, img_rows, img_cols)

print('in shape:', in_shape)

# Make the value floats in [0;1] instead of int in [0;255]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#Display the shapes to check if everything's ok
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

input_img = Input(shape=in_shape)  # adapt this if using non `channels_first` image data format

x = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, kernel_size=(3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, kernel_size=(3,3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print_summary(autoencoder)

autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test))
