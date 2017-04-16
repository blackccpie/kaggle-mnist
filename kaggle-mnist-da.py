import pandas as pd
import numpy as np
np.random.seed(1337) # for reproducibility

from keras import backend as K
from keras.models import Model
from keras.layers import Input, UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
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
test  = pd.read_csv("mnist/test.csv").values

print('train shape:', train.shape)
print('test shape:', test.shape)

# Reshape the data to be used by a Theano CNN. Shape is
# (nb_of_samples, nb_of_color_channels, img_width, img_heigh)
X_train = train[:, 1:].reshape(train.shape[0], 1, img_rows, img_cols)
X_test = test[:, 1:].reshape(test.shape[0], 1, img_rows, img_cols)
in_shape = (1, img_rows, img_cols)

# Make the value floats in [0;1] instead of int in [0;255]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#Display the shapes to check if everything's ok
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

input_img = Input(shape=in_shape)  # adapt this if using non `channels_first` image data format

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print_summary(autoencoder.layers)

autoencoder.fit(X_train, X_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test))
