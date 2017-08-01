'''
EMNIST Digit recognizer using Keras
keras 1.2.2
'''

import pandas as pd
import numpy as np
np.random.seed(1337) # for reproducibility

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import Callback, RemoteMonitor
from keras.utils import np_utils

# enable multi-CPU
#import theano
#theano.config.openmp = True

monitor = RemoteMonitor(root='http://localhost:9000')

# input image dimensions
img_rows, img_cols = 28, 28

batch_size = 128 # Number of images used in each optimization step
nb_classes = 62 # One class per digit/lowercase letter/uppercase letter
nb_epoch = 70 # Number of times the whole data is used to learn

# Read the train and test datasets
train = pd.read_csv("emnist/train.csv").values
test  = pd.read_csv("emnist/test.csv").values

print('train shape:', train.shape)
print('test shape:', test.shape)

# Reshape the data to be used by a Theano CNN. Shape is
# (nb_of_samples, nb_of_color_channels, img_width, img_heigh)
X_train = train[:, 1:].reshape(train.shape[0], 1, img_rows, img_cols)
X_test = test[:, 1:].reshape(test.shape[0], 1, img_rows, img_cols)
in_shape = (1, img_rows, img_cols)
y_train = train[:, 0] # First data is label (already removed from X_train)
y_test = test[:, 0] # First data is label (already removed from Y_train)

# Make the value floats in [0;1] instead of int in [0;255]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices (ie one-hot vectors)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Display the shapes to check if everything's ok
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

model = Sequential()
# For an explanation on conv layers see http://cs231n.github.io/convolutional-networks/#conv
# By default the stride/subsample is 1 and there is no zero-padding.
# If you want zero-padding add a ZeroPadding layer or, if stride is 1 use border_mode="same"
model.add(Convolution2D(15, 5, 5, activation = 'relu', input_shape=in_shape, init='he_normal'))

# For an explanation on pooling layers see http://cs231n.github.io/convolutional-networks/#pool
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(30, 5, 5, activation = 'relu', init='he_normal'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 3D output to 1D tensor for a fully connected layer to accept the input
model.add(Flatten())
model.add(Dense(250, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(125, activation = 'relu', init='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation = 'softmax', init='he_normal')) #Last layer with one output per class

# The function to optimize is the cross entropy between the true label and the output (softmax) of the model
# We will use adadelta to do the gradient descent see http://cs231n.github.io/neural-networks-3/#ada
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])

model.summary()

# Make the model learn
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[monitor,TestCallback((X_test, Y_test))], verbose=1)

# Predict the label for X_test
yPred = model.predict_classes(X_test)

# Save prediction in file for Kaggle submission
np.savetxt('emnist-pred.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
