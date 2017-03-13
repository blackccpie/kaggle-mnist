## kaggle mnist

Inspired from:
[Keras CNN inspired by LeNet-5](http://www.kaggle.com/ftence/digit-recognizer/keras-cnn-inspired-by-lenet-5)

##### The dataset

The used dataset is based on the [official Kaggle MNIST dataset](https://www.kaggle.com/c/digit-recognizer/data), except that true classifications have been incorporated into *test.csv* in order to compute testing scores.

This has been possible given the fact that the kaggle dataset is based on the [original MNIST dataset](http://yann.lecun.com/exdb/mnist), except that the total 70,000 samples has been splitted differently: 42,000 train samples + 28,000 test samples instead of 60,000 train samples + 10,000 test samples.
