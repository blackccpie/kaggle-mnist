## kaggle mnist

Inspiration:
[Keras CNN inspired by LeNet-5](http://www.kaggle.com/ftence/digit-recognizer/keras-cnn-inspired-by-lenet-5)

### The datasets

#### mnist

The used dataset is based on the [official Kaggle MNIST dataset](https://www.kaggle.com/c/digit-recognizer/data), except that true classifications have been incorporated into *test.csv* in order to compute testing scores.

This has been possible given the fact that the kaggle dataset is based on the [original MNIST dataset](http://yann.lecun.com/exdb/mnist), except that the total 70,000 samples has been splitted differently: 42,000 train samples + 28,000 test samples instead of 60,000 train samples + 10,000 test samples.

#### emnist

EMNIST stands for Extended MNIST dataset. It constitutes a more challenging classification task involving letters and digits, and that shares the same image structure and parameters as the original MNIST task, allowing for direct compatibility with all existing classifiers and systems.
The used dataset is based on the [official EMNIST dataset](https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist), which has been convert to kaggle like csv file format.

### Visualizing training

I'm using [hualos](https://github.com/blackccpie/hualos) as a remote monitor, allowing to view the training progression on a web page (http://localhost:9000/).
