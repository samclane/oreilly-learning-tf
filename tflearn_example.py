import tflearn
import numpy as np

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist

X, Y, X_test, Y_test = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
X_test = X_test.reshape([-1, 28, 28, 1])

# Building the network
CNN = input_data(shape=[None, 28, 28, 1], name='input')
CNN = conv_2d(CNN, 32, 5, activation='relu', regularizer='L2')
CNN = max_pool_2d(CNN, 2)
CNN = local_response_normalization(CNN)
CNN = conv_2d(CNN, 64, 5, activation='relu', regularizer='L2')
CNN = max_pool_2d(CNN, 2)
CNN = local_response_normalization(CNN)
CNN = fully_connected(CNN, 1024, activation=None)
CNN = dropout(CNN, 0.5)
CNN = fully_connected(CNN, 10, activation='softmax')
CNN = regression(CNN, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='target')

# Training the network
model = tflearn.DNN(CNN, tensorboard_verbose=0, tensorboard_dir='MNIST_tflearn_board/',
                    checkpoint_path='MNIST_tflearn_checkpoints/checkpoint')
model.fit({'input': X}, {'target': Y}, n_epoch=3, validation_set=({'input': X_test}, {'target': Y_test}),
          snapshot_step=1000, show_metric=True, run_id='convnet_mnist')
evaluation = model.evaluate({'input': X_test}, {'target': Y_test})
print(evaluation)

pred = model.predict({'input': X_test})
print((np.argmax(Y_test, 1)==np.argmax(pred, 1)).mean())