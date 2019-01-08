import tensorflow as tf
from tensorflow.contrib import slim

W = slim.variable('w', shape=[7, 7, 3, 3], initializer=tf.truncated_normal_initializer(stddev=0.1),
                  regularizer=slim.l2_regularizer(0.07),
                  device='/CPU:0')

net = slim.conv2d(inputs, 64, [11, 11], 4, padding='SAME',
                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0007), scope='conv1')

net = slim.repeat(net, 5, slim.conv2d, 128, [3, 3], scope='con1')

slim.stack(net, slim.conv2d, [(64, [3, 3]), (64, [1, 1]), (128, [3, 3]), (128, [1, 1]), (256, [3, 3])], scope='con')

with slim.arg_scope([slim.conv2d],
                    padding='VALID',
                    activation=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    weights_regularizer=slim.l2_regularizer(0.0007)):
    net = slim.conv2d(inputs, 64, [11, 11], scope='con1')
    net = slim