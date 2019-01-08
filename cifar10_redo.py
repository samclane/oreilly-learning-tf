import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = "./cifar-10-batches-py/"


def model_fn(x, target, mode, params):
    y_ = tf.cast(target, tf.float32)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Conv layer 1
    conv1 = layers.convolution2d(x_image, 32, [5, 5],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.1),
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    pool1 = layers.max_pool2d(conv1, [2, 2])

    # Conv layer 2
    conv2 = layers.convolution2d(pool1, 64, [5, 5],
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.1),
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    pool2 = layers.max_pool2d(conv2, [2, 2])

    # FC Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc1 = layers.fully_connected(pool2_flat, 1024,
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.1),
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc1_drop = layers.dropout(fc1, keep_prob=params["dropout"],
                              is_training=(mode == 'train'))

    # Readout layer
    y_conv = layers.fully_connected(fc1_drop, 10, activation_fn=None)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_op = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="Adam"
    )

    predictions = tf.argmax(y_conv, 1)
    return predictions, cross_entropy, train_op


if __name__ == "__main__":
    data = input_data.read_data_sets(DATA_DIR, one_hot=True)
    x_data, y_data = data.train.images, np.int32(data.train.labels)
    tf.cast(x_data, tf.float32)
    tf.cast(y_data, tf.float32)

    model_params = {"learning_rate": 1e-4, "dropout": 0.5}

    CNN = tf.contrib.learn.Estimator(
        model_fn=model_fn, params=model_params
    )

    print("Starting training for %s steps max" % 5000)
    CNN.fit(x=data.train.images,
            y=data.train.labels, batch_size=50,
            max_steps=5000)

    test_acc = 0

    for ii in range(5):
        batch = data.test.next_batch(2000)
        predictions = list(CNN.predict(batch[0], as_iterable=True))
        test_acc = test_acc + (np.argmax(batch[1], 1) == predictions).mean()

    print(test_acc/5)
