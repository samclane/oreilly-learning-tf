import tensorflow as tf
from tensorflow.contrib import learn, layers
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target

def part1():
    global boston, x_data, y_data
    x = tf.placeholder(tf.float64, shape=(None, 13))
    y_true = tf.placeholder(tf.float64, shape=(None))

    with tf.name_scope('inference') as scope:
        w = tf.Variable(tf.zeros([1, 13], dtype=tf.float64, name='weights'))
        b = tf.Variable(0, dtype=tf.float64, name='bias')
        y_pred = tf.matmul(w, tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    with tf.name_scope('train') as scope:
        learning_rate = 0.1
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(200):
            sess.run(train, {x: x_data, y_true: y_data})

        MSE = sess.run(loss, {x: x_data, y_true: y_data})
    print(MSE)

####

def part2():
    global boston, x_data, y_data
    NUM_STEPS = 200
    MINIBATCH_SIZE = 506

    feature_columns = learn.infer_real_valued_columns_from_input(x_data)

    reg = learn.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
    )

    reg.fit(x_data, boston.target, steps=NUM_STEPS, batch_size=MINIBATCH_SIZE)

    MSE = reg.evaluate(x_data, boston.target, steps=1)

    print(MSE)

    ###


def part3():
    global boston, x_data, y_data
    import sys
    import numpy as np
    from tensorflow.examples.tutorials.mnist import input_data
    DATA_DIR = 'c:\\tmp\\data'
    data = input_data.read_data_sets(DATA_DIR, one_hot=False)
    x_data, y_data = data.train.images, data.train.labels.astype(np.int32)
    x_test, y_test = data.test.images, data.test.labels.astype(np.int32)

    NUM_STEPS = 2000
    MINIBATCH_SIZE = 128

    feature_columns = learn.infer_real_valued_columns_from_input(x_data)

    dnn = learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[200],
        n_classes=10,
        optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.2)
    )

    dnn.fit(x=x_data, y=y_data, steps=NUM_STEPS, batch_size=MINIBATCH_SIZE)

    test_acc = dnn.evaluate(x=x_test, y=y_test, steps=1)["accuracy"]
    print(f"test accuracy {test_acc}")

    from sklearn.metrics import confusion_matrix

    y_pred = dnn.predict(x=x_test, as_iterable=False)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)

def part4():
    global boston, x_data, y_data
    import pandas as pd
    import numpy as np
    N = 10000

    weight = np.random.randn(N)*5+70
    spec_id = np.random.randint(0, 3, N)
    bias = [0.9, 1, 1.1]
    height = np.array([weight[i]/100 + bias[b] for i, b in enumerate(spec_id)])
    spec_name = ['Goblin', 'Human', 'ManBear']
    spec = [spec_name[s] for s in spec_id]

    df = pd.DataFrame({'Species': spec, 'Weight': weight, 'Height': height})

    from tensorflow.contrib import layers
    Weight = layers.real_valued_column("Weight")
    Species = layers.sparse_column_with_keys(column_name="Species", keys=spec_name)
    reg = learn.LinearRegressor(feature_columns=[Weight, Species])

    def input_fn(df):
        feature_cols = {}
        feature_cols['Weight'] = tf.constant(df['Weight'].values)

        feature_cols['Species'] = tf.SparseTensor(
            indices=[[i, 0] for i in range(df['Species'].size)],
            values=df['Species'].values,
            dense_shape=[df['Species'].size, 1]
        )

        labels = tf.constant(df['Height'].values)

        return feature_cols, labels

    reg.fit(input_fn=lambda: input_fn(df), steps=50000)

    w_w = reg.get_variable_value('linear/Weight/weight')
    print(f"Estimation for Weight: {w_w}")

    v = reg.get_variable_names()
    print(f"Classes: {v}")

    s_w = reg.get_variable_value('linear/Species/weights')
    b = reg.get_variable_value('linear/bias_weight')
    print(f"Estimation for Species: {s_w + b}")


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
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    fc1 = layers.fully_connected(pool2_flat, 1024,
                                 activation_fn=tf.nn.relu,
                                 biases_initializer=tf.constant_initializer(0.1),
                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc1_drop = layers.dropout(fc1, keep_prob=["dropout"],
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

    preditions = tf.argmax(y_conv, 1)
    return preditions, cross_entropy, train_op



if __name__ == "__main__":
    part4()