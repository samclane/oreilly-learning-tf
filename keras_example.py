from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


# Load
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Get Automobile Pictures (label==1)
x_train = x_train[np.where(y_train==1)[0],:,:,:]
x_test = x_test[np.where(y_test==1)[0],:,:,:]

# Type conversion and normalization
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Add Gaussian noise
x_train_n = x_train + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_train.shape)
x_test_n = x_test + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_test.shape)

# Truncate noise'd data [0, 1]
x_train_n = np.clip(x_train_n, 0., 1.)
x_test_n = np.clip(x_test_n, 0., 1.)

# Input layer (image)
inp_img = Input(shape=(32, 32, 3))

# Autoencoder
img = Conv2D(32, (3, 3), activation='relu', padding='same')(inp_img)
img = MaxPooling2D((2, 2), padding='same')(img)
img = Conv2D(32, (3, 3), activation='relu', padding='same')(img)
img = UpSampling2D((2, 2))(img)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(img)

autoencoder = Model(inp_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

tensorboard = TensorBoard(log_dir='./keras_ex/logs', histogram_freq=0, write_graph=True, write_images=True)
model_saver = ModelCheckpoint(filepath='./keras_ex/checkpoint', verbose=0, period=2)

autoencoder.fit(x_train_n, x_train, epochs=10, batch_size=64, shuffle=True, validation_data=(x_test_n, x_test),
                callbacks=[tensorboard, model_saver])


n_imgs = 10
f, axarr = plt.subplots(2, n_imgs, figsize=[20, 5])
decoded_imgs = autoencoder.predict(x_test_n)

for i in range(n_imgs):
    ax = axarr[0, i]
    ax.get_yaxis().set_visible(False)
    ax.imshow(x_test_n[i, :, :, :])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = axarr[1, i]
    ax.get_yaxis().set_visible(False)
    ax.imshow(decoded_imgs[i, :, :, :])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()