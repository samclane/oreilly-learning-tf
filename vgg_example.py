import sys
sys.path.append(r"./models/research/slim")

from datasets import dataset_utils
import tensorflow as tf
target_dir = r"vgg/vgg_checkpoints"

import requests
url = ("https://i.stack.imgur.com/o1z7p.jpg")
im_as_string = requests.get(url).content
image = tf.image.decode_jpeg(im_as_string, channels=3)

from nets import vgg
image_size = vgg.vgg_16.default_image_size

from preprocessing import vgg_preprocessing
processed_im = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)

processed_images = tf.expand_dims(processed_im, 0)

from tensorflow.contrib import slim
with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
probabilities = tf.nn.softmax(logits)

import os

load_vars = slim.assign_from_checkpoint_fn(os.path.join(target_dir, 'vgg_16.ckpt'), slim.get_model_variables('vgg_16'))

from datasets import imagenet
imagenet.create_readable_names_for_imagenet_labels()

import numpy as np
names = []
with tf.Session() as sess:
    load_vars(sess)
    network_input, probabilities = sess.run([processed_images, probabilities])
    probabilities = probabilities[0, 0:]
    names_ = imagenet.create_readable_names_for_imagenet_labels()
    idxs = np.argsort(-probabilities)[:5]
    probs = probabilities[idxs]
    classes = np.array(list(names_.values()))[idxs + 1]
    for c, p in zip(classes, probs):
        print("Class: " + c + "|Prob: " + str(p))