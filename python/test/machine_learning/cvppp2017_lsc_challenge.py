# Path to libcudnn.so.5.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#%%------------------------------------------------------------------

import sys
sys.path.insert(0, '../../src/machine_learning')

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from loss import dice_coeff, dice_coeff_loss
from cvppp_image_loader import CvpppImageLoader
from unet_model import UNet

#%%------------------------------------------------------------------

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# This means that Keras will use the session we registered to initialize all variables that it creates internally.
K.set_session(sess)
K.set_learning_phase(0)

#%%------------------------------------------------------------------
# Load data.

#dataset_home_dir_path = "/home/sangwook/my_dataset"
#dataset_home_dir_path = "/home/HDD1/sangwook/my_dataset"
dataset_home_dir_path = "D:/dataset"

dataset_dir_path = dataset_home_dir_path + "/phenotyping/cvppp2017_lsc_lcc_challenge/package/CVPPP2017_LSC_training/training/A1"

image_loader = CvpppImageLoader()
train_images, train_labels = image_loader.load(dataset_dir_path, img_suffix = '_rgb', img_extension = 'png', label_suffix = '_label', label_extension = 'png')

# Change the dimension of labels.
if train_images.ndim == train_labels.ndim:
    pass
elif 1 == train_images.ndim - train_labels.ndim:
    train_labels = train_labels.reshape(train_labels.shape + (1,))
else:
    raise ValueError('train_images.ndim or train_labels.ndim is invalid.')

# Change labels from grayscale values to indexes.
for train_label in train_labels:
    unique_lbls = np.unique(train_label).tolist()
    for lbl in unique_lbls:
        train_label[train_label == lbl] = unique_lbls.index(lbl)

assert train_images.shape[0] == train_labels.shape[0] and train_images.shape[1] == train_labels.shape[1] and train_images.shape[2] == train_labels.shape[2], "ERROR: Image and label size mismatched."

batch_size = 32
steps_per_epoch = 2000
num_epoch = 50

#%%------------------------------------------------------------------
# Create a data generator.

# REF [site] >> https://keras.io/preprocessing/image/

data_generator = ImageDataGenerator(
	rescale=1./255.,
	featurewise_center=True,
	featurewise_std_normalization=True,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	vertical_flip=True)

data_generator.fit(train_images)

#%%------------------------------------------------------------------
# Create a U-Net model.

# REF [site] >> https://github.com/zizhaozhang/unet-tensorflow-keras
# REF [file] >> https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/train.py

train_image_shape = (None,) + train_images.shape[1:]
train_label_shape = (None,) + train_labels.shape[1:]
train_images_tf = tf.placeholder(tf.float32, shape=train_image_shape)
train_labels_tf = tf.placeholder(tf.float32, shape=train_label_shape)

with tf.name_scope('unet'):
    unet_model = UNet().create_model(train_image_shape, backend='tf', tf_input=train_images_tf)

#%%------------------------------------------------------------------
# Train the U-Net model.

# Define a loss.
with tf.name_scope('cross_entropy'):
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_labels_tf, logits=unet_model))

# Define an optimzer.
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(0.0001, global_step, steps_per_epoch*3, 0.5, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, global_step=global_step)

# Compute dice score for simple evaluation during training.
with tf.name_scope('dice_eval'):
    dice_evaluator = tf.reduce_mean(dice_coeff(train_labels_tf, unet_model))
