# Path to libcudnn.so.5.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#%%------------------------------------------------------------------

import os
os.chdir('D:/work/swl_github/python/test/machine_learning/keras')

import sys
sys.path.append('../../../src')

#%%------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Input
from keras.preprocessing.image import ImageDataGenerator
from swl.machine_learning.keras.unet import UNet
from swl.machine_learning.keras.loss import dice_coeff, dice_coeff_loss
from swl.machine_learning.data_loader import DataLoader

#%%------------------------------------------------------------------

config = tf.ConfigProto()
#config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4  # only allocate 40% of the total memory of each GPU.
sess = tf.Session(config=config)

# This means that Keras will use the session we registered to initialize all variables that it creates internally.
K.set_session(sess)
K.set_learning_phase(0)

#%%------------------------------------------------------------------
# Load data.

#dataset_home_dir_path = "/home/sangwook/my_dataset"
#dataset_home_dir_path = "/home/HDD1/sangwook/my_dataset"
dataset_home_dir_path = "D:/dataset"

train_dataset_dir_path = dataset_home_dir_path + "/biomedical_imaging/isbi2012_em_segmentation_challenge/train"
test_dataset_dir_path = dataset_home_dir_path + "/biomedical_imaging/isbi2012_em_segmentation_challenge/test"

# NOTICE [caution] >>
#	If the size of data is changed, labels may be dense.

data_loader = DataLoader()
#data_loader = DataLoader(224, 224)
train_dataset = data_loader.load(train_dataset_dir_path, data_suffix='', data_extension='tif', label_suffix='_mask', label_extension='tif')

# Change the dimension of labels.
if train_dataset.data.ndim == train_dataset.labels.ndim:
    pass
elif 1 == train_dataset.data.ndim - train_dataset.labels.ndim:
    train_dataset.labels = train_dataset.labels.reshape(train_dataset.labels.shape + (1,))
else:
    raise ValueError('train_dataset.data.ndim or train_dataset.labels.ndim is invalid.')

# Change labels from grayscale values to indexes.
for train_label in train_dataset.labels:
    unique_lbls = np.unique(train_label).tolist()
    for lbl in unique_lbls:
        train_label[train_label == lbl] = unique_lbls.index(lbl)

#print(np.max(train_dataset.labels))
#print(np.unique(train_dataset.labels).shape[0])
#print(np.max(train_dataset.labels[0]))
#print(np.unique(train_dataset.labels[0]).shape[0])

assert train_dataset.data.shape[0] == train_dataset.labels.shape[0] and train_dataset.data.shape[1] == train_dataset.labels.shape[1] and train_dataset.data.shape[2] == train_dataset.labels.shape[2], "ERROR: Image and label size mismatched."

#%%------------------------------------------------------------------

keras_backend = 'tf'

num_examples = train_dataset.num_examples
num_classes = 2  #np.unique(train_dataset.labels).shape[0]

batch_size = 32
num_epochs = 50
steps_per_epoch = num_examples / batch_size

train_data_shape = (None,) + train_dataset.data.shape[1:]
train_label_shape = (None,) + train_dataset.labels.shape[1:]
train_data_tf = tf.placeholder(tf.float32, shape=train_data_shape)
train_labels_tf = tf.placeholder(tf.float32, shape=train_label_shape)

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

data_generator.fit(train_dataset.data)

#%%------------------------------------------------------------------
# Create a U-Net model.

with tf.name_scope('unet'):
	unet_model = UNet().create_model(num_classes, backend=keras_backend, input_shape=train_data_shape, tf_input=train_data_tf)
	if 'tf' == keras_backend:
		Model(inputs=Input(tensor=train_data_tf), outputs=unet_model).summary()
	else:   
		unet_model.summary()

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
