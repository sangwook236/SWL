# Path to libcudnn.so.5.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#%%------------------------------------------------------------------

import os
os.chdir('D:/work/swl_github/python/test/machine_learning')

import sys
sys.path.insert(0, '../../src/machine_learning')

#%%------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model, Input
from keras.preprocessing.image import ImageDataGenerator
from deconvnet import DeconvNet
from data_loader import DataLoader
from loss import dice_coeff, dice_coeff_loss

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

train_data_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/train"
train_label_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/trainannot"
validation_data_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/val"
validation_label_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/valannot"
test_data_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/test"
test_label_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/testannot"

# NOTICE [caution] >>
#	If the size of data is changed, labels may be dense.

data_loader = DataLoader()
#data_loader = DataLoader(width=480, height=360)
train_dataset = data_loader.load(data_dir_path=train_data_dir_path, label_dir_path=train_label_dir_path, data_extension ='png', label_extension='png')
validation_dataset = data_loader.load(data_dir_path=validation_data_dir_path, label_dir_path=validation_label_dir_path, data_extension ='png', label_extension='png')
test_dataset = data_loader.load(data_dir_path=test_data_dir_path, label_dir_path=test_label_dir_path, data_extension ='png', label_extension='png')

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
# Create a DeconvNet model.

with tf.name_scope('deconvnet'):
	deconv_model = DeconvNet().create_model(num_classes, backend=keras_backend, input_shape=train_data_shape, tf_input=train_data_tf)
	if 'tf' == keras_backend:
		Model(inputs=Input(tensor=train_data_tf), outputs=deconv_model).summary()
	else:   
		deconv_model.summary()

#%%------------------------------------------------------------------

[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]

#images_placeholder = tf.get_default_graph().get_tensor_by_name("input_10:0")
#seg_score = tf.get_default_graph().get_tensor_by_name('deconvnet\\seg-score:0')

#%%------------------------------------------------------------------
# Train the DeconvNet model.

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
