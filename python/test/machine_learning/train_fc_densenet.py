# REF [paper] >> "Densely Connected Convolutional Networks", arXiv 2016.
# REF [paper] >> "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation", arXiv 2016.
# REF [site] >> https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation

#%%------------------------------------------------------------------

import os
os.chdir('D:/work/swl_github/python/test/machine_learning')

#lib_home_dir_path = "/home/sangwook/lib_repo/python"
#lib_home_dir_path = "D:/lib_repo/python"
lib_home_dir_path = "D:/lib_repo/python/rnd"

lib_dir_path = lib_home_dir_path + "/Fully-Connected-DenseNets-Semantic-Segmentation_github"

import sys
sys.path.insert(0, '../../src/machine_learning')
sys.path.insert(0, lib_dir_path)

#%%------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from tensorflow.examples.tutorials.mnist import input_data
import densenet_fc as dc

#%%------------------------------------------------------------------

keras_backend = 'tf'
num_classes = 10
batch_size = 50
steps_per_epoch = 2000
num_epochs = 50

#%%------------------------------------------------------------------

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# This means that Keras will use the session we registered to initialize all variables that it creates internally.
K.set_session(sess)
K.set_learning_phase(0)

#%%------------------------------------------------------------------
# Load data.

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

#%%------------------------------------------------------------------
# Create a FCN-DenseNet model.

fc_densenet_model = dc.DenseNetFCN((32, 32, 3), nb_dense_block=5, growth_rate=16, nb_layers_per_block=4, upsampling_type='upsampling', classes=num_classes)
fc_densenet_model.summary()

#%%------------------------------------------------------------------

images = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
labels = tf.placeholder(tf.float32, shape=(None, num_classes))
output = fc_densenet_model(images)

#%%------------------------------------------------------------------
# Train the FC-DenseNet model.

# REF [site] >> https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

loss = tf.reduce_mean(categorical_crossentropy(labels, fc_densenet_model.output))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
	for i in range(num_epochs):
		batch = mnist_data.train.next_batch(batch_size)
		train_step.run(feed_dict={images: batch[0], labels: batch[1]})

#%%------------------------------------------------------------------
# Evaluate the FC-DenseNet model.

acc_value = accuracy(labels, fc_densenet_model.output)
with sess.as_default():
	print(acc_value.eval(feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels}))
