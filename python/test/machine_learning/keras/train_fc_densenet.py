# REF [paper] >> "Densely Connected Convolutional Networks", arXiv 2016.
# REF [paper] >> "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation", arXiv 2016.
# REF [site] >> https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation

#%%------------------------------------------------------------------

import os
os.chdir('D:/work/swl_github/python/test/machine_learning/keras')

#lib_home_dir_path = "/home/sangwook/lib_repo/python"
lib_home_dir_path = "D:/lib_repo/python"
#lib_home_dir_path = "D:/lib_repo/python/rnd"

lib_dir_path = lib_home_dir_path + "/Fully-Connected-DenseNets-Semantic-Segmentation_github"

import sys
sys.path.append('../../../src')
sys.path.append(lib_dir_path)

#%%------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
import densenet_fc as dc
from swl.machine_learning.keras.preprocessing import ImageDataGeneratorWithCrop
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

# Change the dimension of labels.
if train_dataset.data.ndim == train_dataset.labels.ndim:
    pass
elif 1 == train_dataset.data.ndim - train_dataset.labels.ndim:
    train_dataset.labels = train_dataset.labels.reshape(train_dataset.labels.shape + (1,))
else:
    raise ValueError('train_dataset.data.ndim or train_dataset.labels.ndim is invalid.')
if validation_dataset.data.ndim == validation_dataset.labels.ndim:
    pass
elif 1 == validation_dataset.data.ndim - validation_dataset.labels.ndim:
    validation_dataset.labels = validation_dataset.labels.reshape(validation_dataset.labels.shape + (1,))
else:
    raise ValueError('validation_dataset.data.ndim or validation_dataset.labels.ndim is invalid.')
if test_dataset.data.ndim == test_dataset.labels.ndim:
    pass
elif 1 == test_dataset.data.ndim - test_dataset.labels.ndim:
    test_dataset.labels = test_dataset.labels.reshape(test_dataset.labels.shape + (1,))
else:
    raise ValueError('test_dataset.data.ndim or test_dataset.labels.ndim is invalid.')

#%%------------------------------------------------------------------

keras_backend = 'tf'

num_examples = train_dataset.num_examples
num_classes = np.unique(train_dataset.labels).shape[0]  # 11 + 1.

batch_size = 3
num_epochs = 50
steps_per_epoch = num_examples / batch_size

train_data_shape = (None,) + train_dataset.data.shape[1:]
train_label_shape = (None,) + train_dataset.labels.shape[1:]
train_data_tf = tf.placeholder(tf.float32, shape=train_data_shape)
train_labels_tf = tf.placeholder(tf.float32, shape=train_label_shape)

#%%------------------------------------------------------------------
# Create a data generator.

# REF [site] >> https://keras.io/preprocessing/image/
# REF [site] >> https://github.com/fchollet/keras/issues/3338

train_data_gen_args = dict(rescale=1./255.,
	#featurewise_center=True,
	#featurewise_std_normalization=True,
	vertical_flip=True,
	random_crop_size = (224, 333),  # (height, width).
	center_crop_size = None,
	fill_mode='reflect')
test_data_gen_args = dict(rescale=1./255)

train_data_generator = ImageDataGeneratorWithCrop(**train_data_gen_args)
train_label_generator = ImageDataGeneratorWithCrop(**train_data_gen_args)
test_data_generator = ImageDataGeneratorWithCrop(**test_data_gen_args)
test_label_generator = ImageDataGeneratorWithCrop(**test_data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods.

# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
seed = 1
#train_data_generator.fit(train_dataset.data, augment=True, seed=seed)
#train_label_generator.fit(train_dataset.labels, augment=True, seed=seed)

train_data_gen = train_data_generator.flow_from_directory(
	train_data_dir_path,
	target_size=(360, 480),  # (height, width).
	batch_size=batch_size,
	class_mode=None,
	seed=seed)
train_label_gen = train_label_generator.flow_from_directory(
	train_label_dir_path,
	target_size=(360, 480),  # (height, width).
	batch_size=batch_size,
	class_mode=None,
	seed=seed)
validation_data_gen = test_data_generator.flow_from_directory(
	validation_data_dir_path,
	target_size=(360, 480),  # (height, width).
	batch_size=batch_size,
	class_mode=None,
	seed=seed)
validation_label_gen = test_label_generator.flow_from_directory(
	validation_label_dir_path,
	target_size=(360, 480),  # (height, width).
	batch_size=batch_size,
	class_mode=None,
	seed=seed)
test_data_gen = test_data_generator.flow_from_directory(
	test_data_dir_path,
	target_size=(360, 480),  # (height, width).
	batch_size=batch_size,
	class_mode='categorical')
test_label_gen = test_label_generator.flow_from_directory(
	test_label_dir_path,
	target_size=(360, 480),  # (height, width).
	batch_size=batch_size,
	class_mode=None,
	seed=seed)

# Combine generators into one which yields image and labels
train_generator = zip(train_data_gen, train_label_gen)
validation_generator = zip(validation_data_gen, validation_label_gen)
test_generator = zip(test_data_gen, test_label_gen)

#%%------------------------------------------------------------------
# Create a FC-DenseNet model.

fc_densenet_model = dc.DenseNetFCN((32, 32, 3), nb_dense_block=5, growth_rate=16, nb_layers_per_block=4, upsampling_type='upsampling', classes=num_classes)
fc_densenet_model.summary()

#fc_densenet_model_output = fc_densenet_model(train_data_tf)

#%%------------------------------------------------------------------
# Train the FC-DenseNet model.

fc_densenet_model.fit_generator(
	train_generator,
	steps_per_epoch=steps_per_epoch,
	epochs=num_epochs,
	validation_data=validation_generator,
	validation_steps=800)

#%%------------------------------------------------------------------
# REF [site] >> https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

loss = tf.reduce_mean(categorical_crossentropy(train_labels_tf, fc_densenet_model.output))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
	for i in range(num_epochs):
		batch = mnist_data.train.next_batch(batch_size)
		train_step.run(feed_dict={train_data_tf: batch[0], train_labels_tf: batch[1]})

#%%------------------------------------------------------------------
# Evaluate the FC-DenseNet model.

acc_value = accuracy(train_labels_tf, fc_densenet_model.output)
with sess.as_default():
	print(acc_value.eval(feed_dict={train_data_tf: mnist_data.test.images, train_labels_tf: mnist_data.test.labels}))
