# REF [paper] >> "Densely Connected Convolutional Networks", arXiv 2016.
# REF [paper] >> "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation", arXiv 2016.
# REF [site] >> https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation

# Path to libcudnn.so.5.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#%%------------------------------------------------------------------

import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
	lib_home_dir_path = "/home/sangwook/lib_repo/python"
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
	lib_home_dir_path = "D:/lib_repo/python"
	#lib_home_dir_path = "D:/lib_repo/python/rnd"
lib_dir_path = lib_home_dir_path + "/Fully-Connected-DenseNets-Semantic-Segmentation_github"

sys.path.append(swl_python_home_dir_path + '/src')
sys.path.append(lib_dir_path)

os.chdir(swl_python_home_dir_path + '/test/machine_learning/keras')

#%%------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
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

if 'posix' == os.name:
	#dataset_home_dir_path = "/home/sangwook/my_dataset"
	dataset_home_dir_path = "/home/HDD1/sangwook/my_dataset"
else:
	dataset_home_dir_path = "D:/dataset"

train_data_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/train"
train_label_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/trainannot"
validation_data_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/val"
validation_label_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/valannot"
test_data_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/test"
test_label_dir_path = dataset_home_dir_path + "/pattern_recognition/camvid/tmp/testannot"

model_dir_path = './result/fc_densenet/model'
prediction_dir_path = './result/foreign_body_fc_densenet/prediction'
train_summary_dir_path = './log/fc_densenet/train'
test_summary_dir_path = './log/fc_densenet/test'

if not os.path.exists(model_dir_path):
	try:
		os.makedirs(model_dir_path)
	except OSError as exception:
		if exception.errno != os.errno.EEXIST:
			raise
if not os.path.exists(prediction_dir_path):
	try:
		os.makedirs(prediction_dir_path)
	except OSError as exception:
		if exception.errno != os.errno.EEXIST:
			raise
if not os.path.exists(train_summary_dir_path):
	try:
		os.makedirs(train_summary_dir_path)
	except OSError as exception:
		if exception.errno != os.errno.EEXIST:
			raise
if not os.path.exists(test_summary_dir_path):
	try:
		os.makedirs(test_summary_dir_path)
	except OSError as exception:
		if exception.errno != os.errno.EEXIST:
			raise

# NOTICE [caution] >>
#	If the size of data is changed, labels in label images may be changed.

#data_loader = DataLoader()
##data_loader = DataLoader(width=480, height=360)
#train_dataset = data_loader.load(data_dir_path=train_data_dir_path, label_dir_path=train_label_dir_path, data_extension ='png', label_extension='png')
#validation_dataset = data_loader.load(data_dir_path=validation_data_dir_path, label_dir_path=validation_label_dir_path, data_extension ='png', label_extension='png')
#test_dataset = data_loader.load(data_dir_path=test_data_dir_path, label_dir_path=test_label_dir_path, data_extension ='png', label_extension='png')
#
## Change the dimension of labels.
#if train_dataset.data.ndim == train_dataset.labels.ndim:
#	pass
#elif 1 == train_dataset.data.ndim - train_dataset.labels.ndim:
#	train_dataset.labels = train_dataset.labels.reshape(train_dataset.labels.shape + (1,))
#else:
#	raise ValueError('train_dataset.data.ndim or train_dataset.labels.ndim is invalid.')
#if validation_dataset.data.ndim == validation_dataset.labels.ndim:
#	pass
#elif 1 == validation_dataset.data.ndim - validation_dataset.labels.ndim:
#	validation_dataset.labels = validation_dataset.labels.reshape(validation_dataset.labels.shape + (1,))
#else:
#	raise ValueError('validation_dataset.data.ndim or validation_dataset.labels.ndim is invalid.')
#if test_dataset.data.ndim == test_dataset.labels.ndim:
#	pass
#elif 1 == test_dataset.data.ndim - test_dataset.labels.ndim:
#	test_dataset.labels = test_dataset.labels.reshape(test_dataset.labels.shape + (1,))
#else:
#	raise ValueError('test_dataset.data.ndim or test_dataset.labels.ndim is invalid.')

#%%------------------------------------------------------------------

keras_backend = 'tf'

#num_examples = train_dataset.num_examples
num_examples = 0
#num_classes = np.unique(train_dataset.labels).shape[0]
num_classes = 12  # 11 + 1.

batch_size = 10
num_epochs = 1000
steps_per_epoch = num_examples // batch_size if num_examples > 0 else 50
if steps_per_epoch < 1:
	steps_per_epoch = 1

#resized_input_size = train_dataset.data.shape[1:3]  # (height, width).
resized_input_size = (360, 480)  # (height, width).
cropped_input_size = (224, 224)  # (height, width).

#tf_data_shape = (None,) + cropped_input_size + (train_dataset.data.shape[3],)
tf_data_shape = (None,) + cropped_input_size + (3,)
#tf_label_shape = (None,) + cropped_input_size + (1 if 2 == num_classes else num_classes,)
tf_label_shape = (None,) + cropped_input_size + (num_classes,)
tf_data_ph = tf.placeholder(tf.float32, shape=tf_data_shape)
tf_label_ph = tf.placeholder(tf.float32, shape=tf_label_shape)

# Convert label types from uint16 to float32, and convert label IDs to one-hot encoding.
#train_dataset.labels = train_dataset.labels.astype(np.float32)
#validation_dataset.labels = validation_dataset.labels.astype(np.float32)
#test_dataset.labels = test_dataset.labels.astype(np.float32)
## NOTICE [info] >> Axis 3 has to be 1, 3, or 4 for label images to be transformed by ImageDataGenerator.
##if num_classes > 2:
##	train_dataset.labels = keras.utils.to_categorical(train_dataset.labels, num_classes).reshape(train_dataset.labels.shape[:-1] + (-1,))
##	validation_dataset.labels = keras.utils.to_categorical(validation_dataset.labels, num_classes).reshape(validation_dataset.labels.shape[:-1] + (-1,))
##	test_dataset.labels = keras.utils.to_categorical(test_dataset.labels, num_classes).reshape(test_dataset.labels.shape[:-1] + (-1,))

#%%------------------------------------------------------------------
# Create a data generator.

# REF [site] >> https://keras.io/preprocessing/image/
# REF [site] >> https://github.com/fchollet/keras/issues/3338

print('Create a data generator.')

train_data_generator = ImageDataGeneratorWithCrop(
	rescale=1./255.,
	preprocessing_function=None,
	featurewise_center=False,
	featurewise_std_normalization=False,
	samplewise_center=False,
	samplewise_std_normalization=False,
	zca_whitening=False,
	zca_epsilon=1e-6,
	#rotation_range=20,
	#width_shift_range=0.2,
	#height_shift_range=0.2,
	#horizontal_flip=True,
	vertical_flip=True,
	#zoom_range=0.2,
	#shear_range=0.,
	#channel_shift_range=0.,
	random_crop_size=cropped_input_size,
	center_crop_size=None,
	fill_mode='reflect',
	cval=0.)
train_label_generator = ImageDataGeneratorWithCrop(
	#rescale=1./255.,
	#preprocessing_function=None,
	#featurewise_center=False,
	#featurewise_std_normalization=False,
	#samplewise_center=False,
	#samplewise_std_normalization=False,
	#zca_whitening=False,
	#zca_epsilon=1e-6,
	#rotation_range=20,
	#width_shift_range=0.2,
	#height_shift_range=0.2,
	#horizontal_flip=True,
	vertical_flip=True,
	#zoom_range=0.2,
	#shear_range=0.,
	#channel_shift_range=0.,
	random_crop_size=cropped_input_size,
	center_crop_size=None,
	fill_mode='reflect',
	cval=0.)
test_data_generator = ImageDataGeneratorWithCrop(rescale=1./255.)
test_label_generator = ImageDataGeneratorWithCrop()

# Provide the same seed and keyword arguments to the fit and flow methods.
seed = 1

# Compute the internal data stats related to the data-dependent transformations, based on an array of sample data.
# Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
#train_data_generator.fit(train_dataset.data, augment=True, seed=seed)
##train_label_generator.fit(train_dataset.labels, augment=True, seed=seed)
##test_data_generator.fit(test_dataset.data, augment=True, seed=seed)
##test_label_generator.fit(test_dataset.labels, augment=True, seed=seed)

train_data_gen = train_data_generator.flow_from_directory(
	train_data_dir_path,
	target_size=resized_input_size,
	color_mode='rgb',
	#classes=None,
	class_mode=None,  # NOTICE [important] >>
	batch_size=batch_size,
	shuffle=True,
	seed=seed)
train_label_gen = train_label_generator.flow_from_directory(
	train_label_dir_path,
	target_size=resized_input_size,
	color_mode='grayscale',
	#classes=None,
	class_mode=None,  # NOTICE [important] >>
	batch_size=batch_size,
	shuffle=True,
	seed=seed)
validation_data_gen = test_data_generator.flow_from_directory(
	validation_data_dir_path,
	target_size=resized_input_size,
	color_mode='rgb',
	#classes=None,
	class_mode=None,  # NOTICE [important] >>
	batch_size=batch_size,
	shuffle=True,
	seed=seed)
validation_label_gen = test_label_generator.flow_from_directory(
	validation_label_dir_path,
	target_size=resized_input_size,
	color_mode='grayscale',
	#classes=None,
	class_mode=None,  # NOTICE [important] >>
	batch_size=batch_size,
	shuffle=True,
	seed=seed)
test_data_gen = test_data_generator.flow_from_directory(
	test_data_dir_path,
	target_size=resized_input_size,
	color_mode='rgb',
	#classes=None,
	class_mode=None,  # NOTICE [important] >>
	batch_size=num_examples if num_examples > 0 else 100,
	shuffle=True,
	seed=seed)
test_label_gen = test_label_generator.flow_from_directory(
	test_label_dir_path,
	target_size=resized_input_size,
	color_mode='grayscale',
	#classes=None,
	class_mode=None,  # NOTICE [important] >>
	batch_size=num_examples if num_examples > 0 else 100,
	shuffle=True,
	seed=seed)

# Combine generators into one which yields image and labels.
train_dataset_gen = zip(train_data_gen, train_label_gen)
validation_dataset_gen = zip(validation_data_gen, validation_label_gen)
test_dataset_gen = zip(test_data_gen, test_label_gen)

#%%------------------------------------------------------------------
# Create a FC-DenseNet model.

print('Create a FC-DenseNet model.')

with tf.name_scope('fc-densenet'):
	fc_densenet_model = dc.DenseNetFCN(tf_data_shape[1:], nb_dense_block=5, growth_rate=16, nb_layers_per_block=4, upsampling_type='upsampling', classes=num_classes)
fc_densenet_model_output = fc_densenet_model(tf_data_ph)

#%%------------------------------------------------------------------
# Display.

#fc_densenet_model.summary()

#%%------------------------------------------------------------------
# Prepare training.

print('Prepare training.')

# Define a loss.
with tf.name_scope('loss'):
	#loss = tf.reduce_mean(keras.objectives.categorical_crossentropy(tf_label_ph, fc_densenet_model_output))  # Error.
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_label_ph, logits=fc_densenet_model_output))
	tf.summary.scalar('loss', loss)

# Define a metric.
with tf.name_scope('metric'):
	#metric = keras.metrics.categorical_accuracy(tf_label_ph, fc_densenet_model_output)
	correct_prediction = tf.equal(tf.argmax(fc_densenet_model_output, 1), tf.argmax(tf_label_ph, 1))
	metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('metric', metric)

# Define an optimzer.
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
	learning_rate = tf.train.exponential_decay(1.0e-3, global_step, 1, 0.995, staircase=True)
	tf.summary.scalar('learning_rate', learning_rate)
train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

# Merge all the summaries and write them out to a directory.
merged_summary = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter(train_summary_dir_path, sess.graph)
test_summary_writer = tf.summary.FileWriter(test_summary_dir_path)

# Saves a model every 2 hours and maximum 5 latest models are saved.
saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

#%%------------------------------------------------------------------
# Train the FC-DenseNet model.

TRAINING_MODE = 0  # Start training a model.
#TRAINING_MODE = 1  # Resume training a model.
#TRAINING_MODE = 2  # Use a trained model.

if 0 == TRAINING_MODE:
	print('Start training...')
elif 1 == TRAINING_MODE:
	print('Resume training...')
elif 2 == TRAINING_MODE:
	print('Use a trained model.')
else:
	raise Exception('Invalid TRAINING_MODE')

# Use Keras ==> Cannot train.
#fc_densenet_model.fit_generator(
#	train_generator,
#	steps_per_epoch=steps_per_epoch,
#	epochs=num_epochs,
#	validation_data=validation_generator,
#	validation_steps=800)

# Use TensorFlow.
# REF [site] >> https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

# Initialize all variables.
sess.run(tf.global_variables_initializer())

# Run training loop.
with sess.as_default():
	if 1 == TRAINING_MODE or 2 == TRAINING_MODE:
		# Restore the model.
		# REF [site] >> http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		print('Restore a FC-DenseNet model.')

		ckpt = tf.train.get_checkpoint_state(model_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(model_dir_path))
		print('Model restored from directory:', model_dir_path)

	if 0 == TRAINING_MODE or 1 == TRAINING_MODE:
		for epoch in range(1, num_epochs + 1):
			print('Epoch %d/%d' % (epoch, num_epochs))
			steps = 0
			for data_batch, label_batch in train_dataset_gen:
				if num_classes >= 2:
					label_batch = keras.utils.to_categorical(label_batch, num_classes).reshape(label_batch.shape[:-1] + (-1,))
				summary, _ = sess.run([merged_summary, train_step], feed_dict={tf_data_ph: data_batch, tf_label_ph: label_batch})
				train_summary_writer.add_summary(summary, epoch)
				#print('data batch: (shape, dtype, min, max) =', data_batch.shape, data_batch.dtype, np.min(data_batch), np.max(data_batch))
				#print('label batch: (shape, dtype, min, max) =', label_batch.shape, label_batch.dtype, np.min(label_batch), np.max(label_batch))
				steps += 1
				if steps >= steps_per_epoch:
					break
			if 0 == epoch % 10:
				for data_batch, label_batch in train_dataset_gen:
					if num_classes >= 2:
						label_batch = keras.utils.to_categorical(label_batch, num_classes).reshape(label_batch.shape[:-1] + (-1,))
					break;
				summary, test_metric = sess.run([merged_summary, metric], feed_dict={tf_data_ph: data_batch, tf_label_ph: label_batch})
				test_summary_writer.add_summary(summary, epoch)
				print('Epoch %d: test metric = %g' % (epoch, test_metric))

			# Save the model.
			if 0 == epoch % 10:
				model_saved_path = saver.save(sess, model_dir_path + '/fc_densenet.ckpt', global_step=global_step)
				print('Model saved in file:', model_saved_path)

print('End training...')

#%%------------------------------------------------------------------
# Evaluate the FC-DenseNet model.

print('Start testing...')

import matplotlib.pyplot as plt

with sess.as_default():
	for data_batch, label_batch in test_dataset_gen:
		if num_classes >= 2:
			label_batch = keras.utils.to_categorical(label_batch, num_classes).reshape(label_batch.shape[:-1] + (-1,))
		break
	test_metric = metric.eval(feed_dict={tf_data_ph: data_batch, tf_label_ph: label_batch})
	print('Test metric = %g' % test_metric)

	predictions = fc_densenet_model.predict(data_batch)
	for idx in range(predictions.shape[0]):
		prediction = np.argmax(predictions[idx], axis=2)

		plt.imshow(prediction, cmap='gray')
		plt.imsave(prediction_dir_path + '/prediction' + str(idx) + '.jpg', prediction, cmap='gray')

print('End testing...')

#%%------------------------------------------------------------------

train_summary_writer.close()
test_summary_writer.close()
