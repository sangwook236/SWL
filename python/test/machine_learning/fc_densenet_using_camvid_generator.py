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

#%%------------------------------------------------------------------

import math
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import models
from keras import optimizers, callbacks
import densenet_fc as dc
import matplotlib.pyplot as plt
from swl.machine_learning.camvid_dataset import create_camvid_generator2

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

#keras_backend = 'tf'

#%%------------------------------------------------------------------
# Prepare directories.

output_dir_path = './result/fc_densenet_using_camvid_generator'
log_dir_path = './log/fc_densenet_using_camvid_generator'

model_dir_path = output_dir_path + '/model'
prediction_dir_path = output_dir_path + '/prediction'
train_summary_dir_path = log_dir_path + '/train'
test_summary_dir_path = log_dir_path + '/test'

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

model_checkpoint_best_filepath = model_dir_path + "/fc_densenet_using_camvid_generator_best.hdf5"  # For a best model.
model_checkpoint_filepath = model_dir_path + "/fc_densenet_using_camvid_generator_weight_{epoch:02d}-{val_loss:.2f}.hdf5"
model_json_filepath = model_dir_path + "/fc_densenet_using_camvid_generator.json"
model_weight_filepath = model_dir_path + "/fc_densenet_using_camvid_generator_weight.hdf5"
#model_filepath = model_dir_path + "/fc_densenet_using_camvid_generator_epoch{}.hdf5"  # For a full model.
model_filepath = model_checkpoint_best_filepath

#%%------------------------------------------------------------------
# Parameters.

np.random.seed(7)

num_examples = 367
num_classes = 12  # 11 + 1.

batch_size = 11  # Number of samples per gradient update.
num_epochs = 500  # Number of times to iterate over training data.
steps_per_epoch = num_examples // batch_size if num_examples > 0 else 50
if steps_per_epoch < 1:
	steps_per_epoch = 1

shuffle = False

max_queue_size = 10
workers = 4
use_multiprocessing = False

#%%------------------------------------------------------------------
# Prepare dataset.

if 'posix' == os.name:
	#dataset_home_dir_path = '/home/sangwook/my_dataset'
	dataset_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	dataset_home_dir_path = 'D:/dataset'

train_image_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/train/image'
train_label_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/trainannot/image'
val_image_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/val/image'
val_label_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/valannot/image'
test_image_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/test/image'
test_label_dir_path = dataset_home_dir_path + '/pattern_recognition/camvid/tmp/testannot/image'

image_suffix = ''
image_extension = 'png'
label_suffix = ''
label_extension = 'png'

original_image_size = (360, 480)  # (height, width).
#resized_image_size = None
resized_image_size = (224, 224)  # (height, width).
random_crop_size = None
#random_crop_size = (224, 224)  # (height, width).
center_crop_size = None

if center_crop_size is not None:
	image_size = center_crop_size
elif random_crop_size is not None:
	image_size = random_crop_size
elif resized_image_size is not None:
	image_size = resized_image_size
else:
	image_size = original_image_size
image_shape = image_size + (3,)

use_loaded_dataset = True

# Provide the same seed and keyword arguments to the fit and flow methods.
seed = 1

# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/keras/camvid_dataset_test.py
#train_dataset_gen, val_dataset_gen, test_dataset_gen = create_camvid_generator(
#		train_image_dir_path, train_label_dir_path, val_image_dir_path, val_label_dir_path, test_image_dir_path, test_label_dir_path,
#		data_suffix=image_suffix, data_extension=image_extension, label_suffix=label_suffix, label_extension=label_extension,
#		batch_size=batch_size, resized_image_size=resized_image_size, random_crop_size=random_crop_size, center_crop_size=center_crop_size, use_loaded_dataset=use_loaded_dataset, shuffle=shuffle, seed=seed)
train_dataset_gen, val_dataset_gen, test_dataset_gen = create_camvid_generator2(
		train_image_dir_path, train_label_dir_path, val_image_dir_path, val_label_dir_path, test_image_dir_path, test_label_dir_path,
		data_suffix=image_suffix, data_extension=image_extension, label_suffix=label_suffix, label_extension=label_extension,
		batch_size=batch_size, width=image_shape[1], height=image_shape[0], shuffle=shuffle)

#%%------------------------------------------------------------------
# Create a FC-DenseNet model.

print('Create a FC-DenseNet model.')

with tf.name_scope('fc-densenet'):
	fc_densenet_model = dc.DenseNetFCN(image_shape, nb_dense_block=5, growth_rate=16, nb_layers_per_block=4, upsampling_type='upsampling', classes=num_classes)

# Display the model summary.
#fc_densenet_model.summary()

#%%------------------------------------------------------------------
# Prepare training.

class_weighting = [
	0.2595,
	0.1826,
	4.5640,
	0.1417,
	0.5051,
	0.3826,
	9.6446,
	1.8418,
	6.6823,
	6.2478,
	3.0,
	7.3614
]

# Learning rate scheduler.
def step_decay(epoch):
	initial_learning_rate = 0.001
	drop = 0.00001
	epochs_drop = 10.0
	learning_rate = initial_learning_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
	return learning_rate

# Learning schedule callback.
learning_rate_callback = callbacks.LearningRateScheduler(step_decay)

# Checkpoint.
tensor_board_callback = callbacks.TensorBoard(log_dir=train_summary_dir_path, histogram_freq=5, write_graph=True, write_images=True)
reduce_lr_on_plateau_callback = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

model_checkpoint_callback = callbacks.ModelCheckpoint(model_checkpoint_best_filepath, monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=False, mode='max')
#model_checkpoint_callback = callbacks.ModelCheckpoint(model_checkpoint_filepath, monitor='val_acc', verbose=2, save_best_only=False, save_weights_only=False, mode='max')

# NOTICE [caution] >> Out of memory.
#callback_list = [learning_rate_callback, tensor_board_callback, reduce_lr_on_plateau_callback, model_checkpoint_callback]
#callback_list = [tensor_board_callback, model_checkpoint_callback]
callback_list = [model_checkpoint_callback]

#optimizer = optimizers.SGD(lr=0.01, decay=1.0e-7, momentum=0.95, nesterov=False)
optimizer = optimizers.RMSprop(lr=1.0e-5, decay=1.0e-9, rho=0.9, epsilon=1.0e-8)
#optimizer = optimizers.Adagrad(lr=0.01, decay=1.0e-7, epsilon=1.0e-8)
#optimizer = optimizers.Adadelta(lr=1.0, decay=0.0, rho=0.95, epsilon=1.0e-8)
#optimizer = optimizers.Adam(lr=1.0e-5, decay=1.0e-9, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)
#optimizer = optimizers.Adamax(lr=0.002, decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)
#optimizer = optimizers.Nadam(lr=0.002, schedule_decay=0.004, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8)

#%%------------------------------------------------------------------
# Train the FC-DenseNet model.

TRAINING_MODE = 0  # Start training a model.
#TRAINING_MODE = 1  # Resume training a model.
#TRAINING_MODE = 2  # Use a trained model.

if 0 == TRAINING_MODE:
	initial_epoch = 0
	print('Start training...')
elif 1 == TRAINING_MODE:
	initial_epoch = 1000
	print('Resume training...')
elif 2 == TRAINING_MODE:
	initial_epoch = 0
	print('Use a trained model.')
else:
	raise Exception('Invalid TRAINING_MODE')

if 1 == TRAINING_MODE or 2 == TRAINING_MODE:
	# Deserialize a model from JSON.
	#with open(model_json_filepath, 'r') as json_file:
	#	fc_densenet_model = models.model_from_json(json_file.read())
	# Deserialize weights into the model.
	#fc_densenet_model.load_weights(model_weight_filepath)
	# Load a full model.
	fc_densenet_model = models.load_model(model_filepath)
	#fc_densenet_model = models.load_model(model_filepath.format(num_epochs))

	print('Restored a FC-DenseNet model.')

if 0 == TRAINING_MODE or 1 == TRAINING_MODE:
	fc_densenet_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	history = fc_densenet_model.fit_generator(train_dataset_gen,
			steps_per_epoch=steps_per_epoch, epochs=num_epochs, initial_epoch=initial_epoch,
			#validation_data=val_dataset_gen, validation_steps=steps_per_epoch,
			validation_data=test_dataset_gen, validation_steps=steps_per_epoch,
			#max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing,
			class_weight=class_weighting, callbacks=callback_list, verbose=1)

	# List all data in history.
	print(history.history.keys())

	# Summarize history for accuracy.
	fig = plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	fig.savefig(output_dir_path + '/model_accuracy.png')
	plt.close(fig)
	# Summarize history for loss.
	fig = plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	fig.savefig(output_dir_path + '/model_loss.png')
	plt.close(fig)

	# Serialize a model to JSON.
	#with open(model_json_filepath, 'w') as json_file:
	#	json_file.write(fc_densenet_model.to_json())
	# Serialize weights to HDF5.
	#fc_densenet_model.save_weights(model_weight_filepath)  # Save the model's weights.
	# Save a full model.
	#fc_densenet_model.save(model_filepath.format(num_epochs))

	print('Saved a FC-DenseNet model.')

if 0 == TRAINING_MODE or 1 == TRAINING_MODE:
	print('End training...')

#%%------------------------------------------------------------------
# Evaluate the FC-DenseNet model.

print('Start testing...')

num_test_examples = 233
steps_per_epoch = num_test_examples // batch_size if num_test_examples > 0 else 50
if steps_per_epoch < 1:
	steps_per_epoch = 1

test_loss, test_accuracy = fc_densenet_model.evaluate_generator(test_dataset_gen, steps=steps_per_epoch) #, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
print('Test loss = {}, test accuracy = {}'.format(test_loss, test_accuracy))

print('End testing...')

#%%------------------------------------------------------------------
# Predict.

print('Start prediction...')

predictions = fc_densenet_model.predict_generator(test_dataset_gen, steps=steps_per_epoch, verbose=0) #, max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)

for idx in range(predictions.shape[0]):
	prediction = np.argmax(predictions[idx], axis=-1)

	#plt.imshow(prediction, cmap='gray')
	plt.imsave(prediction_dir_path + '/prediction' + str(idx) + '.jpg', prediction, cmap='gray')

print('End prediction...')

#%%------------------------------------------------------------------
# Display.

for batch_images, batch_labels in test_dataset_gen:
	break
batch_predictions = fc_densenet_model.predict(batch_images, batch_size=batch_size, verbose=0)

idx = 0
#plt.figure(figsize=(7,7))
plt.subplot(131)
plt.imshow((batch_images[idx] - np.min(batch_images[idx])) / (np.max(batch_images[idx]) - np.min(batch_images[idx])))
plt.subplot(132)
plt.imshow(np.argmax(batch_labels[idx], axis=-1), cmap='gray')
plt.subplot(133)
plt.imshow(np.argmax(batch_predictions[idx], axis=-1), cmap='gray')
