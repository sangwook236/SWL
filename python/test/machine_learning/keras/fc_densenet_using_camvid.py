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

import math
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import models
from keras import optimizers, callbacks
import densenet_fc as dc
import matplotlib.pyplot as plt

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

model_dir_path = './result/fc_densenet_using_camvid/model'
prediction_dir_path = './result/fc_densenet_using_camvid/prediction'
train_summary_dir_path = './log/fc_densenet_using_camvid/train'
test_summary_dir_path = './log/fc_densenet_using_camvid/test'

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

model_checkpoint_best_filepath = model_dir_path + "/fc_densenet_using_camvid_10e-7_decay_best.hdf5"  # For a best model.
model_checkpoint_filepath = model_dir_path + "/fc_densenet_using_camvid_10e-7_decay_weight_{epoch:02d}-{val_loss:.2f}.hdf5"
model_json_filepath = model_dir_path + "/fc_densenet_using_camvid_10e-7_decay.json"
model_weight_filepath = model_dir_path + "/fc_densenet_using_camvid_10e-7_decay_weight.hdf5"
#model_filepath = model_dir_path + "/fc_densenet_using_camvid_10e-7_decay_epoch{}.hdf5"  # For a full model.
model_filepath = model_checkpoint_best_filepath

# REF [file] >> ${SWL_PYTHON_HOME}/test/image_processing/util_test.py
train_data = np.load('./camvid_data/train_images.npy')
train_labels = np.load('./camvid_data/train_labels.npy')
val_data = np.load('./camvid_data/val_images.npy')
val_labels = np.load('./camvid_data/val_labels.npy')
test_data = np.load('./camvid_data/test_images.npy')
test_labels = np.load('./camvid_data/test_labels.npy')

#train_labels = train_labels.reshape((train_labels.shape[0], -1, train_labels.shape[-1]))
#val_labels = val_labels.reshape((val_labels.shape[0], -1, val_labels.shape[-1]))
#test_labels = test_labels.reshape((test_labels.shape[0], -1, test_labels.shape[-1]))

#%%------------------------------------------------------------------

np.random.seed(7)

#keras_backend = 'tf'

num_examples = train_data.shape[0]
#num_classes = np.max([np.max(np.unique(train_labels)), np.max(np.unique(val_labels)), np.max(np.unique(test_labels))]) + 1
num_classes = np.max([train_labels.shape[-1], val_labels.shape[-1], test_labels.shape[-1]])
#num_classes = 12  # 11 + 1.

batch_size = 2  # Number of samples per gradient update.
num_epochs = 1000  # Number of times to iterate over training data.
#steps_per_epoch = num_examples // batch_size if num_examples > 0 else 50
#if steps_per_epoch < 1:
#	steps_per_epoch = 1

#%%------------------------------------------------------------------
# Create a FC-DenseNet model.

print('Create a FC-DenseNet model.')

with tf.name_scope('fc-densenet'):
	fc_densenet_model = dc.DenseNetFCN(train_data.shape[1:], nb_dense_block=5, growth_rate=16, nb_layers_per_block=4, upsampling_type='upsampling', classes=num_classes)

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
#callback_list = [tensor_board_callback, reduce_lr_on_plateau_callback, model_checkpoint_callback]
callback_list = [model_checkpoint_callback]

optimizer = optimizers.RMSprop(lr=0.001, decay=0.0000001)
#optimizer = optimizers.SGD(lr=0.01)
#optimizer = optimizers.Adam(lr=1e-3, decay=0.995)

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
	fc_densenet_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

	history = fc_densenet_model.fit(train_data, train_labels, batch_size=batch_size, epochs=num_epochs,
                    				callbacks=callback_list, class_weight=class_weighting, verbose=1, validation_data=(val_data, val_labels), shuffle=True)  # Validation split = 0.33.

	# List all data in history.
	print(history.history.keys())

	# Summarize history for accuracy.
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# Summarize history for loss.
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

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

score = fc_densenet_model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('End testing...')

#%%------------------------------------------------------------------
# Predict.

predictions = fc_densenet_model.predict(test_data, batch_size=batch_size, verbose=0)

for idx in range(predictions.shape[0]):
	prediction = np.argmax(predictions[idx], axis=-1)

	plt.imshow(prediction, cmap='gray')
	plt.imsave(prediction_dir_path + '/prediction' + str(idx) + '.jpg', prediction, cmap='gray')
