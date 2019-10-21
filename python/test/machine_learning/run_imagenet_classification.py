#!/usr/bin/env python

import os, math, time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras_applications
import torch
import torchvision
import PIL.Image

class MyDataSequence(tf.keras.utils.Sequence):
	def __init__(self, data_loader, num_classes):
		self._data_loader = data_loader
		self._num_classes = num_classes

	def __len__(self):
		return len(self._data_loader)

	def __getitem__(self, idx):
		for batch_data in self._data_loader:
			# batch_data: torch.Tensor & torch.Tensor.
			#	REF [function] >> imagenet_dataset_test() in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_data_loading_and_processing.py

			# (batch, channel, height, width) -> (batch, height, width, channel).
			inputs = np.transpose(batch_data[0].numpy(), (0, 2, 3, 1))
			# One-hot encoding.
			outputs = tf.keras.utils.to_categorical(batch_data[1].numpy(), num_classes=self._num_classes)
			return inputs, outputs

def create_data_loader(input_shape, batch_size, shuffle, num_workers=0):
	if 'posix' == os.name:
		#data_dir_path = '/home/sangwook/my_dataset'
		data_dir_path = '/nipa/sangwook/my_dataset'
	else:
		data_dir_path = 'E:/dataset'
	imagenet_dir_path = data_dir_path + '/pattern_recognition/imagenet'

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(size=input_shape[:2], interpolation=PIL.Image.BILINEAR),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(0.5, 0.5, 0.5))
	])

	#--------------------
	train_set = torchvision.datasets.ImageNet(root=imagenet_dir_path, split='train', download=True, transform=transform)
	train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	val_set = torchvision.datasets.ImageNet(root=imagenet_dir_path, split='val', download=True, transform=transform)
	val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

	return train_data_loader, val_data_loader

# REF [function] >> resnet_test() in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/keras/keras_applications_test.py.
def create_resnet(input_shape, num_classes):
	kwargs = {'backend': tf.keras.backend, 'layers': tf.keras.layers, 'models': tf.keras.models, 'utils': tf.keras.utils}

	# ResNet50V2, ResNet101V2, ResNet152V2.
	model = keras_applications.resnet_v2.ResNet50V2(
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		input_shape=input_shape,
		pooling=None,
		classes=num_classes,
		**kwargs
	)
	#print(model.summary())

	return model

# REF [function] >> densenet_test() in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/keras/keras_applications_test.py.
def create_densenet(input_shape, num_classes):
	kwargs = {'backend': tf.keras.backend, 'layers': tf.keras.layers, 'models': tf.keras.models, 'utils': tf.keras.utils}

	# DenseNet121, DenseNet169, DenseNet201.
	model = keras_applications.densenet.DenseNet121(
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		input_shape=input_shape,
		pooling=None,
		classes=num_classes,
		**kwargs
	)
	#print(model.summary())

	return model

# REF [function] >> nasnet_test() in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/keras/keras_applications_test.py.
def create_nasnet(input_shape, num_classes):
	kwargs = {'backend': tf.keras.backend, 'layers': tf.keras.layers, 'models': tf.keras.models, 'utils': tf.keras.utils}

	# NASNetLarge, NASNetMobile.
	model = keras_applications.nasnet.NASNetLarge(
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		input_shape=input_shape,
		pooling=None,
		classes=num_classes,
		**kwargs
	)
	#print(model.summary())

	return model

def train(model, train_data_loader, val_data_loader, num_classes, num_epochs, batch_size, shuffle, model_checkpoint_filepath, num_workers=8):
	loss = tf.keras.losses.categorical_crossentropy

	#optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True)
	#optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.995, amsgrad=False)  # Not good.
	optimizer = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
	lr_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
	csv_logger_callback = tf.keras.callbacks.CSVLogger('./train_log.csv')  # epoch, acc, loss, lr, val_acc, val_loss.
	#callbacks = [model_checkpoint_callback, early_stopping_callback, lr_reduce_callback, csv_logger_callback]
	callbacks = [model_checkpoint_callback, csv_logger_callback]

	train_steps_per_epoch, val_steps_per_epoch = len(train_data_loader), len(val_data_loader)
	train_data_sequence, val_data_sequence = MyDataSequence(train_data_loader, num_classes), MyDataSequence(val_data_loader, num_classes)
	initial_epoch = 0
	class_weights = None
	max_queue_size, use_multiprocessing = 10, True

	history = model.fit_generator(train_data_sequence, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch, validation_data=val_data_sequence, validation_steps=val_steps_per_epoch, shuffle=shuffle, initial_epoch=initial_epoch, class_weight=class_weights, max_queue_size=max_queue_size, workers=num_workers, use_multiprocessing=use_multiprocessing, callbacks=callbacks)

	return history.history

def resnet_imagenet_test():
	input_shape = (224, 224, 3)  # Height, width, channel.
	num_classes = 1000
	num_epochs, batch_size = 10, 32
	shuffle = True
	num_workers = 8
	model_checkpoint_filepath = './resnet_imagenet'

	print('Start creating ImageNet datasets and data loaders...')
	start_time = time.time()
	train_data_loader, val_data_loader = create_data_loader(input_shape, batch_size, shuffle, num_workers=0)
	print('End creating ImageNet datasets and data loaders: {} secs.'.format(time.time() - start_time))

	print('Start creating ResNet...')
	start_time = time.time()
	model = create_resnet(input_shape, num_classes)
	print('End creating ResNet: {} secs.'.format(time.time() - start_time))

	print('Start training ResNet...')
	start_time = time.time()
	history = train(model, train_data_loader, val_data_loader, num_classes, num_epochs, batch_size, shuffle, model_checkpoint_filepath, num_workers)
	print('End training ResNet: {} secs.'.format(time.time() - start_time))

def densenet_imagenet_test():
	input_shape = (224, 224, 3)  # Height, width, channel.
	num_classes = 1000
	num_epochs, batch_size = 10, 32
	shuffle = True
	num_workers = 8
	model_checkpoint_filepath = './densenet_imagenet'

	print('Start creating ImageNet datasets and data loaders...')
	start_time = time.time()
	train_data_loader, val_data_loader = create_data_loader(input_shape, batch_size, shuffle, num_workers=0)
	print('End creating ImageNet datasets and data loaders: {} secs.'.format(time.time() - start_time))

	print('Start creating DenseNet...')
	start_time = time.time()
	model = create_densenet(input_shape, num_classes)
	print('End creating DenseNet: {} secs.'.format(time.time() - start_time))

	print('Start training DenseNet...')
	start_time = time.time()
	history = train(model, train_data_loader, val_data_loader, num_classes, num_epochs, batch_size, shuffle, model_checkpoint_filepath, num_workers)
	print('End training DenseNet: {} secs.'.format(time.time() - start_time))

def nasnet_imagenet_test():
	input_shape = (224, 224, 3)  # Height, width, channel.
	num_classes = 1000
	num_epochs, batch_size = 10, 32
	shuffle = True
	num_workers = 8
	model_checkpoint_filepath = './nasnet_imagenet'

	print('Start creating ImageNet datasets and data loaders...')
	start_time = time.time()
	train_data_loader, val_data_loader = create_data_loader(input_shape, batch_size, shuffle, num_workers=0)
	print('End creating ImageNet datasets and data loaders: {} secs.'.format(time.time() - start_time))

	print('Start creating NASNet...')
	start_time = time.time()
	model = create_nasnet(input_shape, num_classes)
	print('End creating NASNet: {} secs.'.format(time.time() - start_time))

	print('Start training NASNet...')
	start_time = time.time()
	history = train(model, train_data_loader, val_data_loader, num_classes, num_epochs, batch_size, shuffle, model_checkpoint_filepath, num_workers)
	print('End training NASNet: {} secs.'.format(time.time() - start_time))

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	resnet_imagenet_test()
	#densenet_imagenet_test()
	#nasnet_imagenet_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
