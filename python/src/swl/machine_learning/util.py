import keras
import numpy as np
import math

def to_one_hot_encoding(label_indexes, num_classes=None):
	if None == num_classes:
		num_classes = np.max(label_indexes) + 1
	elif num_classes <= np.max(label_indexes):
		raise ValueError('num_classes has to be greater than np.max(label_indexes)')
	if 1 == label_indexes.ndim:
		return np.eye(num_classes)[label_indexes]
		#return np.transpose(np.eye(num_classes)[label_indexes])
	else:
		return np.eye(num_classes)[label_indexes].reshape(label_indexes.shape[:-1] + (-1,))
		#return np.transpose(np.eye(num_classes)[label_indexes].reshape(label_indexes.shape[:-1] + (-1,)))

# Time-based learning rate schedule.
# REF [site] >> http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def time_based_learning_rate(epoch, initial_learning_rate, decay_rate):
	return initial_learning_rate / (1.0 + decay_rate * epoch)

# Drop-based learning rate schedule.
# REF [site] >> http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def drop_based_learning_rate(epoch, initial_learning_rate, drop_rate, epoch_drop):
	return initial_learning_rate * math.pow(drop_rate, math.floor((1.0 + epoch) / epoch_drop))

def generate_batch_from_dataset(X, Y, batch_size, shuffle=False):
	num_steps = np.ceil(len(X) / batch_size).astype(np.int)
	if shuffle is True:
		indexes = np.arange(len(X))
		np.random.shuffle(indexes)
		for idx in range(num_steps):
			batch_x = X[indexes[idx*batch_size:(idx+1)*batch_size]]
			batch_y = Y[indexes[idx*batch_size:(idx+1)*batch_size]]
			#yield({'input': batch_x}, {'output': batch_y})
			yield(batch_x, batch_y)
	else:
		for idx in range(num_steps):
			batch_x = X[idx*batch_size:(idx+1)*batch_size]
			batch_y = Y[idx*batch_size:(idx+1)*batch_size]
			#yield({'input': batch_x}, {'output': batch_y})
			yield(batch_x, batch_y)

def generate_batch_from_image_augmentation_sequence(seq, X, Y, batch_size, shuffle=False):
	while True:
		seq_det = seq.to_deterministic()  # Call this for each batch again, NOT only once at the start.
		X_aug = seq_det.augment_images(X)
		Y_aug = seq_det.augment_images(Y)

		num_steps = np.ceil(len(X) / batch_size).astype(np.int)
		if shuffle is True:
			indexes = np.arange(len(X_aug))
			np.random.shuffle(indexes)
			for idx in range(num_steps):
				batch_x = X_aug[indexes[idx*batch_size:(idx+1)*batch_size]]
				batch_y = Y_aug[indexes[idx*batch_size:(idx+1)*batch_size]]
				#yield({'input': batch_x}, {'output': batch_y})
				yield(batch_x, batch_y)
		else:
			for idx in range(num_steps):
				batch_x = X_aug[idx*batch_size:(idx+1)*batch_size]
				batch_y = Y_aug[idx*batch_size:(idx+1)*batch_size]
				#yield({'input': batch_x}, {'output': batch_y})
				yield(batch_x, batch_y)
