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
