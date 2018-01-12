import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
	lib_home_dir_path = "/home/sangwook/lib_repo/python"
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
	#lib_home_dir_path = "D:/lib_repo/python"
	lib_home_dir_path = "D:/lib_repo/python/rnd"
lib_dir_path = lib_home_dir_path + "/tflearn_github"

sys.path.append(swl_python_home_dir_path + '/src')
sys.path.append(lib_dir_path)

#-------------------
import tflearn
import tensorflow as tf

#%%------------------------------------------------------------------

class TfLearnCnnModel:
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.model_output = None

		#tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

	def __call__(self, input_tensor, is_training=True):
		self.model_output = self._create_model(input_tensor, self.num_classes, is_training)
		return self.model_output

	def train(self, train_data, train_labels, batch_size, num_epochs, shuffle, initial_epoch=0):
		pass
		#return history

	def load(self, model_filepath):
		pass

	def save(self, model_filepath):
		pass

	def _create_model(self, input_tensor, num_classes, is_training=True):
		keep_prob = 0.25 if is_training is True else 1.0

		with tf.variable_scope('tflearn_cnn_model', reuse=True):
			#net = tflearn.input_data(shape=input_shape)

			net = tflearn.conv_2d(input_tensor, nb_filter=32, filter_size=5, strides=1, padding='same', activation='relu', name='conv1_1')
			net = tflearn.max_pool_2d(net, kernel_size=2, strides=2, name='maxpool1_1')

			net = tflearn.conv_2d(net, nb_filter=64, filter_size=3, strides=1, padding='same', activation='relu', name='conv2_1')
			net = tflearn.max_pool_2d(net, kernel_size=2, strides=2, name='maxpool2_1')

			net = tflearn.flatten(net)

			net = tflearn.fully_connected(net, n_units=1024, activation='relu', name='fc1_1')
			net = tflearn.dropout(net, keep_prob=keep_prob, name='dropout1_1')

			if 2 == num_classes:
				net = tflearn.fully_connected(net, n_units=num_classes, activation='sigmoid', name='fc2_1')
			else:
				net = tflearn.fully_connected(net, n_units=num_classes, activation='softmax', name='fc2_1')

			return net
