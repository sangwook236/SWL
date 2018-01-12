from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers, callbacks

#%%------------------------------------------------------------------

class KerasCnnModel:
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.model_output = None

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
		input_shape = input_tensor.get_shape()

		model = Sequential()
		model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Flatten())

		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(keep_prob))

		if 2 == num_classes:
			model.add(Dense(1, activation='sigmoid'))
			#model.add(Dense(1, activation='sigmoid', activity_regularizer=keras.regularizers.activity_l2(0.0001)))
		else:
			model.add(Dense(num_classes, activation='softmax'))
			#model.add(Dense(num_classes, activation='softmax', activity_regularizer=keras.regularizers.activity_l2(0.0001)))

		return model(input_tensor)
